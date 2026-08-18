# NeoSVG — per-region gradient fitting
#
# The stacked-layer engine fills every region with ONE flat colour, so a smooth
# ramp can only ever be approximated by a stack of flat bands.  That is visible
# banding by construction: no amount of tuning removes it, because the primitive
# being used (a solid fill) cannot represent a ramp.
#
# This module supplies the missing primitive.  It works on the regions the
# engine has ALREADY segmented, which matters: those contours were traced with
# sub-pixel marching squares and follow the real shape (a curved ribbon, a
# letterform).  A gradient fitted here is clipped to that shape and its axis can
# point in any direction — unlike the tile-based GradientDetector, which can
# only emit 0°/90° ramps over axis-aligned rectangles.
#
# Two steps:
#
#   coalesce_bands()      A smooth ramp arrives as N adjacent near-flat bands.
#                         Fitting one band is pointless — it has no variation to
#                         fit.  Adjacent regions within MERGE_DELTA of each
#                         other are chained by union-find; because each step is
#                         small the chain walks the entire ramp, while a real
#                         colour edge (a large step) stops it.
#
#   fit_region_gradient() Finds the axis along which the region's ORIGINAL
#                         pixels vary most, then samples binned mean colours
#                         along it.  Stops are binned means rather than a
#                         straight-line fit, so a multi-hue ramp
#                         (cyan→blue→purple→magenta) is reproduced exactly
#                         instead of being averaged into mud.
#
# A fit is only accepted when it explains the region's colour better than a flat
# fill would by a wide margin (R² ≥ MIN_R2).  On rejection the caller keeps the
# original bands, so a bad fit can never make the output worse than before.

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from config import Config

logger = logging.getLogger("neosvg.region_gradient")


def _region_mean_colors(labels: np.ndarray, rgb: np.ndarray, n: int) -> np.ndarray:
    """Mean RGB per label, shape (n, 3)."""
    flat   = labels.ravel()
    counts = np.bincount(flat, minlength=n).astype(np.float64)
    means  = np.empty((n, 3), dtype=np.float64)
    for c in range(3):
        means[:, c] = (
            np.bincount(flat, weights=rgb[:, :, c].ravel().astype(np.float64),
                        minlength=n) / np.maximum(counts, 1.0)
        )
    return means


def _adjacency_pairs(labels: np.ndarray, n: int) -> Tuple[np.ndarray, np.ndarray]:
    """Unique 4-connected label pairs, excluding background (label 0)."""
    pa = np.concatenate([labels[:, :-1].ravel(), labels[:-1, :].ravel()])
    pb = np.concatenate([labels[:, 1:].ravel(),  labels[1:, :].ravel()])
    m  = (pa != pb) & (pa != 0) & (pb != 0)
    pa, pb = pa[m], pb[m]
    if len(pa) == 0:
        return np.empty(0, dtype=int), np.empty(0, dtype=int)
    lo   = np.minimum(pa, pb).astype(np.int64)
    hi   = np.maximum(pa, pb).astype(np.int64)
    uniq = np.unique(lo * n + hi)
    return (uniq // n).astype(int), (uniq % n).astype(int)


def coalesce_bands(
    labels: np.ndarray,
    rgb: np.ndarray,
    delta: float,
) -> Optional[np.ndarray]:
    """
    Chain adjacent regions whose mean colours differ by less than `delta` into
    groups, using union-find.

    Returns an array `roots` indexed by label, where roots[L] is the group
    representative of label L (roots[0] = 0, background untouched).  Returns
    None when there is nothing to merge.
    """
    n = int(labels.max()) + 1
    if n <= 2:
        return None

    lo_u, hi_u = _adjacency_pairs(labels, n)
    if len(lo_u) == 0:
        return None

    means = _region_mean_colors(labels, rgb, n)
    dist  = np.linalg.norm(means[lo_u] - means[hi_u], axis=1)

    # Discard over-delta pairs BEFORE sorting — on a detailed image most
    # adjacencies are real edges, and sorting/walking them all dominates the
    # runtime of the whole stage.
    keep = dist <= delta
    if not keep.any():
        return None
    lo_u, hi_u, dist = lo_u[keep], hi_u[keep], dist[keep]

    parent = np.arange(n)

    def find(x: int) -> int:
        r = x
        while parent[r] != r:
            r = parent[r]
        while parent[x] != r:      # path compression
            parent[x], x = r, parent[x]
        return r

    # Merge in ascending colour distance so the smoothest steps bind first and
    # the chain follows the ramp rather than jumping across a weak edge.
    for i in np.argsort(dist):
        ra, rb = find(int(lo_u[i])), find(int(hi_u[i]))
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    # Resolve every label to its root by pointer jumping.  A per-label Python
    # find() is O(n) calls and was costing more than the merge itself.
    roots = parent.copy()
    while True:
        nxt = roots[roots]
        if np.array_equal(nxt, roots):
            break
        roots = nxt
    roots[0] = 0
    return roots.astype(np.int64)


def _binned_stops(
    t: np.ndarray,
    colors: np.ndarray,
    n_stops: int,
) -> Optional[np.ndarray]:
    """
    Mean colour in each of `n_stops` equal bins along normalised position `t`.
    Empty bins are filled by linear interpolation from their filled neighbours.
    Returns (n_stops, 3) or None if too few bins carry data.
    """
    idx = np.clip((t * n_stops).astype(np.int64), 0, n_stops - 1)
    cnt = np.bincount(idx, minlength=n_stops).astype(np.float64)
    out = np.empty((n_stops, 3), dtype=np.float64)
    for c in range(3):
        s = np.bincount(idx, weights=colors[:, c], minlength=n_stops)
        out[:, c] = s / np.maximum(cnt, 1.0)

    filled = cnt > 0
    if filled.sum() < 2:
        return None
    if not filled.all():
        xs = np.arange(n_stops, dtype=np.float64)
        for c in range(3):
            out[:, c] = np.interp(xs, xs[filled], out[filled, c])
    return out


def _axis_residual(
    coords: np.ndarray,
    colors: np.ndarray,
    u: np.ndarray,
    n_stops: int,
) -> Tuple[float, Optional[np.ndarray], float, float]:
    """
    Project pixels onto unit axis `u`, model colour as binned means along it,
    and return (sum of squared residuals, stops, t_min, t_max).
    """
    proj  = coords @ u
    t_min = float(proj.min())
    t_max = float(proj.max())
    span  = t_max - t_min
    if span < 1e-6:
        return float("inf"), None, 0.0, 0.0

    t     = (proj - t_min) / span
    stops = _binned_stops(t, colors, n_stops)
    if stops is None:
        return float("inf"), None, 0.0, 0.0

    # Predict each pixel by interpolating the stop curve at its own position.
    xs   = (np.arange(n_stops, dtype=np.float64) + 0.5) / n_stops
    pred = np.empty_like(colors)
    for c in range(3):
        pred[:, c] = np.interp(t, xs, stops[:, c])

    ss_res = float(np.sum((colors - pred) ** 2))
    return ss_res, stops, t_min, t_max


def _radial_residual(
    coords: np.ndarray,
    colors: np.ndarray,
    centre: np.ndarray,
    n_stops: int,
    theta: float = 0.0,
    aspect: float = 1.0,
) -> Tuple[float, Optional[np.ndarray], float]:
    """As `_axis_residual`, but colour is modelled against DISTANCE from a
    centre point rather than position along an axis.

    `theta`/`aspect` make the iso-colour contours ELLIPSES rather than circles:
    distance is measured along an axis rotated by `theta`, with the
    perpendicular direction compressed by `aspect`.  Real shading is rarely
    perfectly circular — a sphere seen off-axis, a rounded end cap, any
    perspective at all produces an ellipse, and forcing it circular leaves a
    visible flat disc in the middle.
    """
    d = coords - centre
    if aspect != 1.0 or theta != 0.0:
        ct, st = math.cos(theta), math.sin(theta)
        u =  d[:, 0] * ct + d[:, 1] * st
        v = -d[:, 0] * st + d[:, 1] * ct
        r = np.sqrt(u * u + (aspect * v) ** 2)
    else:
        r = np.sqrt((d * d).sum(axis=1))
    rmax = float(r.max())
    if rmax < 1e-6:
        return float("inf"), None, 0.0

    t     = r / rmax
    stops = _binned_stops(t, colors, n_stops)
    if stops is None:
        return float("inf"), None, 0.0

    xs   = (np.arange(n_stops, dtype=np.float64) + 0.5) / n_stops
    pred = np.empty_like(colors)
    for c in range(3):
        pred[:, c] = np.interp(t, xs, stops[:, c])

    return float(np.sum((colors - pred) ** 2)), stops, rmax


def _fit_radial(
    coords: np.ndarray,
    colors: np.ndarray,
    n_stops: int,
) -> Tuple[float, Optional[np.ndarray], Optional[np.ndarray], float]:
    """
    Search for the centre whose distance field best explains the region's
    colour.  A coarse grid over the bounding box, then two refinement rounds
    around the winner — the residual surface is smooth in the centre position,
    so local refinement is reliable and far cheaper than a fine global grid.

    Returns (ss_res, stops, centre, rmax).
    """
    lo, hi = coords.min(axis=0), coords.max(axis=0)
    span   = np.maximum(hi - lo, 1e-6)

    def centre_search(theta: float, aspect: float, seed=None):
        """Coarse grid then local refinement, for a fixed ellipse shape."""
        bss, bstops, bc, br = float("inf"), None, None, 0.0
        if seed is None:
            cands = [np.array([x, y], dtype=np.float64)
                     for x in np.linspace(lo[0], hi[0], 5)
                     for y in np.linspace(lo[1], hi[1], 5)]
        else:
            bc = seed
            cands = [seed]
        step = span / 4.0
        for _ in range(4):
            for c in cands:
                ss, stops, rmax = _radial_residual(
                    coords, colors, c, n_stops, theta, aspect)
                if ss < bss:
                    bss, bstops, bc, br = ss, stops, c, rmax
            if bc is None:
                break
            step = step / 2.0
            cx, cy = bc
            cands = [np.array([cx + dx, cy + dy], dtype=np.float64)
                     for dx in (-step[0], 0.0, step[0])
                     for dy in (-step[1], 0.0, step[1])]
        return bss, bstops, bc, br

    # Stage 1 — circular centre.
    best_ss, best_stops, best_c, best_r = centre_search(0.0, 1.0)
    if best_c is None:
        return float("inf"), None, None, 0.0, 0.0, 1.0
    best_th, best_asp = 0.0, 1.0

    # Stage 2 — orientation and elongation, at that centre.
    for th in np.linspace(0.0, np.pi, 6, endpoint=False):
        for asp in (1.35, 1.8, 2.4):
            ss, stops, rmax = _radial_residual(
                coords, colors, best_c, n_stops, float(th), float(asp))
            if ss < best_ss:
                best_ss, best_stops, best_r = ss, stops, rmax
                best_th, best_asp = float(th), float(asp)

    # Stage 3 — the centre that best fits a CIRCLE is not the centre that best
    # fits an ellipse, and for a strongly elongated ramp it can be far off.  So
    # once the shape is known, search the centre again from scratch rather than
    # nudging the circular one.
    if best_asp != 1.0:
        ss, stops, c, rmax = centre_search(best_th, best_asp)
        if c is not None and ss < best_ss:
            best_ss, best_stops, best_c, best_r = ss, stops, c, rmax

    # Stage 4 — refine the ellipse around the winner, at the settled centre.
    for th in (best_th - 0.26, best_th, best_th + 0.26):
        for asp in (best_asp * 0.8, best_asp, best_asp * 1.25):
            if asp < 1.0:
                continue
            ss, stops, rmax = _radial_residual(
                coords, colors, best_c, n_stops, float(th), float(asp))
            if ss < best_ss:
                best_ss, best_stops, best_r = ss, stops, rmax
                best_th, best_asp = float(th), float(asp)

    return best_ss, best_stops, best_c, best_r, best_th, best_asp


def _stops_to_list(stops: np.ndarray) -> List[Dict]:
    out: List[Dict] = []
    n = len(stops)
    for i, (r, g, b) in enumerate(stops):
        out.append({
            "offset": round(i / (n - 1), 4),
            "color":  "#%02x%02x%02x" % (
                int(np.clip(round(r), 0, 255)),
                int(np.clip(round(g), 0, 255)),
                int(np.clip(round(b), 0, 255)),
            ),
        })
    return out


def fit_region_gradient(
    coords: np.ndarray,
    colors: np.ndarray,
    n_stops: int = 0,
    n_angles: int = 0,
    min_r2: float = -1.0,
    min_range: float = -1.0,
) -> Optional[Dict]:
    """
    Fit a linear gradient to one region's pixels.

    Parameters
    ----------
    coords : (N, 2) float — pixel (x, y) in full-image space.
    colors : (N, 3) float — that pixel's colour in the ORIGINAL image.

    Returns a dict with userSpaceOnUse endpoints and colour stops:
        {'x1','y1','x2','y2','stops': [{'offset','color'}...], 'r2'}
    or None when the region is better served by a flat fill.
    """
    n_stops   = n_stops   or Config.REGION_GRADIENT_STOPS
    n_angles  = n_angles  or Config.REGION_GRADIENT_ANGLE_STEPS
    min_r2    = min_r2    if min_r2    >= 0 else Config.REGION_GRADIENT_MIN_R2
    min_range = min_range if min_range >= 0 else Config.REGION_GRADIENT_MIN_RANGE

    if len(coords) < 16:
        return None

    # A region with no colour spread is a flat fill by definition — bail before
    # doing any work, and before a near-zero SS_tot makes R² meaningless.
    spread = float(np.max(colors, axis=0).max() - np.min(colors, axis=0).min())
    if spread < min_range:
        return None

    centre = coords.mean(axis=0)
    cen    = coords - centre

    # Candidate axes: a coarse angular sweep (0..180°, direction sign handled by
    # the projection itself) plus the least-squares direction of steepest colour
    # change, which is usually already optimal and costs one solve.
    angles = np.linspace(0.0, np.pi, n_angles, endpoint=False)
    axes   = [np.array([np.cos(a), np.sin(a)]) for a in angles]

    design = np.column_stack([cen, np.ones(len(cen))])
    try:
        coef, *_ = np.linalg.lstsq(design, colors, rcond=None)
        g = coef[:2]                              # (2, 3) — d(colour)/d(x, y)
        # Rows of `g` span the (x, y) domain, so the dominant direction there is
        # the first LEFT singular vector (length 2).  The right singular vectors
        # live in colour space and are the wrong length entirely.
        u_svd, _, _ = np.linalg.svd(g, full_matrices=False)
        dom = u_svd[:, 0]
        nrm = float(np.linalg.norm(dom))
        if nrm > 1e-9:
            axes.append(dom / nrm)
    except np.linalg.LinAlgError:
        pass

    ss_tot = float(np.sum((colors - colors.mean(axis=0)) ** 2))
    if ss_tot < 1e-9:
        return None

    best = (float("inf"), None, 0.0, 0.0, None)
    for u in axes:
        ss_res, stops, t_min, t_max = _axis_residual(cen, colors, u, n_stops)
        if ss_res < best[0]:
            best = (ss_res, stops, t_min, t_max, u)

    lin_ss, lin_stops, t_min, t_max, u = best

    # Shading that varies about a CENTRE (a spherical highlight, the round end
    # of a tube) has no single axis along which colour is monotonic, so the
    # linear model scores poorly no matter which angle is chosen.  Fit a radial
    # model too and keep whichever explains the region better — the comparison
    # is on the same residual, so it is a fair contest.
    rad_ss, rad_stops, rad_c, rad_r, rad_th, rad_asp = _fit_radial(cen, colors, n_stops)

    use_radial = rad_stops is not None and rad_ss < lin_ss
    ss_res     = rad_ss if use_radial else lin_ss
    stops      = rad_stops if use_radial else lin_stops
    if stops is None:
        return None

    r2 = 1.0 - ss_res / ss_tot
    if r2 < min_r2:
        return None

    if use_radial:
        c = centre + rad_c
        return {
            "type":   "radial",
            "cx":     float(c[0]),
            "cy":     float(c[1]),
            "r":      float(rad_r),
            # Orientation/elongation of the iso-colour ellipse.  The assembler
            # turns these into a gradientTransform; they are angles and ratios,
            # so they must NOT be touched by coordinate rescaling.
            "angle":  round(math.degrees(rad_th), 3),
            "aspect": round(float(rad_asp), 4),
            "stops":  _stops_to_list(stops),
            "r2":     round(float(r2), 4),
        }

    p1 = centre + u * t_min
    p2 = centre + u * t_max
    return {
        "type":  "linear",
        "x1":    float(p1[0]),
        "y1":    float(p1[1]),
        "x2":    float(p2[0]),
        "y2":    float(p2[1]),
        "stops": _stops_to_list(stops),
        "r2":    round(float(r2), 4),
    }


def subsample(idx: np.ndarray, cap: int) -> np.ndarray:
    """Evenly stride a pixel-index array down to at most `cap` entries."""
    if len(idx) <= cap:
        return idx
    step = len(idx) // cap + 1
    return idx[::step]
