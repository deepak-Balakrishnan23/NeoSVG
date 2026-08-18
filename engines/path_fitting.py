# NeoSVG — shared path-fitting helpers
# Bezier curve fitting, corner detection and SVG path serialisation
# used by the NeoSVG Engine. All algorithms are NeoSVG originals; no
# third-party vectorization library is used.
#   * Path fitting           (pixel / polygon / spline modes)
#   * SVG path serialisation  (relative coordinates, configurable precision)

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import cv2
import numpy as np


# Spline fitting works on the raw sub-pixel contour.  Beyond this many points a
# contour is decimated first, purely to bound fitting cost — never enough to
# flatten curvature (see _RDP_SAFE_EPSILON).
_RAW_FIT_MAX_POINTS = 1200
_RDP_SAFE_EPSILON   = 0.5   # sub-pixel: removes collinear runs, keeps curves

# Upper bound on a Bezier tangent magnitude, as a multiple of the segment's
# chord length.  Beyond this the fit has diverged and the endpoint-derived
# fallback is used instead.
_MAX_TANGENT_CHORDS = 3.0


# ── helpers ──────────────────────────────────────────────────────────────────

def _rgb_to_hex(r: int, g: int, b: int) -> str:
    return f"#{r:02x}{g:02x}{b:02x}"


def _chord_lengths(pts: np.ndarray) -> np.ndarray:
    """Cumulative chord-length parameterisation for a sequence of 2-D points."""
    diffs = np.diff(pts, axis=0)
    dists = np.linalg.norm(diffs, axis=1)
    t = np.concatenate([[0.0], np.cumsum(dists)])
    total = t[-1]
    if total < 1e-10:
        return np.linspace(0.0, 1.0, len(pts))
    return t / total


def _bernstein(t: float) -> Tuple[float, float, float, float]:
    """Cubic Bernstein basis values at parameter *t*."""
    s = 1.0 - t
    return s**3, 3 * s**2 * t, 3 * s * t**2, t**3


def _fit_cubic_bezier(
    pts: np.ndarray,
    t_hat1: np.ndarray,
    t_hat2: np.ndarray,
) -> Optional[np.ndarray]:
    """
    Fit a single cubic Bezier to *pts* with tangent constraints.
    Returns a (4, 2) array [P0, P1, P2, P3] or None if the fit is degenerate.
    """
    t = _chord_lengths(pts)

    # Vectorised normal equations.  This used to be two Python loops over every
    # contour point, which is why the caller had to decimate the contour with
    # RDP before fitting — and that decimation was what turned curves into
    # polygons.  Fitting straight from the raw sub-pixel contour is only
    # affordable once this is array work.
    s  = 1.0 - t
    b0 = s * s * s
    b1 = 3.0 * s * s * t
    b2 = 3.0 * s * t * t
    b3 = t * t * t

    A0 = b1[:, None] * t_hat1[None, :]          # (n, 2)
    A1 = b2[:, None] * t_hat2[None, :]
    tmp = (pts
           - pts[0][None, :]  * (b0 + b1)[:, None]
           - pts[-1][None, :] * (b2 + b3)[:, None])

    c00 = float(np.sum(A0 * A0))
    c01 = float(np.sum(A0 * A1))
    c11 = float(np.sum(A1 * A1))
    C = np.array([[c00, c01], [c01, c11]])
    X = np.array([float(np.sum(A0 * tmp)), float(np.sum(A1 * tmp))])

    det = C[0, 0] * C[1, 1] - C[0, 1] * C[1, 0]
    fallback = np.linalg.norm(pts[-1] - pts[0]) / 3.0

    # The tangent magnitudes must be bounded ABOVE as well as below.  On a raw
    # sub-pixel contour a nearly-straight run with noisy end tangents drives the
    # least-squares solution to a huge alpha, placing a control point far
    # outside the shape — which renders as a spike shooting off the outline.
    # The decimated contours this fitter used to receive hid the problem.
    max_alpha = max(np.linalg.norm(pts[-1] - pts[0]), 1e-6) * _MAX_TANGENT_CHORDS

    if abs(det) < 1e-10:
        alpha1 = alpha2 = fallback
    else:
        alpha1 = (X[0] * C[1, 1] - X[1] * C[0, 1]) / det
        alpha2 = (C[0, 0] * X[1] - C[1, 0] * X[0]) / det
        if not (1e-6 <= alpha1 <= max_alpha) or not (1e-6 <= alpha2 <= max_alpha):
            alpha1 = alpha2 = fallback

    return np.array([
        pts[0],
        pts[0]  + alpha1 * t_hat1,
        pts[-1] + alpha2 * t_hat2,
        pts[-1],
    ])


def _max_fit_error(pts: np.ndarray, bez: np.ndarray) -> Tuple[float, int]:
    """Return (max_error, split_index) of the cubic Bezier approximation.

    split_at is clamped to [1, len(pts)-1] so callers can always split into
    two non-empty sub-segments without risking an empty left or right slice.
    """
    t  = _chord_lengths(pts)
    s  = 1.0 - t
    approx = (
        (s * s * s)[:, None]       * bez[0][None, :]
        + (3.0 * s * s * t)[:, None] * bez[1][None, :]
        + (3.0 * s * t * t)[:, None] * bez[2][None, :]
        + (t * t * t)[:, None]       * bez[3][None, :]
    )
    err = np.linalg.norm(pts - approx, axis=1)
    i   = int(np.argmax(err))
    # Keep split_at in a valid range: never 0 (empty left) or len(pts)
    # (empty right).
    return float(err[i]), max(1, min(i, len(pts) - 1))


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-10 else v


def _left_tangent(pts: np.ndarray) -> np.ndarray:
    return _unit(pts[1] - pts[0])


def _right_tangent(pts: np.ndarray) -> np.ndarray:
    return _unit(pts[-2] - pts[-1])


def _fit_spline_segment(
    pts: np.ndarray,
    t_hat1: np.ndarray,
    t_hat2: np.ndarray,
    error_threshold: float,
    depth: int = 0,
    max_depth: int = 12,        # Phase 13b: was hardcoded at 8
) -> List[np.ndarray]:
    """
    Recursively fit cubic Bezier curves to a segment, splitting on error.
    Returns a list of (4, 2) Bezier control-point arrays.

    Phase 13b: max_depth raised from 8 → 12 (more recursion = better fit on
    complex curves) and exposed as a parameter so callers can tune per
    region size (large regions afford deeper recursion).
    """
    if len(pts) < 2:
        return []
    if len(pts) == 2:
        dist = np.linalg.norm(pts[-1] - pts[0]) / 3.0
        bez = np.array([pts[0], pts[0] + dist * t_hat1,
                        pts[-1] + dist * t_hat2, pts[-1]])
        return [bez]

    bez = _fit_cubic_bezier(pts, t_hat1, t_hat2)
    if bez is None:
        return []

    max_err, split_at = _max_fit_error(pts, bez)

    if max_err < error_threshold or depth >= max_depth:
        return [bez]

    mid = split_at
    t_mid_l = _right_tangent(pts[:mid + 1])
    t_mid_r = _left_tangent(pts[mid:])
    left  = _fit_spline_segment(pts[:mid + 1], t_hat1,  t_mid_l, error_threshold, depth + 1, max_depth)
    right = _fit_spline_segment(pts[mid:],     t_mid_r, t_hat2,  error_threshold, depth + 1, max_depth)
    return left + right


def _detect_corners(pts: np.ndarray, threshold_deg: float,
                    window: Optional[int] = None) -> List[int]:
    """
    Return indices of points where the contour genuinely turns.

    The angle is measured across a WINDOW of points either side, not between
    the two adjacent segments.  On a raw sub-pixel contour, adjacent segments
    are roughly one pixel long and their direction is dominated by marching-
    squares quantisation, so an adjacent-pair test reports corners all along a
    smooth arc.  Measuring across a window averages that noise out and leaves
    only real corners.

    Corners are then non-maximum suppressed within the window, so a single
    sharp turn yields one break rather than a cluster of them.
    """
    n = len(pts)
    if n < 3:
        return [0, n - 1]

    if window is None:
        window = max(1, min(6, n // 24))

    i  = np.arange(1, n - 1)
    a  = pts[np.maximum(i - window, 0)]
    b  = pts[i]
    c  = pts[np.minimum(i + window, n - 1)]

    v1 = b - a
    v2 = c - b
    n1 = np.linalg.norm(v1, axis=1)
    n2 = np.linalg.norm(v2, axis=1)
    ok = (n1 > 1e-10) & (n2 > 1e-10)
    cos_a = np.ones(len(i))
    cos_a[ok] = np.sum(v1[ok] * v2[ok], axis=1) / (n1[ok] * n2[ok])

    hit = np.flatnonzero(cos_a < math.cos(math.radians(threshold_deg)))

    corners = [0]
    last = -window
    for j in hit:                      # sharpest-first is unnecessary: the
        idx = int(i[j])                # window already isolates each turn
        if idx - last >= window:
            corners.append(idx)
            last = idx
    corners.append(n - 1)
    return corners


# ── SVG path serialisation ────────────────────────────────────────────────────

def _beziers_to_path(beziers: List[np.ndarray], precision: int,
                     closed: bool = True) -> str:
    """Convert a list of cubic Bezier arrays to an SVG path *d* string."""
    if not beziers:
        return ""
    fmt = f".{precision}f"

    def f(v): return format(v, fmt)

    parts = [f"M {f(beziers[0][0][0])} {f(beziers[0][0][1])}"]
    for bez in beziers:
        p1, p2, p3 = bez[1], bez[2], bez[3]
        parts.append(f"c {f(p1[0]-bez[0][0])} {f(p1[1]-bez[0][1])} "
                     f"{f(p2[0]-bez[0][0])} {f(p2[1]-bez[0][1])} "
                     f"{f(p3[0]-bez[0][0])} {f(p3[1]-bez[0][1])}")
    if closed:
        parts.append("z")
    return " ".join(parts)


def _polygon_to_path(pts: np.ndarray, precision: int,
                     closed: bool = True) -> str:
    """Relative polygon path from contour points."""
    if len(pts) < 2:
        return ""
    fmt = f".{precision}f"

    def f(v): return format(v, fmt)

    parts = [f"M {f(pts[0][0])} {f(pts[0][1])}"]
    for i in range(1, len(pts)):
        dx = pts[i][0] - pts[i - 1][0]
        dy = pts[i][1] - pts[i - 1][1]
        parts.append(f"l {f(dx)} {f(dy)}")
    if closed:
        parts.append("z")
    return " ".join(parts)


def _pixel_to_path(pts: np.ndarray, precision: int) -> str:
    """Raw pixel-outline path (all corners kept)."""
    return _polygon_to_path(pts, precision, closed=True)


# ── contour → path ────────────────────────────────────────────────────────────

def _contour_to_path(
    contour: np.ndarray,
    mode: str,
    corner_threshold_deg: float,
    segment_length: float,
    error_threshold: float,
    precision: int,
    epsilon_override: Optional[float] = None,
    max_depth: int = 12,
) -> str:
    pts = contour.squeeze()
    if pts.ndim != 2 or len(pts) < 3:
        return ""
    pts = pts.astype(float)

    if mode == 'pixel':
        return _pixel_to_path(pts, precision)

    # Polygon: RDP simplification via OpenCV
    # epsilon_override lets callers pass an area-adaptive value; the default
    # preserves sub-pixel detail from skimage's marching-squares extractor.
    epsilon = epsilon_override if epsilon_override is not None else max(0.3, segment_length * 0.4)
    poly = cv2.approxPolyDP(contour, epsilon, closed=True)
    poly_pts = poly.squeeze().astype(float)
    if poly_pts.ndim != 2 or len(poly_pts) < 3:
        return ""

    if mode == 'polygon':
        return _polygon_to_path(poly_pts, precision)

    # Spline mode does NOT fit the RDP output.  RDP keeps only direction-change
    # vertices, so every point it returns reads as a corner to the detector
    # below — the curve is then broken into a hard segment at each one and
    # rendered as a polygon.  A large region gets a large epsilon, which is why
    # big smooth shapes suffered worst, and why tightening `bezier_error` made
    # things worse rather than better: it only subdivided an already-flattened
    # outline more finely.
    #
    # So fit the RAW sub-pixel contour and let `error_threshold` alone decide
    # how many curves are needed.  RDP is kept only as a cheap decimation for
    # very long contours, capped so it can never remove real curvature.
    fit_pts = pts
    if len(pts) > _RAW_FIT_MAX_POINTS:
        coarse = cv2.approxPolyDP(contour, min(epsilon, _RDP_SAFE_EPSILON), closed=True)
        cp = coarse.squeeze().astype(float)
        if cp.ndim == 2 and len(cp) >= 3:
            fit_pts = cp

    # Spline: detect corners, fit Bezier segments between them
    corners = _detect_corners(fit_pts, corner_threshold_deg)
    poly_pts = fit_pts
    all_beziers: List[np.ndarray] = []

    for k in range(len(corners) - 1):
        seg = poly_pts[corners[k]: corners[k + 1] + 1]
        if len(seg) < 2:
            continue
        t1 = _left_tangent(seg)
        t2 = _right_tangent(seg)
        beziers = _fit_spline_segment(seg, t1, t2, error_threshold, max_depth=max_depth)
        all_beziers.extend(beziers)

    if not all_beziers:
        return _polygon_to_path(poly_pts, precision)

    return _beziers_to_path(all_beziers, precision, closed=True)
