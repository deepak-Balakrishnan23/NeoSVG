# NeoSVG Engine — local-colour stacked-layer vectorizer
#
# NeoSVG's high-quality original engine. Photograph-grade raster-to-vector
# pipeline. No third-party vectorization code; all implementation is original
# NeoSVG work.
#
# Algorithm (faithful local-colour layered tracing):
#
#   Phase 1 — LOCAL-COLOUR CLUSTERING
#     a. Optional edge-preserving bilateral pre-filter so cluster boundaries
#        follow real edges instead of JPEG noise.
#     b. Bit-shift quantization (`color_precision` bits per channel).  Unlike
#        a GLOBAL palette (k-means / median-cut), bit-shift keeps each pixel's
#        OWN local colour.  This is the critical property for translucent
#        subjects (water, glass): "background" and "background-seen-through-
#        water" round to DIFFERENT buckets, so the water body stays distinct
#        instead of snapping into the background and vanishing.
#     c. Single-pass connected-component labelling of equal-colour regions
#        (skimage.measure.label connects neighbours that share a value), so
#        every coherent colour patch becomes one region in O(W·H) — no slow
#        per-colour loop.
#
#   Phase 2 — STACKED LAYERS (painter's order)
#     Each region is filled solid (interior holes filled) and emitted as one
#     path, largest area first.  Smaller regions that occupy a larger region's
#     "holes" are drawn later and repaint those areas with their own colour —
#     the stacked composite reconstructs gradients and fine detail seamlessly.
#
#   Phase 3 — SUB-PIXEL CONTOUR + ADAPTIVE BEZIER FIT
#     Marching-squares at level 0.5 for half-pixel boundary accuracy.
#     Recursive cubic Bezier fitting with area-adaptive error tolerance.
#     Region fill colour is the TRUE mean of the original pixels (locally
#     accurate), not the quantized bucket value.

from __future__ import annotations

import logging
from typing import List

import cv2
import numpy as np

from config import Config
from engines.path_fitting import _contour_to_path, _rgb_to_hex
from engines.subpixel_contours import subpixel_contours as _sp_contours

try:
    from skimage.measure import label as _sk_label, regionprops as _regionprops
except ImportError:
    raise RuntimeError(
        "scikit-image is required for the NeoSVG Engine. Run: pip install scikit-image"
    )

logger = logging.getLogger("neosvg.engine")


def _smoothness_thresholds(
    labels: np.ndarray,
    rgb: np.ndarray,
    min_busy: float,
    min_smooth: float,
    busy_std: float = 10.0,
) -> np.ndarray:
    """
    Per-label minimum-area threshold driven by NEIGHBOURHOOD smoothness.

    The hard problem with a smooth photographic gradient is that flat-colour
    tiling produces two competing artefacts:
      * too few wide tiles  -> visible step "banding";
      * too many thin tiles -> hard-edged "scratch"/"dash" specks (faint sensor
        streaks and micro-noise traced as crisp shapes that pop on a smooth
        ramp).

    The resolution: in a SMOOTH NEIGHBOURHOOD only LARGE regions are real (the
    wide gradient bands, which span much of the frame); anything small there is
    noise.  In a BUSY NEIGHBOURHOOD (the subject) small regions are genuine fine
    detail and must be kept.  So the threshold keys off the *context* — a
    large-window blur of the local luminance std — NOT the region's own std (a
    dash is itself high-contrast, yet sits in a smooth context):

      * context std <= std_lo  (open gradient)  -> `min_smooth`  (LARGE: merge
                                                    every small speck away, keep
                                                    only the big bands)
      * context std >= std_hi  (busy subject)   -> `min_busy`    (small: keep
                                                    fine detail)
      * in between, linear blend.

    Returns an array `thr` indexed by label (thr[0] = background, unused).
    """
    n = int(labels.max()) + 1
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    mean = cv2.blur(gray, (15, 15))
    sq   = cv2.blur(gray * gray, (15, 15))
    std  = np.sqrt(np.maximum(sq - mean * mean, 0.0))

    # Build a BUSY mask, then morphologically OPEN it: opening erases small
    # isolated busy spots (faint sensor streaks / dust traced as dashes) but the
    # large connected subject survives.  An open-only test on the raw std blur
    # fails because an isolated dash spikes its own neighbourhood; opening keys
    # off connectivity instead of magnitude, so the dash's location is
    # reclassified as smooth context and merged away — while the subject keeps a
    # small threshold and all its fine detail.
    busy = (std > busy_std).astype(np.uint8)
    k    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    busy = cv2.morphologyEx(busy, cv2.MORPH_OPEN, k)
    busy = cv2.dilate(busy, k)                       # keep a halo around the subject
    soft = cv2.GaussianBlur(busy.astype(np.float32), (31, 31), 0)  # 0..1, 1 = busy

    flat   = labels.ravel()
    counts = np.bincount(flat, minlength=n).astype(np.float64)
    bsum   = np.bincount(flat, weights=soft.ravel().astype(np.float64), minlength=n)
    bmean  = bsum / np.maximum(counts, 1.0)

    t = np.clip(bmean, 0.0, 1.0)                      # 0 = smooth ctx, 1 = busy ctx
    thr = min_smooth + t * (min_busy - min_smooth)
    return thr


def _merge_small_regions(
    labels: np.ndarray,
    rgb: np.ndarray,
    min_area,
) -> np.ndarray:
    """
    Absorb every sub-`min_area` region into its most COLOUR-SIMILAR adjacent
    region (not merely the spatially nearest one).

    `min_area` may be a scalar (uniform threshold) OR a per-label array (see
    `_smoothness_thresholds`) so smooth zones keep thin bands while busy zones
    merge aggressively.

    This is a colour-aware small-cluster absorption step.  A
    spatial nearest-neighbour fill (e.g. expand_labels) can dump a small
    region into a wrong-coloured neighbour, producing chunky off-colour
    "voronoi" blobs.  Merging by colour similarity instead means a ripple
    speck joins the ripple next to it — so fine detail stays coherent and the
    partition is gap-free (no fragmentation).

    Background (label 0 = transparent) is never merged or used as a target.
    """
    n = int(labels.max()) + 1
    if n <= 1:
        return labels

    # Accept a scalar threshold or a per-label array (smoothness-aware).
    thr = np.asarray(min_area, dtype=np.float64)
    if thr.ndim == 0:
        thr = np.full(n, float(min_area), dtype=np.float64)

    flat   = labels.ravel()
    counts = np.bincount(flat, minlength=n).astype(np.float64)
    means  = np.empty((n, 3), dtype=np.float64)
    for c in range(3):
        means[:, c] = (
            np.bincount(flat, weights=rgb[:, :, c].ravel().astype(np.float64),
                        minlength=n) / np.maximum(counts, 1.0)
        )

    # Unique 4-connected adjacency pairs, excluding background (label 0).
    pa = np.concatenate([labels[:, :-1].ravel(), labels[:-1, :].ravel()])
    pb = np.concatenate([labels[:, 1:].ravel(),  labels[1:, :].ravel()])
    m  = (pa != pb) & (pa != 0) & (pb != 0)
    pa, pb = pa[m], pb[m]
    if len(pa) == 0:
        return labels
    lo = np.minimum(pa, pb).astype(np.int64)
    hi = np.maximum(pa, pb).astype(np.int64)
    uniq = np.unique(lo * n + hi)
    lo_u = (uniq // n).astype(int)
    hi_u = (uniq % n).astype(int)

    nb: dict = {}
    for a, b in zip(lo_u, hi_u):
        nb.setdefault(a, []).append(b)
        nb.setdefault(b, []).append(a)

    parent = np.arange(n)

    def find(x: int) -> int:
        r = x
        while parent[r] != r:
            r = parent[r]
        while parent[x] != r:
            parent[x], x = r, parent[x]
        return r

    # Merge smallest regions first; cascading is handled via union-find.
    # A region uses its OWN smoothness threshold (thr[i]); the grown root uses
    # the root's threshold so a band growing inside a smooth zone stops at the
    # smooth (small) target instead of bloating to the subject threshold.
    order = [i for i in range(1, n) if counts[i] < thr[i]]
    order.sort(key=lambda i: counts[i])
    for i in order:
        ri = find(i)
        if counts[ri] >= thr[ri]:
            continue                       # already grew past threshold
        best, best_d = None, 1e18
        for j in nb.get(i, ()):
            rj = find(j)
            if rj == ri:
                continue
            d = float(np.sum((means[ri] - means[rj]) ** 2))
            if d < best_d:
                best_d, best = d, rj
        if best is not None:
            parent[ri] = best
            tot = counts[best] + counts[ri]
            means[best] = (means[best] * counts[best] + means[ri] * counts[ri]) / max(tot, 1.0)
            counts[best] = tot
            nb.setdefault(best, []).extend(nb.get(i, ()))

    lut = np.array([find(i) for i in range(n)], dtype=labels.dtype)
    return lut[labels]


def _fill_all_holes(sub: np.ndarray) -> np.ndarray:
    """
    Fill ALL interior holes of a binary region mask, returning a solid
    silhouette.

    In stacked painter's order the regions that occupy those holes are always
    smaller (hence drawn later) and repaint them with their own colours, so
    filling here just gives each region a single clean outer contour instead
    of one contour per enclosed neighbour — which is what would otherwise make
    a noisy photo explode into tens of thousands of paths.

    `sub` must carry a 1-px background border so the corner flood seed is
    guaranteed to be exterior.
    """
    h, w = sub.shape
    ff = sub.copy()
    ffmask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(ff, ffmask, (0, 0), 1)   # flood exterior background → 1
    holes = (ff == 0)                       # unreached bg pixels = interior holes
    out = sub.copy()
    out[holes] = 1
    return out


# ── Public API ───────────────────────────────────────────────────────────────

def vectorize(
    image: np.ndarray,
    detail: str = "high",
    overrides: "dict | None" = None,
) -> List[dict]:
    """
    NeoSVG Engine vectorizer — faithful local-colour stacked layer tracer.

    Quantizes by bit-depth (preserving each region's LOCAL colour), labels
    connected same-colour regions in a single pass, and emits each as a
    spline-fitted path filled with its true mean colour, in painter's order
    (largest first).  Smooth gradients are reconstructed from many stacked
    bands; translucent subjects keep their body because their local colour
    stays distinct from the background, preserving fine detail (droplets,
    highlights).

    Parameters
    ----------
    image    : RGBA uint8 ndarray
    detail   : quality preset key ('low' | 'medium' | 'high' | 'ultra')
    overrides: optional per-run overrides, e.g. {'color_precision': 7,
               'bilateral_sigma_color': 12}.  Takes precedence over the preset.

    Returns: list of {'d', 'color', 'opacity', 'area', 'bbox'} in painter's
    order (largest regions first). Coordinates are in input-image space.
    """
    preset = Config.NEOSVG_LEVELS.get(detail) or Config.NEOSVG_LEVELS["high"]
    overrides = overrides or {}

    if image.ndim != 3 or image.shape[2] != 4:
        raise ValueError("the NeoSVG Engine expects an RGBA image")

    h, w = image.shape[:2]
    rgb   = np.ascontiguousarray(image[:, :, :3])
    alpha = image[:, :, 3]

    color_precision = int(overrides.get(
        "color_precision", preset.get("color_precision", 6)))
    color_precision = max(1, min(8, color_precision))
    meanshift_sp     = int(preset.get("meanshift_sp", 0))
    meanshift_sr     = int(preset.get("meanshift_sr", 0))
    median_k         = int(preset.get("median_blur", 0))
    bilateral_d      = preset.get("bilateral_d", 0)
    bilateral_sigma  = overrides.get(
        "bilateral_sigma_color", preset.get("bilateral_sigma_color", 25))
    min_area_px      = preset["min_area_px"]
    min_area_frac    = preset.get("min_area_frac", 0.0)
    min_area_smooth  = preset.get("min_area_smooth", None)
    corner_deg       = preset["corner_threshold_deg"]
    seg_len          = preset["segment_length"]
    error_thr        = preset["bezier_error"]
    prec_path        = Config.DEFAULT_PATH_PRECISION

    image_area   = float(h * w)
    min_area_eff = max(float(min_area_px), image_area * min_area_frac)

    # ── Phase 1a: flatten smooth zones, preserve edges ───────────────────
    # Mean-shift is the key to clean SMOOTH areas (gradient backgrounds, dark
    # zones): it averages each pixel with spatially+chromatically nearby
    # pixels until convergence, collapsing low-contrast 2-D sensor/JPEG
    # texture into flat colour while leaving high-contrast detail edges sharp.
    # A bilateral alone leaves that texture as fine "crinkle" regions; mean-
    # shift removes it, cutting smooth-area region counts ~4×.  A light
    # bilateral afterwards cleans any residual edge noise.
    work = rgb
    if meanshift_sp > 0 and meanshift_sr > 0:
        work = cv2.pyrMeanShiftFiltering(work, meanshift_sp, meanshift_sr)
        logger.info("Mean-shift pre-filter: sp=%d sr=%d", meanshift_sp, meanshift_sr)
    # Median filter removes the THIN residual slivers that survive mean-shift
    # (the faint "scratch" lines in smooth/dark zones).  It deletes thin
    # structures while leaving solid regions and step edges intact.
    if median_k >= 3 and median_k % 2 == 1:
        work = cv2.medianBlur(work, median_k)
        logger.info("Median pre-filter: ksize=%d", median_k)
    if bilateral_d > 0:
        rgb_f = cv2.bilateralFilter(
            work, d=bilateral_d,
            sigmaColor=float(bilateral_sigma),
            sigmaSpace=float(bilateral_d),
        )
        logger.info(
            "Bilateral pre-filter: d=%d sigma_color=%d", bilateral_d, bilateral_sigma)
    else:
        rgb_f = work

    # ── Phase 1b: bit-shift local-colour quantization ────────────────────
    shift = 8 - color_precision
    q     = ((rgb_f >> shift) << shift).astype(np.int64)
    code  = (q[:, :, 0] << 16) | (q[:, :, 1] << 8) | q[:, :, 2]
    code[alpha < 128] = -1   # transparent pixels → background sentinel

    # ── Phase 1c: single-pass equal-colour connected components ──────────
    # skimage.label assigns label 0 to the background sentinel (-1, i.e. the
    # transparent pixels); real colour regions are labelled 1..N.
    labels   = _sk_label(code, background=-1, connectivity=2)
    n_labels = int(labels.max())
    if n_labels == 0:
        return []

    # ── Phase 1d: absorb sub-threshold regions into best-colour neighbour ──
    # THIS is the step that turns a noisy photo into clean coherent shapes
    # (and the piece the engine was missing).  Bit-shift clustering produces
    # many tiny noise/detail regions; DROPPING them (the old behaviour) left
    # gaps that showed as fragmentation/speckle.  Instead each small region is
    # merged into its most COLOUR-SIMILAR neighbour, giving a gap-free
    # partition with no off-colour blobs.
    # Smoothness-aware per-label thresholds: in a SMOOTH NEIGHBOURHOOD only the
    # large gradient bands are real, so a LARGE threshold (`min_area_smooth`)
    # merges every small speck/scratch/dash into its band — no banding (bands
    # are far bigger than the threshold) and no specks.  In a BUSY (subject)
    # neighbourhood the normal small `min_area_eff` keeps fine detail.  When
    # min_area_smooth is unset, this collapses to the uniform legacy behaviour.
    if min_area_smooth is not None and float(min_area_smooth) > min_area_eff:
        thr_arr = _smoothness_thresholds(
            labels, rgb, min_busy=min_area_eff, min_smooth=float(min_area_smooth))
    else:
        thr_arr = np.full(int(labels.max()) + 1, min_area_eff, dtype=np.float64)

    # Per-region "smoothness" denominator: thr_arr ranges from min_area_eff
    # (busy/subject context) up to min_area_smooth (flat gradient context).
    # We reuse it below to tell the assembler how aggressively to seal the
    # anti-aliased seams between flat tiles — wide overlap on smooth gradient
    # bands (where the background would otherwise bleed through as scratch
    # hairlines), hairline-thin on busy detail (where wide strokes would blob
    # fine features).  See stages/vectorize_stage._seal_seams.
    _smooth_denom = (float(min_area_smooth) - min_area_eff) \
        if (min_area_smooth is not None and float(min_area_smooth) > min_area_eff) \
        else 0.0

    counts  = np.bincount(labels.ravel())
    n_small = int((counts[1:] < thr_arr[1:len(counts)]).sum()) if len(counts) > 1 else 0
    if n_small:
        labels = _merge_small_regions(labels, rgb, thr_arr)
    logger.info(
        "Clustering: color_precision=%d → %d regions (%d small merged)",
        color_precision, n_labels, n_small)

    # Region area / bbox / cropped mask / TRUE mean colour (from original rgb).
    props = _regionprops(labels, intensity_image=rgb)

    path_records: List[dict] = []
    for rp in props:
        # Per-region threshold (smoothness-aware): a surviving region keeps the
        # threshold of its root label, so thin smooth-zone bands clear the bar
        # while sub-threshold speckle in busy zones is already merged away.
        emit_thr = thr_arr[rp.label] if rp.label < len(thr_arr) else min_area_eff
        if rp.area < emit_thr:
            continue

        # Smoothness context in [0,1]: 0 = busy/subject, 1 = flat gradient.
        if _smooth_denom > 0.0:
            smooth_ctx = (float(emit_thr) - min_area_eff) / _smooth_denom
            smooth_ctx = float(min(1.0, max(0.0, smooth_ctx)))
        else:
            smooth_ctx = 0.0

        minr, minc, maxr, maxc = rp.bbox

        # Cropped mask with a 1-px border (for hole flood + contour walk).
        sub = np.zeros((maxr - minr + 2, maxc - minc + 2), np.uint8)
        sub[1:-1, 1:-1] = rp.image.astype(np.uint8)
        sub = _fill_all_holes(sub)

        contours = _sp_contours(sub, min_length=3)
        if not contours:
            continue

        mean = rp.mean_intensity
        color_hex = _rgb_to_hex(
            int(round(mean[0])), int(round(mean[1])), int(round(mean[2])))

        # Adaptive fit tolerance by region size:
        #   • SMALL detail regions (ripples, droplets, highlights) → TIGHT fit
        #     so fine shape is preserved.
        #   • LARGE smooth regions (gradient-background bands) → LOOSE fit:
        #     far fewer Bezier nodes and straighter boundaries.  This both
        #     shrinks the SVG dramatically (long band contours were dominating
        #     file size) and removes the faint wiggle/streak look on smooth
        #     zones.  A smooth band genuinely needs few control points.
        big = rp.area / 25000.0
        error_thr_adapt  = max(0.3, error_thr * (0.6 + big))
        epsilon_override = min(3.0, max(0.2, seg_len * (0.3 + big)))
        max_depth_adapt  = 12

        # Offset from cropped sub-array space back to full-image (x, y).
        off = np.array([[minc - 1, minr - 1]], dtype=np.float32)

        # Sort a region's contours largest-first; with holes filled there is
        # normally just one outer contour per region.  We do NOT apply a
        # polygon-area filter here — a thin feature (e.g. the water stream) has
        # many pixels but a tiny polygon area, and the region already passed
        # the pixel-count threshold above.
        for contour in contours:
            if len(contour) < 3:
                continue
            pts = contour.reshape(-1, 2) + off
            ci  = pts.round().astype(np.int32).reshape(-1, 1, 2)
            d = _contour_to_path(
                pts.reshape(-1, 1, 2).astype(np.float32),
                "spline", corner_deg, seg_len,
                error_thr_adapt, prec_path,
                epsilon_override=epsilon_override,
                max_depth=max_depth_adapt,
            )
            if not d:
                continue
            bx, by, bw, bh = cv2.boundingRect(ci)
            path_records.append({
                "d":          d,
                "color":      color_hex,
                "opacity":    1.0,
                "area":       float(rp.area),
                "bbox":       (bx, by, bw, bh),
                "smooth_ctx": smooth_ctx,
            })

    # Painter's order: largest first so smaller detail regions composite on top.
    path_records.sort(key=lambda r: r["area"], reverse=True)
    logger.info(
        "Trace: %d paths emitted (%d distinct colors)",
        len(path_records), len(set(r["color"] for r in path_records)),
    )
    return path_records
