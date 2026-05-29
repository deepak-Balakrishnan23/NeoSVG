# NeoSVG — GradientDetector stage
# Scans image regions for smooth colour transitions and fits SVG gradient
# parameters. Matched regions are masked before vectorisation.

import logging
from typing import List

import numpy as np

from config import Config
from context import Context, GradientRegion

logger = logging.getLogger("neosvg.gradient_detector")


def _scan_color_profile(region: np.ndarray, axis: int) -> np.ndarray:
    """
    Mean colour along *axis* (0 = rows→column profile, 1 = cols→row profile).
    Returns (N, 3) float64.
    """
    return region[:, :, :3].mean(axis=axis).astype(np.float64)


def _r_squared(y: np.ndarray, y_fit: np.ndarray) -> float:
    ss_res = float(np.sum((y - y_fit) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - ss_res / (ss_tot + 1e-12)


def _linear_gradient_r2(profile: np.ndarray) -> float:
    """Fit a straight line to each channel; return mean R²."""
    n    = len(profile)
    t    = np.linspace(0.0, 1.0, n)
    r2s  = []
    for c in range(3):
        y = profile[:, c]
        coeffs = np.polyfit(t, y, 1)
        y_fit  = np.polyval(coeffs, t)
        r2s.append(_r_squared(y, y_fit))
    return float(np.mean(r2s))


def _extract_gradient_stops(profile: np.ndarray, n_stops: int = 4) -> List[dict]:
    """Sample evenly spaced colour stops from a 1-D colour profile."""
    n      = len(profile)
    idxs   = np.linspace(0, n - 1, n_stops, dtype=int)
    stops  = []
    for i, idx in enumerate(idxs):
        r, g, b = int(round(profile[idx, 0])), int(round(profile[idx, 1])), int(round(profile[idx, 2]))
        stops.append({
            "offset": round(i / (n_stops - 1), 3),
            "color":  f"#{r:02x}{g:02x}{b:02x}",
        })
    return stops


def _test_region(
    region: np.ndarray,
    x: int, y: int, w: int, h: int,
    threshold: float,
) -> "GradientRegion | None":
    """
    Try to classify a rectangular region as a linear or radial gradient.
    Returns a GradientRegion if the fit is good, else None.
    """
    if w < 4 or h < 4:
        return None

    # Sample along horizontal and vertical axes
    h_profile = _scan_color_profile(region, axis=0)  # shape (W, 3)
    v_profile = _scan_color_profile(region, axis=1)  # shape (H, 3)

    # Skip near-uniform tiles — a tile going from 0→5 gets R²≈1.0 but is
    # indistinguishable from solid fill and would render as an ugly block.
    color_range = max(
        float(h_profile.max() - h_profile.min()),
        float(v_profile.max() - v_profile.min()),
    )
    if color_range < Config.GRADIENT_MIN_COLOR_RANGE:
        return None

    r2_h = _linear_gradient_r2(h_profile)
    r2_v = _linear_gradient_r2(v_profile)

    best_r2 = max(r2_h, r2_v)
    if best_r2 < threshold:
        return None

    if r2_h >= r2_v:
        # Horizontal gradient → angle=90° (left→right)
        angle = 90.0
        stops = _extract_gradient_stops(h_profile)
    else:
        # Vertical gradient → angle=0° (top→bottom)
        angle = 0.0
        stops = _extract_gradient_stops(v_profile)

    return GradientRegion(
        bbox          = (x, y, w, h),
        gradient_type = "linear",
        params        = {"angle": angle, "stops": stops, "r2": round(best_r2, 3)},
    )


def detect_gradients(ctx: Context) -> Context:
    """
    Scan the working image (text-masked) in a grid of overlapping tiles.
    Any tile with a sufficiently smooth linear colour ramp is recorded as a
    gradient region and zeroed out of the working image.
    """
    if ctx.quality == "fast":
        logger.info("Gradient detection skipped in fast mode")
        return ctx

    img = ctx.text_masked_image if ctx.text_masked_image is not None else ctx.preprocessed_image
    if img is None:
        return ctx

    h, w = img.shape[:2]
    regions: List[GradientRegion] = []

    # Tile size: ~25% of image, min 64px; non-overlapping step reduces
    # the tile count and avoids the blocky grid pattern from overlapping tiles.
    tile_w = max(64, w // 4)
    tile_h = max(64, h // 4)
    step_x = tile_w
    step_y = tile_h

    threshold = Config.GRADIENT_SMOOTHNESS_CORR

    for ty in range(0, h - tile_h + 1, step_y):
        for tx in range(0, w - tile_w + 1, step_x):
            tile   = img[ty: ty + tile_h, tx: tx + tile_w]
            result = _test_region(tile, tx, ty, tile_w, tile_h, threshold)
            if result is None:
                continue

            # Avoid duplicating heavily overlapping tiles
            duplicate = any(
                abs(r.bbox[0] - result.bbox[0]) < step_x and
                abs(r.bbox[1] - result.bbox[1]) < step_y
                for r in regions
            )
            if not duplicate:
                regions.append(result)
                logger.debug(
                    "Gradient detected at (%d,%d,%d,%d) type=%s r2=%.2f",
                    tx, ty, tile_w, tile_h,
                    result.gradient_type,
                    result.params["r2"],
                )

    # Build masks for downstream stages
    for gr in regions:
        x, y, gw, gh = gr.bbox
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[y: y + gh, x: x + gw] = 255
        gr.mask = mask

    logger.info("Detected %d gradient region(s)", len(regions))
    ctx.gradient_regions = regions
    return ctx
