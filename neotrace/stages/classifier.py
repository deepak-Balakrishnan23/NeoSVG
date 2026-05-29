# NeoSVG — ImageClassifier stage
# Examines a thumbnail of the input and decides which image archetype it is,
# then sets recommended vectorization parameters on the context.

import logging

import cv2
import numpy as np

from config import Config
from context import Context

logger = logging.getLogger("neosvg.classifier")


def _count_quantized_colors(thumb: np.ndarray, n: int = 16) -> int:
    """K-means on a small thumbnail; returns cluster count actually needed."""
    h, w = thumb.shape[:2]
    pixels = thumb[:, :, :3].reshape(-1, 3).astype(np.float32)
    k = min(n, len(np.unique(pixels.view(np.uint8).reshape(-1, 3), axis=0)))
    k = max(2, k)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, _ = cv2.kmeans(pixels, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    return int(np.unique(labels).size)


def _avg_saturation(thumb: np.ndarray) -> float:
    """Mean HSV saturation of a thumbnail (0–255 scale)."""
    hsv = cv2.cvtColor(thumb[:, :, :3], cv2.COLOR_RGB2HSV)
    return float(hsv[:, :, 1].mean())


def _pixelart_score(img: np.ndarray) -> float:
    """
    Fraction of gradient-edge pixels that align to a small integer grid.
    High score → pixel art.
    """
    gray  = cv2.cvtColor(img[:, :, :3], cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    ys, xs = np.where(edges > 0)
    if len(xs) < 20:
        return 0.0

    # Check alignment to grids of size 2, 3, 4, 8.
    # Use AND so both x and y must be on-grid — OR gives ~75% for any image
    # with g=2 just by chance and causes systematic misclassification.
    best = 0.0
    for g in (2, 3, 4, 8):
        aligned = np.sum((xs % g == 0) & (ys % g == 0))
        ratio   = aligned / len(xs)
        best    = max(best, ratio)
    return best


def _color_variance(img: np.ndarray) -> float:
    """Coefficient of variation of per-pixel luminance."""
    gray = cv2.cvtColor(img[:, :, :3], cv2.COLOR_RGB2GRAY).astype(float)
    mu   = gray.mean()
    if mu < 1e-6:
        return 0.0
    return float(gray.std() / mu)


def classify(ctx: Context) -> Context:
    """
    Classify the input image and populate ctx.image_type and
    ctx.recommended_params.  Uses ctx.preprocessed_image if available,
    otherwise ctx.original_image.
    """
    img = ctx.preprocessed_image if ctx.preprocessed_image is not None else ctx.original_image
    if img is None:
        logger.warning("No image on context; defaulting to LOGO")
        ctx.image_type = "LOGO"
        ctx.recommended_params = Config.PARAMS_BY_TYPE["LOGO"].copy()
        return ctx

    h, w = img.shape[:2]

    # Work on a small thumbnail for speed
    scale  = 64.0 / min(h, w)
    tw, th = max(1, int(w * scale)), max(1, int(h * scale))
    thumb  = cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA)

    # --- features -----------------------------------------------------------
    n_colors  = _count_quantized_colors(thumb, n=32)
    avg_sat   = _avg_saturation(thumb)
    px_score  = _pixelart_score(thumb)
    variance  = _color_variance(img)
    short_dim = min(h, w)

    logger.info(
        "Classifier features — n_colors=%d avg_sat=%.1f px_score=%.2f "
        "variance=%.2f short_dim=%d",
        n_colors, avg_sat, px_score, variance, short_dim,
    )

    # --- decision tree -------------------------------------------------------
    if avg_sat < Config.LINEART_MAX_AVG_SATURATION:
        image_type = "LINEART"
        reason = f"avg_sat={avg_sat:.1f} < {Config.LINEART_MAX_AVG_SATURATION}"

    elif px_score >= Config.PIXELART_GRID_RATIO:
        image_type = "PIXELART"
        reason = f"pixel-grid alignment {px_score:.2f}"

    elif short_dim <= Config.ICON_MAX_DIM and n_colors <= Config.LOGO_MAX_UNIQUE_COLORS:
        image_type = "ICON"
        reason = f"short_dim={short_dim} ≤ {Config.ICON_MAX_DIM}, n_colors={n_colors}"

    elif n_colors <= Config.LOGO_MAX_UNIQUE_COLORS:
        image_type = "LOGO"
        reason = f"n_colors={n_colors} ≤ {Config.LOGO_MAX_UNIQUE_COLORS}"

    elif n_colors <= Config.CARTOON_MAX_UNIQUE_COLORS and variance < 0.60:
        # Texture-vs-edge guard: water/specular photos can appear low-variance
        # on the thumbnail (smooth gradients → small std/mean) yet carry
        # significant interior Laplacian energy from surface texture and
        # specular highlights.  Detecting this prevents the CARTOON bilateral
        # pre-filter from flattening the gradient before engine sees it.
        _gray_thumb = cv2.cvtColor(thumb[:, :, :3], cv2.COLOR_RGB2GRAY)
        _lap = cv2.Laplacian(_gray_thumb.astype(np.float32), cv2.CV_32F)
        _abs_lap = np.abs(_lap)
        _interior_mask = _abs_lap < 10.0
        if _interior_mask.sum() > 100:
            _interior_energy = float(_abs_lap[_interior_mask].mean())
            if _interior_energy > 1.5:
                image_type = "PHOTO"
                reason = (
                    f"n_colors={n_colors} ≤ {Config.CARTOON_MAX_UNIQUE_COLORS} "
                    f"but interior_energy={_interior_energy:.2f} > 1.5 "
                    f"(photo texture — not CARTOON)"
                )
            else:
                image_type = "CARTOON"
                reason = f"n_colors={n_colors} ≤ {Config.CARTOON_MAX_UNIQUE_COLORS}, variance={variance:.2f}"
        else:
            image_type = "CARTOON"
            reason = f"n_colors={n_colors} ≤ {Config.CARTOON_MAX_UNIQUE_COLORS}, variance={variance:.2f}"

    else:
        image_type = "PHOTO"
        reason = f"n_colors={n_colors}, variance={variance:.2f}"

    logger.info("Classified as %s — %s", image_type, reason)

    ctx.image_type         = image_type
    ctx.recommended_params = Config.PARAMS_BY_TYPE[image_type].copy()

    # Override with --mode flag if user explicitly chose
    if ctx.mode != "auto":
        mode_map = {
            "logo":    "LOGO",
            "photo":   "PHOTO",
            "cartoon": "CARTOON",
        }
        forced = mode_map.get(ctx.mode.lower())
        if forced:
            logger.info("User forced mode %s, overriding classifier result %s",
                        forced, image_type)
            ctx.image_type         = forced
            ctx.recommended_params = Config.PARAMS_BY_TYPE[forced].copy()

    return ctx
