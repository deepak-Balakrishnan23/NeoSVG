# NeoSVG — Preprocessor stage
# Cleans the input image: denoise, JPEG artefact removal, histogram stretch,
# and upscaling.  Every decision is logged so users can understand what changed.

import logging

import cv2
import numpy as np

from config import Config
from context import Context

logger = logging.getLogger("neosvg.preprocessor")


def _needs_upscale(h: int, w: int) -> bool:
    return min(h, w) < Config.MIN_DIM_FOR_UPSCALE


def _upscale(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    short = min(h, w)
    scale = Config.UPSCALE_TARGET / short
    nw    = max(1, int(round(w * scale)))
    nh    = max(1, int(round(h * scale)))
    logger.info("Upscaling %dx%d → %dx%d (LANCZOS4)", w, h, nw, nh)
    return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LANCZOS4)


def _stretch_histogram(img: np.ndarray) -> np.ndarray:
    """Per-channel contrast stretch if dynamic range is narrow."""
    out  = img.copy()
    for c in range(3):
        ch = img[:, :, c]
        lo = int(np.percentile(ch, Config.HISTOGRAM_FLAT_PERCENTILE))
        hi = int(np.percentile(ch, 100 - Config.HISTOGRAM_FLAT_PERCENTILE))
        if hi - lo < 30:
            logger.info("Channel %d: histogram flat [%d,%d], stretching", c, lo, hi)
            out[:, :, c] = np.clip(
                (ch.astype(float) - lo) * 255.0 / max(hi - lo, 1), 0, 255
            ).astype(np.uint8)
    return out


def _denoise_photo(img: np.ndarray) -> np.ndarray:
    rgb   = img[:, :, :3]
    alpha = img[:, :, 3:4]
    denoised = cv2.fastNlMeansDenoisingColored(
        rgb,
        h           = Config.PHOTO_DENOISE_H,
        hColor      = Config.PHOTO_DENOISE_H,
        templateWindowSize = Config.PHOTO_DENOISE_TEMPLATE,
        searchWindowSize   = Config.PHOTO_DENOISE_SEARCH,
    )
    logger.info("Applied fastNlMeansDenoisingColored (h=%d)", Config.PHOTO_DENOISE_H)
    return np.concatenate([denoised, alpha], axis=2)


def _denoise_cartoon(img: np.ndarray) -> np.ndarray:
    rgb   = img[:, :, :3]
    alpha = img[:, :, 3:4]
    filtered = cv2.bilateralFilter(
        rgb,
        d      = Config.CARTOON_BILATERAL_D,
        sigmaColor = Config.CARTOON_BILATERAL_SIGMA_C,
        sigmaSpace = Config.CARTOON_BILATERAL_SIGMA_S,
    )
    logger.info("Applied bilateral filter (d=%d, sigmaC=%.0f)",
                Config.CARTOON_BILATERAL_D, Config.CARTOON_BILATERAL_SIGMA_C)
    return np.concatenate([filtered, alpha], axis=2)


def _guided_filter(guide: np.ndarray, src: np.ndarray, r: int = 8, eps: float = 0.01) -> np.ndarray:
    """
    Guided image filter for JPEG artifact removal.
    Implemented from scratch (He et al. 2010, "Guided Image Filtering").
    """
    guide_f = guide.astype(np.float64) / 255.0
    src_f   = src.astype(np.float64) / 255.0

    ksize = (2 * r + 1, 2 * r + 1)
    mean_I  = cv2.boxFilter(guide_f, -1, ksize)
    mean_p  = cv2.boxFilter(src_f,   -1, ksize)
    mean_Ip = cv2.boxFilter(guide_f * src_f, -1, ksize)
    cov_Ip  = mean_Ip - mean_I * mean_p
    mean_II = cv2.boxFilter(guide_f * guide_f, -1, ksize)
    var_I   = mean_II - mean_I * mean_I

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    mean_a = cv2.boxFilter(a, -1, ksize)
    mean_b = cv2.boxFilter(b, -1, ksize)
    q = mean_a * guide_f + mean_b
    return np.clip(q * 255.0, 0, 255).astype(np.uint8)


def _remove_jpeg_artifacts(img: np.ndarray) -> np.ndarray:
    """Guided filter pass to soften JPEG block artefacts."""
    rgb   = img[:, :, :3]
    alpha = img[:, :, 3:4]
    # Use grayscale guide to avoid colour bleeding
    gray  = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    out   = np.stack([
        _guided_filter(gray, rgb[:, :, c]) for c in range(3)
    ], axis=2)
    logger.info("Applied guided filter for JPEG artifact removal")
    return np.concatenate([out, alpha], axis=2)


def preprocess(ctx: Context) -> Context:
    """
    Apply appropriate preprocessing based on ctx.image_type.
    Sets ctx.preprocessed_image.
    """
    img = ctx.original_image
    if img is None:
        logger.error("No original image; skipping preprocessing")
        return ctx

    # Ensure RGBA
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGBA)
    elif img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)

    h, w = img.shape[:2]
    image_type = ctx.image_type

    # ── upscale first ────────────────────────────────────────────────────────
    if _needs_upscale(h, w):
        img = _upscale(img)

    # ── denoising (type-specific) ─────────────────────────────────────────────
    if image_type == "PHOTO":
        img = _denoise_photo(img)
        img = _remove_jpeg_artifacts(img)
    elif image_type in ("CARTOON",):
        img = _denoise_cartoon(img)
    # LOGO, LINEART, PIXELART, ICON: no denoising — would degrade crisp edges

    # ── colour normalisation ──────────────────────────────────────────────────
    img = _stretch_histogram(img)

    ctx.preprocessed_image = img
    logger.info("Preprocessing done. Final size: %dx%d", img.shape[1], img.shape[0])
    return ctx
