# NeoSVG — shared engine preprocessing
#
# These transforms run before the NeoSVG Engine traces the image.
#
# Two operations:
#   1. flatten_alpha() — composite anti-aliased alpha onto a solid background
#      and hard-mask alpha to 0/255. Eliminates the phantom grey "dots"
#      that scatter along the silhouette of transparent-background PNGs.
#   2. boost_edges() — gentle unsharp mask + 2× upscale. Sharpens crisp
#      line content for cleaner traces; off by default for photographs.

from __future__ import annotations

import logging
from typing import Tuple

import cv2
import numpy as np

from config import Config

logger = logging.getLogger("neosvg.preprocessing")


def has_partial_alpha(image: np.ndarray) -> bool:
    """True if any pixel's alpha is between 1 and 254 (anti-aliased)."""
    if image.ndim != 3 or image.shape[2] != 4:
        return False
    a = image[:, :, 3]
    return bool(np.any((a > 0) & (a < 255)))


#: Key color used for transparent pixels. Chosen so it's vanishingly rare in
#: real photos/designs but is a single distinct color that engines will trace
#: as one region (rather than blending into adjacent subject colors).
KEY_COLOR = (255, 0, 255)   # neon magenta — never appears in apparel mockups
KEY_COLOR_HEX = "#ff00ff"


def flatten_alpha(image: np.ndarray, use_key_color: bool = False,
                  key_rgb: Tuple[int, int, int] = KEY_COLOR) -> np.ndarray:
    """
    Hard-mask the alpha channel to 0/255. Optionally replace the RGB of
    transparent pixels with a KEY color so engines that don't respect alpha
    don't blend transparent corners into subject regions.

    Two modes:
      use_key_color=False (default mode):
        Preserve transparency. Hard-mask alpha to 0 or 255. RGB is unchanged.
        The NeoSVG Engine detects alpha=0 pixels and handles keying internally.

      use_key_color=True (for engines that ignore alpha):
        Set alpha=255 everywhere. Replace RGB of transparent pixels with the
        KEY color (neon magenta). cv2.floodFill / region growing will trace
        those pixels as one large KEY-colored region which we strip later.

    Both modes eliminate anti-aliased edge dots by binarising alpha first.
    """
    if image.ndim != 3 or image.shape[2] != 4:
        return image

    hard_alpha = (image[:, :, 3] >= 128).astype(np.uint8) * 255
    transparent_mask = hard_alpha == 0

    rgb_out = image[:, :, :3].copy()

    if use_key_color:
        rgb_out[transparent_mask] = np.array(key_rgb, dtype=np.uint8)
        # alpha=255 everywhere so engines see a fully-opaque image
        return np.dstack([rgb_out, np.full_like(hard_alpha, 255)])

    return np.dstack([rgb_out, hard_alpha])


def boost_edges(image: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Pre-process for higher fidelity tracing:
      1. Unsharp mask on RGB so anti-aliased edges become crisp transitions.
      2. Upscale by FIDELITY_UPSCALE_FACTOR (capped on the biggest side) so
         the region grower sees more boundary detail.

    Returns (processed_image, scale_factor). Engines that emit absolute
    coordinates should divide them by the scale factor to bring the SVG
    back into the original coordinate space.
    """
    if image.ndim != 3 or image.shape[2] != 4:
        raise ValueError("boost_edges expects an RGBA image")

    h, w = image.shape[:2]
    rgb   = image[:, :, :3]
    alpha = image[:, :, 3]

    # Gentle unsharp mask — strong sharpening creates dark halos that get
    # traced as separate "ink" outline paths.
    blurred = cv2.GaussianBlur(rgb, (0, 0), Config.FIDELITY_SHARPEN_RADIUS)
    sharpened = cv2.addWeighted(
        rgb, 1.0 + Config.FIDELITY_SHARPEN_AMOUNT,
        blurred, -Config.FIDELITY_SHARPEN_AMOUNT,
        0,
    )
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)

    # 2× upscale (cap on biggest side to avoid memory blow-up)
    scale = Config.FIDELITY_UPSCALE_FACTOR
    max_side = max(h, w) * scale
    if max_side > Config.FIDELITY_UPSCALE_MAX_SIDE:
        scale = Config.FIDELITY_UPSCALE_MAX_SIDE / max(h, w)
    if scale > 1.0:
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        sharpened = cv2.resize(sharpened, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        alpha     = cv2.resize(alpha,     (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return np.dstack([sharpened, alpha]), scale


def prepare_for_engine(image: np.ndarray, max_fidelity: bool = False,
                       use_key_color: bool = False) -> Tuple[np.ndarray, float]:
    """
    Apply flatten_alpha + (optionally) boost_edges. Returns (image, scale).

    use_key_color=True is needed for engines that don't respect the alpha
    channel during region growing (anything using cv2.floodFill or similar).
    For the NeoSVG Engine (which handles keying internally), use_key_color=False.

    scale is the factor by which the output is larger than the input.
    """
    scale = 1.0
    if has_partial_alpha(image) or use_key_color:
        image = flatten_alpha(image, use_key_color=use_key_color)
        logger.info(
            "Alpha flattened — %s",
            "key-color mode" if use_key_color else "hard mask",
        )

    if max_fidelity:
        image, scale = boost_edges(image)
        logger.info("Boost edges: scale=%.2fx, sharpen=on", scale)

    return image, scale
