# NeoSVG — Segmenter stage
# Separates foreground from background for PHOTO/CARTOON images using rembg.
# On failure the pipeline falls back to treating the whole image as one layer.

import logging

import cv2
import numpy as np

from config import Config
from context import Context, LayerData

logger = logging.getLogger("neosvg.segmenter")


def _mask_to_layer(img: np.ndarray, mask: np.ndarray,
                   name: str) -> LayerData:
    """Apply a boolean mask to *img*, returning a LayerData."""
    out       = img.copy()
    alpha_ch  = (mask * 255).astype(np.uint8)
    if out.shape[2] == 3:
        out = cv2.cvtColor(out, cv2.COLOR_RGB2RGBA)
    out[:, :, 3] = np.where(mask, out[:, :, 3], 0)
    return LayerData(name=name, image=out)


def segment(ctx: Context) -> Context:
    """
    Split the working image into foreground + background layers.

    PHOTO / CARTOON  → use rembg to estimate subject mask
    Others           → single layer (no segmentation needed)
    Skip flag        → single layer
    """
    working = ctx.text_masked_image if ctx.text_masked_image is not None \
        else ctx.preprocessed_image

    if working is None:
        return ctx

    skip = (
        ctx.skip_segmentation
        or ctx.image_type in ("LOGO", "ICON", "PIXELART")
        or ctx.quality == "fast"
    )

    if skip:
        logger.info("Segmentation skipped (type=%s, fast=%s, flag=%s)",
                    ctx.image_type, ctx.quality == "fast", ctx.skip_segmentation)
        ctx.layers = [LayerData(name="full", image=working.copy())]
        return ctx

    try:
        from rembg import remove as rembg_remove
        import PIL.Image

        pil_in  = PIL.Image.fromarray(working[:, :, :3], "RGB")
        pil_out = rembg_remove(pil_in)                # returns RGBA
        fg_arr  = np.array(pil_out)                   # RGBA
        alpha   = fg_arr[:, :, 3]
        fg_mask = alpha >= Config.REMBG_ALPHA_THRESHOLD
        bg_mask = ~fg_mask

        logger.info(
            "rembg segmentation: fg=%d px  bg=%d px",
            int(fg_mask.sum()), int(bg_mask.sum()),
        )

        fg_layer = _mask_to_layer(working, fg_mask, "foreground")
        bg_layer = _mask_to_layer(working, bg_mask, "background")
        ctx.layers = [bg_layer, fg_layer]   # background drawn first

    except Exception as exc:
        logger.warning("Segmentation failed (%s) — using single layer", exc)
        ctx.layers = [LayerData(name="full", image=working.copy())]

    return ctx
