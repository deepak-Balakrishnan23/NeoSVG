# NeoSVG — QualityValidator
# Rasterises the output SVG back to PNG and computes similarity / size metrics.

import logging
import os

import numpy as np

from config import Config
from context import Context

logger = logging.getLogger("neosvg.validator")


def _rasterize_svg(svg_string: str, width: int, height: int) -> np.ndarray:
    """
    Rasterise *svg_string* to an RGB ndarray (height × width × 3) uint8.
    Requires cairosvg.
    """
    import cairosvg
    import io
    from PIL import Image

    png_bytes = cairosvg.svg2png(
        bytestring=svg_string.encode("utf-8"),
        output_width=width,
        output_height=height,
    )
    img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
    return np.array(img)


def _ssim(a: np.ndarray, b: np.ndarray) -> float:
    """
    Structural Similarity Index (SSIM) — lightweight implementation.
    Uses scikit-image if available, otherwise a simplified local version.
    """
    try:
        from skimage.metrics import structural_similarity
        score, _ = structural_similarity(a, b, full=True, channel_axis=2)
        return float(score)
    except ImportError:
        pass

    # Fallback: mean absolute error mapped to [0,1]
    diff = np.abs(a.astype(float) - b.astype(float))
    return float(1.0 - diff.mean() / 255.0)


def validate(ctx: Context) -> Context:
    """
    Compute quality metrics and log the quality report.
    Stores results in ctx (ssim, path_count, node_count, svg_size_bytes,
    png_size_bytes).
    """
    if not ctx.final_svg:
        logger.warning("No SVG to validate")
        return ctx

    svg_bytes     = ctx.final_svg.encode("utf-8")
    ctx.svg_size_bytes = len(svg_bytes)

    # Original PNG size
    if ctx.input_path and os.path.exists(ctx.input_path):
        ctx.png_size_bytes = os.path.getsize(ctx.input_path)

    # SSIM against original
    original = ctx.original_image
    if original is None:
        logger.info(
            "Quality report — Paths: %d | Nodes: %d | SVG: %d B | PNG: %d B",
            ctx.path_count, ctx.node_count,
            ctx.svg_size_bytes, ctx.png_size_bytes,
        )
        return ctx

    h, w = original.shape[:2]
    try:
        raster = _rasterize_svg(ctx.final_svg, w, h)
        # Keep original in RGB for SSIM (cairosvg rasterises to RGB)
        orig_rgb = original[:, :, :3]
        ctx.ssim = _ssim(orig_rgb, raster)
        warn = " ⚠  SSIM below threshold" if ctx.ssim < Config.MIN_ACCEPTABLE_SSIM else ""
    except Exception as exc:
        logger.warning("SVG rasterisation failed (%s) — SSIM not computed", exc)
        ctx.ssim = -1.0
        warn = ""

    logger.info(
        "Quality report — SSIM: %.3f%s | Paths: %d | Nodes: %d | "
        "SVG: %d B vs PNG: %d B",
        ctx.ssim, warn,
        ctx.path_count, ctx.node_count,
        ctx.svg_size_bytes, ctx.png_size_bytes,
    )
    return ctx
