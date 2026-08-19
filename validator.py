# NeoSVG — QualityValidator
# Rasterises the output SVG back to PNG and computes similarity / size metrics.

import logging
import os

import numpy as np

from config import Config
from context import Context

logger = logging.getLogger("neosvg.validator")

#: Where a Homebrew/MacPorts Cairo actually lives. cairocffi resolves the
#: library through ctypes.util.find_library(), which on macOS only searches the
#: system paths — so a perfectly good `brew install cairo` is invisible to it
#: and every run reports SSIM as n/a. DYLD_* environment variables cannot fix
#: this either: System Integrity Protection strips them from the hardened
#: /usr/bin/python3. Resolving the path ourselves is the only reliable route.
_CAIRO_SEARCH_DIRS = ("/opt/homebrew/lib", "/usr/local/lib", "/opt/local/lib")
_CAIRO_SONAMES = ("libcairo.2.dylib", "libcairo.so.2", "libcairo-2.dll", "cairo")
_cairo_patched = False


def _ensure_cairo_findable() -> None:
    """
    Teach ctypes.util.find_library() about Cairo installed outside the system
    prefix. No-op when the default lookup already succeeds, so a Linux box or a
    system-wide Cairo behaves exactly as before.
    """
    global _cairo_patched
    if _cairo_patched:
        return
    _cairo_patched = True

    import ctypes.util

    original = ctypes.util.find_library
    if original("cairo"):
        return

    resolved = {}
    for directory in _CAIRO_SEARCH_DIRS:
        candidate = os.path.join(directory, "libcairo.2.dylib")
        if os.path.exists(candidate):
            resolved = {name: candidate for name in _CAIRO_SONAMES}
            logger.debug("Cairo resolved to %s", candidate)
            break
    if not resolved:
        return

    def find_library(name):
        return resolved.get(name) or original(name)

    ctypes.util.find_library = find_library


def _rasterize_svg(svg_string: str, width: int, height: int) -> np.ndarray:
    """
    Rasterise *svg_string* to an RGB ndarray (height × width × 3) uint8.
    Requires cairosvg.
    """
    _ensure_cairo_findable()

    import cairosvg
    import io
    from PIL import Image

    png_bytes = cairosvg.svg2png(
        bytestring=svg_string.encode("utf-8"),
        output_width=width,
        output_height=height,
    )
    # cairosvg renders onto transparency; .convert("RGB") would drop alpha and
    # leave every unpainted pixel BLACK, which scores a white-background image
    # as wildly wrong. Composite onto white to match how the SVG is viewed.
    img = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
    canvas = Image.new("RGBA", img.size, (255, 255, 255, 255))
    return np.array(Image.alpha_composite(canvas, img).convert("RGB"))


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
        # Composite the source over white too, so both sides of the comparison
        # treat transparency identically (see _rasterize_svg).
        if original.ndim == 3 and original.shape[2] == 4:
            a = original[:, :, 3:4].astype(np.float32) / 255.0
            orig_rgb = (original[:, :, :3].astype(np.float32) * a +
                        255.0 * (1.0 - a)).astype(np.uint8)
        else:
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
