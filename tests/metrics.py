"""
Quality metrics for the regression suite.

SSIM alone is a poor judge of this pipeline: it under-weights the wrong-hue
patches that are the most visible failure, and it moves when band COUNT changes
even if nothing looks different. So the suite also measures perceptual colour
error (CIE Lab dE), the area of contiguous wrong-hue blotches, and error split
between edge and interior — which is what localises a regression to the tracer
versus the segmenter.
"""
from __future__ import annotations

import io
import os

import numpy as np

try:
    import cv2
except ImportError:                                    # pragma: no cover
    cv2 = None


def _find_cairo() -> None:
    """Same Cairo resolution the validator does — see validator._ensure_cairo_findable."""
    import ctypes.util
    original = ctypes.util.find_library
    if original("cairo"):
        return
    for d in ("/opt/homebrew/lib", "/usr/local/lib", "/opt/local/lib"):
        cand = os.path.join(d, "libcairo.2.dylib")
        if os.path.exists(cand):
            ctypes.util.find_library = lambda n, _c=cand, _o=original: (
                _c if "cairo" in n else _o(n))
            return


def rasterizer_available() -> bool:
    try:
        _find_cairo()
        import cairosvg           # noqa: F401
        return True
    except Exception:
        return False


def load_rgb(path: str) -> np.ndarray:
    """Load an image composited over white, as float RGB."""
    from PIL import Image
    im = Image.open(path).convert("RGBA")
    bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
    return np.asarray(Image.alpha_composite(bg, im).convert("RGB")).astype(np.float64)


def render_svg(svg_path: str, w: int, h: int) -> np.ndarray:
    """Rasterise an SVG over white at exactly w×h."""
    _find_cairo()
    import cairosvg
    from PIL import Image
    png = cairosvg.svg2png(url=svg_path, output_width=w, output_height=h)
    im = Image.open(io.BytesIO(png)).convert("RGBA")
    bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
    return np.asarray(Image.alpha_composite(bg, im).convert("RGB")).astype(np.float64)


def compare(ref_path: str, svg_path: str) -> dict:
    """Every metric the suite asserts on, for one (source, output) pair."""
    ref = load_rgb(ref_path)
    h, w = ref.shape[:2]
    out = render_svg(svg_path, w, h)

    err = np.abs(ref - out).mean(2)

    from skimage.metrics import structural_similarity
    ssim = float(structural_similarity(ref.astype(np.uint8), out.astype(np.uint8),
                                       channel_axis=2))

    # Perceptual colour error, and the area of contiguous visibly-wrong patches.
    reflab = cv2.cvtColor(ref.astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float64)
    outlab = cv2.cvtColor(out.astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float64)
    dE = np.sqrt(((reflab - outlab) ** 2).sum(2))
    mask = cv2.morphologyEx((dE > 15).astype(np.uint8), cv2.MORPH_OPEN,
                            np.ones((5, 5), np.uint8))
    _, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    blotch = int(sum(s[cv2.CC_STAT_AREA] for s in stats[1:]
                     if s[cv2.CC_STAT_AREA] >= 200))

    # Edge band = within ~2px of a real edge in the SOURCE.
    grey = ref.mean(2)
    gy, gx = np.gradient(grey)
    edge = cv2.dilate((np.hypot(gx, gy) > 12).astype(np.uint8),
                      np.ones((5, 5), np.uint8)).astype(bool)

    return {
        "ssim":      ssim,
        "mae":       float(err.mean()),
        "mean_dE":   float(dE.mean()),
        "blotch_px": blotch,
        "edge_mae":  float(err[edge].mean()) if edge.any() else 0.0,
        "int_mae":   float(err[~edge].mean()) if (~edge).any() else 0.0,
        "bytes":     os.path.getsize(svg_path),
    }
