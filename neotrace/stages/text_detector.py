# NeoSVG — TextDetector stage
# Uses PaddleOCR to find text regions, masks them in the working image,
# and stores them for reinsertion as real <text> SVG elements later.

import logging
from typing import List

import numpy as np

from context import Context, TextRegion

logger = logging.getLogger("neosvg.text_detector")


def _estimate_font_size(bbox) -> float:
    """Estimate font size in px from bounding box coordinates."""
    pts = np.array(bbox, dtype=float)
    h   = float(np.linalg.norm(pts[0] - pts[3]))  # left edge height
    h2  = float(np.linalg.norm(pts[1] - pts[2]))  # right edge height
    return (h + h2) / 2.0


def _bbox_to_xywh(bbox) -> tuple:
    pts = np.array(bbox, dtype=int)
    x   = int(pts[:, 0].min())
    y   = int(pts[:, 1].min())
    w   = int(pts[:, 0].max() - x)
    h   = int(pts[:, 1].max() - y)
    return x, y, w, h


def detect_text(ctx: Context) -> Context:
    """
    Detect text in ctx.preprocessed_image using PaddleOCR.
    On failure (library not installed, no GPU, etc.) logs a warning and
    continues — the pipeline always produces output.
    """
    if ctx.skip_text:
        logger.info("Text detection skipped (--no-text)")
        ctx.text_masked_image = ctx.preprocessed_image
        return ctx

    img = ctx.preprocessed_image
    if img is None:
        ctx.text_masked_image = img
        return ctx

    try:
        from paddleocr import PaddleOCR
        ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
        rgb = img[:, :, :3]
        result = ocr.ocr(rgb, cls=True)
    except Exception as exc:
        logger.warning("PaddleOCR failed (%s) — text detection skipped", exc)
        ctx.text_masked_image = img.copy()
        return ctx

    masked = img.copy()
    regions: List[TextRegion] = []

    if result and result[0]:
        for line in result[0]:
            bbox, (text, conf) = line[0], line[1]
            if conf < 0.5:
                continue
            font_sz = _estimate_font_size(bbox)
            xywh    = _bbox_to_xywh(bbox)
            regions.append(TextRegion(bbox=xywh, text=text,
                                      font_size_estimate=font_sz))

            # Mask the region with its dominant background colour
            x, y, bw, bh = xywh
            patch = img[y: y + bh, x: x + bw]
            # Use the border pixel average as fill
            border = np.concatenate([
                patch[0, :, :3].reshape(-1, 3),
                patch[-1, :, :3].reshape(-1, 3),
                patch[:, 0, :3].reshape(-1, 3),
                patch[:, -1, :3].reshape(-1, 3),
            ], axis=0)
            fill_rgb = border.mean(axis=0).astype(np.uint8)
            masked[y: y + bh, x: x + bw, :3] = fill_rgb
            logger.debug("Masked text '%s' at (%d,%d,%d,%d) font≈%.0fpx",
                         text, x, y, bw, bh, font_sz)

    logger.info("Detected %d text regions", len(regions))
    ctx.text_regions     = regions
    ctx.text_masked_image = masked
    return ctx
