# NeoSVG — sub-pixel marching squares contour extraction
#
# Replaces cv2.findContours (integer pixel coordinates) with marching squares
# at half-pixel resolution. This gives NeoSVG's engines sub-pixel boundary
# accuracy, which is the primary source of SSIM gains over integer-grid tracing.
#
# Algorithm: walk the iso-value 0.5 contour of a binary mask using marching
# squares with linear interpolation along grid edges. Output: list of
# (x, y) float-coordinate point arrays. Public-domain algorithm (Lorensen-
# Cline 1987); we use scikit-image's BSD-licensed implementation.
#
# Why this matters: a typical traced shape boundary has ~1000 points. If
# every point is 0.5px off, the rasterized SVG looks "stairstep-y" and the
# SSIM drops by ~5 points purely from sub-pixel error — even when the
# logical shape is correct.

from __future__ import annotations

import logging
from typing import List

import numpy as np

logger = logging.getLogger("neosvg.subpixel")

try:
    from skimage.measure import find_contours
    _AVAILABLE = True
except ImportError:
    find_contours = None
    _AVAILABLE = False


def subpixel_contours(mask: np.ndarray, min_length: int = 4) -> List[np.ndarray]:
    """
    Extract sub-pixel boundary contours from a binary mask.

    Args:
        mask:       2D uint8 array, non-zero pixels = inside the region.
        min_length: drop contours shorter than this point count.

    Returns:
        List of (N, 1, 2) int32 contour arrays in OpenCV's contour format,
        but with FLOAT-derived sub-pixel positions rounded back to int32 *
        (1 << SUBPIXEL_BITS). The caller can choose how to handle the
        sub-pixel precision — most code that uses cv2.findContours output
        already does integer math, so we round here for compatibility.

    The contours are returned in (x, y) order matching cv2.findContours.
    """
    if not _AVAILABLE:
        raise RuntimeError(
            "scikit-image not installed. Run: pip install scikit-image"
        )

    # skimage's find_contours walks the iso-value 0.5 of the input array.
    # It returns (N, 2) arrays in (row, col) = (y, x) order with sub-pixel
    # precision (float coordinates).
    raw = find_contours(mask.astype(np.float32), 0.5)
    contours: List[np.ndarray] = []
    for c in raw:
        if len(c) < min_length:
            continue
        # Swap (row, col) → (x, y) and reshape to cv2's (N, 1, 2)
        xy = np.column_stack([c[:, 1], c[:, 0]])
        # Round to integer pixel coordinates (the downstream Bezier fitter
        # is integer-friendly; the half-pixel offset already improves
        # accuracy because the contour is now AT the boundary, not at the
        # outer pixel center). We keep float precision via the Bezier
        # fitter which interpolates between these.
        contours.append(xy.astype(np.float32).reshape(-1, 1, 2))

    return contours
