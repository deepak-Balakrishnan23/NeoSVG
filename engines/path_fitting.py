# NeoSVG — shared path-fitting helpers
# Bezier curve fitting, corner detection and SVG path serialisation
# used by the NeoSVG Engine. All algorithms are NeoSVG originals; no
# third-party vectorization library is used.
#   * Path fitting           (pixel / polygon / spline modes)
#   * SVG path serialisation  (relative coordinates, configurable precision)

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import cv2
import numpy as np


# ── helpers ──────────────────────────────────────────────────────────────────

def _rgb_to_hex(r: int, g: int, b: int) -> str:
    return f"#{r:02x}{g:02x}{b:02x}"


def _chord_lengths(pts: np.ndarray) -> np.ndarray:
    """Cumulative chord-length parameterisation for a sequence of 2-D points."""
    diffs = np.diff(pts, axis=0)
    dists = np.linalg.norm(diffs, axis=1)
    t = np.concatenate([[0.0], np.cumsum(dists)])
    total = t[-1]
    if total < 1e-10:
        return np.linspace(0.0, 1.0, len(pts))
    return t / total


def _bernstein(t: float) -> Tuple[float, float, float, float]:
    """Cubic Bernstein basis values at parameter *t*."""
    s = 1.0 - t
    return s**3, 3 * s**2 * t, 3 * s * t**2, t**3


def _fit_cubic_bezier(
    pts: np.ndarray,
    t_hat1: np.ndarray,
    t_hat2: np.ndarray,
) -> Optional[np.ndarray]:
    """
    Fit a single cubic Bezier to *pts* with tangent constraints.
    Returns a (4, 2) array [P0, P1, P2, P3] or None if the fit is degenerate.
    """
    n = len(pts)
    t = _chord_lengths(pts)

    # Build A matrix (n × 2 × 2)
    A = np.zeros((n, 2, 2))
    for i in range(n):
        b = _bernstein(t[i])
        A[i, 0] = b[1] * t_hat1
        A[i, 1] = b[2] * t_hat2

    C = np.zeros((2, 2))
    X = np.zeros(2)
    for i in range(n):
        b = _bernstein(t[i])
        # contribution of the fixed end-points
        tmp = (pts[i]
               - pts[0]  * (b[0] + b[1])
               - pts[-1] * (b[2] + b[3]))
        C[0, 0] += A[i, 0] @ A[i, 0]
        C[0, 1] += A[i, 0] @ A[i, 1]
        C[1, 0] += A[i, 1] @ A[i, 0]
        C[1, 1] += A[i, 1] @ A[i, 1]
        X[0] += A[i, 0] @ tmp
        X[1] += A[i, 1] @ tmp

    det = C[0, 0] * C[1, 1] - C[0, 1] * C[1, 0]
    fallback = np.linalg.norm(pts[-1] - pts[0]) / 3.0

    if abs(det) < 1e-10:
        alpha1 = alpha2 = fallback
    else:
        alpha1 = (X[0] * C[1, 1] - X[1] * C[0, 1]) / det
        alpha2 = (C[0, 0] * X[1] - C[1, 0] * X[0]) / det
        if alpha1 < 1e-6 or alpha2 < 1e-6:
            alpha1 = alpha2 = fallback

    return np.array([
        pts[0],
        pts[0]  + alpha1 * t_hat1,
        pts[-1] + alpha2 * t_hat2,
        pts[-1],
    ])


def _max_fit_error(pts: np.ndarray, bez: np.ndarray) -> Tuple[float, int]:
    """Return (max_error, split_index) of the cubic Bezier approximation.

    split_at is clamped to [1, len(pts)-1] so callers can always split into
    two non-empty sub-segments without risking an empty left or right slice.
    """
    t = _chord_lengths(pts)
    max_err = 0.0
    split_at = max(1, len(pts) // 2)
    for i, ti in enumerate(t):
        b = _bernstein(ti)
        approx = (b[0] * bez[0] + b[1] * bez[1]
                  + b[2] * bez[2] + b[3] * bez[3])
        err = float(np.linalg.norm(pts[i] - approx))
        if err > max_err:
            max_err = err
            # Keep split_at in a valid range: never 0 (empty left) or
            # len(pts) (empty right).
            split_at = max(1, min(i, len(pts) - 1))
    return max_err, split_at


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-10 else v


def _left_tangent(pts: np.ndarray) -> np.ndarray:
    return _unit(pts[1] - pts[0])


def _right_tangent(pts: np.ndarray) -> np.ndarray:
    return _unit(pts[-2] - pts[-1])


def _fit_spline_segment(
    pts: np.ndarray,
    t_hat1: np.ndarray,
    t_hat2: np.ndarray,
    error_threshold: float,
    depth: int = 0,
    max_depth: int = 12,        # Phase 13b: was hardcoded at 8
) -> List[np.ndarray]:
    """
    Recursively fit cubic Bezier curves to a segment, splitting on error.
    Returns a list of (4, 2) Bezier control-point arrays.

    Phase 13b: max_depth raised from 8 → 12 (more recursion = better fit on
    complex curves) and exposed as a parameter so callers can tune per
    region size (large regions afford deeper recursion).
    """
    if len(pts) < 2:
        return []
    if len(pts) == 2:
        dist = np.linalg.norm(pts[-1] - pts[0]) / 3.0
        bez = np.array([pts[0], pts[0] + dist * t_hat1,
                        pts[-1] + dist * t_hat2, pts[-1]])
        return [bez]

    bez = _fit_cubic_bezier(pts, t_hat1, t_hat2)
    if bez is None:
        return []

    max_err, split_at = _max_fit_error(pts, bez)

    if max_err < error_threshold or depth >= max_depth:
        return [bez]

    mid = split_at
    t_mid_l = _right_tangent(pts[:mid + 1])
    t_mid_r = _left_tangent(pts[mid:])
    left  = _fit_spline_segment(pts[:mid + 1], t_hat1,  t_mid_l, error_threshold, depth + 1, max_depth)
    right = _fit_spline_segment(pts[mid:],     t_mid_r, t_hat2,  error_threshold, depth + 1, max_depth)
    return left + right


def _detect_corners(pts: np.ndarray, threshold_deg: float) -> List[int]:
    """
    Return indices of points where the angle between the incoming and
    outgoing direction vectors is less than *threshold_deg*.
    """
    corners = [0]
    n = len(pts)
    thresh_cos = math.cos(math.radians(threshold_deg))
    for i in range(1, n - 1):
        v1 = _unit(pts[i]     - pts[i - 1])
        v2 = _unit(pts[i + 1] - pts[i])
        cos_a = float(np.dot(v1, v2))
        if cos_a < thresh_cos:
            corners.append(i)
    corners.append(n - 1)
    return corners


# ── SVG path serialisation ────────────────────────────────────────────────────

def _beziers_to_path(beziers: List[np.ndarray], precision: int,
                     closed: bool = True) -> str:
    """Convert a list of cubic Bezier arrays to an SVG path *d* string."""
    if not beziers:
        return ""
    fmt = f".{precision}f"

    def f(v): return format(v, fmt)

    parts = [f"M {f(beziers[0][0][0])} {f(beziers[0][0][1])}"]
    for bez in beziers:
        p1, p2, p3 = bez[1], bez[2], bez[3]
        parts.append(f"c {f(p1[0]-bez[0][0])} {f(p1[1]-bez[0][1])} "
                     f"{f(p2[0]-bez[0][0])} {f(p2[1]-bez[0][1])} "
                     f"{f(p3[0]-bez[0][0])} {f(p3[1]-bez[0][1])}")
    if closed:
        parts.append("z")
    return " ".join(parts)


def _polygon_to_path(pts: np.ndarray, precision: int,
                     closed: bool = True) -> str:
    """Relative polygon path from contour points."""
    if len(pts) < 2:
        return ""
    fmt = f".{precision}f"

    def f(v): return format(v, fmt)

    parts = [f"M {f(pts[0][0])} {f(pts[0][1])}"]
    for i in range(1, len(pts)):
        dx = pts[i][0] - pts[i - 1][0]
        dy = pts[i][1] - pts[i - 1][1]
        parts.append(f"l {f(dx)} {f(dy)}")
    if closed:
        parts.append("z")
    return " ".join(parts)


def _pixel_to_path(pts: np.ndarray, precision: int) -> str:
    """Raw pixel-outline path (all corners kept)."""
    return _polygon_to_path(pts, precision, closed=True)


# ── contour → path ────────────────────────────────────────────────────────────

def _contour_to_path(
    contour: np.ndarray,
    mode: str,
    corner_threshold_deg: float,
    segment_length: float,
    error_threshold: float,
    precision: int,
    epsilon_override: Optional[float] = None,
    max_depth: int = 12,
) -> str:
    pts = contour.squeeze()
    if pts.ndim != 2 or len(pts) < 3:
        return ""
    pts = pts.astype(float)

    if mode == 'pixel':
        return _pixel_to_path(pts, precision)

    # Polygon: RDP simplification via OpenCV
    # epsilon_override lets callers pass an area-adaptive value; the default
    # preserves sub-pixel detail from skimage's marching-squares extractor.
    epsilon = epsilon_override if epsilon_override is not None else max(0.3, segment_length * 0.4)
    poly = cv2.approxPolyDP(contour, epsilon, closed=True)
    poly_pts = poly.squeeze().astype(float)
    if poly_pts.ndim != 2 or len(poly_pts) < 3:
        return ""

    if mode == 'polygon':
        return _polygon_to_path(poly_pts, precision)

    # Spline: detect corners, fit Bezier segments between them
    corners = _detect_corners(poly_pts, corner_threshold_deg)
    all_beziers: List[np.ndarray] = []

    for k in range(len(corners) - 1):
        seg = poly_pts[corners[k]: corners[k + 1] + 1]
        if len(seg) < 2:
            continue
        t1 = _left_tangent(seg)
        t2 = _right_tangent(seg)
        beziers = _fit_spline_segment(seg, t1, t2, error_threshold, max_depth=max_depth)
        all_beziers.extend(beziers)

    if not all_beziers:
        return _polygon_to_path(poly_pts, precision)

    return _beziers_to_path(all_beziers, precision, closed=True)
