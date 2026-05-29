# NeoSVG — PrimitiveDetector stage
# Replaces freeform paths with exact SVG primitives where the fit is tight.
# This is the single biggest quality/filesize improvement in the pipeline.

from __future__ import annotations

import logging
import math
import re
from typing import List, Optional, Tuple

import cv2
import numpy as np

from config import Config
from context import Context

logger = logging.getLogger("neosvg.primitives")


# ── path → point list ─────────────────────────────────────────────────────────

_TOKEN = re.compile(r"([MmLlCcZz])|(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)")


def _path_to_points(d: str) -> Optional[np.ndarray]:
    """
    Extract the on-curve points from an SVG path *d* attribute.
    Supports absolute M/L and relative m/l/c/z only.
    Returns (N, 2) float32 array or None if path is too short.
    """
    tokens = _TOKEN.findall(d)
    cmd    = None
    x = y  = 0.0
    pts: List[Tuple[float, float]] = []
    nums: List[float] = []
    i = 0

    while i < len(tokens):
        letter, number = tokens[i]
        i += 1
        if letter:
            if nums:
                _flush(cmd, nums, pts, x, y)
                if pts:
                    x, y = pts[-1]
            cmd = letter
            nums = []
        else:
            nums.append(float(number))

    _flush(cmd, nums, pts, x, y)
    if len(pts) < 3:
        return None
    return np.array(pts, dtype=np.float32)


def _flush(cmd, nums, pts, cx, cy):
    if cmd is None or not nums:
        return
    c = cmd.lower()
    is_rel = cmd.islower()
    idx = 0
    if c == 'm' or c == 'l':
        while idx + 1 < len(nums):
            nx = nums[idx] + (cx if is_rel else 0)
            ny = nums[idx + 1] + (cy if is_rel else 0)
            pts.append((nx, ny))
            cx, cy = nx, ny
            idx += 2
    elif c == 'c':
        while idx + 5 < len(nums):
            # skip control points; take endpoint
            ex = nums[idx + 4] + (cx if is_rel else 0)
            ey = nums[idx + 5] + (cy if is_rel else 0)
            pts.append((ex, ey))
            cx, cy = ex, ey
            idx += 6


# ── primitive fit functions ───────────────────────────────────────────────────

def _fit_circle(pts: np.ndarray) -> Optional[dict]:
    """
    Algebraic least-squares circle fit (Kåsa method).
    Returns {cx, cy, r, error} or None.
    """
    n = len(pts)
    if n < 5:
        return None
    x, y = pts[:, 0], pts[:, 1]
    A = np.column_stack([2 * x, 2 * y, np.ones(n)])
    b = x**2 + y**2
    try:
        result, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return None

    cx, cy = float(result[0]), float(result[1])
    r      = math.sqrt(max(0.0, float(result[2]) + cx**2 + cy**2))
    if r < 1.0:
        return None

    residuals = np.abs(np.sqrt((x - cx)**2 + (y - cy)**2) - r)
    diag      = math.sqrt((x.max()-x.min())**2 + (y.max()-y.min())**2) + 1e-6
    error     = float(residuals.mean() / diag)
    return {"cx": cx, "cy": cy, "r": r, "error": error}


def _fit_ellipse(pts: np.ndarray) -> Optional[dict]:
    """cv2.fitEllipse wrapper. Returns {cx,cy,rx,ry,angle,error} or None."""
    if len(pts) < 5:
        return None
    try:
        (cx, cy), (w, h), angle = cv2.fitEllipse(pts)
    except cv2.error:
        return None

    rx, ry = w / 2.0, h / 2.0
    cos_a  = math.cos(math.radians(angle))
    sin_a  = math.sin(math.radians(angle))
    x, y   = pts[:, 0] - cx, pts[:, 1] - cy
    xr =  x * cos_a + y * sin_a
    yr = -x * sin_a + y * cos_a
    residuals = np.abs(np.sqrt((xr / (rx + 1e-6))**2 + (yr / (ry + 1e-6))**2) - 1)
    diag  = math.sqrt((pts[:, 0].max() - pts[:, 0].min())**2 +
                      (pts[:, 1].max() - pts[:, 1].min())**2) + 1e-6
    error = float(residuals.mean() / diag * max(rx, ry))

    return {"cx": cx, "cy": cy, "rx": rx, "ry": ry,
            "angle": angle, "error": error}


def _fit_rect(pts: np.ndarray) -> Optional[dict]:
    """
    Check whether the convex hull of *pts* is close to a rectangle.
    Returns {x,y,w,h,angle,rounded,error} or None.
    """
    if len(pts) < 4:
        return None
    hull = cv2.convexHull(pts).squeeze()
    if hull.ndim != 2 or len(hull) < 4:
        return None

    rect  = cv2.minAreaRect(pts)
    box   = cv2.boxPoints(rect).astype(float)

    # Measure how far the actual hull points deviate from the box
    dists = []
    for hp in hull:
        min_d = min(
            float(cv2.pointPolygonTest(box.reshape(-1, 1, 2).astype(np.float32),
                                        tuple(hp.astype(float)), True))
            for _ in (1,)
        )
        dists.append(abs(min_d))

    diag  = math.sqrt(rect[1][0]**2 + rect[1][1]**2) + 1e-6
    error = float(np.mean(dists) / diag)

    # Verify corner angles ≈ 90°
    n = len(box)
    max_angle_err = 0.0
    for i in range(n):
        a = box[(i - 1) % n]
        b = box[i]
        c = box[(i + 1) % n]
        v1 = a - b
        v2 = c - b
        nv1 = np.linalg.norm(v1) + 1e-9
        nv2 = np.linalg.norm(v2) + 1e-9
        cos_a = float(np.dot(v1, v2) / (nv1 * nv2))
        angle_err = abs(math.degrees(math.acos(max(-1.0, min(1.0, cos_a)))) - 90.0)
        max_angle_err = max(max_angle_err, angle_err)

    if max_angle_err > Config.RECT_ANGLE_TOLERANCE_DEG:
        return None

    cx, cy  = float(rect[0][0]), float(rect[0][1])
    rw, rh  = float(rect[1][0]), float(rect[1][1])
    angle   = float(rect[2])
    x_ul    = cx - rw / 2
    y_ul    = cy - rh / 2
    return {"x": x_ul, "y": y_ul, "w": rw, "h": rh,
            "angle": angle, "error": error}


def _fit_line(pts: np.ndarray) -> Optional[dict]:
    """Return {x1,y1,x2,y2,error} if points are collinear."""
    if len(pts) < 2:
        return None
    [vx, vy, x0, y0] = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
    dx, dy = float(vx), float(vy)
    # Distance of each point to the line
    perp = np.abs((pts[:, 0] - x0) * (-dy) + (pts[:, 1] - y0) * dx)
    error = float(perp.mean())
    if error > Config.LINE_COLLINEAR_TOL:
        return None
    t     = (pts[:, 0] - x0) * dx + (pts[:, 1] - y0) * dy
    t_min, t_max = float(t.min()), float(t.max())
    return {
        "x1": x0 + t_min * dx, "y1": y0 + t_min * dy,
        "x2": x0 + t_max * dx, "y2": y0 + t_max * dy,
        "error": error,
    }


# ── SVG primitive builders ───────────────────────────────────────────────────

def _circle_svg(c: dict, color: str, opacity: float, prec: int = 2) -> str:
    f = f".{prec}f"
    return (f'<circle cx="{format(c["cx"], f)}" cy="{format(c["cy"], f)}" '
            f'r="{format(c["r"], f)}" fill="{color}" '
            f'fill-opacity="{opacity:.2f}"/>')


def _ellipse_svg(e: dict, color: str, opacity: float, prec: int = 2) -> str:
    f = f".{prec}f"
    return (f'<ellipse cx="{format(e["cx"], f)}" cy="{format(e["cy"], f)}" '
            f'rx="{format(e["rx"], f)}" ry="{format(e["ry"], f)}" '
            f'transform="rotate({format(e["angle"], f)},{format(e["cx"], f)},{format(e["cy"], f)})" '
            f'fill="{color}" fill-opacity="{opacity:.2f}"/>')


def _rect_svg(r: dict, color: str, opacity: float, prec: int = 2) -> str:
    f = f".{prec}f"
    if abs(r["angle"]) < 0.5:
        return (f'<rect x="{format(r["x"], f)}" y="{format(r["y"], f)}" '
                f'width="{format(r["w"], f)}" height="{format(r["h"], f)}" '
                f'fill="{color}" fill-opacity="{opacity:.2f}"/>')
    cx = r["x"] + r["w"] / 2
    cy = r["y"] + r["h"] / 2
    return (f'<rect x="{format(r["x"], f)}" y="{format(r["y"], f)}" '
            f'width="{format(r["w"], f)}" height="{format(r["h"], f)}" '
            f'transform="rotate({format(r["angle"], f)},{format(cx, f)},{format(cy, f)})" '
            f'fill="{color}" fill-opacity="{opacity:.2f}"/>')


def _line_svg(l: dict, color: str, opacity: float, prec: int = 2) -> str:
    f = f".{prec}f"
    return (f'<line x1="{format(l["x1"], f)}" y1="{format(l["y1"], f)}" '
            f'x2="{format(l["x2"], f)}" y2="{format(l["y2"], f)}" '
            f'stroke="{color}" stroke-opacity="{opacity:.2f}" stroke-width="1"/>')


# ── main stage ────────────────────────────────────────────────────────────────

def detect_primitives(ctx: Context) -> Context:
    """
    For every collected path, try to replace it with a simpler SVG primitive.
    Modifies path records in-place: adds a 'primitive_svg' key when a match
    is found; the assembler uses that instead of the <path> element.
    """
    if ctx.skip_primitives:
        logger.info("Primitive detection skipped (--no-primitives)")
        return ctx

    img = ctx.preprocessed_image if ctx.preprocessed_image is not None else ctx.original_image
    img_area = float(img.shape[0] * img.shape[1]) if img is not None else 1.0
    max_path_area = img_area * Config.MAX_PRIMITIVE_AREA_FRAC

    replaced = 0
    for rec in ctx.collected_paths:
        d       = rec.get("d", "")
        color   = rec.get("color", "#000000")
        opacity = rec.get("opacity", 1.0)

        pts = _path_to_points(d)
        if pts is None or len(pts) < Config.MIN_PRIMITIVE_POINTS:
            continue

        path_area = float(cv2.contourArea(cv2.convexHull(pts)))
        too_large_for_rect = path_area > max_path_area

        primitive_svg = None

        # Try circle first — circles are accurate even when large
        c = _fit_circle(pts)
        if c and c["error"] < Config.CIRCLE_FIT_TOLERANCE:
            primitive_svg = _circle_svg(c, color, opacity)
            logger.debug("Replaced path with circle (err=%.3f)", c["error"])

        if primitive_svg is None:
            # Try line
            l = _fit_line(pts)
            if l:
                primitive_svg = _line_svg(l, color, opacity)
                logger.debug("Replaced path with line (err=%.2f)", l["error"])

        if primitive_svg is None and not too_large_for_rect:
            # Rect fitting only for small-ish paths — large irregular shapes
            # (e.g. garment silhouettes) pass the convex-hull rect test but the
            # rect destroys all the real shape detail.
            r = _fit_rect(pts)
            if r and r["error"] < Config.ELLIPSE_FIT_TOLERANCE:
                primitive_svg = _rect_svg(r, color, opacity)
                logger.debug("Replaced path with rect (err=%.3f)", r["error"])

        if primitive_svg is None:
            # Try ellipse
            e = _fit_ellipse(pts)
            if e and e["error"] < Config.ELLIPSE_FIT_TOLERANCE:
                primitive_svg = _ellipse_svg(e, color, opacity)
                logger.debug("Replaced path with ellipse (err=%.3f)", e["error"])

        if primitive_svg:
            rec["primitive_svg"] = primitive_svg
            replaced += 1

    logger.info("Primitive detection: %d/%d paths replaced",
                replaced, len(ctx.collected_paths))
    return ctx
