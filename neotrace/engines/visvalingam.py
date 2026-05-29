# NeoSVG — Visvalingam-Whyatt path simplifier
# Original implementation. Algorithm concept: M. Visvalingam & J. D. Whyatt
# (1993) "Line generalisation by repeated elimination of points",
# Cartographic Journal 30(1). Public-domain algorithm — no library used.

import heapq
from typing import List, Tuple

import numpy as np


def _triangle_area(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Signed area of the triangle formed by three 2-D points."""
    return abs(
        (b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1])
    ) * 0.5


def simplify(points: List[Tuple[float, float]],
             tolerance: float,
             keep_endpoints: bool = True) -> List[Tuple[float, float]]:
    """
    Reduce a polyline to the fewest points whose triangle-area contribution
    exceeds *tolerance* px².

    Parameters
    ----------
    points      : ordered (x, y) list
    tolerance   : area threshold — points with effective area < this are removed
    keep_endpoints : always preserve first and last point (default True)

    Returns
    -------
    Simplified list of (x, y) tuples.
    """
    n = len(points)
    if n <= 2:
        return list(points)

    pts = [np.array(p, dtype=float) for p in points]

    # ── doubly-linked list so we can remove mid-points in O(1) ──────────────
    prev = list(range(-1, n - 1))  # prev[i] = index before i
    nxt  = list(range(1, n + 1))   # nxt[i]  = index after  i
    nxt[n - 1] = n - 1             # sentinel: last point points to itself
    prev[0]    = 0                 # sentinel: first point points to itself
    removed = [False] * n

    # ── initial areas ────────────────────────────────────────────────────────
    areas = [0.0] * n
    for i in range(1, n - 1):
        areas[i] = _triangle_area(pts[prev[i]], pts[i], pts[nxt[i]])

    # min-heap of (area, index)
    heap: List[Tuple[float, int]] = []
    for i in range(1, n - 1):
        heapq.heappush(heap, (areas[i], i))

    max_area_seen = 0.0

    while heap:
        area, idx = heapq.heappop(heap)

        # stale entry — skip
        if removed[idx]:
            continue
        if abs(area - areas[idx]) > 1e-10:
            continue

        # Visvalingam's monotonicity rule: effective area never decreases
        effective = max(area, max_area_seen)
        if effective >= tolerance:
            # everything remaining is above tolerance
            break

        max_area_seen = effective
        removed[idx] = True

        # re-link neighbours — update unconditionally; endpoints have
        # sentinel self-links so updating them is harmless.
        p = prev[idx]
        nx = nxt[idx]
        nxt[p] = nx
        prev[nx] = p

        # recompute area for affected neighbours
        for affected in (p, nx):
            if 0 < affected < n - 1 and not removed[affected]:
                pp = prev[affected]
                nn = nxt[affected]
                new_area = _triangle_area(pts[pp], pts[affected], pts[nn])
                areas[affected] = new_area
                heapq.heappush(heap, (new_area, affected))

    result = [points[i] for i in range(n) if not removed[i]]

    if keep_endpoints and result:
        if result[0] != points[0]:
            result.insert(0, points[0])
        if result[-1] != points[-1]:
            result.append(points[-1])

    return result


def simplify_svg_path(path_d: str, tolerance: float) -> str:
    """
    Parse an SVG path *d* attribute (absolute M/L/Z only), simplify the
    polyline vertices, and return a new *d* string with relative commands.

    Bezier paths (C/S/Q commands) are left unchanged — only line segments
    benefit from simplification.
    """
    tokens = path_d.strip().split()
    points: List[Tuple[float, float]] = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok in ('M', 'L'):
            x = float(tokens[i + 1])
            y = float(tokens[i + 2])
            points.append((x, y))
            i += 3
        elif tok == 'Z':
            i += 1
        else:
            # Contains curves — return unchanged
            return path_d

    if len(points) < 3:
        return path_d

    simplified = simplify(points, tolerance)
    if not simplified:
        return path_d

    # Emit relative path (lowercase commands)
    parts = [f"M {simplified[0][0]:.2f} {simplified[0][1]:.2f}"]
    for prev_pt, cur_pt in zip(simplified, simplified[1:]):
        dx = cur_pt[0] - prev_pt[0]
        dy = cur_pt[1] - prev_pt[1]
        parts.append(f"l {dx:.2f} {dy:.2f}")
    parts.append("z")
    return " ".join(parts)
