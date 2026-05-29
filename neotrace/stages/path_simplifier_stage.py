# NeoSVG — PathSimplifier stage
# Runs the Visvalingam-Whyatt simplifier on every freeform path.
# Paths that already have a primitive_svg replacement are skipped.

import logging
from typing import List, Tuple

import numpy as np

from config import Config
from context import Context, TextRegion
from engines.visvalingam import simplify_svg_path

logger = logging.getLogger("neosvg.simplifier")


def _bbox_distance(bbox: Tuple[int, int, int, int],
                   px: float, py: float) -> float:
    """Minimum distance from point (px, py) to axis-aligned bounding box."""
    x, y, w, h = bbox
    cx = max(x, min(px, x + w))
    cy = max(y, min(py, y + h))
    return float(np.hypot(px - cx, py - cy))


def _path_centroid(d: str) -> Tuple[float, float]:
    """Very rough centroid: mean of all numeric pairs in the path string."""
    import re
    nums = [float(n) for n in re.findall(r"-?\d+(?:\.\d+)?", d)]
    if len(nums) < 2:
        return 0.0, 0.0
    xs = nums[0::2]
    ys = nums[1::2]
    return float(np.mean(xs)), float(np.mean(ys))


def _choose_tolerance(rec: dict, text_regions: List[TextRegion]) -> float:
    """
    Select simplification tolerance for a single path record.
    Paths near text bboxes get a tighter tolerance to preserve context.
    Large background-layer paths get a looser tolerance.
    """
    layer = rec.get("layer", "")
    if "background" in layer:
        return Config.BACKGROUND_SIMPLIFY_TOLERANCE

    if not text_regions:
        return Config.DEFAULT_SIMPLIFY_TOLERANCE

    cx, cy = _path_centroid(rec.get("d", ""))
    for tr in text_regions:
        d = _bbox_distance(tr.bbox, cx, cy)
        if d < Config.TEXT_PROXIMITY_PX:
            return Config.TEXT_ADJACENT_TOLERANCE

    return Config.DEFAULT_SIMPLIFY_TOLERANCE


def simplify_paths(ctx: Context) -> Context:
    """
    Apply Visvalingam-Whyatt simplification to all freeform paths.
    Counts saved nodes and logs the reduction ratio.

    Skipped entirely when the engine emits already-fitted Bezier paths
    (the NeoSVG Engine compound-path output) — running V-W on those destroys
    the curve fit and tanks SSIM.
    """
    engine = (getattr(ctx, "engine", "") or "").lower()
    if engine == "neosvg":
        logger.info("Path simplification skipped (engine=%s emits fitted paths)", engine)
        return ctx

    before = sum(p.get("d", "").count(" ") for p in ctx.collected_paths)
    simplified = 0

    for rec in ctx.collected_paths:
        if rec.get("primitive_svg"):
            continue

        d = rec.get("d", "")
        if not d:
            continue

        tol  = _choose_tolerance(rec, ctx.text_regions)
        new_d = simplify_svg_path(d, tol)

        if new_d != d:
            rec["d"] = new_d
            simplified += 1

    after = sum(p.get("d", "").count(" ") for p in ctx.collected_paths)
    reduction = (1 - after / max(before, 1)) * 100

    logger.info(
        "Path simplification: %d paths modified, ~%.1f%% node reduction",
        simplified, reduction,
    )
    return ctx
