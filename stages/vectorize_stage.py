# NeoSVG — VectorizeStage
# Dispatches each layer to the user-selected engine and collects SVG paths.
#
# Universal preprocessing:
#   * Alpha-flatten — runs whenever the input has anti-aliased alpha edges.
#     Eliminates the phantom grey "dots" along transparent-PNG silhouettes.
#   * Boost edges — runs when the user enables the toggle. 2× upscale +
#     gentle unsharp mask. Applied uniformly across all engines.
#
# Engine:
#   * 'neosvg'  — The NeoSVG Engine: high-fidelity hierarchical region-growing
#                 with sub-pixel Bezier curve fitting. This is the only engine.

import logging
import re

import numpy as np

from context import Context, GradientRegion
from engines.preprocessing import KEY_COLOR_HEX, prepare_for_engine

logger = logging.getLogger("neosvg.vectorize")

_NUM_RE = re.compile(r'-?\d+(?:\.\d+)?')

# The NeoSVG Engine emits a PARTITION of flat-color tiles — every pixel belongs to
# exactly one region, so neighboring tiles meet along a shared boundary.  When
# the SVG is rendered, those boundaries are anti-aliased, leaving a sub-pixel
# seam where neither tile is fully opaque.  On a transparent input that seam is
# invisible (the neighbor's color shows through), but the assembler paints an
# opaque background rect behind opaque inputs — and that background bleeds
# through every seam as a bright hairline.  Across a smooth photographic
# gradient this reads as a field of thin "scratch" strokes.
#
# The fix is to stroke each tile with its OWN fill color so adjacent tiles
# overlap along their shared edge and seal the seam.  Because stroke == fill
# the visible color is unchanged; only the previously-leaking seam pixels are
# covered.  Width is expressed in FINAL (post-scale) coordinates, so it is
# applied AFTER any coordinate rescale below.
#
# A single global width can't win: a SMOOTH gradient band needs a WIDE overlap
# to fully bury its seams (and the pin-holes left where the Bezier fitter drops
# thin slivers), while a BUSY detail tile (droplet, thin stripe) needs a
# HAIRLINE stroke or wide overlap would blob the fine feature.  So the width is
# interpolated by each tile's `smooth_ctx` (0 = busy subject, 1 = flat
# gradient), which the engine tags onto every record.
_SEAM_STROKE_MIN = 0.6   # busy/detail tiles — keep edges crisp
_SEAM_STROKE_MAX = 4.0   # flat gradient bands — fully seal seams + pin-holes


def _seal_seams(rec: dict) -> None:
    """Replace a flat-tile record's rendering with a stroke=fill <path> so the
    background can't bleed through the anti-aliased seam between tiles.  Stroke
    width scales with the tile's smoothness context."""
    d = rec.get("d")
    if not d:
        return
    t = float(rec.get("smooth_ctx", 0.0))
    sw = _SEAM_STROKE_MIN + max(0.0, min(1.0, t)) * (_SEAM_STROKE_MAX - _SEAM_STROKE_MIN)
    color = rec.get("color", "#000000")
    op = float(rec.get("opacity", 1.0))
    extra = (f' fill-opacity="{op:.2f}" stroke-opacity="{op:.2f}"'
             if op < 1.0 else "")
    tr = rec.get("transform")
    tr_attr = f' transform="{tr}"' if tr else ""
    rec["primitive_svg"] = (
        f'<path d="{d}" fill="{color}" stroke="{color}" '
        f'stroke-width="{sw:.2f}" stroke-linejoin="round"'
        f'{extra}{tr_attr}/>'
    )


def _select_engine(name: str):
    """Return the NeoSVG Engine vectorize() function, plus its label.

    NeoSVG ships a single engine; the ``name`` argument is accepted for
    backwards compatibility but is always resolved to the NeoSVG Engine."""
    from engines.hierarchical_grow_vectorizer import vectorize
    return vectorize, "neosvg"


def _scale_path_coords(d: str, factor: float) -> str:
    """Multiply every number in an SVG path d-string by factor (no-op if 1.0)."""
    if factor == 1.0:
        return d
    return _NUM_RE.sub(lambda m: f"{float(m.group(0)) * factor:.3f}", d)


def _scale_record(rec: dict, factor: float) -> dict:
    if factor == 1.0:
        return rec
    if rec.get("d"):
        rec["d"] = _scale_path_coords(rec["d"], factor)
    tr = rec.get("transform")
    if tr:
        m = re.match(r'translate\(([-\d.]+),\s*([-\d.]+)\)', tr)
        if m:
            tx = float(m.group(1)) * factor
            ty = float(m.group(2)) * factor
            rec["transform"] = f"translate({tx:.3f},{ty:.3f})"
    return rec


def run_vectorize(ctx: Context) -> Context:
    """
    Vectorize via the selected engine. Alpha-flatten and (optional) Boost-edges
    preprocessing are applied uniformly before any engine runs.
    """
    detail = ctx.detail or "high"
    engine_name = (getattr(ctx, "engine", None) or "neosvg").lower()
    max_fidelity = bool(getattr(ctx, "max_fidelity", False))
    vec, label = _select_engine(engine_name)

    # Mark whether the original had transparency so the assembler can skip
    # the white background fill and produce a transparent SVG instead.
    if ctx.original_image is not None and ctx.original_image.shape[2] == 4:
        a = ctx.original_image[:, :, 3]
        ctx.has_transparency = bool(np.any(a < 255))

    # The NeoSVG Engine is high-fidelity — it applies its own perceptual color
    # clustering and needs to see the un-preprocessed original.  In particular,
    # the preprocess stage applies a heavy CARTOON bilateral filter
    # (sigmaColor=75) on smooth-gradient images, which collapses subtle color
    # variation into flat blocks before clustering — exactly the posterization
    # we want to avoid.  So we feed it the original whole image directly.
    if ctx.original_image is not None:
        source = ctx.original_image
        per_layer = False
    else:
        source = None
        per_layer = True

    if per_layer:
        logger.info(
            "Vectorizing %d layer(s) — engine=%s detail=%s max_fidelity=%s",
            len(ctx.layers), label, detail, max_fidelity,
        )
        # The key-color trick keeps transparent corners from blending into
        # subject regions during region growing.
        use_key = ctx.has_transparency
        all_paths = []
        for layer in ctx.layers:
            if layer.image is None or layer.image.size == 0:
                logger.warning("Layer '%s' has no image; skipping", layer.name)
                continue

            prepped, scale = prepare_for_engine(
                layer.image, max_fidelity=max_fidelity, use_key_color=use_key,
            )
            paths = vec(image=prepped, detail=detail)

            inv = 1.0 / scale if scale != 1.0 else 1.0
            for p in paths:
                p["layer"] = layer.name
                if inv != 1.0:
                    _scale_record(p, inv)
            layer.svg_paths = paths
            all_paths.extend(paths)
            logger.info("Layer '%s': %d paths", layer.name, len(paths))

        # Strip key-color paths (they represented the transparent corners)
        if use_key:
            before = len(all_paths)
            all_paths = [p for p in all_paths
                         if (p.get("color") or "").lower() != KEY_COLOR_HEX]
            logger.info("Stripped %d key-color paths (transparent regions)",
                        before - len(all_paths))

        ctx.collected_paths.extend(all_paths)
        logger.info("Total paths after vectorization: %d", len(ctx.collected_paths))
        return ctx

    # Single-source path: hand the NeoSVG Engine the original whole image.
    logger.info(
        "Engine=%s on original image — max_fidelity=%s detail=%s",
        label, max_fidelity, detail,
    )
    prepped, scale = prepare_for_engine(source, max_fidelity=max_fidelity)

    # The NeoSVG Engine's bilateral strength is tuned per detail level in Config
    # (a STRONG bilateral is what keeps photographic noise from fragmenting
    # the output), so no per-image-type override is applied here.  We already
    # applied boost_edges via preprocessing — pass max_fidelity=False to the
    # engine to avoid double-applying.
    try:
        paths = vec(image=prepped, detail=detail, max_fidelity=False)
    except TypeError:
        paths = vec(image=prepped, detail=detail)

    inv = 1.0 / scale if scale != 1.0 else 1.0

    # Extract any gradient-region records into ctx.gradient_regions.  The
    # NeoSVG Engine may emit a <linearGradient> stand-in for the smooth
    # photographic background — the assembler renders that as a continuous
    # SVG gradient instead of N banded fill polygons.
    real_paths = []
    for p in paths:
        if p.get("type") == "linear_gradient":
            x, y, bw, bh = p["bbox"]
            if inv != 1.0:
                x, y, bw, bh = (int(round(v * inv)) for v in (x, y, bw, bh))
            ctx.gradient_regions.append(GradientRegion(
                bbox=(x, y, bw, bh),
                gradient_type="linear",
                params={"angle": p["angle"], "stops": p["stops"]},
            ))
            continue
        p["layer"] = "foreground"
        if inv != 1.0:
            _scale_record(p, inv)
        # Seal anti-aliased tile seams so the assembler's background fill
        # can't bleed through them as scratch hairlines.
        if label == "neosvg":
            _seal_seams(p)
        real_paths.append(p)

    ctx.collected_paths.extend(real_paths)
    logger.info(
        "Total paths after vectorization: %d (+ %d gradient regions)",
        len(real_paths), len(ctx.gradient_regions),
    )
    return ctx
