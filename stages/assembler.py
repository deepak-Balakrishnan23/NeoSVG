# NeoSVG — SVGAssembler stage
# Composes the final SVG from all pipeline pieces in the correct paint order.
# Layer order (bottom → top):
#   1. Background colour rect
#   2. Gradient regions (<defs> + filled <rect>/<circle>)
#   3. Traced background layer
#   4. Traced foreground layers
#   5. Primitives (already embedded in path records)
#   6. Freeform simplified paths
#   7. Text elements as real <text> nodes

from __future__ import annotations

import logging
from typing import List, Optional

from context import Context, GradientRegion, TextRegion

logger = logging.getLogger("neosvg.assembler")


# ── gradient <defs> builders ─────────────────────────────────────────────────

def _linear_gradient_def(gid: str, region: GradientRegion) -> str:
    angle  = region.params.get("angle", 90.0)
    stops  = region.params.get("stops", [])
    rad    = angle * (3.14159265 / 180.0)
    x2     = round(0.5 + 0.5 * (0 if abs(angle - 90) < 1 else
                                  (1 if abs(angle) < 1 else -1)), 4)
    y2     = round(0.5 + 0.5 * (1 if abs(angle) < 1 else 0), 4)

    stop_els = "\n".join(
        f'    <stop offset="{s["offset"]}" stop-color="{s["color"]}"/>'
        for s in stops
    )
    return (f'  <linearGradient id="{gid}" '
            f'x1="0" y1="0" x2="{x2}" y2="{y2}" '
            f'gradientUnits="objectBoundingBox">\n'
            f'{stop_els}\n'
            f'  </linearGradient>')


def _radial_gradient_def(gid: str, region: GradientRegion) -> str:
    stops = region.params.get("stops", [])
    stop_els = "\n".join(
        f'    <stop offset="{s["offset"]}" stop-color="{s["color"]}"/>'
        for s in stops
    )
    return (f'  <radialGradient id="{gid}" '
            f'cx="0.5" cy="0.5" r="0.5" '
            f'gradientUnits="objectBoundingBox">\n'
            f'{stop_els}\n'
            f'  </radialGradient>')


def _gradient_rect_el(gid: str, region: GradientRegion) -> str:
    x, y, w, h = region.bbox
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" '
            f'fill="url(#{gid})"/>')


# ── SVG path element ─────────────────────────────────────────────────────────

def _path_el(d: str, color: str, opacity: float, transform: Optional[str] = None) -> str:
    op_attr = f' fill-opacity="{opacity:.2f}"' if opacity < 1.0 else ""
    tr_attr = f' transform="{transform}"' if transform else ""
    return f'<path d="{d}" fill="{color}"{op_attr}{tr_attr}/>'


# ── text element ─────────────────────────────────────────────────────────────

def _text_el(region: TextRegion) -> str:
    x, y, _, _ = region.bbox
    fs = max(8, int(round(region.font_size_estimate)))
    safe = (region.text
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;"))
    return (f'<text x="{x}" y="{y + fs}" '
            f'font-size="{fs}" '
            f'font-family="sans-serif" '
            f'fill="#000000">{safe}</text>')


def _compute_viewbox(paths: List[dict], img_w: int, img_h: int) -> str:
    return f"0 0 {img_w} {img_h}"


# ── main stage ────────────────────────────────────────────────────────────────

def assemble(ctx: Context) -> Context:
    """
    Build the final SVG string and store it in ctx.final_svg.
    """
    img = ctx.preprocessed_image if ctx.preprocessed_image is not None \
        else ctx.original_image
    img_h, img_w = (img.shape[:2] if img is not None else (512, 512))

    # ── collect gradient defs ─────────────────────────────────────────────────
    defs_lines: List[str] = []
    gradient_rects: List[str] = []
    for gi, gr in enumerate(ctx.gradient_regions):
        gid = f"nt-grad-{gi}"
        if gr.gradient_type == "linear":
            defs_lines.append(_linear_gradient_def(gid, gr))
        else:
            defs_lines.append(_radial_gradient_def(gid, gr))
        gradient_rects.append(_gradient_rect_el(gid, gr))

    # ── sort paths into layers ────────────────────────────────────────────────
    bg_paths: List[str]   = []
    fg_paths: List[str]   = []
    other_paths: List[str] = []

    for rec in ctx.collected_paths:
        layer = rec.get("layer", "")
        d     = rec.get("d", "")
        color = rec.get("color", "#000000")
        op    = float(rec.get("opacity", 1.0))

        tr = rec.get("transform")
        el = rec.get("primitive_svg") or (_path_el(d, color, op, tr) if d else None)
        if not el:
            continue

        if "background" in layer:
            bg_paths.append(el)
        elif "foreground" in layer:
            fg_paths.append(el)
        else:
            other_paths.append(el)

    # ── count statistics for quality report ───────────────────────────────────
    ctx.path_count = len(ctx.collected_paths)
    total_d = " ".join(r.get("d", "") for r in ctx.collected_paths)
    ctx.node_count = total_d.count(" ")

    # ── build viewBox ─────────────────────────────────────────────────────────
    viewbox = _compute_viewbox(ctx.collected_paths, img_w, img_h)

    # ── assemble SVG ──────────────────────────────────────────────────────────
    parts: List[str] = []
    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="{viewbox}" '
        f'width="{img_w}" height="{img_h}">'
    )
    parts.append("<!-- Generated by NeoSVG — original AI vectorization pipeline -->")

    if defs_lines:
        parts.append("<defs>")
        parts.extend(defs_lines)
        parts.append("</defs>")

    # 1. Background colour — skipped when the input PNG had transparency so
    # the output SVG stays transparent (otherwise a white t-shirt on a
    # transparent background becomes white-on-white and is invisible).
    if not getattr(ctx, "has_transparency", False):
        parts.append(f'<g id="layer-background-fill">')
        parts.append(f'  <rect width="{img_w}" height="{img_h}" '
                     f'fill="{ctx.background_color}"/>')
        parts.append("</g>")

    # 2. Gradient regions
    if gradient_rects:
        parts.append('<g id="layer-gradients">')
        parts.extend(f"  {el}" for el in gradient_rects)
        parts.append("</g>")

    # 3. Background traced layer
    if bg_paths:
        parts.append('<g id="layer-background">')
        parts.extend(f"  {el}" for el in bg_paths)
        parts.append("</g>")

    # 4. Foreground layers
    if fg_paths:
        parts.append('<g id="layer-foreground">')
        parts.extend(f"  {el}" for el in fg_paths)
        parts.append("</g>")

    # 5+6. Other paths (includes primitives already inlined)
    if other_paths:
        parts.append('<g id="layer-shapes">')
        parts.extend(f"  {el}" for el in other_paths)
        parts.append("</g>")

    # 7. Text elements
    if ctx.text_regions and not ctx.skip_text:
        parts.append('<g id="layer-text">')
        for tr in ctx.text_regions:
            parts.append(f"  {_text_el(tr)}")
        parts.append("</g>")

    parts.append("</svg>")

    ctx.final_svg = "\n".join(parts)
    logger.info("SVG assembled: %d paths, %d nodes, %d bytes",
                ctx.path_count, ctx.node_count, len(ctx.final_svg))
    return ctx
