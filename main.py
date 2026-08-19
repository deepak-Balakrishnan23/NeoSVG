#!/usr/bin/env python3
# NeoSVG — main CLI entry point
# Usage: python main.py input.png output.svg [options]

import logging
import os
import sys
import tempfile
import time

import click
import cv2
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table

from config import Config
from context import Context
import stages

console = Console()

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(console=console, rich_tracebacks=True, show_path=False)],
)
logger = logging.getLogger("neosvg")


def _load_image(path: str):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGBA)
    elif img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
    return img


def _ramp_hard_ratio(image) -> float:
    """
    Ratio of smooth-ramp pixels to hard-edge pixels in the luma gradient.

    High  → the image is mostly continuous tone (gradients, soft shading), the
            content max-fidelity tracing improves.
    Low   → the image is mostly text, hairlines and flat fills, the content
            max-fidelity tracing degrades (sharpen halos) or merely inflates.

    Transparent pixels are composited onto white first so that the alpha edge
    of a cut-out subject reads as one silhouette edge, not as a colour cliff.
    """
    import numpy as np

    img = image.astype(np.float32)
    if img.ndim == 3 and img.shape[2] == 4:
        a = img[:, :, 3:4] / 255.0
        img = img[:, :, :3] * a + 255.0 * (1.0 - a)

    luma = img.mean(axis=2)
    gy, gx = np.gradient(luma)
    mag = np.hypot(gx, gy)

    ramp = np.count_nonzero((mag > Config.AUTO_FIDELITY_RAMP_LO) &
                            (mag < Config.AUTO_FIDELITY_RAMP_HI))
    hard = np.count_nonzero(mag > Config.AUTO_FIDELITY_HARD_MAG)
    return ramp / max(hard, 1)


def _dominant_color(image) -> str:
    """
    Most common colour in the image, as an SVG hex string.

    Counted on a coarsely quantised copy so that a smooth ramp's thousands of
    near-identical shades sum into one bucket instead of splitting the vote.
    Fully transparent pixels do not count — they are not part of the artwork.
    """
    import numpy as np

    img = image
    if img.ndim == 3 and img.shape[2] == 4:
        opaque = img[:, :, 3] > 0
        if not opaque.any():
            return "#ffffff"
        px = img[:, :, :3][opaque]
    else:
        px = img[:, :, :3].reshape(-1, 3)

    q = (px.astype(np.uint16) >> 4)                  # 16 levels per channel
    keys = (q[:, 0] << 8) | (q[:, 1] << 4) | q[:, 2]
    top = np.bincount(keys).argmax()
    r, g, b = (int(top) >> 8) & 0xF, (int(top) >> 4) & 0xF, int(top) & 0xF
    # Bucket centre, so the colour sits mid-bucket rather than at its floor.
    return "#%02x%02x%02x" % ((r << 4) + 8, (g << 4) + 8, (b << 4) + 8)


def _auto_max_fidelity(image) -> bool:
    """Decide whether to trace at max fidelity based on image content."""
    if not Config.AUTO_FIDELITY_ENABLED:
        return False

    h, w = image.shape[:2]
    if h * w > Config.AUTO_FIDELITY_MAX_PIXELS:
        logger.info("Max-fidelity auto: OFF (%d×%d exceeds pixel budget)", w, h)
        return False

    ratio = _ramp_hard_ratio(image)
    on = ratio >= Config.AUTO_FIDELITY_MIN_RATIO
    logger.info("Max-fidelity auto: %s (ramp/hard %.1f vs threshold %.1f)",
                "ON" if on else "OFF", ratio, Config.AUTO_FIDELITY_MIN_RATIO)
    return on


def run_pipeline(
    input_path: str,
    output_path: str,
    mode: str = "auto",
    quality: str = "balanced",
    detail: str = "high",
    engine: str = "neosvg",
    max_fidelity: bool = False,
    skip_text: bool = False,
    skip_primitives: bool = True,
    skip_gradients: bool = True,
    skip_segmentation: bool = False,
    test_stage: str = "",
) -> Context:
    start = time.perf_counter()

    ctx = Context(
        input_path        = input_path,
        output_path       = output_path,
        mode              = mode,
        quality           = quality,
        detail            = detail,
        engine            = engine,
        max_fidelity      = max_fidelity,
        skip_text         = skip_text,
        skip_primitives   = skip_primitives,
        skip_gradients    = skip_gradients,
        skip_segmentation = skip_segmentation,
    )

    with tempfile.TemporaryDirectory(prefix="neosvg_") as tmp:
        ctx.temp_dir = tmp

        # ── load ───────────────────────────────────────────────────────────────
        logger.info("Loading %s", input_path)
        ctx.original_image = _load_image(input_path)
        ctx.png_size_bytes = os.path.getsize(input_path)

        # An explicit request always wins; otherwise decide from the content.
        if not ctx.max_fidelity:
            ctx.max_fidelity = _auto_max_fidelity(ctx.original_image)

        # Whatever leaks through an anti-aliased seam between two traced tiles
        # is this rect. Defaulting it to white means every seam in a dark image
        # leaks a bright hairline; painting the image's own dominant colour
        # makes that leakage self-cancelling instead of maximally visible.
        ctx.background_color = _dominant_color(ctx.original_image)

        # If --test-stage, run just that stage and exit
        if test_stage:
            return _run_test_stage(test_stage, ctx)

        # ── classify ───────────────────────────────────────────────────────────
        logger.info("[1/9] Classifying image…")
        ctx = stages.preprocess(stages.classify(ctx))

        # ── text detection ─────────────────────────────────────────────────────
        logger.info("[2/9] Detecting text…")
        ctx = stages.detect_text(ctx)

        # ── gradient detection ─────────────────────────────────────────────────
        if ctx.skip_gradients:
            logger.info("[3/9] Gradient detection skipped")
        else:
            logger.info("[3/9] Detecting gradients…")
            ctx = stages.detect_gradients(ctx)

        # ── segmentation ───────────────────────────────────────────────────────
        logger.info("[4/9] Segmenting layers…")
        ctx = stages.segment(ctx)

        # ── vectorize ──────────────────────────────────────────────────────────
        logger.info("[5/9] Vectorizing…")
        ctx = stages.run_vectorize(ctx)

        # ── primitive detection ────────────────────────────────────────────────
        logger.info("[6/9] Detecting primitives…")
        ctx = stages.detect_primitives(ctx)

        # ── path simplification ────────────────────────────────────────────────
        logger.info("[7/9] Simplifying paths…")
        ctx = stages.simplify_paths(ctx)

        # ── assemble ───────────────────────────────────────────────────────────
        logger.info("[8/9] Assembling SVG…")
        ctx = stages.assemble(ctx)

        # ── validate ───────────────────────────────────────────────────────────
        logger.info("[9/9] Validating quality…")
        from validator import validate
        ctx = validate(ctx)

        # ── write output ───────────────────────────────────────────────────────
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(ctx.final_svg)

    ctx.elapsed_seconds = time.perf_counter() - start
    return ctx


def _run_test_stage(stage_name: str, ctx: Context) -> Context:
    """Run a single stage for debugging."""
    stage_map = {
        "classify":    lambda: stages.classify(ctx),
        "preprocess":  lambda: stages.preprocess(ctx),
        "text":        lambda: stages.detect_text(ctx),
        "gradient":    lambda: stages.detect_gradients(ctx),
        "segment":     lambda: stages.segment(ctx),
        "vectorize":   lambda: stages.run_vectorize(ctx),
        "primitives":  lambda: stages.detect_primitives(ctx),
        "simplify":    lambda: stages.simplify_paths(ctx),
        "assemble":    lambda: stages.assemble(ctx),
    }
    if stage_name not in stage_map:
        console.print(f"[red]Unknown stage '{stage_name}'. "
                      f"Available: {', '.join(stage_map)}")
        sys.exit(1)

    # Run all preceding stages so the target stage has valid input
    order = list(stage_map.keys())
    idx   = order.index(stage_name)
    for s in order[:idx + 1]:
        ctx = stage_map[s]()

    console.print(f"[green]Stage '{stage_name}' complete.")
    return ctx


def _print_report(ctx: Context) -> None:
    table = Table(title="NeoSVG Quality Report", show_header=True)
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Value",  style="green")

    def fsize(n):
        if n >= 1024 * 1024:
            return f"{n / 1024 / 1024:.1f} MB"
        if n >= 1024:
            return f"{n / 1024:.1f} KB"
        return f"{n} B"

    ssim_str = f"{ctx.ssim:.3f}" if ctx.ssim >= 0 else "n/a"
    warn     = " ⚠" if 0 < ctx.ssim < Config.MIN_ACCEPTABLE_SSIM else ""

    table.add_row("Image type",       ctx.image_type)
    table.add_row("SSIM",             ssim_str + warn)
    table.add_row("Path count",       str(ctx.path_count))
    table.add_row("Node count",       str(ctx.node_count))
    table.add_row("SVG size",         fsize(ctx.svg_size_bytes))
    table.add_row("PNG size",         fsize(ctx.png_size_bytes))
    table.add_row("Compression ratio",
                  f"{ctx.svg_size_bytes / max(ctx.png_size_bytes, 1):.2f}×")
    table.add_row("Elapsed",          f"{ctx.elapsed_seconds:.1f} s")

    console.print(table)


@click.command()
@click.argument("input",  metavar="INPUT.png",  type=click.Path(exists=True))
@click.argument("output", metavar="OUTPUT.svg", type=click.Path())
@click.option("--mode",    "-m",
              type=click.Choice(["auto", "logo", "photo", "cartoon"]),
              default="auto", show_default=True,
              help="Force image type or use auto-detection.")
@click.option("--quality", "-q",
              type=click.Choice(["fast", "balanced", "best"]),
              default="balanced", show_default=True,
              help="fast=skip heavy stages, best=run everything.")
@click.option("--detail", "-d",
              type=click.Choice(["low", "medium", "high", "ultra"]),
              default="high", show_default=True,
              help="Vector fidelity level — controls n_colors, smoothing, and simplification.")
@click.option("--engine", "-e",
              type=click.Choice(["neosvg"]),
              default="neosvg", show_default=True,
              help="Vectorization engine. The NeoSVG Engine: high-fidelity hierarchical region-growing with sub-pixel Bezier curve fitting.")
@click.option("--no-text",       is_flag=True,
              help="Skip OCR text detection and preservation.")
@click.option("--primitives", is_flag=True, default=False,
              help="Enable aggressive primitive (circle/rect/ellipse) replacement (off by default — destroys shape detail).")
@click.option("--gradients", is_flag=True, default=False,
              help="Enable tile-based gradient detection (off by default — produces blocky artifacts).")
@click.option("--no-segment",    is_flag=True,
              help="Skip foreground/background segmentation.")
@click.option("--test-stage",    default="",
              help="Run only up to named stage and print result.")
@click.option("--verbose", "-v", is_flag=True, help="Debug logging.")
def main(input, output, mode, quality, detail, engine, no_text, primitives,
         gradients, no_segment, test_stage, verbose):
    """
    NeoSVG — AI-assisted raster-to-vector pipeline.

    Converts INPUT.png to OUTPUT.svg using a multi-stage intelligence
    layer: classifier → preprocessor → OCR → gradient detection →
    segmentation → vectorizer → primitive detection → path simplification
    → SVG assembly → quality validation.
    """
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    console.print(Panel(
        "[bold cyan]NeoSVG[/bold cyan]  AI-assisted vectorization pipeline\n"
        f"  Input  : {input}\n"
        f"  Output : {output}\n"
        f"  Mode   : {mode}  |  Quality: {quality}  |  Detail: {detail}  |  Engine: {engine}",
        expand=False,
    ))

    try:
        ctx = run_pipeline(
            input_path        = input,
            output_path       = output,
            mode              = mode,
            quality           = quality,
            detail            = detail,
            engine            = engine,
            skip_text         = no_text,
            skip_primitives   = not primitives,
            skip_gradients    = not gradients,
            skip_segmentation = no_segment,
            test_stage        = test_stage,
        )
    except Exception as exc:
        console.print(f"[bold red]Pipeline failed:[/bold red] {exc}")
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

    if not test_stage:
        _print_report(ctx)
        console.print(f"[bold green]✓ Saved:[/bold green] {output}")


if __name__ == "__main__":
    main()
