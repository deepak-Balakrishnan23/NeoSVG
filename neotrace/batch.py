#!/usr/bin/env python3
# NeoSVG — batch processor
# Usage: python batch.py ./input_folder/ ./output_folder/ --workers 4

import csv
import logging
import multiprocessing
import os
import sys
from pathlib import Path
from typing import Tuple

import click
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import (
    BarColumn, MofNCompleteColumn,
    Progress, SpinnerColumn, TextColumn, TimeElapsedColumn,
)

console = Console()
logging.basicConfig(
    level=logging.WARNING,        # quiet during batch
    format="%(message)s",
    handlers=[RichHandler(console=console, show_path=False)],
)
logger = logging.getLogger("neosvg.batch")

SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def _worker(args: Tuple) -> dict:
    """
    Process a single file.  Returns a summary dict for the CSV report.
    Called in a subprocess pool — imports are local so each worker has
    its own state.
    """
    input_path, output_path, mode, quality = args
    from main import run_pipeline

    result = {
        "filename":     os.path.basename(input_path),
        "ssim":         -1.0,
        "paths":        0,
        "nodes":        0,
        "time_seconds": 0.0,
        "error":        "",
    }
    try:
        ctx = run_pipeline(
            input_path  = input_path,
            output_path = output_path,
            mode        = mode,
            quality     = quality,
        )
        result["ssim"]         = round(ctx.ssim, 4)
        result["paths"]        = ctx.path_count
        result["nodes"]        = ctx.node_count
        result["time_seconds"] = round(ctx.elapsed_seconds, 2)
    except Exception as exc:
        result["error"] = str(exc)
        logger.error("Failed %s: %s", input_path, exc)

    return result


@click.command()
@click.argument("input_folder",  type=click.Path(exists=True, file_okay=False))
@click.argument("output_folder", type=click.Path())
@click.option("--workers",  "-w", default=4,         show_default=True,
              help="Number of parallel worker processes.")
@click.option("--mode",    "-m",
              type=click.Choice(["auto", "logo", "photo", "cartoon"]),
              default="auto", show_default=True)
@click.option("--quality", "-q",
              type=click.Choice(["fast", "balanced", "best"]),
              default="balanced", show_default=True)
@click.option("--csv-out", default="",
              help="Path for summary CSV (default: output_folder/report.csv)")
def batch(input_folder, output_folder, workers, mode, quality, csv_out):
    """
    Convert all PNG/JPG files in INPUT_FOLDER to SVG in OUTPUT_FOLDER,
    using multiprocessing.  Writes a summary CSV report.
    """
    input_dir  = Path(input_folder)
    output_dir = Path(output_folder)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_files = sorted(
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS
    )
    if not image_files:
        console.print(f"[yellow]No supported images found in {input_folder}[/yellow]")
        sys.exit(0)

    console.print(f"[bold]NeoSVG batch[/bold]  {len(image_files)} files, "
                  f"{workers} worker(s), mode={mode}, quality={quality}")

    tasks = [
        (str(p), str(output_dir / (p.stem + ".svg")), mode, quality)
        for p in image_files
    ]

    results = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task_id = progress.add_task("Processing…", total=len(tasks))

        with multiprocessing.Pool(processes=workers) as pool:
            for res in pool.imap_unordered(_worker, tasks):
                results.append(res)
                progress.advance(task_id)
                status = "ok" if not res["error"] else "ERR"
                progress.console.print(
                    f"  [{status}] {res['filename']}  "
                    f"ssim={res['ssim']:.3f}  t={res['time_seconds']:.1f}s"
                )

    # ── write CSV ─────────────────────────────────────────────────────────────
    csv_path = csv_out or str(output_dir / "report.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["filename", "ssim", "paths", "nodes",
                           "time_seconds", "error"])
        writer.writeheader()
        writer.writerows(results)

    ok    = sum(1 for r in results if not r["error"])
    errs  = len(results) - ok
    avg_t = sum(r["time_seconds"] for r in results) / max(len(results), 1)
    avg_s = sum(r["ssim"] for r in results if r["ssim"] >= 0) / max(ok, 1)

    console.print(
        f"\n[bold green]Done:[/bold green] {ok}/{len(results)} succeeded  "
        f"({errs} error(s))  avg SSIM={avg_s:.3f}  avg time={avg_t:.1f}s\n"
        f"Report: {csv_path}"
    )


if __name__ == "__main__":
    batch()
