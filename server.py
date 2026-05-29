#!/usr/bin/env python3
# NeoSVG — web server
# Wraps the pipeline behind a simple upload API and serves the frontend.

import os
import sys

# rich (and other libs) call os.getcwd() at import time; in the preview
# sandbox the inherited cwd is invalid, so we set a safe cwd first.
try:
    os.getcwd()
except PermissionError:
    os.chdir('/tmp')

import tempfile
import traceback

from flask import Flask, jsonify, request, send_from_directory

# Ensure the neotrace package root is on sys.path regardless of cwd
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

app = Flask(
    __name__,
    static_folder=os.path.join(_HERE, "static"),
    static_url_path="",
    root_path=_HERE,
    instance_path=os.path.join(_HERE, "instance"),
)

# Hard limit on upload size — prevents OOM from giant uploads.
# 50 MB is large enough for 4K raster images but blocks multi-hundred-MB files.
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB

ALLOWED_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


def _allowed(filename: str) -> bool:
    return os.path.splitext(filename.lower())[1] in ALLOWED_EXTS


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/convert", methods=["POST"])
def convert():
    if "image" not in request.files:
        return jsonify(error="No image file uploaded"), 400

    f = request.files["image"]
    if not f.filename or not _allowed(f.filename):
        return jsonify(error="Unsupported file type"), 400

    mode    = request.form.get("mode",    "auto")
    quality = request.form.get("quality", "balanced")
    detail  = request.form.get("detail",  "high")
    engine  = request.form.get("engine",  "neosvg")
    maxfid  = request.form.get("max_fidelity", "false").lower() == "true"
    no_text = request.form.get("no_text", "false").lower() == "true"
    no_prim = request.form.get("no_primitives", "true").lower() == "true"
    no_grad = request.form.get("no_gradients", "true").lower() == "true"
    no_seg  = request.form.get("no_segment", "false").lower() == "true"

    ext = os.path.splitext(f.filename)[1].lower()

    with tempfile.TemporaryDirectory(prefix="neosvg_web_") as tmp:
        in_path  = os.path.join(tmp, f"input{ext}")
        out_path = os.path.join(tmp, "output.svg")
        f.save(in_path)

        try:
            from main import run_pipeline
            ctx = run_pipeline(
                input_path        = in_path,
                output_path       = out_path,
                mode              = mode,
                quality           = quality,
                detail            = detail,
                engine            = engine,
                max_fidelity      = maxfid,
                skip_text         = no_text,
                skip_primitives   = no_prim,
                skip_gradients    = no_grad,
                skip_segmentation = no_seg,
            )
        except Exception as exc:
            traceback.print_exc()
            return jsonify(error=str(exc)), 500

        return jsonify(
            svg            = ctx.final_svg,
            image_type     = ctx.image_type,
            path_count     = ctx.path_count,
            node_count     = ctx.node_count,
            svg_size_bytes = ctx.svg_size_bytes,
            png_size_bytes = ctx.png_size_bytes,
            ssim           = round(ctx.ssim, 3) if ctx.ssim >= 0 else None,
            elapsed        = round(ctx.elapsed_seconds, 2),
        )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"NeoSVG server → http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
