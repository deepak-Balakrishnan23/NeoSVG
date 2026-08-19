"""
Deterministic test images.

Generated in code rather than committed as binaries: the repo stays small, the
inputs are reviewable in a diff, and a seeded RNG makes every run identical.
Each fixture targets a distinct classifier branch and a distinct failure mode.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def gradient_logo(size=(340, 392)) -> Image.Image:
    """
    Smooth multi-hue ramps covering most of the frame — the banding and
    blotching case, and the content the fidelity gate is meant to fire on.

    The ramp deliberately dominates the canvas. An earlier version painted a
    thin ribbon on white, which left hard mask edges outnumbering ramp pixels
    and made the fixture read as line art to the gate.
    """
    w, h = size
    y, x = np.mgrid[0:h, 0:w].astype(np.float64)
    u = x / max(w - 1, 1)
    v = y / max(h - 1, 1)
    t = np.clip(0.5 * (u + v), 0.0, 1.0)
    s2 = np.clip(0.5 + 0.5 * np.sin(3.0 * (u - v)), 0.0, 1.0)
    ramp = np.stack([
        40 + 200 * t,
        30 + 150 * s2 * (1 - 0.4 * t),
        245 - 120 * t + 40 * s2,
    ], -1)
    img = Image.fromarray(np.clip(ramp, 0, 255).astype(np.uint8))
    return img.convert("RGBA")


def flat_shapes(size=(256, 256)) -> Image.Image:
    """Hard-edged flat fills — must stay tiny and near-exact."""
    im = Image.new("RGBA", size, (255, 255, 255, 255))
    d = ImageDraw.Draw(im)
    d.ellipse((30, 30, 150, 150), fill=(220, 50, 40, 255))
    d.rectangle((110, 120, 230, 220), fill=(30, 90, 200, 255))
    d.polygon([(60, 230), (130, 160), (200, 240)], fill=(20, 160, 90, 255))
    return im


def dark_ui(size=(420, 300)) -> Image.Image:
    """Dark background with thin bright rules — the white-seam-leak case."""
    im = Image.new("RGBA", size, (24, 24, 40, 255))
    d = ImageDraw.Draw(im)
    d.rectangle((0, 0, size[0], 26), fill=(16, 16, 28, 255))
    for i in range(6):
        yy = 46 + i * 38
        d.rectangle((16, yy, size[0] - 16, yy + 24), fill=(38, 38, 58, 255))
        d.rectangle((22, yy + 8, 120, yy + 12), fill=(150, 155, 200, 255))
        d.rectangle((size[0] - 96, yy + 6, size[0] - 26, yy + 18),
                    fill=(70, 110, 240, 255))
    return im


def pixel_art(blocks=32, scale=12, seed=3) -> Image.Image:
    """Nearest-neighbour upscaled blocks — the PIXELART classifier case."""
    rng = np.random.default_rng(seed)
    pal = np.array([(30, 30, 40), (220, 60, 60), (60, 180, 90),
                    (240, 220, 80), (80, 120, 240)], np.uint8)
    idx = rng.integers(0, len(pal), (blocks, blocks))
    small = pal[idx]
    return Image.fromarray(small).resize(
        (blocks * scale, blocks * scale), Image.NEAREST).convert("RGBA")


def transparent_logo() -> Image.Image:
    """
    Gradient artwork with a genuine alpha channel — the case where no background
    rect is painted, so nothing sits behind a seam to hide it.

    Alpha is hard 0/255 (no partial coverage), matching what the engine expects
    after flatten_alpha, and the cut-out is a shape rather than a colour test so
    it stays transparent regardless of what the gradient does.
    """
    base = np.asarray(gradient_logo()).copy()
    h, w = base.shape[:2]
    mask = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(mask)
    d.ellipse((w * 0.08, h * 0.08, w * 0.92, h * 0.92), fill=255)
    base[:, :, 3] = np.asarray(mask)
    return Image.fromarray(base)


def line_art(size=(300, 300)) -> Image.Image:
    """Desaturated thin strokes — the LINEART classifier case."""
    im = Image.new("RGBA", size, (255, 255, 255, 255))
    d = ImageDraw.Draw(im)
    for i in range(7):
        d.line((20 + i * 34, 20, 160, 280 - i * 24), fill=(25, 25, 30, 255), width=2)
    d.ellipse((70, 70, 240, 240), outline=(20, 20, 25, 255), width=3)
    return im


ALL = {
    "gradient_logo":     gradient_logo,
    "flat_shapes":       flat_shapes,
    "dark_ui":           dark_ui,
    "pixel_art":         pixel_art,
    "transparent_logo":  transparent_logo,
    "line_art":          line_art,
}


def write_all(directory: str) -> dict:
    """Materialise every fixture into *directory*; returns {name: path}."""
    import os
    os.makedirs(directory, exist_ok=True)
    out = {}
    for name, fn in ALL.items():
        path = os.path.join(directory, f"{name}.png")
        fn().save(path)
        out[name] = path
    return out
