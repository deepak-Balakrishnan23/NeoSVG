"""
End-to-end regression suite.

Runs the real pipeline over every fixture and asserts quality budgets. The
budgets are ceilings measured from a known-good run with headroom added, so an
ordinary refactor passes and a genuine quality loss fails.

Why several metrics rather than SSIM alone: SSIM is insensitive to exactly the
defect users complain about first — a wrong-hue patch inside a smooth ramp — and
it also shifts when band COUNT changes without any visible difference. So each
fixture also budgets perceptual colour error and blotch area, and the edge and
interior errors are budgeted separately, which localises a failure to the
tracer (edge) or the segmenter (interior).

Run:  python3 -m unittest discover -s tests -v
"""
from __future__ import annotations

import logging
import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.disable(logging.CRITICAL)

from config import Config          # noqa: E402
from main import run_pipeline      # noqa: E402
from tests import fixtures         # noqa: E402
from tests import metrics          # noqa: E402


# ceiling per fixture: every value is a MAXIMUM except ssim, which is a MINIMUM.
BUDGETS = {
    "gradient_logo":    dict(ssim=0.940, mae=2.50, mean_dE=3.05, blotch_px=200,
                             int_mae=2.50, bytes=700_000, image_type="CARTOON"),
    "flat_shapes":      dict(ssim=0.960, mae=1.45, mean_dE=1.50, blotch_px=200,
                             int_mae=0.80, bytes=40_000, image_type="LOGO"),
    "dark_ui":          dict(ssim=0.875, mae=5.60, mean_dE=6.50, blotch_px=6000,
                             int_mae=4.20, bytes=200_000, image_type="LOGO"),
    "pixel_art":        dict(ssim=0.895, mae=5.30, mean_dE=7.10, blotch_px=1500,
                             int_mae=5.00, bytes=600_000, image_type="PIXELART"),
    "line_art":         dict(ssim=0.945, mae=3.90, mean_dE=3.80, blotch_px=200,
                             int_mae=2.40, bytes=120_000, image_type="LINEART"),
}


@unittest.skipUnless(metrics.rasterizer_available(),
                     "cairosvg/Cairo unavailable — quality metrics need a rasteriser")
class Quality(unittest.TestCase):
    """One pipeline run per fixture, asserted against its budget."""

    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.mkdtemp(prefix="neosvg-tests-")
        cls.inputs = fixtures.write_all(cls.dir)
        cls.results = {}
        for name in BUDGETS:
            out = os.path.join(cls.dir, f"{name}.svg")
            ctx = run_pipeline(input_path=cls.inputs[name], output_path=out,
                               detail="ultra", skip_text=True,
                               skip_primitives=True, skip_gradients=True)
            cls.results[name] = (ctx, metrics.compare(cls.inputs[name], out), out)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.dir, ignore_errors=True)

    def test_quality_budgets(self):
        for name, budget in BUDGETS.items():
            ctx, m, _ = self.results[name]
            with self.subTest(fixture=name):
                self.assertEqual(ctx.image_type, budget["image_type"])
                self.assertGreaterEqual(
                    m["ssim"], budget["ssim"],
                    f"{name}: SSIM {m['ssim']:.4f} below floor {budget['ssim']}")
                for key in ("mae", "mean_dE", "blotch_px", "int_mae", "bytes"):
                    self.assertLessEqual(
                        m[key], budget[key],
                        f"{name}: {key} {m[key]} exceeds budget {budget[key]}")

    def test_output_dimensions_match_input(self):
        """A core promise of the tool: the SVG is the size of the source."""
        from PIL import Image
        for name, (_, _, out) in self.results.items():
            with self.subTest(fixture=name):
                w, h = Image.open(self.inputs[name]).size
                with open(out, encoding="utf-8") as fh:
                    head = fh.read(400)
                self.assertIn(f'viewBox="0 0 {w} {h}"', head)
                self.assertIn(f'width="{w}" height="{h}"', head)

    def test_node_count_is_a_node_count(self):
        """It once counted spaces, so it tracked formatting, not geometry."""
        for name, (ctx, _, out) in self.results.items():
            with self.subTest(fixture=name):
                with open(out, encoding="utf-8") as fh:
                    body = fh.read()
                commands = sum(body.count(c) for c in "MmCcLlZz")
                self.assertGreater(ctx.node_count, 0)
                self.assertLessEqual(ctx.node_count, commands)


@unittest.skipUnless(metrics.rasterizer_available(), "needs a rasteriser")
class Transparency(unittest.TestCase):
    """Transparent inputs must stay transparent and must not gain a backdrop."""

    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.mkdtemp(prefix="neosvg-alpha-")
        cls.src = os.path.join(cls.dir, "transparent_logo.png")
        fixtures.transparent_logo().save(cls.src)
        cls.out = os.path.join(cls.dir, "transparent_logo.svg")
        run_pipeline(input_path=cls.src, output_path=cls.out, detail="ultra",
                     skip_text=True, skip_primitives=True, skip_gradients=True)
        with open(cls.out, encoding="utf-8") as fh:
            cls.svg = fh.read()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.dir, ignore_errors=True)

    def test_no_background_rect_is_emitted(self):
        self.assertNotIn("layer-background-fill", self.svg)

    def test_transparent_area_is_not_painted_over(self):
        """The seam stroke has nothing behind it here, so over-dilation shows up
        directly as paint spilling into what should be empty space."""
        import io
        import numpy as np
        from PIL import Image
        metrics._find_cairo()
        import cairosvg
        src = np.asarray(Image.open(self.src).convert("RGBA"))
        h, w = src.shape[:2]
        png = cairosvg.svg2png(url=self.out, output_width=w, output_height=h)
        out = np.asarray(Image.open(io.BytesIO(png)).convert("RGBA"))

        empty = src[:, :, 3] == 0
        spill = float((out[empty, 3] > 32).mean())
        self.assertLess(spill, 0.06,
                        f"{spill:.1%} of transparent pixels were painted over")

        opaque_src = int((src[:, :, 3] > 128).sum())
        opaque_out = int((out[:, :, 3] > 128).sum())
        self.assertLess(abs(opaque_out - opaque_src) / max(opaque_src, 1), 0.05,
                        "opaque area drifted more than 5% from the source")


class PresetLadderEndToEnd(unittest.TestCase):
    """Each preset must cost more and deliver more than the one below it. The
    middle preset was once both bigger than the cheap one and lower quality."""

    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.mkdtemp(prefix="neosvg-ladder-")
        cls.src = os.path.join(cls.dir, "gradient_logo.png")
        fixtures.gradient_logo().save(cls.src)
        cls.runs, cls.metrics = {}, {}
        for detail in ("low", "medium", "ultra"):
            out = os.path.join(cls.dir, f"{detail}.svg")
            cls.runs[detail] = run_pipeline(
                input_path=cls.src, output_path=out, detail=detail,
                skip_text=True, skip_primitives=True, skip_gradients=True)
            if metrics.rasterizer_available():
                cls.metrics[detail] = metrics.compare(cls.src, out)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.dir, ignore_errors=True)

    def test_size_increases_with_quality(self):
        sizes = [self.runs[d].svg_size_bytes for d in ("low", "medium", "ultra")]
        self.assertEqual(sizes, sorted(sizes),
                         f"presets must not shrink as quality rises: {sizes}")

    @unittest.skipUnless(metrics.rasterizer_available(), "needs a rasteriser")
    def test_no_preset_is_dominated_by_a_cheaper_one(self):
        """
        Judged on perceptual colour error, NOT SSIM.

        On a smooth ramp SSIM prefers few large flat areas to many accurate
        ones, so it ranks the cheapest preset above the middle one even when
        that preset leaves thousands of pixels of visibly wrong hue. Measured
        on this fixture: low SSIM 0.9493 with 1684px of blotch, medium SSIM
        0.9459 with none. Mean dE and blotch area both order the presets
        correctly, so they are what the ladder is held to.
        """
        order = ("low", "medium", "ultra")
        dE = [self.metrics[d]["mean_dE"] for d in order]
        self.assertLessEqual(dE[1], dE[0],
                             f"'medium' has worse colour error than 'low': {dE}")
        self.assertLessEqual(dE[2], dE[1] + 1e-6,
                             f"'ultra' has worse colour error than 'medium': {dE}")

        blotch = [self.metrics[d]["blotch_px"] for d in order]
        self.assertLessEqual(blotch[1], blotch[0], f"blotch grew at 'medium': {blotch}")
        self.assertLessEqual(blotch[2], blotch[1], f"blotch grew at 'ultra': {blotch}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
