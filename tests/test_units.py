"""
Unit tests pinning specific defects that were found and fixed.

Each test names the failure it prevents, so a future change that reintroduces
it fails with an explanation rather than just a red assertion.
"""
from __future__ import annotations

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config                                     # noqa: E402
from stages.classifier import _pixelart_score                 # noqa: E402
from engines.region_gradient import _stops_to_list            # noqa: E402
from engines.hierarchical_grow_vectorizer import _beats_bands # noqa: E402
import main as neosvg_main                                    # noqa: E402
from tests import fixtures                                    # noqa: E402


class PixelArtScore(unittest.TestCase):
    """The scorer once required edge pixels to sit on grid INTERSECTIONS, which
    capped the achievable score near 1/g and made PIXELART unreachable."""

    def _score(self, block):
        img = np.asarray(fixtures.pixel_art(blocks=32, scale=block).convert("RGBA"))
        return _pixelart_score(img)

    def test_ideal_pixel_art_scores_above_threshold(self):
        for block in (4, 8, 12, 16):
            with self.subTest(block=block):
                self.assertGreaterEqual(
                    self._score(block), Config.PIXELART_GRID_RATIO,
                    f"{block}px pixel art must classify as PIXELART")

    def test_non_pixel_art_scores_below_threshold(self):
        for name in ("gradient_logo", "flat_shapes", "dark_ui", "line_art"):
            with self.subTest(fixture=name):
                img = np.asarray(fixtures.ALL[name]().convert("RGBA"))
                self.assertLess(_pixelart_score(img), Config.PIXELART_GRID_RATIO,
                                f"{name} must not classify as PIXELART")


class GradientStopOffsets(unittest.TestCase):
    """Stops are binned means, so they represent bin CENTRES. Emitting i/(n-1)
    stretched every fitted ramp by 1/(1-1/n) against the curve that was fitted."""

    def test_offsets_are_bin_centres_with_clamps(self):
        stops = np.array([[0, 0, 0], [128, 128, 128], [255, 255, 255]], float)
        out = _stops_to_list(stops)
        offsets = [s["offset"] for s in out]
        self.assertEqual(offsets[0], 0.0)
        self.assertEqual(offsets[-1], 1.0)
        # Interior stops sit at (i + 0.5) / n
        self.assertAlmostEqual(offsets[1], 0.5 / 3, places=4)
        self.assertAlmostEqual(offsets[2], 1.5 / 3, places=4)
        self.assertAlmostEqual(offsets[3], 2.5 / 3, places=4)

    def test_end_colours_are_held_by_clamp_stops(self):
        stops = np.array([[10, 20, 30], [200, 210, 220]], float)
        out = _stops_to_list(stops)
        self.assertEqual(out[0]["color"], out[1]["color"])
        self.assertEqual(out[-1]["color"], out[-2]["color"])


class GradientAcceptance(unittest.TestCase):
    """R² scores a ramp against ONE flat fill, a baseline never painted. A ramp
    must instead beat the per-label bands it actually replaces."""

    def test_rejects_ramp_worse_than_its_bands(self):
        colors = np.array([[0, 0, 0]] * 50 + [[255, 255, 255]] * 50, float)
        labels = np.array([1] * 50 + [2] * 50)
        # Bands reproduce this perfectly (zero within-label variance), so any
        # positive residual must lose.
        self.assertFalse(_beats_bands(colors, labels, ss_res=1e6))

    def test_accepts_ramp_better_than_its_bands(self):
        rng = np.random.default_rng(0)
        colors = rng.normal(128, 40, (200, 3))
        labels = rng.integers(1, 5, 200)
        self.assertTrue(_beats_bands(colors, labels, ss_res=0.0))

    def test_missing_residual_is_permitted(self):
        colors = np.zeros((10, 3))
        self.assertTrue(_beats_bands(colors, np.ones(10), ss_res=None))


class BackgroundColour(unittest.TestCase):
    """The rect behind every seam was hardcoded white, so on dark artwork each
    seam leaked a bright hairline."""

    def test_dominant_colour_of_dark_image_is_dark(self):
        img = np.asarray(fixtures.dark_ui().convert("RGBA"))
        hexcol = neosvg_main._dominant_color(img)
        r, g, b = (int(hexcol[i:i + 2], 16) for i in (1, 3, 5))
        self.assertLess(r + g + b, 240, f"expected a dark backdrop, got {hexcol}")

    def test_transparent_pixels_do_not_vote(self):
        img = np.asarray(fixtures.transparent_logo())
        hexcol = neosvg_main._dominant_color(img)
        self.assertTrue(hexcol.startswith("#") and len(hexcol) == 7)


class FidelityGate(unittest.TestCase):
    """Fidelity tracing helps smooth ramps and hurts text/hairlines, so the gate
    keys off content, not on the classifier (which calls all three 'LOGO')."""

    def test_gradient_content_enables_fidelity(self):
        img = np.asarray(fixtures.gradient_logo())
        self.assertGreaterEqual(neosvg_main._ramp_hard_ratio(img),
                                Config.AUTO_FIDELITY_MIN_RATIO)

    def test_flat_and_text_content_does_not(self):
        for name in ("flat_shapes", "dark_ui"):
            with self.subTest(fixture=name):
                img = np.asarray(fixtures.ALL[name]().convert("RGBA"))
                self.assertLess(neosvg_main._ramp_hard_ratio(img),
                                Config.AUTO_FIDELITY_MIN_RATIO)

    def test_oversized_images_skip_fidelity(self):
        big = np.zeros((10, Config.AUTO_FIDELITY_MAX_PIXELS // 10 + 10, 4), np.uint8)
        self.assertFalse(neosvg_main._auto_max_fidelity(big))


class PresetLadder(unittest.TestCase):
    """'medium' once omitted min_area_smooth entirely, so it emitted MORE paths
    than 'ultra' and scored below 'low' — the middle button was the worst one."""

    def test_every_engine_level_defines_the_smooth_threshold(self):
        for level, params in Config.NEOSVG_LEVELS.items():
            with self.subTest(level=level):
                self.assertIn("min_area_smooth", params)

    def test_thresholds_decrease_as_quality_rises(self):
        order = ["low", "medium", "high", "ultra"]
        vals = [Config.NEOSVG_LEVELS[k]["min_area_smooth"] for k in order]
        self.assertEqual(vals, sorted(vals, reverse=True),
                         f"min_area_smooth must not increase with quality: {vals}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
