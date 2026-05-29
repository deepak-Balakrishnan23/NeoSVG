# NeoSVG — pipeline context
# Every stage reads from and writes to this object. Nothing is global.

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


@dataclass
class TextRegion:
    bbox: Tuple[int, int, int, int]  # x, y, w, h in preprocessed-image coords
    text: str
    font_size_estimate: float        # estimated px height


@dataclass
class GradientRegion:
    bbox: Tuple[int, int, int, int]
    gradient_type: str               # 'linear' | 'radial' | 'complex'
    params: Dict[str, Any]           # keys depend on type (see gradient_detector)
    mask: Optional[np.ndarray] = None  # boolean mask same size as preprocessed_image


@dataclass
class LayerData:
    name: str
    image: np.ndarray        # RGBA masked PNG
    svg_paths: List[dict] = field(default_factory=list)
    # Each path dict: {'d': str, 'color': '#rrggbb', 'opacity': float}


@dataclass
class Context:
    # ── Input / output ──────────────────────────────────────────────────────
    input_path: str  = ""
    output_path: str = ""
    temp_dir: str    = ""

    # ── User options ────────────────────────────────────────────────────────
    mode: str    = "auto"       # auto | logo | photo | cartoon
    quality: str = "balanced"   # fast | balanced | best
    detail: str  = "high"       # low | medium | high | ultra
    engine: str  = "neosvg"       # the NeoSVG Engine (only engine)
    max_fidelity: bool = False    # 2× upscale + edge sharpen before tracing (NeoSVG Engine)
    skip_text:        bool = False
    skip_primitives:  bool = True   # default OFF — destroys fidelity on complex shapes
    skip_gradients:   bool = True   # default OFF — tile gradients look blocky
    skip_segmentation: bool = False

    # ── Classification ───────────────────────────────────────────────────────
    image_type: str             = "LOGO"
    recommended_params: Dict    = field(default_factory=dict)

    # ── Images ───────────────────────────────────────────────────────────────
    original_image:     Optional[np.ndarray] = None   # RGBA, original size
    preprocessed_image: Optional[np.ndarray] = None   # RGBA, cleaned+upscaled
    text_masked_image:  Optional[np.ndarray] = None   # text regions blacked out

    # ── Detections ──────────────────────────────────────────────────────────
    text_regions:     List[TextRegion]     = field(default_factory=list)
    gradient_regions: List[GradientRegion] = field(default_factory=list)

    # ── Segmentation layers ──────────────────────────────────────────────────
    layers: List[LayerData] = field(default_factory=list)

    # ── Assembled SVG pieces ─────────────────────────────────────────────────
    background_color: str        = "#ffffff"
    has_transparency: bool       = False   # input had alpha < 255 somewhere
    # collected path records before assembly; each dict has d/color/layer/opacity
    collected_paths: List[dict]  = field(default_factory=list)

    # ── Final SVG ────────────────────────────────────────────────────────────
    final_svg: str = ""

    # ── Quality metrics ──────────────────────────────────────────────────────
    ssim:            float = 0.0
    path_count:      int   = 0
    node_count:      int   = 0
    svg_size_bytes:  int   = 0
    png_size_bytes:  int   = 0
    elapsed_seconds: float = 0.0
