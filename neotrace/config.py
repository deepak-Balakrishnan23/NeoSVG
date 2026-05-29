# NeoSVG — configuration
# All thresholds are documented here. No magic numbers anywhere else.

class Config:

    # ── ImageClassifier ────────────────────────────────────────────────────
    # Color counts come from k-means on a 64×64 thumbnail.
    LOGO_MAX_UNIQUE_COLORS    = 12    # ≤ this distinct quantized colors → LOGO
    CARTOON_MAX_UNIQUE_COLORS = 40
    # Saturation (0-255 HSV) below this → LINEART
    LINEART_MAX_AVG_SATURATION = 20
    # Percentage of edge pixels that are on a grid → PIXELART
    PIXELART_GRID_RATIO        = 0.65
    # Shorter side ≤ this AND mostly flat → ICON
    ICON_MAX_DIM               = 96

    # ── Preprocessor ───────────────────────────────────────────────────────
    # Upscaling here is DISABLED (set to 0): the assembler sizes the output
    # SVG from the preprocessed image, so any preprocessor upscale would make
    # the output bigger than the input.  The high-fidelity engines also trace
    # the ORIGINAL image (not the preprocessed one) and do their own internal
    # FIDELITY_UPSCALE that they invert afterwards — so a preprocessor upscale
    # only desynchronised the viewBox from the path coordinates.  Keeping the
    # native size end-to-end guarantees output dimensions == input dimensions.
    MIN_DIM_FOR_UPSCALE        = 0     # 0 = never upscale (output size == input size)
    UPSCALE_TARGET             = 512   # (unused while MIN_DIM_FOR_UPSCALE == 0)
    PHOTO_DENOISE_H            = 10    # fastNlMeansDenoisingColored h
    PHOTO_DENOISE_TEMPLATE     = 7     # template window size
    PHOTO_DENOISE_SEARCH       = 21    # search window size
    CARTOON_BILATERAL_D        = 9
    CARTOON_BILATERAL_SIGMA_C  = 75    # sigma for color space
    CARTOON_BILATERAL_SIGMA_S  = 75    # sigma for coordinate space
    HISTOGRAM_FLAT_PERCENTILE  = 5     # stretch if range < this percentile gap

    # ── SVG output ─────────────────────────────────────────────────────────
    DEFAULT_PATH_PRECISION       = 2   # decimal places in SVG path coordinates

    # ── PrimitiveDetector ──────────────────────────────────────────────────
    # Normalised residual = mean_deviation / bounding_diagonal
    CIRCLE_FIT_TOLERANCE     = 0.04
    ELLIPSE_FIT_TOLERANCE    = 0.06
    RECT_ANGLE_TOLERANCE_DEG = 8.0   # max deviation from 90° at corners
    LINE_COLLINEAR_TOL       = 1.5   # pixels: max dist from fitted line
    MIN_PRIMITIVE_POINTS     = 8     # don't attempt fit below this count
    MAX_PRIMITIVE_AREA_FRAC  = 0.10  # skip primitive fitting if path covers > 10% of image

    # ── Visvalingam-Whyatt simplifier ──────────────────────────────────────
    DEFAULT_SIMPLIFY_TOLERANCE    = 1.5  # area threshold in px²
    TEXT_ADJACENT_TOLERANCE       = 0.4  # tighter near text
    BACKGROUND_SIMPLIFY_TOLERANCE = 3.5  # looser for background shapes
    # How close a path needs to be to a text bbox to use tight tolerance
    TEXT_PROXIMITY_PX             = 20

    # ── GradientDetector ───────────────────────────────────────────────────
    GRADIENT_SMOOTHNESS_CORR = 0.92  # min R² for linear/radial gradient fit
    GRADIENT_MIN_COLOR_RANGE = 30    # min color variation (0-255) to qualify as gradient

    # ── Segmenter ──────────────────────────────────────────────────────────
    REMBG_ALPHA_THRESHOLD = 128      # alpha < this → background

    # ── QualityValidator ───────────────────────────────────────────────────
    MIN_ACCEPTABLE_SSIM = 0.75

    # ── Recommended params per image type ─────────────────────────────────
    # These override defaults when --mode auto is used.
    PARAMS_BY_TYPE = {
        'LOGO': {
            'color_count':        8,
            'filter_speckle':     4,
            'curve_mode':         'spline',
            'corner_threshold':   60,
            'segment_length':     4.0,
            'color_precision':    6,
            'layer_difference':   16,
        },
        'CARTOON': {
            'color_count':        16,
            'filter_speckle':     4,
            'curve_mode':         'spline',
            'corner_threshold':   60,
            'segment_length':     4.0,
            'color_precision':    8,
            'layer_difference':   16,
        },
        'PHOTO': {
            'color_count':        32,
            'filter_speckle':     10,
            'curve_mode':         'spline',
            'corner_threshold':   180,
            'segment_length':     4.0,
            'color_precision':    8,
            'layer_difference':   48,
        },
        'LINEART': {
            'color_count':        2,
            'filter_speckle':     4,
            'curve_mode':         'spline',
            'corner_threshold':   60,
            'segment_length':     4.0,
            'color_precision':    6,
            'layer_difference':   16,
        },
        'PIXELART': {
            'color_count':        8,
            'filter_speckle':     1,
            'curve_mode':         'pixel',
            'corner_threshold':   60,
            'segment_length':     4.0,
            'color_precision':    8,
            'layer_difference':   0,
        },
        'ICON': {
            'color_count':        8,
            'filter_speckle':     2,
            'curve_mode':         'spline',
            'corner_threshold':   60,
            'segment_length':     4.0,
            'color_precision':    6,
            'layer_difference':   16,
        },
    }

    # ── Max-fidelity preprocessing ───────────────────────────────────────
    # Optional 2× upscale + unsharp mask before NeoSVG Engine tracing.
    # Sharpening is subtle — strong sharpening creates dark halos at
    # edges that get traced as separate outline paths.
    FIDELITY_UPSCALE_FACTOR   = 2.0   # 2× input image before tracing
    FIDELITY_UPSCALE_MAX_SIDE = 4096  # don't upscale beyond this (memory cap)
    FIDELITY_SHARPEN_AMOUNT   = 0.4   # was 1.2 — gentler to avoid edge halos
    FIDELITY_SHARPEN_RADIUS   = 0.5   # was 0.8 — tighter radius, no halo

    # ── Detail levels for the NeoSVG Engine (local-colour stacked-layer) ──
    # color_precision       : bits/channel for bit-shift quantization (1-8).
    #                         HIGHER = finer local colour = more regions =
    #                         smoother gradients & more detail (and more paths).
    #                         Unlike a global palette, bit-shift preserves each
    #                         region's local colour, so translucent subjects
    #                         (water/glass) keep their body instead of snapping
    #                         into the background.
    # bilateral_d           : Bilateral pre-filter diameter (0 = off) — edge-
    #                         preserving denoise so cluster boundaries follow
    #                         real edges, not JPEG noise.
    # bilateral_sigma_color : Colour distance for bilateral merging (smaller =
    #                         more edge preservation, less smoothing).
    # min_area_px / frac    : Drop regions/contours below this area.  Kept LOW
    #                         so fine detail (droplets, highlights) survives;
    #                         single-pixel noise is handled by the bilateral.
    # corner_threshold_deg, segment_length, bezier_error : Bezier-fit tuning.
    # NOTE on bilateral_sigma_color: this is the single biggest quality lever.
    # A STRONG bilateral (sigma 22-30) collapses JPEG-noise regions ~25× (e.g.
    # 58k → 2.4k regions) BEFORE clustering, while keeping reconstruction SSIM
    # ~0.98.  A weak bilateral leaves tens of thousands of noise regions that
    # then fragment the output.  Combined with the small-region MERGE step in
    # the engine (expand_labels), this yields clean, coherent shapes.
    # meanshift_sp / meanshift_sr : spatial & colour radii for the mean-shift
    #   pre-filter.  Mean-shift is the master lever for SMOOTH-area cleanliness:
    #   it flattens low-contrast 2-D texture (gradient backgrounds, dark zones)
    #   into uniform colour while keeping high-contrast detail edges crisp,
    #   eliminating the fine "crinkle" a bilateral alone leaves behind.  The
    #   bilateral is now a LIGHT secondary cleanup.
    NEOSVG_LEVELS = {
        'low': {
            'color_precision':       5,
            'meanshift_sp':          20,
            'meanshift_sr':          40,
            'median_blur':           5,
            'bilateral_d':           5,
            'bilateral_sigma_color': 15,
            'min_area_px':           48,
            'min_area_frac':         0.00020,
            'corner_threshold_deg':  75,
            'segment_length':        6.0,
            'bezier_error':          4.0,
        },
        'medium': {
            'color_precision':       6,
            'meanshift_sp':          16,
            'meanshift_sr':          30,
            'median_blur':           5,
            'bilateral_d':           5,
            'bilateral_sigma_color': 12,
            'min_area_px':           32,
            'min_area_frac':         0.00010,
            'corner_threshold_deg':  60,
            'segment_length':        4.0,
            'bezier_error':          2.0,
        },
        'high': {
            'color_precision':       7,
            'meanshift_sp':          16,
            'meanshift_sr':          24,
            'median_blur':           7,
            'bilateral_d':           5,
            'bilateral_sigma_color': 10,
            'min_area_px':           20,
            'min_area_frac':         0.00006,
            # In a SMOOTH neighbourhood only the large gradient bands are real,
            # so merge every region below this away (kills scratch texture and
            # dash specks without banding — bands are far larger).  The busy
            # subject keeps min_area_px/frac.  See _smoothness_thresholds.
            'min_area_smooth':       2000,
            'corner_threshold_deg':  60,
            'segment_length':        2.0,
            'bezier_error':          1.0,
        },
        'ultra': {
            # The crinkle/scratch fix is a 3-stage smooth-zone cleanup:
            #   1. mean-shift (sp=16, sr=24) flattens low-contrast texture and
            #      subtle bokeh streaks in the background
            #   2. median_blur=7 deletes the THIN residual slivers that survive
            #      mean-shift (the faint scratch lines) — 7 catches wider ones
            #   3. light bilateral cleans residual edge noise
            # color_precision=7 retains fine ripple/specular detail; the colour-
            # aware small-region merge absorbs leftover specks into their most-
            # similar neighbour.  min_area_px=20 lets thin background streaks
            # (which are long but low-contrast) be absorbed by the merge.
            # The high-contrast subject is unaffected by this bg cleanup.
            'color_precision':       7,
            'meanshift_sp':          16,
            'meanshift_sr':          24,
            'median_blur':           7,
            'bilateral_d':           5,
            'bilateral_sigma_color': 10,
            'min_area_px':           20,
            'min_area_frac':         0.00004,
            # In a SMOOTH neighbourhood only the large gradient bands are real,
            # so merge every region below this away (kills scratch texture and
            # dash specks without banding — bands are far larger).  The busy
            # subject keeps min_area_px/frac.  See _smoothness_thresholds.
            'min_area_smooth':       2000,
            'corner_threshold_deg':  45,
            'segment_length':        1.0,
            'bezier_error':          0.5,
        },
    }

