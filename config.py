# NeoSVG — configuration
# All thresholds are documented here. No magic numbers anywhere else.

class Config:

    # ── ImageClassifier ────────────────────────────────────────────────────
    # A colour is counted as distinct only when it is both populous enough to
    # be part of the design (not an anti-aliasing or dithering by-product) and
    # far enough from the colours already counted to read as a different one.
    COLOR_MIN_POPULATION = 0.004   # min share of pixels for a colour to count
    COLOR_MIN_SEPARATION = 26.0    # min RGB distance between distinct colours

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
    # INERT with the NeoSVG Engine: it emits Beziers whose node count is already
    # set by its own fit tolerance, so the simplifier finds nothing to remove
    # (measured: 0 of 978 paths modified on a full ultra run). These three
    # values change nothing today — see stages/path_simplifier_stage.py.
    DEFAULT_SIMPLIFY_TOLERANCE    = 1.5  # area threshold in px²
    TEXT_ADJACENT_TOLERANCE       = 0.4  # tighter near text
    BACKGROUND_SIMPLIFY_TOLERANCE = 3.5  # looser for background shapes
    # How close a path needs to be to a text bbox to use tight tolerance
    TEXT_PROXIMITY_PX             = 20

    # ── GradientDetector ───────────────────────────────────────────────────
    GRADIENT_SMOOTHNESS_CORR = 0.92  # min R² for linear/radial gradient fit
    GRADIENT_MIN_COLOR_RANGE = 30    # min color variation (0-255) to qualify as gradient

    # ── Per-region gradient fitting (NeoSVG Engine) ────────────────────────
    # The tile-based GradientDetector above scans an axis-aligned grid and can
    # only emit 0°/90° ramps over rectangles, so it never matches a gradient
    # that follows a SHAPE (a curved ribbon, a letterform).  Per-region fitting
    # works on the regions the engine has already segmented — their contours
    # follow the real shape — so the gradient axis is free to point anywhere
    # and the ramp is clipped to the shape, not to a bounding box.
    #
    # Two steps:
    #   1. COALESCE — a smooth ramp is initially labelled as N adjacent
    #      near-flat bands.  Each band alone has almost no colour variation, so
    #      fitting it individually is pointless.  Adjacent regions whose mean
    #      colours are within MERGE_DELTA are chained into one super-region;
    #      because each step is small the chain naturally walks the whole ramp
    #      while a genuine colour edge (large step) stops it.
    #   2. FIT — sample the ORIGINAL pixels of the super-region, find the axis
    #      along which colour varies most, and sample stops along it.  Stops
    #      are binned means, not a straight line fit, so multi-hue ramps
    #      (cyan→blue→purple→magenta) are captured exactly.
    #
    # If the fit is poor the super-region is discarded and its original bands
    # are emitted as flat fills — the previous behaviour, so a bad fit can
    # never make output worse than before.
    REGION_GRADIENT_ENABLED     = True
    # R² only has to be a coarse sanity filter now: REGION_GRADIENT_BAND_MARGIN
    # below is the test that actually decides whether a ramp is kept, and it
    # compares against the real baseline. 0.90 was strict because nothing else
    # caught a bad fit. Measured: 0.90 → 1031 paths/1742 KB, 0.80 → 978/1690 at
    # equal error; below 0.70 error climbs as marginal ramps get through.
    REGION_GRADIENT_MIN_R2      = 0.80   # min R² of the axis model to accept
    REGION_GRADIENT_MIN_RANGE   = 26     # min colour spread (0-255) to bother
    # Stops are binned means along the axis, so this is the ramp's colour
    # resolution: 10 stops quantise a cyan→magenta ramp into 10 visible steps
    # and the renderer interpolates linearly between them, which flattens the
    # hue curve wherever it bends.  32 tracks the curve closely and — because a
    # better-fitting ramp survives acceptance where a coarse one is rejected —
    # replaces more flat bands with gradients, so the file gets SMALLER, not
    # bigger.  Measured on a gradient logo (678×784, detail=ultra + fidelity):
    # 10 stops → SSIM 0.9819, MAE 2.51, 719 KB; 32 stops → SSIM 0.9822,
    # MAE 1.92, 579 KB.  Neutral on images with no fitted gradients.
    REGION_GRADIENT_STOPS       = 32     # stops sampled along the fitted axis
    REGION_GRADIENT_MIN_AREA    = 300    # px² — below this a flat fill is fine
    REGION_GRADIENT_ANGLE_STEPS = 24     # angular sweep resolution for the axis
    REGION_GRADIENT_MAX_SAMPLES = 4000   # pixel subsample cap (keeps the fit fast)

    # Coalescing runs COARSE→FINE.  One fixed delta cannot work for a whole
    # image: a delta wide enough to chain a long subtle ramp also welds two
    # neighbouring ramps into one blob whose colour no longer varies along any
    # single axis, and the fit is then correctly rejected — losing both.  So a
    # rejected group is retried at a tighter delta, which splits it back into
    # its constituent ramps.  Labels claimed by an accepted fit are withheld
    # from later passes, so the coarsest ramp that actually fits always wins.
    REGION_GRADIENT_MERGE_DELTAS = (36, 22, 13)

    # Ceiling on how far a coalesced GROUP may span in colour, as a multiple of
    # the pass's delta. Single-linkage only bounds each STEP, so without this a
    # long ramp chains indefinitely and the coarse pass can absorb a whole shape
    # into one blob that varies along no single axis — the fit then fails and
    # the ramp is lost along with its parts.
    REGION_GRADIENT_GROUP_SPAN  = 6.0

    # R² above scores the ramp against ONE flat fill for the whole group — a
    # baseline the pipeline never paints.  What a ramp actually replaces is the
    # stack of per-label bands, which approximates the ramp far better, so a fit
    # can pass R² and still render worse than the thing it displaced.  Left
    # unchecked, a single oversized bad ramp can repaint a large share of the
    # canvas, which is a much uglier failure than the banding it removed.
    #
    # So the acceptance test also requires the ramp's residual to beat the
    # bands' own within-label residual, measured on the same pixels.  The margin
    # is above 1.0 because a ramp additionally removes every seam between those
    # bands, which is worth real perceptual quality the residual cannot see.
    REGION_GRADIENT_BAND_MARGIN = 1.25

    # ── Bezier fit tolerance ceiling ───────────────────────────────────────
    # The per-region fit tolerance scales with region AREA so large smooth
    # bands do not burn thousands of control points. That scaling is unbounded,
    # though, and the largest regions are exactly the ones whose silhouette is
    # most visible, so cap the maximum deviation any fitted curve may have from
    # the traced contour, in pixels.
    BEZIER_ERROR_MAX = 1.0

    # ── Small-region merge ─────────────────────────────────────────────────
    # The merge absorbs every sub-threshold region into its most colour-similar
    # neighbour, and used to do so unconditionally — if a region had neighbours
    # at all, one of them won, however far away in colour.  A region's pixels
    # are then painted with the root's mean, so an unbounded merge is an
    # unbounded colour error: a cell can be welded into a band 100+ levels away
    # and repainted, which is exactly the isolated wrong-hue island that shows
    # up inside otherwise smooth ramps.
    #
    # This caps how far a region may be recoloured, as Euclidean RGB distance
    # between the region's own mean and the target root's mean.  Above the cap
    # the region simply stays unmerged: one extra small path, no blotch.
    # Set high enough that ordinary ramp accretion (neighbouring bands differ
    # by a few levels) is untouched.
    MERGE_MAX_COLOR_DELTA = 70.0

    # ── Thin-feature protection ────────────────────────────────────────────
    # Local contrast (as a multiple of the busy-context threshold) above which a
    # feature is treated as unambiguous structure and exempted from the
    # smooth-zone size test.  The faint streaks that test exists to remove sit
    # just above the busy threshold; a real rule or letterform sits far above
    # it.  Lower this to protect fainter detail, raise it to merge harder.
    THIN_FEATURE_CONTRAST_MULT = 2.5

    # Local standard deviation above which a pixel is considered real structure
    # and is exempted from the thin-structure median pre-filter.  Faint sensor
    # streaks sit well below this; a rule or letter stroke sits well above it.
    THIN_FEATURE_MEDIAN_GUARD = 25.0

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

    # ── Automatic max-fidelity gate ──────────────────────────────────────
    # Max-fidelity tracing is NOT a free win — it is content-dependent, and
    # measuring it on real inputs (detail=ultra, SSIM vs the source) shows a
    # clean split:
    #
    #   gradient artwork     ramp/hard 13.0  SSIM 0.9714 → 0.9822  ✓ big win
    #   soft/blurred artwork ramp/hard 57.6  SSIM 0.9551 → 0.9656  ✓ big win
    #   dense UI screenshot  ramp/hard  3.5  SSIM 0.8148 → 0.7942  ✗ REGRESSES
    #   flat vector shapes   ramp/hard  0.1  SSIM  ±0.000, 7× bytes ✗ wasteful
    #
    # The 2× upscale + unsharp helps a smooth ramp (more sample points across
    # the ramp ⇒ finer bands and better gradient fits) but hurts small text and
    # hairlines, where the sharpen adds halos the tracer then follows.
    #
    # So gate on image CONTENT rather than on the classifier — the classifier
    # calls a gradient logo, a flat logo and a 1440×1024 UI screenshot all
    # 'LOGO', so it cannot separate these cases.  The discriminator is the ratio
    # of smooth-ramp pixels to hard-edge pixels in the luma gradient magnitude:
    # ramp pixels are what fidelity helps, hard-edge pixels are what it hurts.
    # Observed values cluster far apart (≤4.3 for text/UI, ≥13.0 for ramps), so
    # the threshold sits in a wide empty gap and is not finely tuned.
    #
    # An explicit --max-fidelity / max_fidelity=True still forces it ON.
    AUTO_FIDELITY_ENABLED     = True
    AUTO_FIDELITY_RAMP_LO     = 0.4   # |∇luma| above this = not flat
    AUTO_FIDELITY_RAMP_HI     = 6.0   # …and below this = smooth ramp
    AUTO_FIDELITY_HARD_MAG    = 30.0  # |∇luma| above this = text / hard edge
    AUTO_FIDELITY_MIN_RATIO   = 8.0   # ramp/hard ≥ this → enable fidelity
    # Skip the upscale on large inputs regardless of content. The gain does not
    # scale with resolution — a big image already carries plenty of samples
    # across each ramp — but the cost does. Measured on one gradient image
    # resampled to several sizes, fidelity ON vs OFF:
    #   0.53 Mpx  +0.0055 SSIM    364 KB →  1690 KB    2.5s →  8.3s
    #   1.50 Mpx  +0.0044 SSIM   1063 KB →  4217 KB    5.9s → 19.4s
    #   3.00 Mpx  +0.0030 SSIM   1993 KB →  7616 KB   11.3s → 35.1s
    # Past ~2 Mpx the output stops being a usable web asset, for a gain that has
    # shrunk to a third of what it was at small sizes.
    AUTO_FIDELITY_MAX_PIXELS  = 2_000_000

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
            # Without this the smooth-zone merge never runs at all, and a
            # gradient shatters into hundreds of thin bands — which is how the
            # cheap preset ended up BIGGER than it needed to be and no better.
            # Measured on a gradient logo: absent → 434 paths / 519 KB /
            # SSIM 0.9564; 8000 → 102 paths / 184 KB / SSIM 0.9589. Smaller and
            # better at the same time, because the bands it removes were noise.
            'min_area_smooth':       8000,
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
            # Same omission as 'low'. Without it this preset was strictly worse
            # than the cheap one — MORE paths than 'ultra' (1249 vs 978), twice
            # the bytes of 'low', and a LOWER SSIM than 'low' — so the middle
            # button on the UI was the worst choice available. Measured:
            # absent → 1249 paths / 1061 KB / SSIM 0.9546;
            # 2500   →  345 paths /  514 KB / SSIM 0.9747.
            # Must stay ABOVE the 'high'/'ultra' value: this threshold is a
            # merge floor, so a smaller number keeps more bands. Setting it
            # below theirs made 'medium' preserve more detail than 'high', which
            # is how the ladder ended up non-monotonic (a higher preset scoring
            # BELOW the one under it on soft images).
            'min_area_smooth':       2500,
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

