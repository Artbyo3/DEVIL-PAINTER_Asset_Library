"""Procedural Metals & Armor generator suite (10 physically grounded metal materials).

Built strictly on the committed periodic math primitives in ``tools/generators/common.py``
with self-contained metallic palettes defined directly in this module:
  0. brushed-steel        - Directional satin-brushed stainless steel sheet
  1. polished-chrome      - Mirror-finished chrome with subtle buffing swirls & micro-haze
  2. hammered-iron-plate  - Hand-planished iron armor plate with concave peen dimples & mill scale
  3. riveted-steel-plate  - Running-bond beveled armor steel plates with domed perimeter rivets
  4. chainmail-classic    - Interlocked 3D four-in-one polished steel maille rings
  5. chainmail-darkened   - Interlocked 3D blackened oil-quenched armor maille with worn crowns
  6. antique-brass        - Aged golden-bronze brass with hand-rubbed sheen & tarnished pits
  7. oxidized-copper      - Weathered copper transitioning through cuprite into crusty verdigris
  8. woven-metal-mesh     - Plain-weave crimped metallic wire screen with open dark apertures
  9. damascus-steel       - Pattern-welded acid-etched bilayer steel with raindrop & flow folds
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from PIL import Image

from .. import common
from ..interface import GeneratorSpec, PaletteArg


@dataclass(frozen=True)
class MetalStyleSpec:
    variant: int
    slug: str
    label: str
    default_palette: str
    normal_strength: float
    rough_range: tuple[float, float]
    description: str


METAL_PALETTES: dict[str, list[tuple[int, int, int]]] = {
    "brushed_steel": [
        (86, 92, 100),
        (134, 141, 150),
        (176, 183, 192),
        (210, 216, 224),
        (240, 244, 249),
    ],
    "polished_chrome": [
        (114, 122, 134),
        (164, 172, 184),
        (206, 213, 223),
        (234, 239, 246),
        (252, 253, 255),
    ],
    "hammered_iron_plate": [
        (26, 28, 32),
        (54, 58, 64),
        (94, 100, 108),
        (146, 154, 164),
        (202, 210, 220),
    ],
    "riveted_steel_plate": [
        (30, 34, 40),
        (68, 74, 82),
        (116, 124, 134),
        (166, 174, 184),
        (218, 224, 232),
    ],
    "chainmail_classic": [
        (12, 14, 18),
        (76, 82, 92),
        (138, 146, 158),
        (194, 202, 214),
        (240, 244, 250),
    ],
    "chainmail_darkened": [
        (10, 12, 15),
        (32, 36, 44),
        (58, 64, 74),
        (98, 106, 118),
        (156, 166, 180),
    ],
    "antique_brass": [
        (42, 32, 16),
        (96, 76, 36),
        (158, 128, 62),
        (208, 176, 98),
        (244, 220, 146),
    ],
    "oxidized_copper": [
        (46, 26, 20),
        (112, 56, 34),
        (188, 102, 62),
        (72, 154, 134),
        (148, 212, 192),
    ],
    "woven_metal_mesh": [
        (12, 14, 18),
        (74, 80, 88),
        (136, 144, 154),
        (194, 202, 212),
        (238, 242, 248),
    ],
    "damascus_steel": [
        (20, 23, 28),
        (54, 60, 68),
        (108, 116, 128),
        (178, 186, 198),
        (234, 240, 248),
    ],
}


METAL_STYLES: tuple[MetalStyleSpec, ...] = (
    MetalStyleSpec(
        0,
        "brushed-steel",
        "Brushed Steel",
        "brushed_steel",
        2.3,
        (0.18, 0.44),
        "Directional satin-brushed stainless steel with fine linear abrasive striations, subtle belt-pass banding, and sparse cross-scuffs.",
    ),
    MetalStyleSpec(
        1,
        "polished-chrome",
        "Polished Chrome",
        "polished_chrome",
        1.8,
        (0.04, 0.18),
        "Mirror-finished electroplated chrome with ultra-low roughness, delicate optical waviness, fine buffing swirls, and faint handling haze.",
    ),
    MetalStyleSpec(
        2,
        "hammered-iron-plate",
        "Hammered Iron Plate",
        "hammered_iron_plate",
        7.5,
        (0.20, 0.84),
        "Hand-planished forged iron armor plate with overlapping concave ball-peen dimples, burnished facet ridges, and dark recessed mill scale.",
    ),
    MetalStyleSpec(
        3,
        "riveted-steel-plate",
        "Riveted Steel Plate",
        "riveted_steel_plate",
        7.2,
        (0.18, 0.88),
        "Running-bond beveled steel armor plates with recessed panel seams, domed perimeter rivets, seam grime collars, and burnished edges.",
    ),
    MetalStyleSpec(
        4,
        "chainmail-classic",
        "Chainmail Classic",
        "chainmail_classic",
        8.0,
        (0.14, 0.94),
        "Interlocked 3D four-in-one bare steel maille rings with tubular cylindrical cross-sections, bright worn crowns, and deep dark inter-ring voids.",
    ),
    MetalStyleSpec(
        5,
        "chainmail-darkened",
        "Chainmail Darkened",
        "chainmail_darkened",
        8.2,
        (0.20, 0.96),
        "Oil-quenched blackened steel four-in-one maille rings with riveted overlap bosses, satin dark oxide finish, and rubbed crown highlights.",
    ),
    MetalStyleSpec(
        6,
        "antique-brass",
        "Antique Brass",
        "antique_brass",
        3.2,
        (0.16, 0.80),
        "Aged architectural brass with warm burnished golden highs, hand-rubbed directional buffing grain, mottled bronze patina, and dark tarnished pits.",
    ),
    MetalStyleSpec(
        7,
        "oxidized-copper",
        "Oxidized Copper",
        "oxidized_copper",
        4.8,
        (0.15, 0.94),
        "Weathered copper plate transitioning from burnished reddish-orange metal through dark cuprite oxide into raised porous turquoise verdigris crust.",
    ),
    MetalStyleSpec(
        8,
        "woven-metal-mesh",
        "Woven Metal Mesh",
        "woven_metal_mesh",
        7.0,
        (0.15, 0.95),
        "Plain-weave crimped metallic wire cloth with smooth drawn steel cylinders, pinch-crimp crossings, and crisp open dark square apertures.",
    ),
    MetalStyleSpec(
        9,
        "damascus-steel",
        "Damascus Steel",
        "damascus_steel",
        3.4,
        (0.12, 0.64),
        "Pattern-welded folded bilayer steel with flowing curl-warped raindrop contours, raised polished nickel bands, and matte acid-etched dark valleys.",
    ),
)

STYLE_BY_SLUG: dict[str, MetalStyleSpec] = {s.slug: s for s in METAL_STYLES}


def _smoothstep(edge0: float, edge1: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - edge0) / max(edge1 - edge0, 1e-6), 0.0, 1.0)
    return (t * t * (3.0 - 2.0 * t)).astype(np.float32)


def _resolve_local_palette(
    palette: PaletteArg,
    default_key: str,
    rng: np.random.Generator,
) -> list[tuple[int, int, int]]:
    if palette is None:
        return METAL_PALETTES[default_key]
    if isinstance(palette, str):
        key = palette.strip().lower()
        if key in METAL_PALETTES:
            return METAL_PALETTES[key]
    return common.resolve_palette(palette, "folded_damascus_steel", rng)


def _lerp_rgb(
    a: np.ndarray | tuple[int, int, int],
    b: np.ndarray | tuple[int, int, int],
    t: np.ndarray,
) -> np.ndarray:
    ca = np.asarray(a, dtype=np.float32)
    cb = np.asarray(b, dtype=np.float32)
    if ca.ndim == 1:
        ca = ca[None, None, :]
    if cb.ndim == 1:
        cb = cb[None, None, :]
    tt = np.clip(t, 0.0, 1.0).astype(np.float32)[..., None]
    return (ca * (1.0 - tt) + cb * tt).astype(np.float32)


def _eval_3stop(
    field: np.ndarray,
    c0: tuple[int, int, int],
    c1: tuple[int, int, int],
    c2: tuple[int, int, int],
) -> np.ndarray:
    f = np.clip(field, 0.0, 1.0).astype(np.float32)
    lo = _lerp_rgb(c0, c1, f * 2.0)
    hi = _lerp_rgb(c1, c2, (f - 0.5) * 2.0)
    return np.where((f < 0.5)[..., None], lo, hi).astype(np.float32)


# ---------------------------------------------------------------------------
# 0. brushed-steel
# ---------------------------------------------------------------------------
def _build_brushed_steel(
    u: np.ndarray,
    v: np.ndarray,
    size: int,
    rng: np.random.Generator,
    stops: Sequence[tuple[int, int, int]],
) -> tuple[Image.Image, np.ndarray, np.ndarray]:
    # Multi-scale horizontal abrasive brush grain (parallel to U axis)
    fine_brush = common.periodic_fbm(
        u, v, freqx=14, freqy=340, octaves=5, persistence=0.56, rng=rng, rotate_octaves=False
    )
    med_brush = common.periodic_fbm(
        u, v, freqx=6, freqy=112, octaves=4, persistence=0.54, rng=rng, rotate_octaves=False
    )
    coarse_band = common.periodic_fbm(
        u, v, freqx=3, freqy=28, octaves=3, persistence=0.50, rng=rng, rotate_octaves=False
    )
    # Periodic directional Gabor streaks (omega0 = pi/2 oscillates along V -> lines along U)
    gabor_h = common.periodic_gabor_noise(
        u, v, grid_cells=18, f0=5.5, omega0=math.pi * 0.5, bandwidth=1.8, rng=rng, impulses_per_cell=3
    )

    # Subtle long handling cross-scratches at slight oblique angles
    sc_mask1 = _smoothstep(0.68, 0.86, common.periodic_fbm(u, v, 5, 3, 0.5, rng))
    sc1_raw = common.periodic_gabor_noise(
        u, v, grid_cells=10, f0=4.5, omega0=math.pi * 0.5 + 0.22, bandwidth=1.4, rng=rng, impulses_per_cell=2
    )
    sc1 = np.power(np.clip(1.0 - np.abs(sc1_raw - 0.5) * 2.0, 0.0, 1.0), 4.0) * sc_mask1

    sc_mask2 = _smoothstep(0.72, 0.89, common.periodic_fbm(u, v, 7, 3, 0.5, rng))
    sc2_raw = common.periodic_gabor_noise(
        u, v, grid_cells=12, f0=5.0, omega0=math.pi * 0.5 - 0.17, bandwidth=1.5, rng=rng, impulses_per_cell=2
    )
    sc2 = np.power(np.clip(1.0 - np.abs(sc2_raw - 0.5) * 2.0, 0.0, 1.0), 4.5) * sc_mask2
    scratches = np.clip(sc1 + 0.85 * sc2, 0.0, 1.0)

    # Broad mill-sheet sheen undulation & fine metallic grain
    mill_sheen = common.periodic_fbm(u, v, freqx=2, freqy=5, octaves=3, persistence=0.45, rng=rng, rotate_octaves=False)
    micro = common.periodic_fbm(u, v, 32, 4, 0.52, rng)

    h = common.robust_normalize(
        0.44 * fine_brush
        + 0.24 * med_brush
        + 0.14 * gabor_h
        + 0.10 * coarse_band
        + 0.06 * mill_sheen
        - 0.14 * scratches,
        low_pct=0.2,
        high_pct=99.8,
    )

    color_field = common.robust_normalize(
        0.34 * fine_brush
        + 0.22 * med_brush
        + 0.20 * coarse_band
        + 0.18 * mill_sheen
        + 0.06 * gabor_h
        + 0.08 * scratches,
        low_pct=0.3,
        high_pct=99.7,
    )
    albedo = common.apply_palette_5stop(color_field, micro, stops, contrast=1.14)
    # Brighten sharp cross-scratch glints slightly
    glint = (scratches * 0.18)[..., None]
    albedo = np.clip(albedo * (1.0 - glint * 0.3) + np.array([236.0, 240.0, 246.0], dtype=np.float32) * glint, 0.0, 255.0)

    # Roughness: polished brush crowns low, recessed abrasive grooves and scuff zones higher
    rough_raw = (
        0.42 * (1.0 - fine_brush)
        + 0.22 * (1.0 - med_brush)
        + 0.16 * (1.0 - coarse_band)
        + 0.14 * scratches
        + 0.06 * micro
    )
    rough = common.match_histogram_range(rough_raw, 0.18, 0.44, target_mean=0.29, contrast=1.12)

    base = common.shade_pbr_surface(
        albedo,
        h,
        roughness=rough,
        normal_strength=2.3,
        spec_intensity=0.36,
        ao_strength=0.14,
        cavity_strength=0.12,
    )
    return base, h, rough


# ---------------------------------------------------------------------------
# 1. polished-chrome
# ---------------------------------------------------------------------------
def _build_polished_chrome(
    u: np.ndarray,
    v: np.ndarray,
    size: int,
    rng: np.random.Generator,
    stops: Sequence[tuple[int, int, int]],
) -> tuple[Image.Image, np.ndarray, np.ndarray]:
    # Subtle electroplated mirror surface undulation (very smooth broad optical wave)
    plating_wave = common.periodic_fbm(u, v, 3, 3, 0.42, rng)
    plating_wave = common.periodic_gaussian_blur(plating_wave, max(2.0, size * 0.006))

    # Ultra-delicate rotary buffing swirls via divergence-free curl advection
    vx, vy = common.periodic_curl_noise(u, v, 5, 3, 0.48, rng)
    su = (u + 0.018 * vx) % 1.0
    sv = (v + 0.018 * vy) % 1.0
    buff_swirl_a = common.periodic_gabor_noise(
        su, sv, grid_cells=12, f0=4.0, omega0=0.4, bandwidth=1.5, rng=rng, impulses_per_cell=2, isotropic=True
    )
    buff_swirl_b = common.periodic_fbm(su, sv, 28, 4, 0.46, rng)
    swirl = 0.55 * buff_swirl_a + 0.45 * buff_swirl_b

    # Hairline micro-scratches (sparse, faint, realistic inspection micro-wear)
    hair_mask = _smoothstep(0.76, 0.92, common.periodic_fbm(u, v, 6, 3, 0.5, rng))
    h_line1 = common.periodic_fbm(u, v, 24, 3, 0.5, rng, mode="ridged")
    hairlines = np.power(h_line1, 5.0) * hair_mask

    # Soft handling smudge / micro-haze clouds (drives subtle roughness contrast)
    smudge_field, _, _ = common.periodic_domain_warp_fbm(
        u, v, freq=4, octaves=4, persistence=0.50, rng=rng, warp_strength=0.06, stages=1
    )
    smudge = _smoothstep(0.54, 0.86, smudge_field)

    # Smooth periodic environment-like metallic tonal reflection wave
    refl_a = common.periodic_fbm(u, v, 2, 3, 0.45, rng)
    refl_b = 0.5 + 0.5 * np.sin((u * 1.0 + v * 1.0 + 0.25 * (refl_a - 0.5)) * 2.0 * math.pi)
    refl = common.robust_normalize(0.52 * refl_a + 0.36 * refl_b + 0.12 * plating_wave, low_pct=0.5, high_pct=99.5)

    h = common.robust_normalize(
        0.42 * plating_wave + 0.44 * swirl - 0.14 * hairlines,
        low_pct=0.2,
        high_pct=99.8,
    )

    color_field = common.robust_normalize(
        0.74 * refl + 0.16 * plating_wave + 0.08 * swirl - 0.05 * smudge,
        low_pct=0.4,
        high_pct=99.6,
    )
    albedo = common.apply_palette_5stop(color_field, swirl * 0.4 + 0.3, stops, contrast=1.18)
    # Subtle cool tint modulation in slightly hazed zones
    haze_tint = np.array([186.0, 192.0, 200.0], dtype=np.float32)
    albedo = _lerp_rgb(albedo, haze_tint, smudge * 0.14)

    # Mirror-low roughness: gleaming 0.04-0.08 core, buffing swirls & light haze up to 0.18
    rough_raw = 0.44 * smudge + 0.32 * (1.0 - swirl) + 0.16 * hairlines + 0.08 * (1.0 - plating_wave)
    rough = common.match_histogram_range(rough_raw, 0.04, 0.18, target_mean=0.085, contrast=1.18)

    base = common.shade_pbr_surface(
        albedo,
        h,
        roughness=rough,
        normal_strength=1.8,
        spec_intensity=0.48,
        ao_strength=0.07,
        cavity_strength=0.06,
    )
    return base, h, rough


# ---------------------------------------------------------------------------
# 2. hammered-iron-plate
# ---------------------------------------------------------------------------
def _build_hammered_iron_plate(
    u: np.ndarray,
    v: np.ndarray,
    size: int,
    rng: np.random.Generator,
    stops: Sequence[tuple[int, int, int]],
) -> tuple[Image.Image, np.ndarray, np.ndarray]:
    # Gentle organic domain warp so hammer strike boundaries look hand-forged
    wu = (common.periodic_fbm(u, v, 5, 3, 0.48, rng) - 0.5) * 0.014
    wv = (common.periodic_fbm(u, v, 5, 3, 0.48, rng) - 0.5) * 0.014
    uw = (u + wu) % 1.0
    vw = (v + wv) % 1.0

    # Primary hand-planished ball-peen indentations (R2-seeded Lloyd CVT)
    f1_a, edge_a, cid_a, dx_a, dy_a = common.periodic_lloyd_voronoi(uw, vw, 10, 0.92, rng, lloyd_iters=2)
    # Secondary overlapping smaller planishing strikes
    f1_b, edge_b, cid_b, _, _ = common.periodic_lloyd_voronoi(uw, vw, 19, 0.95, rng, lloyd_iters=1)

    # Concave spherical peen bowl with crisp burnished ridge boundaries
    dimple_a = np.power(np.clip(f1_a * 1.16, 0.0, 1.0), 1.85)
    dimple_b = np.power(np.clip(f1_b * 1.14, 0.0, 1.0), 1.75)
    rim_a = np.exp(-edge_a * 6.8).astype(np.float32)
    rim_b = np.exp(-edge_b * 8.5).astype(np.float32)

    # Combine primary & secondary strikes with subtle organic overlap
    overlap_w = 0.16 + 0.12 * common.periodic_fbm(u, v, 4, 3, 0.5, rng)
    dimple = (1.0 - overlap_w) * dimple_a + overlap_w * dimple_b
    rim = np.clip(0.80 * rim_a + 0.30 * rim_b, 0.0, 1.0)

    # Forged mill scale & fine fire-pitting trapped inside recessed dimple bottoms
    scale_noise = common.periodic_fbm(u, v, 20, 5, 0.56, rng, mode="turbulence")
    fine_pits = common.periodic_fbm(u, v, 44, 4, 0.54, rng, mode="ridged")
    hollow_mask = np.clip(1.0 - dimple * 1.15, 0.0, 1.0)
    mill_scale = _smoothstep(0.32, 0.78, hollow_mask * 0.72 + scale_noise * 0.45)

    # Subtle directional light-catching facet tilt modulation
    facet_tilt = np.clip(0.5 + 0.35 * (dx_a * 0.65 - dy_a * 0.75), 0.0, 1.0)
    micro = common.periodic_fbm(u, v, 16, 4, 0.52, rng)

    h = common.robust_normalize(
        0.74 * dimple
        + 0.12 * rim
        + 0.08 * micro
        - 0.10 * mill_scale * (0.4 + 0.6 * fine_pits),
        low_pct=0.2,
        high_pct=99.8,
    )

    color_field = common.robust_normalize(
        0.44 * dimple
        + 0.20 * rim
        + 0.14 * cid_a
        + 0.10 * facet_tilt
        + 0.12 * micro
        - 0.24 * mill_scale,
        low_pct=0.3,
        high_pct=99.7,
    )
    albedo = common.apply_palette_5stop(color_field, micro, stops, contrast=1.18)
    # Warm dark charcoal-oxide cast in recessed forge scale pockets
    forge_scale_col = np.array([32.0, 28.0, 26.0], dtype=np.float32)
    albedo = _lerp_rgb(albedo, forge_scale_col, mill_scale * 0.52)

    # Roughness: burnished ridges & high shoulders smooth, hammered bowls & mill scale matte
    rough_raw = (
        0.42 * mill_scale
        + 0.32 * (1.0 - dimple)
        + 0.16 * (1.0 - rim)
        + 0.10 * fine_pits
    )
    rough = common.match_histogram_range(rough_raw, 0.20, 0.84, target_mean=0.50, contrast=1.16)

    base = common.shade_pbr_surface(
        albedo,
        h,
        roughness=rough,
        normal_strength=7.5,
        spec_intensity=0.38,
        ao_strength=0.28,
        cavity_strength=0.24,
    )
    return base, h, rough


# ---------------------------------------------------------------------------
# 3. riveted-steel-plate
# ---------------------------------------------------------------------------
def _build_riveted_steel_plate(
    u: np.ndarray,
    v: np.ndarray,
    size: int,
    rng: np.random.Generator,
    stops: Sequence[tuple[int, int, int]],
) -> tuple[Image.Image, np.ndarray, np.ndarray]:
    # Periodic running-bond armor plates: 2 wide columns x 4 rows (even -> seamless wrap)
    # Half-cell phase shift places beveled seams cleanly inside the tile interior
    cols = 2
    rows = 4
    us = (u + 0.125) % 1.0
    vs = (v + 0.125) % 1.0

    gy = vs * rows
    iy = np.floor(gy).astype(np.int32) % rows
    row_shift = (iy % 2).astype(np.float32) * 0.5
    gx = ((us * cols) + row_shift) % cols
    ix = np.floor(gx).astype(np.int32) % cols

    # Local plate coordinates in [0, 1)
    lu = gx - np.floor(gx)
    lv = gy - np.floor(gy)

    # Per-plate deterministic height step, tone variation, and slight brush tint
    plate_h_tbl = rng.uniform(-0.032, 0.032, size=(rows, cols)).astype(np.float32)
    plate_tone_tbl = rng.uniform(-0.09, 0.09, size=(rows, cols)).astype(np.float32)
    plate_h_off = plate_h_tbl[iy, ix]
    plate_tone = plate_tone_tbl[iy, ix]

    # Aspect-corrected distance to plate edges (plate is 2x wider than tall -> U scale = 2, V scale = 1)
    du_edge = np.minimum(lu, 1.0 - lu) * 2.0
    dv_edge = np.minimum(lv, 1.0 - lv)
    edge_dist = np.minimum(du_edge, dv_edge)

    # Deep recessed seam channel + crisp chamfered armor plate bevel
    seam_gap = 0.016
    bevel_w = 0.038
    seam_open = _smoothstep(seam_gap, seam_gap + 0.012, edge_dist)
    bevel_ramp = _smoothstep(seam_gap, seam_gap + bevel_w, edge_dist)
    bevel_edge_wear = np.exp(-((edge_dist - (seam_gap + bevel_w * 0.85)) / 0.024) ** 2).astype(np.float32)

    # Perimeter dome-head rivets in aspect-corrected plate space:
    # x_p in [0, 2], y_p in [0, 1]
    xp = lu * 2.0
    yp = lv
    margin = 0.135
    rivet_centers = [
        # Top edge (5 rivets)
        (margin, margin),
        (0.55, margin),
        (1.00, margin),
        (1.45, margin),
        (2.0 - margin, margin),
        # Bottom edge (5 rivets)
        (margin, 1.0 - margin),
        (0.55, 1.0 - margin),
        (1.00, 1.0 - margin),
        (1.45, 1.0 - margin),
        (2.0 - margin, 1.0 - margin),
        # Left & right mid rivets (2 rivets)
        (margin, 0.50),
        (2.0 - margin, 0.50),
    ]

    rivet_r = 0.056
    rivet_dome = np.zeros_like(u, dtype=np.float32)
    rivet_collar = np.zeros_like(u, dtype=np.float32)
    rivet_mask = np.zeros_like(u, dtype=np.float32)

    # Per-plate slight hand-driven rivet jitter tables
    n_riv = len(rivet_centers)
    jx_tbl = rng.uniform(-0.008, 0.008, size=(rows, cols, n_riv)).astype(np.float32)
    jy_tbl = rng.uniform(-0.008, 0.008, size=(rows, cols, n_riv)).astype(np.float32)
    jr_tbl = rng.uniform(0.93, 1.06, size=(rows, cols, n_riv)).astype(np.float32)

    for k, (rx0, ry0) in enumerate(rivet_centers):
        cx = rx0 + jx_tbl[iy, ix, k]
        cy = ry0 + jy_tbl[iy, ix, k]
        rr = rivet_r * jr_tbl[iy, ix, k]
        d = np.hypot(xp - cx, yp - cy)
        # Hemispherical forged rivet head with subtle flattened top crown
        rn = d / rr
        inside = rn < 1.0
        dome_k = np.where(inside, np.sqrt(np.clip(1.0 - rn * rn, 0.0, 1.0)), 0.0).astype(np.float32)
        # Subtle hammer-peened crown flattening
        dome_k = np.power(dome_k, 0.82) * _smoothstep(1.02, 0.90, rn)
        # Indented seating collar & grime ring right around rivet base
        collar_k = np.exp(-((rn - 1.18) / 0.28) ** 2).astype(np.float32)
        rivet_dome = np.maximum(rivet_dome, dome_k)
        rivet_collar = np.maximum(rivet_collar, collar_k)
        rivet_mask = np.maximum(rivet_mask, _smoothstep(1.02, 0.84, rn))

    # Rolled steel mill grain, subtle scratches, and crevice grime
    rolled_grain = common.periodic_fbm(u, v, freqx=8, freqy=96, octaves=4, persistence=0.52, rng=rng, rotate_octaves=False)
    macro_mottle = common.periodic_fbm(u, v, 6, 4, 0.54, rng)
    fine_pits = common.periodic_fbm(u, v, 36, 4, 0.52, rng, mode="turbulence")
    grime_noise = common.periodic_fbm(u, v, 14, 4, 0.56, rng)

    seam_grime = np.clip((1.0 - _smoothstep(seam_gap, 0.14, edge_dist)) * (0.45 + 0.55 * grime_noise), 0.0, 1.0)
    collar_grime = np.clip(rivet_collar * (0.50 + 0.50 * grime_noise), 0.0, 1.0)
    grime = np.clip(np.maximum(seam_grime, collar_grime * 0.85), 0.0, 1.0)

    # Assemble physical height: deep seam (0) -> beveled plate (~0.52) -> raised domed rivets (1.0)
    plate_surface = (
        0.52
        + plate_h_off
        + 0.045 * (rolled_grain - 0.5)
        + 0.030 * (macro_mottle - 0.5)
        - 0.025 * (1.0 - fine_pits)
        - 0.040 * rivet_collar * (1.0 - rivet_mask)
    )
    h_raw = seam_open * (0.14 + 0.86 * bevel_ramp) * plate_surface + 0.44 * rivet_dome
    h = common.normalize_field(h_raw)

    # Base color with steel mill tone, bright edge/rivet burnish, and warm dark grime in crevices
    color_field = common.robust_normalize(
        0.42 * (0.5 + plate_tone)
        + 0.22 * rolled_grain
        + 0.14 * macro_mottle
        + 0.24 * rivet_dome
        + 0.14 * bevel_edge_wear
        - 0.28 * (1.0 - seam_open)
        - 0.18 * grime,
        low_pct=0.3,
        high_pct=99.7,
    )
    albedo = common.apply_palette_5stop(color_field, fine_pits, stops, contrast=1.18)
    # Subtle warm rust-oil stain inside recessed seams & around rivet collars
    grime_col = np.array([46.0, 36.0, 28.0], dtype=np.float32)
    albedo = _lerp_rgb(albedo, grime_col, grime * 0.48)
    # Darken deep seam gap
    albedo = _lerp_rgb(np.array([16.0, 18.0, 22.0], dtype=np.float32), albedo, seam_open)

    # Roughness: polished rivet crowns & beveled plate edges low, rolled plate satin, grime/seams matte
    rough_raw = (
        0.38
        + 0.16 * (1.0 - rolled_grain)
        + 0.12 * (1.0 - macro_mottle)
        + 0.36 * grime
        + 0.40 * (1.0 - seam_open)
        - 0.24 * rivet_dome
        - 0.14 * bevel_edge_wear
    )
    rough = common.match_histogram_range(rough_raw, 0.16, 0.88, target_mean=0.44, contrast=1.18)

    base = common.shade_pbr_surface(
        albedo,
        h,
        roughness=rough,
        normal_strength=7.2,
        spec_intensity=0.36,
        ao_strength=0.28,
        cavity_strength=0.24,
    )
    return base, h, rough


# ---------------------------------------------------------------------------
# Shared exact 3D interlocked 4-in-1 chainmail geometry engine
# ---------------------------------------------------------------------------
def _compute_chainmail_geometry(
    u: np.ndarray,
    v: np.ndarray,
    cols: int,
    rows: int,
    tilt_deg: float,
    ring_r_ratio: float,
    wire_r_ratio: float,
    rng: np.random.Generator,
    rivet_boss: bool = False,
) -> dict[str, np.ndarray]:
    """Compute exact 3D Z-buffered four-in-one interlocked chainmail rings on T^2.

    Every ring in row j is tilted around the Y-axis by (-1)^j * tilt_deg, guaranteeing:
      - Same-row neighbors lie in parallel planes (zero collision).
      - Adjacent-row neighbors interlock topologically (one OVER crossing, one UNDER crossing).
      - Exact periodic toroidal wrapping when `rows` is even.
    """
    dx = 1.0 / float(cols)
    dy = 1.0 / float(rows)
    alpha = math.radians(tilt_deg)
    cos_a = math.cos(alpha)
    sin_a = math.sin(alpha)

    # Ring physical radius R in UV space and wire tube radius r_w
    R_base = dx * ring_r_ratio
    rw_base = R_base * wire_r_ratio

    # Per-ring subtle physical variation tables (hand-linked authenticity)
    jx_tbl = rng.uniform(-0.035 * dx, 0.035 * dx, size=(rows, cols)).astype(np.float32)
    jy_tbl = rng.uniform(-0.035 * dy, 0.035 * dy, size=(rows, cols)).astype(np.float32)
    jr_tbl = rng.uniform(0.96, 1.04, size=(rows, cols)).astype(np.float32)
    tone_tbl = rng.uniform(0.0, 1.0, size=(rows, cols)).astype(np.float32)
    boss_ang_tbl = rng.uniform(-0.45, 0.45, size=(rows, cols)).astype(np.float32)

    # Base cell index for each pixel
    ix0 = np.floor(u * cols).astype(np.int32) % cols
    iy0 = np.floor(v * rows).astype(np.int32) % rows

    z_buf = np.full_like(u, -1e4, dtype=np.float32)
    tube_prof = np.zeros_like(u, dtype=np.float32)
    ring_tone = np.zeros_like(u, dtype=np.float32)
    ring_phi = np.zeros_like(u, dtype=np.float32)
    crown_z = np.zeros_like(u, dtype=np.float32)
    boss_map = np.zeros_like(u, dtype=np.float32)

    # 3x3 cell neighborhood covers all potentially overlapping rings cleanly
    for oy in (-1, 0, 1):
        ry_idx = (iy0 + oy) % rows
        parity = (ry_idx % 2).astype(np.float32)
        s_sign = np.where((ry_idx % 2) == 0, 1.0, -1.0).astype(np.float32)

        for ox in (-1, 0, 1):
            rx_idx = (ix0 + ox) % cols
            # Toroidal center coordinates for ring (rx_idx, ry_idx), quarter-pitch centered
            cx = (rx_idx.astype(np.float32) + 0.25 + 0.5 * parity) * dx + jx_tbl[ry_idx, rx_idx]
            cy = (ry_idx.astype(np.float32) + 0.5) * dy + jy_tbl[ry_idx, rx_idx]

            # Shortest signed toroidal displacement from ring center to pixel
            px = ((u - cx + 0.5) % 1.0) - 0.5
            py = ((v - cy + 0.5) % 1.0) - 0.5

            R = R_base * jr_tbl[ry_idx, rx_idx]
            rx = R * cos_a
            ry = R
            rw = rw_base * jr_tbl[ry_idx, rx_idx]

            # Nearest point on projected ellipse (1 Newton step refines exact normal distance)
            phi = np.arctan2(py / ry, px / rx)
            c_phi = np.cos(phi)
            s_phi = np.sin(phi)
            ex = rx * c_phi
            ey = ry * s_phi
            # Tangential correction along ellipse to minimize squared distance
            tx = -rx * s_phi
            ty = ry * c_phi
            t_len2 = tx * tx + ty * ty + 1e-12
            dphi = ((px - ex) * tx + (py - ey) * ty) / t_len2
            phi = phi + np.clip(dphi, -0.35, 0.35)
            c_phi = np.cos(phi)
            s_phi = np.sin(phi)
            ex = rx * c_phi
            ey = ry * s_phi

            d_perp = np.hypot(px - ex, py - ey)
            in_wire = d_perp < rw

            # Exact 3D Z height: tilted ring plane Z + cylindrical half-tube dome
            cz_plane = s_sign * R * sin_a * c_phi
            tube_z = np.sqrt(np.clip(rw * rw - d_perp * d_perp, 0.0, rw * rw))
            z_cand = cz_plane + tube_z

            # Optional hammered riveted overlap boss at one side of each ring
            boss_k = np.zeros_like(u, dtype=np.float32)
            if rivet_boss:
                ang_diff = np.abs(np.arctan2(np.sin(phi - boss_ang_tbl[ry_idx, rx_idx]), np.cos(phi - boss_ang_tbl[ry_idx, rx_idx])))
                boss_k = np.exp(-((ang_diff / 0.24) ** 2)) * np.clip(1.0 - (d_perp / (rw * 1.12)) ** 2, 0.0, 1.0)
                z_cand = z_cand + 0.28 * rw * boss_k

            win = in_wire & (z_cand > z_buf)
            z_buf = np.where(win, z_cand, z_buf)
            tube_prof = np.where(win, tube_z / np.maximum(rw, 1e-6), tube_prof)
            ring_tone = np.where(win, tone_tbl[ry_idx, rx_idx], ring_tone)
            ring_phi = np.where(win, phi, ring_phi)
            crown_z = np.where(win, cz_plane, crown_z)
            boss_map = np.where(win, boss_k, boss_map)

    metal_mask = (z_buf > -1e3).astype(np.float32)
    z_min = -(R_base * sin_a + rw_base)
    z_max = +(R_base * sin_a + rw_base * (1.28 if rivet_boss else 1.0))
    z_norm = np.where(
        metal_mask > 0.5,
        np.clip((z_buf - z_min) / max(z_max - z_min, 1e-6), 0.0, 1.0),
        0.0,
    ).astype(np.float32)

    # Anti-alias ring silhouette softly into deep recessed voids (keeps Scharr normals crisp & smooth)
    h_metal = np.where(metal_mask > 0.5, 0.24 + 0.76 * z_norm, 0.0).astype(np.float32)
    crown_norm = np.where(
        metal_mask > 0.5,
        np.clip((crown_z + R_base * sin_a) / max(2.0 * R_base * sin_a, 1e-6), 0.0, 1.0),
        0.0,
    ).astype(np.float32)

    return {
        "height": h_metal,
        "mask": metal_mask,
        "tube": tube_prof,
        "tone": ring_tone,
        "phi": ring_phi,
        "crown": crown_norm,
        "boss": boss_map,
    }


# ---------------------------------------------------------------------------
# 4. chainmail-classic
# ---------------------------------------------------------------------------
def _build_chainmail_classic(
    u: np.ndarray,
    v: np.ndarray,
    size: int,
    rng: np.random.Generator,
    stops: Sequence[tuple[int, int, int]],
) -> tuple[Image.Image, np.ndarray, np.ndarray]:
    # Subtle fabric-like drape displacement so the maille sheet feels alive, never rigid
    du = (common.periodic_fbm(u, v, 3, 3, 0.45, rng) - 0.5) * 0.0045
    dv = (common.periodic_fbm(u, v, 3, 3, 0.45, rng) - 0.5) * 0.0045
    uw = (u + du) % 1.0
    vw = (v + dv) % 1.0

    geom = _compute_chainmail_geometry(
        uw,
        vw,
        cols=14,
        rows=24,
        tilt_deg=38.0,
        ring_r_ratio=0.585,
        wire_r_ratio=0.225,
        rng=rng,
        rivet_boss=False,
    )

    mask = geom["mask"]
    tube = geom["tube"]
    crown = geom["crown"]
    tone = geom["tone"]
    phi = geom["phi"]

    # Drawn steel micro-grain & subtle battle-rub wear along ring surfaces
    micro = common.periodic_fbm(u, v, 28, 4, 0.52, rng)
    wear_mottle = common.periodic_fbm(u, v, 6, 4, 0.50, rng)
    wire_draw = 0.5 + 0.5 * np.sin(phi * 10.0 + micro * 3.0)

    # Soften only sub-pixel edges slightly while keeping full tubular roundness
    h_raw = geom["height"] + mask * (0.025 * (micro - 0.5) + 0.012 * (wire_draw - 0.5))
    h = np.clip(h_raw, 0.0, 1.0).astype(np.float32)

    # Metallic steel brightness: bright gleaming top crowns, shaded lower flanks, deep dark voids
    metal_val = np.clip(
        0.22
        + 0.40 * crown
        + 0.24 * tube
        + 0.09 * tone
        + 0.07 * wear_mottle
        + 0.05 * wire_draw,
        0.18,
        1.0,
    )
    color_field = np.where(mask > 0.5, metal_val, 0.02 * micro).astype(np.float32)
    albedo = common.apply_palette_5stop(color_field, micro * mask, stops, contrast=1.15)
    # Deep shadow cavity floor behind rings
    albedo = _lerp_rgb(np.array([10.0, 12.0, 15.0], dtype=np.float32), albedo, mask)

    # Roughness: polished bare steel ring crowns very low (0.14-0.26), rubbed inner flanks higher, voids matte
    ring_rough = np.clip(
        0.34 - 0.14 * crown - 0.08 * tube + 0.09 * (1.0 - wear_mottle) + 0.05 * micro,
        0.14,
        0.52,
    )
    rough = np.where(mask > 0.5, ring_rough, 0.88 + 0.06 * micro).astype(np.float32)

    base = common.shade_pbr_surface(
        albedo,
        h,
        roughness=rough,
        normal_strength=8.0,
        spec_intensity=0.44,
        ao_strength=0.34,
        cavity_strength=0.28,
    )
    return base, h, rough


# ---------------------------------------------------------------------------
# 5. chainmail-darkened
# ---------------------------------------------------------------------------
def _build_chainmail_darkened(
    u: np.ndarray,
    v: np.ndarray,
    size: int,
    rng: np.random.Generator,
    stops: Sequence[tuple[int, int, int]],
) -> tuple[Image.Image, np.ndarray, np.ndarray]:
    du = (common.periodic_fbm(u, v, 3, 3, 0.45, rng) - 0.5) * 0.0040
    dv = (common.periodic_fbm(u, v, 3, 3, 0.45, rng) - 0.5) * 0.0040
    uw = (u + du) % 1.0
    vw = (v + dv) % 1.0

    # Slightly heavier gauge riveted oil-quenched blackened maille
    geom = _compute_chainmail_geometry(
        uw,
        vw,
        cols=16,
        rows=28,
        tilt_deg=39.0,
        ring_r_ratio=0.590,
        wire_r_ratio=0.235,
        rng=rng,
        rivet_boss=True,
    )

    mask = geom["mask"]
    tube = geom["tube"]
    crown = geom["crown"]
    tone = geom["tone"]
    boss = geom["boss"]

    micro = common.periodic_fbm(u, v, 32, 4, 0.54, rng)
    oxide_mottle = common.periodic_fbm(u, v, 8, 4, 0.52, rng)
    # Rubbed bright metallic edge wear on the highest exposed ring crowns & rivet bosses
    rub_wear = _smoothstep(0.58, 0.92, 0.62 * crown + 0.30 * tube + 0.25 * boss + 0.14 * oxide_mottle) * mask

    h_raw = geom["height"] + mask * (0.032 * (micro - 0.5))
    h = np.clip(h_raw, 0.0, 1.0).astype(np.float32)

    metal_val = np.clip(
        0.16
        + 0.30 * crown
        + 0.20 * tube
        + 0.10 * tone
        + 0.24 * rub_wear
        + 0.08 * boss,
        0.12,
        1.0,
    )
    color_field = np.where(mask > 0.5, metal_val, 0.015 * micro).astype(np.float32)
    albedo = common.apply_palette_5stop(color_field, micro * mask, stops, contrast=1.18)
    # Subtle warm oil-quench patina in deep ring undersides
    quench_patina = np.array([34.0, 28.0, 24.0], dtype=np.float32)
    albedo = _lerp_rgb(albedo, quench_patina, mask * (1.0 - crown) * oxide_mottle * 0.32)
    albedo = _lerp_rgb(np.array([8.0, 9.0, 12.0], dtype=np.float32), albedo, mask)

    # Roughness: darkened oil-oxide satin (0.34-0.56), rubbed crown wear smoother (0.20-0.30), deep voids matte
    ring_rough = np.clip(
        0.54 - 0.28 * rub_wear - 0.08 * tube + 0.08 * (1.0 - oxide_mottle) + 0.05 * micro,
        0.20,
        0.64,
    )
    rough = np.where(mask > 0.5, ring_rough, 0.90 + 0.06 * micro).astype(np.float32)

    base = common.shade_pbr_surface(
        albedo,
        h,
        roughness=rough,
        normal_strength=8.2,
        spec_intensity=0.36,
        ao_strength=0.38,
        cavity_strength=0.30,
    )
    return base, h, rough


# ---------------------------------------------------------------------------
# 6. antique-brass
# ---------------------------------------------------------------------------
def _build_antique_brass(
    u: np.ndarray,
    v: np.ndarray,
    size: int,
    rng: np.random.Generator,
    stops: Sequence[tuple[int, int, int]],
) -> tuple[Image.Image, np.ndarray, np.ndarray]:
    # Gentle antique rolled/cast brass surface undulation
    sheet_wave = common.periodic_fbm(u, v, 4, 3, 0.46, rng)

    # Multi-scale organic sulfur-tarnish & age-patina clouding via 2-stage curl warp
    tarnish_warp, _, _ = common.periodic_domain_warp_fbm(
        u, v, freq=5, octaves=5, persistence=0.54, rng=rng, warp_strength=0.09, stages=2, use_curl=True
    )
    tarnish_fine = common.periodic_fbm(u, v, 18, 5, 0.55, rng, mode="turbulence")
    tarnish_mask = _smoothstep(0.36, 0.78, 0.70 * tarnish_warp + 0.30 * tarnish_fine)

    # Hand-rubbed directional buffing grain (two soft overlapping polish orientations)
    region_blend = _smoothstep(0.30, 0.70, common.periodic_fbm(u, v, 3, 3, 0.50, rng))
    buff_a = common.periodic_gabor_noise(
        u, v, grid_cells=16, f0=5.0, omega0=math.pi * 0.5 + 0.18, bandwidth=1.7, rng=rng, impulses_per_cell=3
    )
    buff_b = common.periodic_gabor_noise(
        u, v, grid_cells=16, f0=4.5, omega0=math.pi * 0.5 - 0.32, bandwidth=1.7, rng=rng, impulses_per_cell=3
    )
    hand_buff = region_blend * buff_a + (1.0 - region_blend) * buff_b
    fine_grain = common.periodic_fbm(u, v, 48, 4, 0.54, rng)

    # Aged corrosion micro-pitting concentrated inside dark tarnished zones
    pit_vor, _, _, _, _ = common.periodic_voronoi(u, v, 34, 0.92, rng)
    pit_spots = _smoothstep(0.28, 0.08, pit_vor) * _smoothstep(0.35, 0.72, tarnish_mask)
    micro_speck = np.power(common.periodic_fbm(u, v, 56, 3, 0.55, rng, mode="ridged"), 3.0) * tarnish_mask

    h = common.robust_normalize(
        0.34 * sheet_wave
        + 0.28 * hand_buff
        + 0.20 * fine_grain
        + 0.14 * (1.0 - tarnish_mask)
        - 0.22 * pit_spots
        - 0.10 * micro_speck,
        low_pct=0.2,
        high_pct=99.8,
    )

    # Rich antique golden-bronze to burnished brass color response
    burnish = np.clip(1.0 - tarnish_mask, 0.0, 1.0)
    color_field = common.robust_normalize(
        0.52 * burnish
        + 0.18 * hand_buff
        + 0.14 * sheet_wave
        + 0.12 * fine_grain
        - 0.26 * pit_spots
        - 0.10 * micro_speck,
        low_pct=0.3,
        high_pct=99.7,
    )
    albedo = common.apply_palette_5stop(color_field, fine_grain, stops, contrast=1.18)
    # Deep olive-umber oxidation tint strictly inside pits and heavy tarnish
    deep_tarnish_col = np.array([46.0, 40.0, 22.0], dtype=np.float32)
    albedo = _lerp_rgb(albedo, deep_tarnish_col, np.clip(0.55 * pit_spots + 0.28 * tarnish_mask * (1.0 - hand_buff), 0.0, 0.75))

    # Roughness: burnished golden brass low (0.16-0.30), aged patina satin (0.40-0.58), dark pits rough (0.68-0.80)
    rough_raw = (
        0.54 * tarnish_mask
        + 0.22 * pit_spots
        + 0.14 * (1.0 - hand_buff)
        + 0.10 * micro_speck
    )
    rough = common.match_histogram_range(rough_raw, 0.16, 0.80, target_mean=0.44, contrast=1.18)

    base = common.shade_pbr_surface(
        albedo,
        h,
        roughness=rough,
        normal_strength=3.2,
        spec_intensity=0.34,
        ao_strength=0.20,
        cavity_strength=0.18,
    )
    return base, h, rough


# ---------------------------------------------------------------------------
# 7. oxidized-copper
# ---------------------------------------------------------------------------
def _build_oxidized_copper(
    u: np.ndarray,
    v: np.ndarray,
    size: int,
    rng: np.random.Generator,
    stops: Sequence[tuple[int, int, int]],
) -> tuple[Image.Image, np.ndarray, np.ndarray]:
    # Slight vertical rain-runoff bias mixed with 2-stage curl-warped oxidation spread
    ox_base, _, _ = common.periodic_domain_warp_fbm(
        u, v, freq=4, octaves=5, persistence=0.55, rng=rng, warp_strength=0.11, stages=2, use_curl=True
    )
    runoff = common.periodic_fbm(u, v, freqx=10, freqy=3, octaves=4, persistence=0.54, rng=rng, rotate_octaves=False)
    detail_ox = common.periodic_fbm(u, v, 18, 5, 0.56, rng, mode="turbulence")
    ox = common.robust_normalize(0.56 * ox_base + 0.22 * runoff + 0.22 * detail_ox, low_pct=0.5, high_pct=99.5)

    # Physical 3-stage oxidation zones:
    #   bare copper -> dark brown cuprite halo -> raised chalky turquoise verdigris crust
    verdigris = _smoothstep(0.48, 0.70, ox)
    cuprite_band = np.clip(_smoothstep(0.30, 0.52, ox) * (1.0 - _smoothstep(0.58, 0.78, ox)), 0.0, 1.0)

    # Exposed metallic copper sheet grain & subtle hammered waviness
    copper_wave = common.periodic_fbm(u, v, 5, 3, 0.45, rng)
    copper_brush = common.periodic_gabor_noise(
        u, v, grid_cells=16, f0=4.5, omega0=0.35, bandwidth=1.7, rng=rng, impulses_per_cell=2
    )
    fine_micro = common.periodic_fbm(u, v, 44, 5, 0.56, rng)

    # Crusty granular mineral texture & flaking rim inside verdigris patina islands
    crust_cells, crust_edge, _, _, _ = common.periodic_lloyd_voronoi(u, v, 24, 0.90, rng, lloyd_iters=1)
    crust_gran = common.periodic_fbm(u, v, 32, 5, 0.58, rng, mode="turbulence")
    crust_crack = np.exp(-crust_edge * 9.0).astype(np.float32) * verdigris
    patina_crust_h = verdigris * (0.24 + 0.16 * crust_gran + 0.10 * crust_cells - 0.14 * crust_crack)

    # Physical height: smooth undulating copper + slight etched cuprite trough + raised porous verdigris crust
    h = common.robust_normalize(
        0.34 * copper_wave
        + 0.14 * copper_brush * (1.0 - verdigris)
        + 0.12 * fine_micro
        - 0.10 * cuprite_band
        + patina_crust_h,
        low_pct=0.2,
        high_pct=99.8,
    )

    # Physically separated multi-material albedo blending (copper vs cuprite vs verdigris)
    copper_t = common.robust_normalize(
        0.48 * copper_wave + 0.30 * copper_brush + 0.22 * fine_micro, low_pct=0.5, high_pct=99.5
    )
    copper_rgb = _eval_3stop(copper_t, (116, 56, 32), (186, 100, 60), (234, 152, 104))

    cuprite_rgb = _eval_3stop(detail_ox, (38, 22, 18), (74, 42, 30), (112, 62, 42))

    verd_t = common.robust_normalize(
        0.46 * ox + 0.32 * crust_gran + 0.22 * fine_micro - 0.24 * crust_crack, low_pct=0.4, high_pct=99.6
    )
    verdigris_rgb = _eval_3stop(verd_t, (38, 96, 84), (76, 158, 138), (154, 214, 194))

    albedo = _lerp_rgb(copper_rgb, cuprite_rgb, np.clip(cuprite_band * 1.15, 0.0, 0.92))
    albedo = _lerp_rgb(albedo, verdigris_rgb, verdigris)

    # Roughness: gleaming bare copper (0.15-0.30), cuprite oxide (0.48-0.66), chalky verdigris crust (0.78-0.94)
    rough_copper = 0.16 + 0.15 * (1.0 - copper_brush) + 0.06 * (1.0 - fine_micro)
    rough_cuprite = 0.50 + 0.16 * detail_ox
    rough_verd = 0.78 + 0.16 * (0.6 * crust_gran + 0.4 * crust_crack)
    rough_blend = rough_copper * (1.0 - np.clip(cuprite_band, 0.0, 1.0)) + rough_cuprite * np.clip(cuprite_band, 0.0, 1.0)
    rough = np.clip(rough_blend * (1.0 - verdigris) + rough_verd * verdigris, 0.15, 0.94).astype(np.float32)

    base = common.shade_pbr_surface(
        albedo,
        h,
        roughness=rough,
        normal_strength=4.8,
        spec_intensity=0.34,
        ao_strength=0.22,
        cavity_strength=0.22,
    )
    return base, h, rough


# ---------------------------------------------------------------------------
# 8. woven-metal-mesh
# ---------------------------------------------------------------------------
def _build_woven_metal_mesh(
    u: np.ndarray,
    v: np.ndarray,
    size: int,
    rng: np.random.Generator,
    stops: Sequence[tuple[int, int, int]],
) -> tuple[Image.Image, np.ndarray, np.ndarray]:
    # Subtle wire-cloth tension waviness (periodic)
    du = (common.periodic_fbm(u, v, 3, 3, 0.45, rng) - 0.5) * 0.0022
    dv = (common.periodic_fbm(u, v, 3, 3, 0.45, rng) - 0.5) * 0.0022
    uw = (u + du) % 1.0
    vw = (v + dv) % 1.0

    # Plain 1/1 crimped metallic wire cloth: even count guarantees seamless toroidal wrap
    wires = 20
    tu = uw * wires
    tv = vw * wires
    iu = np.floor(tu).astype(np.int32) % wires
    iv = np.floor(tv).astype(np.int32) % wires
    fu = tu - np.floor(tu)
    fv = tv - np.floor(tv)

    # Per-wire subtle diameter & tone variation (drawn industrial wire stock)
    wu_r_tbl = rng.uniform(0.232, 0.252, size=wires).astype(np.float32)
    wv_r_tbl = rng.uniform(0.232, 0.252, size=wires).astype(np.float32)
    wu_tone_tbl = rng.uniform(0.0, 1.0, size=wires).astype(np.float32)
    wv_tone_tbl = rng.uniform(0.0, 1.0, size=wires).astype(np.float32)

    rw_u = wu_r_tbl[iu]
    rw_v = wv_r_tbl[iv]

    # Exact cylindrical wire cross-sections with crisp open apertures between wires
    du_c = (fu - 0.5) / rw_u
    dv_c = (fv - 0.5) / rw_v
    in_u = np.abs(du_c) < 1.0
    in_v = np.abs(dv_c) < 1.0
    cyl_u = np.where(in_u, np.sqrt(np.clip(1.0 - du_c * du_c, 0.0, 1.0)), 0.0).astype(np.float32)
    cyl_v = np.where(in_v, np.sqrt(np.clip(1.0 - dv_c * dv_c, 0.0, 1.0)), 0.0).astype(np.float32)

    # Continuous sinusoidal over/under wire crimp waves aligned cleanly at wire crossings (fu=0.5, fv=0.5)
    crimp_u = np.cos((tv - 0.5 + (iu % 2).astype(np.float32)) * math.pi).astype(np.float32)
    crimp_v = -np.cos((tu - 0.5 + (iv % 2).astype(np.float32)) * math.pi).astype(np.float32)

    # 3D Z-buffer of vertical (warp) and horizontal (weft) wires
    z_u = np.where(in_u, 0.50 + 0.21 * crimp_u + 0.27 * cyl_u, -1.0).astype(np.float32)
    z_v = np.where(in_v, 0.50 + 0.21 * crimp_v + 0.27 * cyl_v, -1.0).astype(np.float32)

    warp_top = z_u >= z_v
    wire_mask = (in_u | in_v).astype(np.float32)
    z_max = np.maximum(z_u, z_v)
    cyl_vis = np.where(warp_top, cyl_u, cyl_v)
    crimp_vis = np.where(warp_top, crimp_u, crimp_v)
    wire_tone = np.where(warp_top, wu_tone_tbl[iu], wv_tone_tbl[iv])

    # Longitudinal micro-striations along each drawn wire's axis
    micro = common.periodic_fbm(u, v, 36, 4, 0.52, rng)
    draw_u = 0.5 + 0.5 * np.sin((fu * 14.0 + micro * 0.8) * 2.0 * math.pi)
    draw_v = 0.5 + 0.5 * np.sin((fv * 14.0 + micro * 0.8) * 2.0 * math.pi)
    draw_vis = np.where(warp_top, draw_u, draw_v)

    # Pinch-contact shadow right where the under-wire tucks beneath the over-wire
    crossing_pinch = np.where(in_u & in_v, np.clip(1.0 - np.abs(crimp_vis), 0.0, 1.0), 0.0).astype(np.float32)

    # Height: crisp dark aperture floor (0.0), crimped metallic cylinders (0.22..1.0)
    h_raw = np.where(
        wire_mask > 0.5,
        0.22 + 0.78 * np.clip(z_max, 0.0, 1.0) + 0.018 * (draw_vis - 0.5) + 0.012 * (micro - 0.5),
        0.0,
    ).astype(np.float32)
    h = np.clip(h_raw, 0.0, 1.0)

    # Albedo: gleaming drawn steel crowns, subtle warp/weft anisotropy, dark open apertures
    wire_val = np.clip(
        0.26
        + 0.34 * cyl_vis
        + 0.22 * (0.5 + 0.5 * crimp_vis)
        + 0.08 * wire_tone
        + 0.06 * draw_vis
        + 0.04 * np.where(warp_top, 1.0, 0.65)
        - 0.14 * crossing_pinch,
        0.16,
        1.0,
    )
    color_field = np.where(wire_mask > 0.5, wire_val, 0.015 * micro).astype(np.float32)
    albedo = common.apply_palette_5stop(color_field, micro * wire_mask, stops, contrast=1.14)
    albedo = _lerp_rgb(np.array([10.0, 12.0, 16.0], dtype=np.float32), albedo, wire_mask)

    # Roughness: smooth drawn wire crowns low (0.15-0.28), crimped underside & pinch contacts rougher, apertures matte
    wire_rough = np.clip(
        0.38 - 0.16 * cyl_vis - 0.08 * (0.5 + 0.5 * crimp_vis) + 0.12 * crossing_pinch + 0.05 * micro,
        0.15,
        0.54,
    )
    rough = np.where(wire_mask > 0.5, wire_rough, 0.90 + 0.05 * micro).astype(np.float32)

    base = common.shade_pbr_surface(
        albedo,
        h,
        roughness=rough,
        normal_strength=7.0,
        spec_intensity=0.42,
        ao_strength=0.34,
        cavity_strength=0.28,
    )
    return base, h, rough


# ---------------------------------------------------------------------------
# 9. damascus-steel
# ---------------------------------------------------------------------------
def _build_damascus_steel(
    u: np.ndarray,
    v: np.ndarray,
    size: int,
    rng: np.random.Generator,
    stops: Sequence[tuple[int, int, int]],
) -> tuple[Image.Image, np.ndarray, np.ndarray]:
    # 2-stage divergence-free curl domain warp produces authentic forged bilayer flow eddies
    flow_field, wu, wv = common.periodic_domain_warp_fbm(
        u, v, freq=3, octaves=5, persistence=0.54, rng=rng, warp_strength=0.14, stages=2, use_curl=True
    )
    uw = (u + 0.65 * wu) % 1.0
    vw = (v + 0.65 * wv) % 1.0

    # Forged raindrop/dimple grind wells create closed concentric topographic rings
    vor_f1, _, _, _, _ = common.periodic_lloyd_voronoi(uw, vw, 5, 0.88, rng, lloyd_iters=2)
    raindrop_wells = common.periodic_gaussian_blur(np.power(1.0 - vor_f1, 1.8), max(2.0, size * 0.008))

    # Diagonal toroidal billet coordinate (integer coeffs 2, 1 -> exact seam-free periodicity)
    billet_coord = (
        (2.0 * u + 1.0 * v) * 4.0
        + 2.65 * flow_field
        + 1.75 * raindrop_wells
        + 0.45 * common.periodic_fbm(uw, vw, 6, 4, 0.52, rng, mode="turbulence")
    )

    # Primary forged steel/nickel billet stratification with organic band-width modulation
    phase = billet_coord * 2.0 * math.pi
    wave = np.sin(phase) + 0.30 * np.sin(2.0 * phase + 0.55) + 0.14 * np.sin(3.0 * phase - 0.35)
    # Sharp acid-etch plateau transition: 1.0 = raised bright nickel steel, 0.0 = recessed dark carbon steel
    nickel_bands = (0.5 + 0.5 * np.tanh(wave * 2.75)).astype(np.float32)

    # Ultra-fine secondary forge sub-laminations visible inside each layer
    sub_lam = (0.5 + 0.5 * np.sin(phase * 5.0 + 0.8 * flow_field * math.pi)).astype(np.float32)

    # Directional fine grind polish on raised nickel lands + frosty micro-etch inside dark valleys
    grind_polish = common.periodic_gabor_noise(
        u, v, grid_cells=18, f0=5.0, omega0=0.46, bandwidth=1.8, rng=rng, impulses_per_cell=3
    )
    etch_frost = common.periodic_fbm(u, v, 48, 5, 0.58, rng, mode="turbulence")
    micro = common.periodic_fbm(u, v, 24, 4, 0.52, rng)

    # Acid-etch physical step relief
    h = common.robust_normalize(
        0.64 * nickel_bands
        + 0.12 * sub_lam
        + 0.12 * grind_polish * nickel_bands
        - 0.12 * etch_frost * (1.0 - nickel_bands)
        + 0.06 * micro,
        low_pct=0.2,
        high_pct=99.8,
    )

    color_field = common.robust_normalize(
        0.72 * nickel_bands
        + 0.10 * sub_lam
        + 0.10 * grind_polish * (0.35 + 0.65 * nickel_bands)
        + 0.08 * micro
        - 0.08 * etch_frost * (1.0 - nickel_bands),
        low_pct=0.3,
        high_pct=99.7,
    )
    albedo = common.apply_palette_5stop(color_field, micro, stops, contrast=1.22)

    # Roughness: shiny polished nickel bands low (0.12-0.24), acid-etched carbon steel troughs matte (0.42-0.64)
    rough_raw = (
        0.68 * (1.0 - nickel_bands)
        + 0.16 * etch_frost * (1.0 - nickel_bands)
        + 0.10 * (1.0 - grind_polish) * nickel_bands
        + 0.06 * (1.0 - sub_lam)
    )
    rough = common.match_histogram_range(rough_raw, 0.12, 0.64, target_mean=0.34, contrast=1.24)

    base = common.shade_pbr_surface(
        albedo,
        h,
        roughness=rough,
        normal_strength=3.4,
        spec_intensity=0.40,
        ao_strength=0.18,
        cavity_strength=0.20,
    )
    return base, h, rough


_BUILDERS = (
    _build_brushed_steel,
    _build_polished_chrome,
    _build_hammered_iron_plate,
    _build_riveted_steel_plate,
    _build_chainmail_classic,
    _build_chainmail_darkened,
    _build_antique_brass,
    _build_oxidized_copper,
    _build_woven_metal_mesh,
    _build_damascus_steel,
)


def _compute_bundle(
    seed: int = 1,
    palette: PaletteArg = None,
    size: int = 512,
    variant: int = 0,
) -> tuple[Image.Image, np.ndarray, np.ndarray, MetalStyleSpec]:
    style = METAL_STYLES[int(variant) % len(METAL_STYLES)]
    rng = np.random.default_rng(int(seed))
    u, v = common.make_uv_grid(int(size))
    stops = _resolve_local_palette(palette, style.default_palette, rng)
    base, h, rough = _BUILDERS[style.variant](u, v, int(size), rng, stops)
    return base, np.asarray(h, dtype=np.float32), np.asarray(rough, dtype=np.float32), style


def render_metal_bundle(
    style_or_variant: str | int,
    seed: int = 1,
    size: int = 2048,
    palette: PaletteArg = None,
) -> tuple[Image.Image, Image.Image, Image.Image, np.ndarray, np.ndarray, MetalStyleSpec]:
    """Single-pass PBR bundle renderer returning (base, normal, roughness, height, rough_field, style)."""
    if isinstance(style_or_variant, str):
        style = STYLE_BY_SLUG[style_or_variant]
        variant = style.variant
    else:
        variant = int(style_or_variant) % len(METAL_STYLES)
    base, h, rough, style = _compute_bundle(seed=seed, palette=palette, size=size, variant=variant)
    normal = common.height_to_normal(h, strength=style.normal_strength)
    rough_img = common.roughness_image(rough)
    return base, normal, rough_img, h, rough, style


def generate(seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0) -> Image.Image:
    return _compute_bundle(seed=seed, palette=palette, size=size, variant=variant)[0]


def generate_height(seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0) -> np.ndarray:
    return _compute_bundle(seed=seed, palette=palette, size=size, variant=variant)[1]


def generate_normal(seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0) -> Image.Image:
    _, h, _, style = _compute_bundle(seed=seed, palette=palette, size=size, variant=variant)
    return common.height_to_normal(h, strength=style.normal_strength)


def generate_roughness(seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0) -> np.ndarray:
    return _compute_bundle(seed=seed, palette=palette, size=size, variant=variant)[2]


SPEC = GeneratorSpec(
    name="metal-pack",
    kind="texture",
    tileable=True,
    origin="tools/generators/types/metal_pack.py",
    default_palette="brushed_steel",
    description=(
        "Metals & Armor procedural texture suite (10 variants: brushed-steel, "
        "polished-chrome, hammered-iron-plate, riveted-steel-plate, chainmail-classic, "
        "chainmail-darkened, antique-brass, oxidized-copper, woven-metal-mesh, and "
        "damascus-steel) with self-contained palettes and calibrated PBR normal & roughness maps."
    ),
    render_fn=generate,
    normal_source="native",
    normal_rule="Native per-style metal & armor relief height field (h) converted with calibrated Scharr normal strength.",
    normal_strength=4.5,
    height_fn=generate_height,
    normal_fn=generate_normal,
    roughness_fn=generate_roughness,
    default_roughness=0.38,
)
