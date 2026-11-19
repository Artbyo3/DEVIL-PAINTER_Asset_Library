"""Procedural Wood & Bamboo generator suite (10 tileable PBR materials).

Built strictly on the committed periodic math primitives in ``tools/generators/common.py``
with all material palettes defined locally inside this module:
  0. walnut-polished     - Dark heartwood with flowing cathedral grain, fine vessels, glossy polish.
  1. oak-natural         - Warm ring-porous timber with open earlywood vessels and medullary rays.
  2. dark-ebony          - Dense straight-grained dark heartwood with tight fine rings and satin sheen.
  3. cherry-satin        - Warm reddish-amber flowing cathedral grain with fine diffuse pores and satin luster.
  4. teak-oiled          - Golden-brown heartwood with dark mineral streaks and hand-rubbed oiled finish.
  5. birch-pale          - Creamy pale fine-grained timber with gentle annual bands and subtle flecks.
  6. bamboo-slats        - Segmented vertical culm slats with staggered raised nodes, scars, and vascular fibers.
  7. bamboo-woven-mat    - Interlaced 2/2 twill split-bamboo strips with directional strip grain and woven relief.
  8. weathered-driftwood - Sun-bleached silver-grey eroded timber with proud latewood ridges and shrinkage checks.
  9. cork-natural        - Multi-scale cellular compressed cork granule aggregate with porous lenticel cavities.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from PIL import Image

from .. import common
from ..interface import GeneratorSpec, PaletteArg


@dataclass(frozen=True)
class WoodStyleSpec:
    variant: int
    slug: str
    normal_strength: float
    rough_range: tuple[float, float]
    description: str


WOOD_PALETTES: dict[str, list[tuple[int, int, int]]] = {
    "walnut-polished": [
        (22, 13, 8),
        (52, 31, 18),
        (88, 54, 33),
        (126, 83, 52),
        (164, 117, 78),
    ],
    "oak-natural": [
        (66, 46, 28),
        (118, 88, 56),
        (162, 127, 86),
        (196, 162, 118),
        (222, 193, 150),
    ],
    "dark-ebony": [
        (13, 11, 10),
        (25, 21, 19),
        (43, 35, 30),
        (66, 52, 43),
        (94, 74, 58),
    ],
    "cherry-satin": [
        (54, 19, 12),
        (102, 40, 24),
        (146, 66, 40),
        (184, 98, 60),
        (214, 136, 92),
    ],
    "teak-oiled": [
        (34, 21, 11),
        (80, 48, 23),
        (132, 84, 40),
        (174, 118, 60),
        (208, 152, 86),
    ],
    "birch-pale": [
        (144, 118, 90),
        (182, 158, 126),
        (212, 192, 160),
        (232, 217, 190),
        (246, 237, 218),
    ],
    "bamboo-slats": [
        (48, 33, 16),
        (116, 84, 42),
        (176, 140, 78),
        (212, 180, 114),
        (236, 212, 154),
    ],
    "bamboo-woven-mat": [
        (36, 23, 11),
        (96, 66, 32),
        (158, 118, 62),
        (200, 164, 98),
        (228, 198, 138),
    ],
    "weathered-driftwood": [
        (54, 49, 44),
        (98, 92, 85),
        (142, 136, 127),
        (182, 177, 168),
        (215, 211, 203),
    ],
    "cork-natural": [
        (46, 28, 16),
        (106, 70, 42),
        (156, 112, 72),
        (192, 148, 102),
        (218, 180, 134),
    ],
}


WOOD_STYLES: tuple[WoodStyleSpec, ...] = (
    WoodStyleSpec(
        0,
        "walnut-polished",
        2.2,
        (0.14, 0.34),
        "Dark chocolate heartwood with flowing domain-warped cathedral growth rings, fine vessel pores, and smooth polished finish.",
    ),
    WoodStyleSpec(
        1,
        "oak-natural",
        3.6,
        (0.48, 0.76),
        "Warm honey-tan ring-porous hardwood with pronounced earlywood vessel pores, crisp latewood bands, and pale medullary ray flecks.",
    ),
    WoodStyleSpec(
        2,
        "dark-ebony",
        2.0,
        (0.20, 0.42),
        "Dense dark charcoal-brown heartwood with tight straight growth-ring striping, fine micro-pores, and silky satin surface.",
    ),
    WoodStyleSpec(
        3,
        "cherry-satin",
        2.4,
        (0.26, 0.48),
        "Warm reddish-amber hardwood with smooth cathedral growth arches, fine diffuse pores, subtle pitch flecks, and satin sheen.",
    ),
    WoodStyleSpec(
        4,
        "teak-oiled",
        3.0,
        (0.32, 0.58),
        "Golden-brown heartwood with wavy straight grain, dark mineral oil streaks, medium open pores, and hand-rubbed oiled luster.",
    ),
    WoodStyleSpec(
        5,
        "birch-pale",
        2.3,
        (0.42, 0.66),
        "Pale creamy-blonde fine-grained hardwood with soft low-contrast growth bands, delicate flecks, and smooth matte-satin surface.",
    ),
    WoodStyleSpec(
        6,
        "bamboo-slats",
        4.8,
        (0.26, 0.76),
        "Segmented vertical bamboo slats with beveled seams, staggered raised culm nodes, dark sheath scars, and fine vertical vascular fibers.",
    ),
    WoodStyleSpec(
        7,
        "bamboo-woven-mat",
        4.8,
        (0.28, 0.90),
        "Interlaced 2/2 twill split-bamboo woven mat with beveled flat-convex strips, directional strip fibers, and shadowed weave cavities.",
    ),
    WoodStyleSpec(
        8,
        "weathered-driftwood",
        5.4,
        (0.68, 0.94),
        "Sun-bleached silver-grey weathered timber with deeply eroded earlywood channels, raised latewood ridges, and longitudinal checks.",
    ),
    WoodStyleSpec(
        9,
        "cork-natural",
        5.2,
        (0.64, 0.92),
        "Natural cellular cork aggregate of compressed irregular polygonal granules with per-granule tone variation and dark lenticel pores.",
    ),
)

STYLE_BY_SLUG: dict[str, WoodStyleSpec] = {s.slug: s for s in WOOD_STYLES}


def _resolve_wood_palette(
    palette: PaletteArg,
    style_slug: str,
    rng: np.random.Generator,
) -> list[tuple[int, int, int]]:
    if palette is None:
        return list(WOOD_PALETTES[style_slug])
    if isinstance(palette, str) and palette in WOOD_PALETTES:
        return list(WOOD_PALETTES[palette])
    return common.resolve_palette(palette, "burled_black_walnut", rng)


def _smoothstep(edge0: float, edge1: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - float(edge0)) / max(1e-6, float(edge1 - edge0)), 0.0, 1.0)
    return (t * t * (3.0 - 2.0 * t)).astype(np.float32)


def _interp_stops(field: np.ndarray, stops: list[tuple[int, int, int]]) -> np.ndarray:
    f = np.clip(field, 0.0, 1.0).astype(np.float32)
    n_stops = len(stops)
    pos = np.linspace(0.0, 1.0, n_stops, dtype=np.float32)
    out = np.empty((*f.shape, 3), dtype=np.float32)
    for ch in range(3):
        vals = np.array([float(s[ch]) for s in stops], dtype=np.float32)
        out[..., ch] = np.interp(f, pos, vals)
    return out


def _grain_domain_warp(
    u: np.ndarray,
    v: np.ndarray,
    rng: np.random.Generator,
    amp_u: float = 0.055,
    amp_v: float = 0.018,
    macro_fx: int = 3,
    macro_fy: int = 2,
    med_fx: int = 7,
    med_fy: int = 3,
    med_ratio: float = 0.35,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Periodic anisotropic domain warp preserving consistent vertical grain direction."""
    wu_macro = common.periodic_fbm(
        u, v, macro_fx, 3, 0.50, rng, mode="fbm", freqy=macro_fy, rotate_octaves=False
    )
    wu_med = common.periodic_fbm(
        u, v, med_fx, 3, 0.52, rng, mode="fbm", freqy=med_fy, rotate_octaves=False
    )
    wv_macro = common.periodic_fbm(
        u, v, max(2, macro_fx), 3, 0.50, rng, mode="fbm", freqy=max(2, macro_fy), rotate_octaves=False
    )

    du = ((1.0 - med_ratio) * (wu_macro - 0.5) + med_ratio * (wu_med - 0.5)) * (2.0 * float(amp_u))
    dv = (wv_macro - 0.5) * (2.0 * float(amp_v))
    uw = (u + du) % 1.0
    vw = (v + dv) % 1.0
    return uw.astype(np.float32), vw.astype(np.float32), du.astype(np.float32), dv.astype(np.float32)


def _growth_ring_profile(
    u: np.ndarray,
    v: np.ndarray,
    du: np.ndarray,
    dv: np.ndarray,
    rng: np.random.Generator,
    linear_rings: int = 8,
    cathedral_rings: float = 10.0,
    arch_cols: int = 2,
    arch_rows: int = 1,
    arch_v_scale: float = 0.24,
    ring_wobble: float = 1.6,
    late_cut: float = 0.74,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Strictly periodic sawn-timber annual growth rings combining straight & cathedral arches.

    Returns:
        (rings_light, earlywood_band, latewood_band)
        - rings_light: high in pale earlywood, low in dense dark latewood.
        - earlywood_band: peaks cleanly inside the porous earlywood transition zone.
        - latewood_band: peaks cleanly inside the dense latewood band.
    """
    pu = float(rng.uniform(0.0, 1.0))
    pv = float(rng.uniform(0.0, 1.0))

    u_eff = u + du
    v_eff = v + dv

    # Periodic log-pith radius proxy: strictly periodic on T^2 for integer arch_cols, arch_rows.
    sin_u = np.sin(math.pi * float(max(1, arch_cols)) * (u_eff + pu))
    sin_v = np.sin(math.pi * float(max(1, arch_rows)) * (v_eff + pv))
    r_cathedral = np.sqrt(sin_u * sin_u + float(arch_v_scale) * (sin_v * sin_v) + 0.035)

    # Fine periodic wobble breaks geometric regularity while preserving seam continuity.
    wobble = common.periodic_fbm(
        u_eff % 1.0,
        v_eff % 1.0,
        freqx=6,
        octaves=4,
        persistence=0.52,
        rng=rng,
        mode="fbm",
        freqy=2,
        rotate_octaves=False,
    )
    phase = (
        float(int(linear_rings)) * u_eff
        + float(cathedral_rings) * r_cathedral
        + float(ring_wobble) * (wobble - 0.5)
    )
    frac = (phase % 1.0).astype(np.float32)

    # Asymmetric earlywood -> abrupt latewood density transition.
    lc = float(np.clip(late_cut, 0.50, 0.90))
    t_rise = np.clip(frac / lc, 0.0, 1.0)
    t_drop = np.clip((1.0 - frac) / (1.0 - lc), 0.0, 1.0)
    ew_curve = t_rise * t_rise * (3.0 - 2.0 * t_rise)
    lw_drop = t_drop * t_drop * (3.0 - 2.0 * t_drop)
    rings_base = np.where(frac < lc, 1.0 - 0.78 * ew_curve, 0.22 + 0.78 * (1.0 - lw_drop))

    # Secondary sub-ring harmonic gives realistic internal density banding inside wide rings.
    sub_band = 0.5 + 0.5 * np.cos(2.0 * math.pi * (2.0 * frac + 0.15))
    rings_light = np.clip(0.86 * rings_base + 0.14 * sub_band, 0.0, 1.0).astype(np.float32)

    # Earlywood porous band concentrates right after the latewood boundary (frac in ~[0.02, 0.36]).
    earlywood_band = (
        _smoothstep(0.0, 0.08, frac) * (1.0 - _smoothstep(0.24, 0.56, frac))
    ).astype(np.float32)
    latewood_band = (1.0 - rings_light).astype(np.float32)
    return rings_light, earlywood_band, latewood_band


def _directional_fibers(
    uw: np.ndarray,
    vw: np.ndarray,
    rng: np.random.Generator,
    coarse_fx: int = 32,
    coarse_fy: int = 3,
    fine_fx: int = 128,
    fine_fy: int = 9,
) -> tuple[np.ndarray, np.ndarray]:
    """Multi-scale vertical directional wood fiber streak fields."""
    coarse = common.periodic_fbm(
        uw, vw, coarse_fx, 4, 0.56, rng, mode="fbm", freqy=coarse_fy, rotate_octaves=False
    )
    fine = common.periodic_fbm(
        uw, vw, fine_fx, 3, 0.58, rng, mode="fbm", freqy=fine_fy, rotate_octaves=False
    )
    return coarse, fine


def _vessel_pores(
    uw: np.ndarray,
    vw: np.ndarray,
    rng: np.random.Generator,
    fx: int = 160,
    fy: int = 18,
    threshold: float = 0.67,
    gain: float = 4.8,
) -> np.ndarray:
    """Elongated vertical wood vessel pores aligned cleanly with the warped fiber flow."""
    p1 = common.periodic_fbm(
        uw, vw, fx, 3, 0.58, rng, mode="fbm", freqy=fy, rotate_octaves=False
    )
    p2 = common.periodic_gradient_perlin(
        uw,
        vw,
        int(round(fx * 1.35)),
        max(4, int(round(fy * 1.45))),
        rng,
        phase_u=float(rng.uniform()),
        phase_v=float(rng.uniform()),
    )
    combined = 0.62 * p1 + 0.38 * p2
    return np.clip((combined - float(threshold)) * float(gain), 0.0, 1.0).astype(np.float32)


def _heartwood_bands(
    uw: np.ndarray,
    vw: np.ndarray,
    rng: np.random.Generator,
    fx: int = 5,
    fy: int = 2,
) -> np.ndarray:
    """Broad longitudinal heartwood / mineral color variation along grain direction."""
    b1 = common.periodic_fbm(
        uw, vw, fx, 4, 0.52, rng, mode="fbm", freqy=fy, rotate_octaves=False
    )
    b2 = common.periodic_fbm(
        uw, vw, fx * 2 + 1, 3, 0.50, rng, mode="fbm", freqy=max(2, fy), rotate_octaves=False
    )
    return common.robust_normalize(0.68 * b1 + 0.32 * b2, low_pct=0.4, high_pct=99.6, soft=True)


# ---------------------------------------------------------------------------
# 0. walnut-polished
# ---------------------------------------------------------------------------
def _build_walnut_polished(
    u: np.ndarray,
    v: np.ndarray,
    size: int,
    rng: np.random.Generator,
    stops: list[tuple[int, int, int]],
) -> tuple[Image.Image, np.ndarray, np.ndarray]:
    _ = size
    uw, vw, du, dv = _grain_domain_warp(
        u, v, rng, amp_u=0.072, amp_v=0.024, macro_fx=3, macro_fy=2, med_fx=6, med_fy=3, med_ratio=0.38
    )
    rings, ew_band, lw_band = _growth_ring_profile(
        u,
        v,
        du,
        dv,
        rng,
        linear_rings=4,
        cathedral_rings=11.5,
        arch_cols=2,
        arch_rows=1,
        arch_v_scale=0.22,
        ring_wobble=1.75,
        late_cut=0.73,
    )
    streaks, fine_fibers = _directional_fibers(uw, vw, rng, coarse_fx=36, coarse_fy=3, fine_fx=148, fine_fy=10)
    heartwood = _heartwood_bands(uw, vw, rng, fx=4, fy=2)

    # Semi-ring-porous dark walnut vessel pores, subtly more frequent in earlywood.
    pores_raw = _vessel_pores(uw, vw, rng, fx=155, fy=16, threshold=0.68, gain=5.2)
    pores = (pores_raw * (0.45 + 0.55 * ew_band)).astype(np.float32)

    # Smooth lacquered polished height: subtle grain undulation + crisp shallow pore dips.
    h_raw = (
        0.46 * rings
        + 0.24 * streaks
        + 0.18 * fine_fibers
        + 0.12 * heartwood
        - 0.34 * pores
    )
    h = common.robust_normalize(h_raw, low_pct=0.2, high_pct=99.8, soft=True)

    # Rich dark chocolate to warm golden-brown tonal flow with dark latewood & pores.
    color_field = common.robust_normalize(
        0.46 * rings
        + 0.26 * heartwood
        + 0.18 * streaks
        + 0.10 * fine_fibers
        - 0.28 * pores
        - 0.08 * lw_band * (1.0 - heartwood),
        low_pct=0.3,
        high_pct=99.7,
        soft=True,
    )
    albedo = _interp_stops(color_field, stops)

    # High-gloss to satin-polished lacquer: smooth over solid wood, slightly rougher inside open pores.
    rough_raw = 0.22 * (1.0 - rings) + 0.20 * (1.0 - fine_fibers) + 0.58 * pores
    rough = common.match_histogram_range(rough_raw, 0.14, 0.34, target_mean=0.22, contrast=1.15)
    base = common.shade_pbr_surface(
        albedo, h, roughness=rough, normal_strength=2.2, spec_intensity=0.22, ao_strength=0.14, cavity_strength=0.14
    )
    return base, h, rough


# ---------------------------------------------------------------------------
# 1. oak-natural
# ---------------------------------------------------------------------------
def _build_oak_natural(
    u: np.ndarray,
    v: np.ndarray,
    size: int,
    rng: np.random.Generator,
    stops: list[tuple[int, int, int]],
) -> tuple[Image.Image, np.ndarray, np.ndarray]:
    _ = size
    uw, vw, du, dv = _grain_domain_warp(
        u, v, rng, amp_u=0.052, amp_v=0.018, macro_fx=3, macro_fy=2, med_fx=7, med_fy=3, med_ratio=0.32
    )
    rings, ew_band, lw_band = _growth_ring_profile(
        u,
        v,
        du,
        dv,
        rng,
        linear_rings=6,
        cathedral_rings=10.5,
        arch_cols=2,
        arch_rows=1,
        arch_v_scale=0.26,
        ring_wobble=1.45,
        late_cut=0.76,
    )
    streaks, fine_fibers = _directional_fibers(uw, vw, rng, coarse_fx=42, coarse_fy=4, fine_fx=164, fine_fy=11)
    heartwood = _heartwood_bands(uw, vw, rng, fx=5, fy=2)

    # Strong ring-porous open vessels concentrated in earlywood bands.
    pores_coarse = _vessel_pores(uw, vw, rng, fx=132, fy=14, threshold=0.61, gain=5.6)
    pores_fine = _vessel_pores(uw, vw, rng, fx=210, fy=22, threshold=0.66, gain=5.0)
    vessels = np.clip(
        pores_coarse * (0.20 + 0.80 * ew_band) + 0.45 * pores_fine * (0.35 + 0.65 * ew_band),
        0.0,
        1.0,
    ).astype(np.float32)

    # Medullary ray flecks: pale cross-grain horizontal shimmer ribbons characteristic of oak.
    ray_noise = common.periodic_fbm(
        u, v, freqx=10, octaves=3, persistence=0.52, rng=rng, mode="fbm", freqy=88, rotate_octaves=False
    )
    ray_mask_macro = common.periodic_fbm(
        u, v, freqx=4, octaves=2, persistence=0.50, rng=rng, mode="fbm", freqy=4, rotate_octaves=False
    )
    medullary_rays = np.clip((ray_noise - 0.68) * 4.8, 0.0, 1.0) * _smoothstep(0.32, 0.72, ray_mask_macro)
    medullary_rays = medullary_rays.astype(np.float32)

    # Natural unvarnished tactile relief: proud dense latewood, recessed porous earlywood vessels.
    h_raw = (
        0.38 * (1.0 - rings)
        + 0.24 * streaks
        + 0.16 * fine_fibers
        + 0.08 * medullary_rays
        - 0.56 * vessels
    )
    h = common.robust_normalize(h_raw, low_pct=0.2, high_pct=99.8, soft=True)

    color_field = common.robust_normalize(
        0.48 * rings
        + 0.22 * streaks
        + 0.14 * heartwood
        + 0.10 * fine_fibers
        + 0.16 * medullary_rays
        - 0.42 * vessels
        - 0.06 * lw_band,
        low_pct=0.3,
        high_pct=99.7,
        soft=True,
    )
    albedo = _interp_stops(color_field, stops)

    # Matte-satin natural raw oak: slightly smoother on medullary rays & latewood, rougher inside pores.
    rough_raw = (
        0.26 * rings
        + 0.22 * (1.0 - streaks)
        + 0.52 * vessels
        - 0.18 * medullary_rays
    )
    rough = common.match_histogram_range(rough_raw, 0.48, 0.76, target_mean=0.61, contrast=1.10)
    base = common.shade_pbr_surface(
        albedo, h, roughness=rough, normal_strength=3.6, spec_intensity=0.10, ao_strength=0.22, cavity_strength=0.22
    )
    return base, h, rough


# ---------------------------------------------------------------------------
# 2. dark-ebony
# ---------------------------------------------------------------------------
def _build_dark_ebony(
    u: np.ndarray,
    v: np.ndarray,
    size: int,
    rng: np.random.Generator,
    stops: list[tuple[int, int, int]],
) -> tuple[Image.Image, np.ndarray, np.ndarray]:
    _ = size
    # Very dense, straight-grained hardwood with gentle longitudinal undulation.
    uw, vw, du, dv = _grain_domain_warp(
        u, v, rng, amp_u=0.026, amp_v=0.010, macro_fx=3, macro_fy=2, med_fx=8, med_fy=3, med_ratio=0.30
    )
    rings, _, lw_band = _growth_ring_profile(
        u,
        v,
        du,
        dv,
        rng,
        linear_rings=20,
        cathedral_rings=3.2,
        arch_cols=1,
        arch_rows=1,
        arch_v_scale=0.18,
        ring_wobble=1.15,
        late_cut=0.68,
    )
    streaks, fine_fibers = _directional_fibers(uw, vw, rng, coarse_fx=56, coarse_fy=4, fine_fx=210, fine_fy=14)
    mineral_ribbons = _heartwood_bands(uw, vw, rng, fx=8, fy=2)

    # Very fine, sparse closed micro-pores typical of dense dark heartwood.
    pores = _vessel_pores(uw, vw, rng, fx=230, fy=26, threshold=0.73, gain=5.0)

    h_raw = (
        0.34 * rings
        + 0.30 * streaks
        + 0.24 * fine_fibers
        + 0.12 * mineral_ribbons
        - 0.26 * pores
    )
    h = common.robust_normalize(h_raw, low_pct=0.2, high_pct=99.8, soft=True)

    # Subtle smoky espresso-brown striping over deep charcoal-black body.
    stripe_mod = _smoothstep(0.32, 0.84, mineral_ribbons)
    color_field = common.robust_normalize(
        0.42 * rings * (0.45 + 0.55 * stripe_mod)
        + 0.28 * mineral_ribbons
        + 0.18 * streaks
        + 0.12 * fine_fibers
        - 0.24 * pores
        - 0.10 * lw_band,
        low_pct=0.3,
        high_pct=99.7,
        soft=True,
    )
    albedo = _interp_stops(color_field, stops)

    # Dense fine-sanded silky satin sheen.
    rough_raw = 0.26 * (1.0 - fine_fibers) + 0.24 * (1.0 - streaks) + 0.50 * pores
    rough = common.match_histogram_range(rough_raw, 0.20, 0.42, target_mean=0.29, contrast=1.12)
    base = common.shade_pbr_surface(
        albedo, h, roughness=rough, normal_strength=2.0, spec_intensity=0.18, ao_strength=0.14, cavity_strength=0.14
    )
    return base, h, rough


# ---------------------------------------------------------------------------
# 3. cherry-satin
# ---------------------------------------------------------------------------
def _build_cherry_satin(
    u: np.ndarray,
    v: np.ndarray,
    size: int,
    rng: np.random.Generator,
    stops: list[tuple[int, int, int]],
) -> tuple[Image.Image, np.ndarray, np.ndarray]:
    _ = size
    uw, vw, du, dv = _grain_domain_warp(
        u, v, rng, amp_u=0.064, amp_v=0.022, macro_fx=3, macro_fy=2, med_fx=6, med_fy=3, med_ratio=0.36
    )
    rings, ew_band, _ = _growth_ring_profile(
        u,
        v,
        du,
        dv,
        rng,
        linear_rings=5,
        cathedral_rings=10.0,
        arch_cols=2,
        arch_rows=1,
        arch_v_scale=0.23,
        ring_wobble=1.55,
        late_cut=0.75,
    )
    streaks, fine_fibers = _directional_fibers(uw, vw, rng, coarse_fx=38, coarse_fy=3, fine_fx=160, fine_fy=11)
    heartwood = _heartwood_bands(uw, vw, rng, fx=4, fy=2)

    # Fine diffuse-porous vessels plus characteristic tiny dark gum/pitch flecks along the grain.
    fine_pores = _vessel_pores(uw, vw, rng, fx=195, fy=22, threshold=0.71, gain=4.8)
    pitch_flecks = _vessel_pores(uw, vw, rng, fx=74, fy=16, threshold=0.77, gain=7.2)

    h_raw = (
        0.44 * rings
        + 0.26 * streaks
        + 0.20 * fine_fibers
        + 0.10 * heartwood
        - 0.24 * fine_pores
        - 0.30 * pitch_flecks
    )
    h = common.robust_normalize(h_raw, low_pct=0.2, high_pct=99.8, soft=True)

    # Warm reddish-amber satin glow with gentle contrast and scattered dark pitch flecks.
    color_field = common.robust_normalize(
        0.44 * rings
        + 0.28 * heartwood
        + 0.18 * streaks
        + 0.10 * fine_fibers
        - 0.16 * fine_pores * (0.5 + 0.5 * ew_band)
        - 0.46 * pitch_flecks,
        low_pct=0.3,
        high_pct=99.7,
        soft=True,
    )
    albedo = _interp_stops(color_field, stops)

    rough_raw = 0.24 * (1.0 - rings) + 0.24 * (1.0 - fine_fibers) + 0.34 * fine_pores + 0.30 * pitch_flecks
    rough = common.match_histogram_range(rough_raw, 0.26, 0.48, target_mean=0.35, contrast=1.08)
    base = common.shade_pbr_surface(
        albedo, h, roughness=rough, normal_strength=2.4, spec_intensity=0.16, ao_strength=0.15, cavity_strength=0.15
    )
    return base, h, rough


# ---------------------------------------------------------------------------
# 4. teak-oiled
# ---------------------------------------------------------------------------
def _build_teak_oiled(
    u: np.ndarray,
    v: np.ndarray,
    size: int,
    rng: np.random.Generator,
    stops: list[tuple[int, int, int]],
) -> tuple[Image.Image, np.ndarray, np.ndarray]:
    _ = size
    # Straight to gently wavy grain with rich dark mineral/oleoresin longitudinal ribbons.
    uw, vw, du, dv = _grain_domain_warp(
        u, v, rng, amp_u=0.040, amp_v=0.014, macro_fx=3, macro_fy=2, med_fx=7, med_fy=3, med_ratio=0.34
    )
    rings, ew_band, _ = _growth_ring_profile(
        u,
        v,
        du,
        dv,
        rng,
        linear_rings=13,
        cathedral_rings=5.2,
        arch_cols=1,
        arch_rows=1,
        arch_v_scale=0.20,
        ring_wobble=1.35,
        late_cut=0.72,
    )
    streaks, fine_fibers = _directional_fibers(uw, vw, rng, coarse_fx=46, coarse_fy=3, fine_fx=168, fine_fy=11)

    # Characteristic dark mineral/oil streaks running along the vertical grain.
    mineral_raw = common.periodic_fbm(
        uw, vw, freqx=14, octaves=4, persistence=0.56, rng=rng, mode="ridged", freqy=2, rotate_octaves=False
    )
    mineral_streaks = _smoothstep(0.56, 0.90, mineral_raw)
    heartwood = _heartwood_bands(uw, vw, rng, fx=5, fy=2)

    # Medium-open longitudinal oily pores.
    pores = (_vessel_pores(uw, vw, rng, fx=145, fy=15, threshold=0.64, gain=5.2) * (0.40 + 0.60 * ew_band)).astype(
        np.float32
    )

    h_raw = (
        0.38 * rings
        + 0.26 * streaks
        + 0.20 * fine_fibers
        - 0.44 * pores
        - 0.08 * mineral_streaks
    )
    h = common.robust_normalize(h_raw, low_pct=0.2, high_pct=99.8, soft=True)

    color_field = common.robust_normalize(
        0.40 * rings
        + 0.24 * heartwood
        + 0.18 * streaks
        + 0.10 * fine_fibers
        - 0.36 * mineral_streaks
        - 0.28 * pores,
        low_pct=0.3,
        high_pct=99.7,
        soft=True,
    )
    albedo = _interp_stops(color_field, stops)

    # Hand-rubbed oil finish: warm satin sheen on dense grain, slightly glossier on oil-rich mineral bands, rougher in open pores.
    rough_raw = (
        0.24 * (1.0 - rings)
        + 0.22 * (1.0 - fine_fibers)
        + 0.54 * pores
        - 0.14 * mineral_streaks
    )
    rough = common.match_histogram_range(rough_raw, 0.32, 0.58, target_mean=0.43, contrast=1.10)
    base = common.shade_pbr_surface(
        albedo, h, roughness=rough, normal_strength=3.0, spec_intensity=0.14, ao_strength=0.18, cavity_strength=0.18
    )
    return base, h, rough


# ---------------------------------------------------------------------------
# 5. birch-pale
# ---------------------------------------------------------------------------
def _build_birch_pale(
    u: np.ndarray,
    v: np.ndarray,
    size: int,
    rng: np.random.Generator,
    stops: list[tuple[int, int, int]],
) -> tuple[Image.Image, np.ndarray, np.ndarray]:
    _ = size
    uw, vw, du, dv = _grain_domain_warp(
        u, v, rng, amp_u=0.048, amp_v=0.016, macro_fx=3, macro_fy=2, med_fx=7, med_fy=3, med_ratio=0.30
    )
    rings, _, _ = _growth_ring_profile(
        u,
        v,
        du,
        dv,
        rng,
        linear_rings=9,
        cathedral_rings=7.5,
        arch_cols=2,
        arch_rows=1,
        arch_v_scale=0.25,
        ring_wobble=1.30,
        late_cut=0.78,
    )
    streaks, fine_fibers = _directional_fibers(uw, vw, rng, coarse_fx=48, coarse_fy=4, fine_fx=185, fine_fy=12)
    heartwood = _heartwood_bands(uw, vw, rng, fx=4, fy=2)

    # Very fine diffuse pores + delicate sparse brownish pith flecks characteristic of pale birch.
    fine_pores = _vessel_pores(uw, vw, rng, fx=215, fy=24, threshold=0.72, gain=4.5)
    flecks = _vessel_pores(uw, vw, rng, fx=64, fy=14, threshold=0.79, gain=7.0)

    # Soft, tight, fine-sanded surface relief.
    h_raw = (
        0.38 * rings
        + 0.28 * streaks
        + 0.24 * fine_fibers
        + 0.10 * heartwood
        - 0.22 * fine_pores
        - 0.18 * flecks
    )
    h = common.robust_normalize(h_raw, low_pct=0.2, high_pct=99.8, soft=True)

    # Soft pale creamy-blonde tonal field with gentle low-contrast bands and delicate flecks.
    color_field = common.robust_normalize(
        0.42 * rings
        + 0.28 * heartwood
        + 0.18 * streaks
        + 0.12 * fine_fibers
        - 0.14 * fine_pores
        - 0.34 * flecks,
        low_pct=0.4,
        high_pct=99.6,
        soft=True,
    )
    # Bias gently toward the pale blonde stops[1..4] while letting flecks dip to stops[0].
    color_toned = np.clip(0.18 + 0.82 * color_field - 0.22 * flecks, 0.0, 1.0).astype(np.float32)
    albedo = _interp_stops(color_toned, stops)

    rough_raw = 0.28 * (1.0 - fine_fibers) + 0.26 * (1.0 - rings) + 0.34 * fine_pores + 0.18 * flecks
    rough = common.match_histogram_range(rough_raw, 0.42, 0.66, target_mean=0.53, contrast=1.06)
    base = common.shade_pbr_surface(
        albedo, h, roughness=rough, normal_strength=2.3, spec_intensity=0.10, ao_strength=0.14, cavity_strength=0.14
    )
    return base, h, rough


# ---------------------------------------------------------------------------
# 6. bamboo-slats
# ---------------------------------------------------------------------------
def _build_bamboo_slats(
    u: np.ndarray,
    v: np.ndarray,
    size: int,
    rng: np.random.Generator,
    stops: list[tuple[int, int, int]],
) -> tuple[Image.Image, np.ndarray, np.ndarray]:
    _ = size
    n_slats = 7  # odd count avoids half-tile symmetry

    # Subtle periodic edge micro-wobble so slat joints feel natural, not sterile pixels.
    joint_wobble = common.periodic_fbm(
        u, v, freqx=14, octaves=2, persistence=0.5, rng=rng, mode="fbm", freqy=4, rotate_octaves=False
    )
    su = (u + (joint_wobble - 0.5) * 0.0022) * float(n_slats)
    slat_id = np.floor(su).astype(np.int32) % n_slats
    fu = (su - np.floor(su)).astype(np.float32)

    # Per-slat deterministic offsets and tone variations.
    slat_u_off = rng.uniform(0.0, 1.0, size=n_slats).astype(np.float32)
    slat_v_off = rng.uniform(0.0, 1.0, size=n_slats).astype(np.float32)
    slat_tone = rng.uniform(-0.10, 0.10, size=n_slats).astype(np.float32)
    # Two organically spaced culm nodes per slat height, staggered across adjacent slats.
    node_pos_a = rng.uniform(0.0, 1.0, size=n_slats).astype(np.float32)
    node_pos_b = (node_pos_a + rng.uniform(0.42, 0.58, size=n_slats)).astype(np.float32) % 1.0

    u_off = slat_u_off[slat_id]
    v_off = slat_v_off[slat_id]
    tone_shift = slat_tone[slat_id]

    # Slat cross-section: gently crowned culm face + crisp beveled V-groove seam at fu=0/1.
    edge_dist = np.minimum(fu, 1.0 - fu)
    seam_groove = _smoothstep(0.008, 0.055, edge_dist)
    slat_crown = 0.88 + 0.12 * np.sqrt(np.clip(1.0 - ((fu - 0.5) * 1.92) ** 2, 0.0, 1.0))
    slat_profile = (seam_groove * slat_crown).astype(np.float32)

    # Staggered horizontal culm nodes: double-lip raised ridge + thin recessed dark sheath scar.
    node_wobble = (
        common.periodic_fbm(
            (u + u_off) % 1.0,
            (v + v_off) % 1.0,
            freqx=18,
            octaves=2,
            persistence=0.5,
            rng=rng,
            mode="fbm",
            freqy=6,
            rotate_octaves=False,
        )
        - 0.5
    ) * 0.006

    def _toroidal_signed_dist(coord: np.ndarray, center: np.ndarray) -> np.ndarray:
        return (((coord - center + 0.5) % 1.0) - 0.5).astype(np.float32)

    da = _toroidal_signed_dist(v + node_wobble, node_pos_a[slat_id])
    db = _toroidal_signed_dist(v + node_wobble, node_pos_b[slat_id])
    d_node = np.where(np.abs(da) < np.abs(db), da, db)
    abs_d = np.abs(d_node)

    # Characteristic bamboo node anatomy: two raised shoulders flanking a sharp recessed ring scar.
    node_ridge = (
        np.exp(-((d_node - 0.0085) / 0.0062) ** 2)
        + 0.85 * np.exp(-((d_node + 0.0075) / 0.0058) ** 2)
    ).astype(np.float32)
    node_scar = np.exp(-((d_node) / 0.0024) ** 2).astype(np.float32)
    node_halo = np.exp(-(abs_d / 0.042) ** 2).astype(np.float32)

    # Slight longitudinal fiber flare right across the culm node.
    u_fib = ((u + u_off + 0.003 * node_halo * np.sin(fu * 2.0 * math.pi)) % 1.0).astype(np.float32)
    v_fib = ((v + v_off) % 1.0).astype(np.float32)

    # Straight bamboo parenchyma fibers + short dark vascular bundle flecks.
    fiber_base = common.periodic_fbm(
        u_fib, v_fib, freqx=110, octaves=4, persistence=0.58, rng=rng, mode="fbm", freqy=8, rotate_octaves=False
    )
    bundle_noise = common.periodic_fbm(
        u_fib, v_fib, freqx=175, octaves=3, persistence=0.56, rng=rng, mode="fbm", freqy=24, rotate_octaves=False
    )
    bundle_dash = np.clip((bundle_noise - 0.64) * 5.2, 0.0, 1.0).astype(np.float32)
    # Vascular bundles cluster denser near outer culm margins and around nodes.
    edge_density = 0.55 + 0.45 * np.abs(fu - 0.5) * 2.0
    vascular_flecks = np.clip(bundle_dash * (edge_density + 0.35 * node_halo), 0.0, 1.0).astype(np.float32)

    # Real 3D segmented height field: beveled slats + raised node ridges - recessed scars & seams.
    h_raw = (
        0.64 * slat_profile
        + 0.20 * node_ridge * seam_groove
        - 0.24 * node_scar * seam_groove
        + 0.06 * fiber_base * seam_groove
        - 0.05 * vascular_flecks * seam_groove
    )
    h = common.robust_normalize(h_raw, low_pct=0.15, high_pct=99.85, soft=True)

    # Golden-straw internodes, toasted caramel node halos, dark scars & seams, amber vascular flecks.
    color_raw = (
        0.66
        + tone_shift
        + 0.14 * (fiber_base - 0.5)
        - 0.24 * vascular_flecks
        - 0.16 * node_halo
        - 0.42 * node_scar
        - 0.58 * (1.0 - seam_groove)
        + 0.07 * node_ridge * (1.0 - node_scar)
    )
    color_field = np.clip(color_raw, 0.0, 1.0).astype(np.float32)
    albedo = _interp_stops(color_field, stops)

    # Waxy satin bamboo cuticle on faces, matte rougher inside seams, scars, and vascular flecks.
    rough_raw = (
        0.18 * (1.0 - fiber_base)
        + 0.22 * vascular_flecks
        + 0.26 * node_halo
        + 0.44 * node_scar
        + 0.62 * (1.0 - seam_groove)
    )
    rough = common.match_histogram_range(rough_raw, 0.26, 0.76, target_mean=0.39, contrast=1.18)
    base = common.shade_pbr_surface(
        albedo, h, roughness=rough, normal_strength=4.8, spec_intensity=0.18, ao_strength=0.26, cavity_strength=0.26
    )
    return base, h, rough


# ---------------------------------------------------------------------------
# 7. bamboo-woven-mat
# ---------------------------------------------------------------------------
def _build_bamboo_woven_mat(
    u: np.ndarray,
    v: np.ndarray,
    size: int,
    rng: np.random.Generator,
    stops: list[tuple[int, int, int]],
) -> tuple[Image.Image, np.ndarray, np.ndarray]:
    _ = size
    # 16x16 interlaced 2/2 diagonal twill split-bamboo strips (exact multiple of 4).
    n_u = 16
    n_v = 16

    # Gentle organic mat tension wave so weave isn't rigidly synthetic.
    mat_wu = common.periodic_fbm(u, v, freqx=3, octaves=3, persistence=0.5, rng=rng, mode="fbm")
    mat_wv = common.periodic_fbm(u, v, freqx=3, octaves=3, persistence=0.5, rng=rng, mode="fbm")
    uw = (u + (mat_wu - 0.5) * 0.008) % 1.0
    vw = (v + (mat_wv - 0.5) * 0.008) % 1.0

    tu = uw * float(n_u)
    tv = vw * float(n_v)
    iu = np.floor(tu).astype(np.int32) % n_u
    iv = np.floor(tv).astype(np.int32) % n_v
    fu = (tu - np.floor(tu)).astype(np.float32)
    fv = (tv - np.floor(tv)).astype(np.float32)

    # Per-strip tone, offset, and subtle width jitter.
    warp_tone = rng.uniform(-0.08, 0.08, size=n_u).astype(np.float32)[iu]
    weft_tone = rng.uniform(-0.08, 0.08, size=n_v).astype(np.float32)[iv]
    warp_v_off = rng.uniform(0.0, 1.0, size=n_u).astype(np.float32)[iu]
    weft_u_off = rng.uniform(0.0, 1.0, size=n_v).astype(np.float32)[iv]

    # Flat-convex beveled split-bamboo strip cross-sections with crisp ~9% inter-strip gaps.
    half_width = 0.455
    du_abs = np.abs(fu - 0.5)
    dv_abs = np.abs(fv - 0.5)

    # Sharp beveled edge drop + gently crowned split-bamboo top face.
    warp_mask = 1.0 - _smoothstep(half_width - 0.055, half_width + 0.012, du_abs)
    weft_mask = 1.0 - _smoothstep(half_width - 0.055, half_width + 0.012, dv_abs)
    warp_crown = 0.86 + 0.14 * np.sqrt(np.clip(1.0 - (du_abs / max(half_width, 1e-4)) ** 2, 0.0, 1.0))
    weft_crown = 0.86 + 0.14 * np.sqrt(np.clip(1.0 - (dv_abs / max(half_width, 1e-4)) ** 2, 0.0, 1.0))
    warp_xsec = (warp_mask * warp_crown).astype(np.float32)
    weft_xsec = (weft_mask * weft_crown).astype(np.float32)

    # Exact continuous 2/2 diagonal twill longitudinal float waves (zero sawtooth seams):
    # Warp column iu is OVER when (iu - iv) % 4 in {0, 1}; Weft row iv is OVER when {2, 3}.
    s_warp = (tv - (iu.astype(np.float32) - 1.0)) % 4.0
    s_weft = (tu - (iv.astype(np.float32) + 2.0)) % 4.0

    # Stiff flat-topped arch for bamboo ribbons via shaped cosine wave.
    k_stiff = 1.55
    inv_tanh = 1.0 / math.tanh(k_stiff)
    wave_warp = np.tanh(k_stiff * np.cos(0.5 * math.pi * (s_warp - 1.0))) * inv_tanh
    wave_weft = np.tanh(k_stiff * np.cos(0.5 * math.pi * (s_weft - 1.0))) * inv_tanh

    # Physical 3D elevation of warp and weft strips; max resolves over/under cleanly & reveals under-strip in gaps.
    z_warp = (warp_xsec * (0.56 + 0.36 * wave_warp)).astype(np.float32)
    z_weft = (weft_xsec * (0.56 + 0.36 * wave_weft)).astype(np.float32)

    h_weave = np.maximum(z_warp, z_weft)
    gap_mask = np.clip(np.maximum(warp_mask, weft_mask), 0.0, 1.0)

    # Smooth ownership blend (1 = warp strip visible on top, 0 = weft strip visible on top).
    warp_top = _smoothstep(-0.04, 0.04, z_warp - z_weft)

    # Directional bamboo fibers running ALONG each individual strip's length!
    # Vertical strips (warp) have vertical fibers (fx >> fy); horizontal strips (weft) have horizontal fibers (fy >> fx).
    warp_fib = common.periodic_fbm(
        uw, (vw + warp_v_off) % 1.0, freqx=140, octaves=3, persistence=0.58, rng=rng, mode="fbm", freqy=12, rotate_octaves=False
    )
    warp_fleck = np.clip(
        (common.periodic_fbm(uw, (vw + warp_v_off) % 1.0, freqx=200, octaves=2, persistence=0.56, rng=rng, mode="fbm", freqy=28, rotate_octaves=False) - 0.65) * 4.8,
        0.0,
        1.0,
    ).astype(np.float32)

    weft_fib = common.periodic_fbm(
        (uw + weft_u_off) % 1.0, vw, freqx=12, octaves=3, persistence=0.58, rng=rng, mode="fbm", freqy=140, rotate_octaves=False
    )
    weft_fleck = np.clip(
        (common.periodic_fbm((uw + weft_u_off) % 1.0, vw, freqx=28, octaves=2, persistence=0.56, rng=rng, mode="fbm", freqy=200, rotate_octaves=False) - 0.65) * 4.8,
        0.0,
        1.0,
    ).astype(np.float32)

    strip_fib = warp_top * warp_fib + (1.0 - warp_top) * weft_fib
    strip_fleck = warp_top * warp_fleck + (1.0 - warp_top) * weft_fleck
    strip_tone = warp_top * warp_tone + (1.0 - warp_top) * weft_tone

    # Subtle edge bevel wear/darkening along strip margins.
    strip_edge_shadow = warp_top * _smoothstep(half_width - 0.12, half_width, du_abs) + (
        1.0 - warp_top
    ) * _smoothstep(half_width - 0.12, half_width, dv_abs)

    # Combined height with real strip relief + fine directional bamboo fibers.
    h_raw = 0.88 * h_weave + 0.08 * strip_fib * gap_mask - 0.04 * strip_fleck * gap_mask
    h = common.robust_normalize(h_raw, low_pct=0.15, high_pct=99.85, soft=True)

    # Warm honey split-bamboo crowns, toasted vascular streaks, shadowed tuck-unders and deep weave gaps.
    under_ao = np.clip(0.32 + 0.68 * h_weave, 0.0, 1.0)
    color_raw = (
        (0.72 + strip_tone + 0.14 * (strip_fib - 0.5) - 0.22 * strip_fleck - 0.14 * strip_edge_shadow)
        * under_ao
        * (0.14 + 0.86 * gap_mask)
    )
    color_field = np.clip(color_raw, 0.0, 1.0).astype(np.float32)
    albedo = _interp_stops(color_field, stops)

    # Smooth waxy split-bamboo crowns vs rough beveled edges & deep matte weave crevices.
    rough_raw = (
        0.54 * (1.0 - h_weave)
        + 0.20 * strip_edge_shadow
        + 0.14 * strip_fleck
        + 0.12 * (1.0 - strip_fib)
        + 0.35 * (1.0 - gap_mask)
    )
    rough = common.match_histogram_range(rough_raw, 0.28, 0.90, target_mean=0.49, contrast=1.18)
    base = common.shade_pbr_surface(
        albedo, h, roughness=rough, normal_strength=4.8, spec_intensity=0.16, ao_strength=0.30, cavity_strength=0.30
    )
    return base, h, rough


# ---------------------------------------------------------------------------
# 8. weathered-driftwood
# ---------------------------------------------------------------------------
def _build_weathered_driftwood(
    u: np.ndarray,
    v: np.ndarray,
    size: int,
    rng: np.random.Generator,
    stops: list[tuple[int, int, int]],
) -> tuple[Image.Image, np.ndarray, np.ndarray]:
    _ = size
    # Stronger organic grain twist & weathered flow along consistent vertical direction.
    uw, vw, du, dv = _grain_domain_warp(
        u, v, rng, amp_u=0.078, amp_v=0.028, macro_fx=3, macro_fy=2, med_fx=7, med_fy=3, med_ratio=0.40
    )
    rings, ew_band, lw_band = _growth_ring_profile(
        u,
        v,
        du,
        dv,
        rng,
        linear_rings=9,
        cathedral_rings=8.5,
        arch_cols=2,
        arch_rows=1,
        arch_v_scale=0.22,
        ring_wobble=2.1,
        late_cut=0.70,
    )
    streaks, fine_fibers = _directional_fibers(uw, vw, rng, coarse_fx=40, coarse_fy=3, fine_fx=160, fine_fy=10)

    # Deep wind/water erosion grooves carving softer earlywood channels while dense latewood ridges stand proud.
    erosion_ridge = common.periodic_fbm(
        uw, vw, freqx=26, octaves=4, persistence=0.58, rng=rng, mode="ridged", freqy=3, rotate_octaves=False
    )
    splinter_fib = common.periodic_fbm(
        uw, vw, freqx=96, octaves=4, persistence=0.60, rng=rng, mode="turbulence", freqy=6, rotate_octaves=False
    )

    # Sharp longitudinal desiccation checks / shrinkage splits running along the warped grain.
    check_field_a = common.periodic_fbm(
        uw, vw, freqx=14, octaves=3, persistence=0.55, rng=rng, mode="turbulence", freqy=2, rotate_octaves=False
    )
    check_field_b = common.periodic_fbm(
        uw, vw, freqx=28, octaves=3, persistence=0.54, rng=rng, mode="turbulence", freqy=3, rotate_octaves=False
    )
    check_mask = common.periodic_fbm(
        uw, vw, freqx=4, octaves=2, persistence=0.50, rng=rng, mode="fbm", freqy=3, rotate_octaves=False
    )
    checks_major = (1.0 - _smoothstep(0.015, 0.085, check_field_a)) * _smoothstep(0.30, 0.72, check_mask)
    checks_minor = (1.0 - _smoothstep(0.012, 0.068, check_field_b)) * 0.55
    checks = np.clip(checks_major + checks_minor, 0.0, 1.0).astype(np.float32)

    # Strong sculptural relief: proud dense latewood ridges, deeply eroded earlywood & sharp shrinkage splits.
    h_raw = (
        0.44 * lw_band
        + 0.24 * erosion_ridge
        + 0.18 * streaks
        + 0.14 * fine_fibers
        - 0.22 * ew_band * (1.0 - splinter_fib)
        - 0.62 * checks
    )
    h = common.robust_normalize(h_raw, low_pct=0.2, high_pct=99.8, soft=True)

    # Sun-bleached silver-grey on proud weathered ridges, darker grey-taupe inside eroded valleys & checks.
    bleach_patch = common.periodic_fbm(
        uw, vw, freqx=4, octaves=3, persistence=0.52, rng=rng, mode="fbm", freqy=2, rotate_octaves=False
    )
    color_field = common.robust_normalize(
        0.42 * h
        + 0.22 * bleach_patch
        + 0.18 * erosion_ridge
        + 0.14 * fine_fibers
        - 0.52 * checks,
        low_pct=0.3,
        high_pct=99.7,
        soft=True,
    )
    albedo = _interp_stops(color_field, stops)

    # Dry, fibrous, powdery salt-weathered matte roughness across the entire surface.
    rough_raw = (
        0.34 * (1.0 - h)
        + 0.26 * (1.0 - fine_fibers)
        + 0.24 * splinter_fib
        + 0.36 * checks
    )
    rough = common.match_histogram_range(rough_raw, 0.68, 0.94, target_mean=0.82, contrast=1.10)
    base = common.shade_pbr_surface(
        albedo, h, roughness=rough, normal_strength=5.4, spec_intensity=0.05, ao_strength=0.30, cavity_strength=0.32
    )
    return base, h, rough


# ---------------------------------------------------------------------------
# 9. cork-natural
# ---------------------------------------------------------------------------
def _build_cork_natural(
    u: np.ndarray,
    v: np.ndarray,
    size: int,
    rng: np.random.Generator,
    stops: list[tuple[int, int, int]],
) -> tuple[Image.Image, np.ndarray, np.ndarray]:
    _ = size
    # Domain-warp coordinates slightly so compressed cork granules have irregular organic edges.
    wx = common.periodic_fbm(u, v, freqx=10, octaves=3, persistence=0.55, rng=rng, mode="fbm")
    wy = common.periodic_fbm(u, v, freqx=10, octaves=3, persistence=0.55, rng=rng, mode="fbm")
    uw = (u + (wx - 0.5) * 0.022) % 1.0
    vw = (v + (wy - 0.5) * 0.022) % 1.0

    # Multi-scale cellular Lloyd Voronoi aggregate: large compressed cork chips + finer interstitial granules.
    f1_a, edge_a, cid_a, rx_a, ry_a = common.periodic_lloyd_voronoi(
        uw, vw, grid_cells=18, jitter=0.92, rng=rng, lloyd_iters=2
    )
    f1_b, edge_b, cid_b, _, _ = common.periodic_lloyd_voronoi(
        (uw + 0.37) % 1.0, (vw + 0.61) % 1.0, grid_cells=34, jitter=0.95, rng=rng, lloyd_iters=1
    )

    # Blend coarse and fine granules organically so granule sizes vary naturally across the sheet.
    size_mix = _smoothstep(0.36, 0.68, common.periodic_fbm(u, v, freqx=6, octaves=3, persistence=0.52, rng=rng, mode="fbm"))
    edge_a_gap = 1.0 - _smoothstep(0.015, 0.095, edge_a)
    edge_b_gap = 1.0 - _smoothstep(0.018, 0.105, edge_b)

    use_fine = (size_mix > 0.52) | (edge_a < 0.075)
    granule_id = np.where(use_fine, cid_b, cid_a).astype(np.float32)
    granule_seam = np.where(use_fine, np.maximum(edge_a_gap * 0.85, edge_b_gap), edge_a_gap).astype(np.float32)
    granule_f1 = np.where(use_fine, f1_b, f1_a).astype(np.float32)

    # Per-granule internal cellular orientation & spongy micro-texture.
    angle = (granule_id * 2.0 * math.pi).astype(np.float32)
    internal_dir = rx_a * np.cos(angle) + ry_a * np.sin(angle)
    sponge_mid = common.periodic_fbm(u, v, freqx=48, octaves=4, persistence=0.60, rng=rng, mode="billow")
    sponge_fine = common.periodic_fbm(u, v, freqx=150, octaves=3, persistence=0.62, rng=rng, mode="fbm")

    # Dark lenticel pits & binder voids along granule junctions and scattered across cork cells.
    lenticel_raw = common.periodic_fbm(uw, vw, freqx=28, octaves=4, persistence=0.58, rng=rng, mode="turbulence")
    lenticel_pits = (1.0 - _smoothstep(0.04, 0.19, lenticel_raw)).astype(np.float32)
    junction_voids = np.clip(granule_seam * (0.45 + 0.55 * (1.0 - sponge_mid)), 0.0, 1.0).astype(np.float32)
    cavities = np.clip(np.maximum(junction_voids, lenticel_pits * 0.92), 0.0, 1.0).astype(np.float32)

    # Per-granule slight elevation step (pressed chips don't sit at identical micro-height).
    granule_elev = 0.55 + 0.45 * np.sin(granule_id * 17.3 + 0.4) * 0.5 + 0.14 * np.clip(internal_dir, -1.0, 1.0)
    granule_crown = np.clip(1.0 - 0.25 * granule_f1, 0.0, 1.0)

    h_raw = (
        0.38 * granule_crown
        + 0.18 * granule_elev
        + 0.18 * sponge_mid
        + 0.14 * sponge_fine
        - 0.58 * cavities
    )
    h = common.robust_normalize(h_raw, low_pct=0.2, high_pct=99.8, soft=True)

    # Distinct per-granule warm tan/ochre/sienna variation + dark porous lenticel & seam cavities.
    # Quantized-like non-linear mapping gives crisp warm/dark granule contrast.
    chip_tone = np.clip(
        0.24 + 0.68 * (0.5 + 0.5 * np.sin(granule_id * 29.7)),
        0.15,
        0.96,
    ).astype(np.float32)
    macro_warmth = common.periodic_fbm(u, v, freqx=4, octaves=3, persistence=0.50, rng=rng, mode="fbm")
    color_raw = (
        0.54 * chip_tone
        + 0.16 * macro_warmth
        + 0.16 * sponge_mid
        + 0.14 * sponge_fine
        - 0.64 * cavities
    )
    color_field = common.robust_normalize(color_raw, low_pct=0.3, high_pct=99.7, soft=True)
    albedo = _interp_stops(color_field, stops)

    # Warm matte cork roughness: flattened granule crowns slightly smoother, spongy cavities very rough.
    rough_raw = (
        0.46 * cavities
        + 0.24 * (1.0 - sponge_fine)
        + 0.18 * (1.0 - sponge_mid)
        + 0.12 * (1.0 - chip_tone)
    )
    rough = common.match_histogram_range(rough_raw, 0.64, 0.92, target_mean=0.78, contrast=1.12)
    base = common.shade_pbr_surface(
        albedo, h, roughness=rough, normal_strength=5.2, spec_intensity=0.06, ao_strength=0.28, cavity_strength=0.30
    )
    return base, h, rough


_BUILDERS = (
    _build_walnut_polished,
    _build_oak_natural,
    _build_dark_ebony,
    _build_cherry_satin,
    _build_teak_oiled,
    _build_birch_pale,
    _build_bamboo_slats,
    _build_bamboo_woven_mat,
    _build_weathered_driftwood,
    _build_cork_natural,
)


def _compute_bundle(
    seed: int = 1,
    palette: PaletteArg = None,
    size: int = 512,
    variant: int = 0,
) -> tuple[Image.Image, np.ndarray, np.ndarray, WoodStyleSpec]:
    style = WOOD_STYLES[int(variant) % len(WOOD_STYLES)]
    rng = np.random.default_rng(int(seed))
    u, v = common.make_uv_grid(int(size))
    stops = _resolve_wood_palette(palette, style.slug, rng)
    base, h, rough = _BUILDERS[style.variant](u, v, int(size), rng, stops)
    return base, np.asarray(h, dtype=np.float32), np.asarray(rough, dtype=np.float32), style


def render_wood_bundle(
    style_or_variant: str | int,
    seed: int = 1,
    size: int = 2048,
    palette: PaletteArg = None,
) -> tuple[Image.Image, Image.Image, Image.Image, np.ndarray, np.ndarray, WoodStyleSpec]:
    """Single-pass PBR bundle renderer returning (base, normal, roughness, height, rough_field, style)."""
    if isinstance(style_or_variant, str):
        style = STYLE_BY_SLUG[style_or_variant]
        variant = style.variant
    else:
        variant = int(style_or_variant) % len(WOOD_STYLES)
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
    name="wood-pack",
    kind="texture",
    tileable=True,
    origin="tools/generators/types/wood_pack.py",
    default_palette="walnut-polished",
    description=(
        "Wood and bamboo procedural suite (10 tileable materials: walnut-polished, "
        "oak-natural, dark-ebony, cherry-satin, teak-oiled, birch-pale, bamboo-slats, "
        "bamboo-woven-mat, weathered-driftwood, cork-natural) with calibrated PBR height, "
        "OpenGL normal, and roughness."
    ),
    render_fn=generate,
    normal_source="native",
    normal_rule="Native per-material height field (h) converted with calibrated Scharr normal strength.",
    normal_strength=3.6,
    height_fn=generate_height,
    normal_fn=generate_normal,
    roughness_fn=generate_roughness,
    default_roughness=0.48,
)
