"""Procedural apparel leather & suede generator suite (12 physically grounded styles).

Built on the upgraded periodic math primitives in ``tools/generators/common.py``:
  - Domain-warped R2-seeded free toroidal Lloyd CVT dermal papillae with exact IQ
    perpendicular bisector valleys (no square-grid harmonics, uniform crease width).
  - Nyquist-aware Gaussian-integer rotated fBm & analytic domain warping for natural
    hide growth ripples, supple flex creases, and crisp sub-8px micro-grain at 2048px.
  - Junction-localized hair-follicle pores and micro-collagen stipple.
  - Multi-orientation sparse Gabor noise for brushed split suede & velvety nubuck nap.
  - Calibrated per-style PBR roughness maps via ``match_histogram_range`` and exact
    per-style Scharr normal strengths.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from PIL import Image

from .. import common
from ..interface import GeneratorSpec, PaletteArg


@dataclass(frozen=True)
class LeatherStyleSpec:
    variant: int
    slug: str
    label: str
    default_palette: str
    normal_strength: float
    rough_range: tuple[float, float]
    description: str


LEATHER_STYLES: tuple[LeatherStyleSpec, ...] = (
    LeatherStyleSpec(
        0,
        "leather-full-grain-smooth-espresso",
        "Full-Grain Smooth Espresso",
        "espresso_fullgrain_leather",
        3.6,
        (0.30, 0.54),
        "Tight calfskin Nappa with fine dermal grain, supple flex creases, and pinpoint follicle pores.",
    ),
    LeatherStyleSpec(
        1,
        "leather-pebbled-grain-cognac",
        "Pebbled Grain Cognac",
        "cognac_pebbled_leather",
        5.6,
        (0.24, 0.68),
        "Tumbled full-grain cowhide with plump domain-warped papillae, burnished crowns, and matte valleys.",
    ),
    LeatherStyleSpec(
        2,
        "leather-suede-camel",
        "Brushed Suede Camel",
        "camel_brushed_suede",
        2.5,
        (0.80, 0.96),
        "Soft napped split suede with multi-angle fibrous Gabor micro-nap and directional writing sheen.",
    ),
    LeatherStyleSpec(
        3,
        "leather-nubuck-charcoal",
        "Velvet Nubuck Charcoal",
        "charcoal_velvet_nubuck",
        2.4,
        (0.68, 0.91),
        "Fine-sanded top-grain nubuck combining powdery micro-nap, faint ghost pores, and burnished rub marks.",
    ),
    LeatherStyleSpec(
        4,
        "leather-patent-obsidian",
        "Gloss Patent Obsidian",
        "obsidian_patent_leather",
        3.4,
        (0.05, 0.20),
        "Mirror-lacquered dress leather with clearcoat-damped hide grain, gentle flex billows, and micro orange-peel.",
    ),
    LeatherStyleSpec(
        5,
        "leather-cracked-vintage-chestnut",
        "Cracked Vintage Chestnut",
        "vintage_cracked_chestnut",
        6.4,
        (0.26, 0.90),
        "Distressed heirloom hide with curl-warped craze fissures exposing dry lighter suede corium.",
    ),
    LeatherStyleSpec(
        6,
        "leather-perforated-jet-black",
        "Perforated Moto Jet-Black",
        "perforated_moto_black",
        6.2,
        (0.30, 0.92),
        "Full-grain moto calfskin punched with staggered beveled micro-perforations and dark matte bores.",
    ),
    LeatherStyleSpec(
        7,
        "leather-woven-strips-mahogany",
        "Woven Strips Mahogany",
        "mahogany_woven_leather",
        6.8,
        (0.26, 0.78),
        "Intrecciato flat leather basketweave with skived beveled edges, per-strip hide grain, and tuck shadows.",
    ),
    LeatherStyleSpec(
        8,
        "leather-tooled-embossed-saddle",
        "Tooled Embossed Saddle",
        "saddle_tooled_tan",
        7.0,
        (0.28, 0.80),
        "Vegetable-tanned saddle leather with beveled floral-diamond tooling, antique grooves, and stippled background.",
    ),
    LeatherStyleSpec(
        9,
        "leather-matte-aniline-oxblood",
        "Matte Aniline Oxblood",
        "oxblood_aniline_leather",
        4.4,
        (0.42, 0.70),
        "Naked drum-dyed aniline cowhide with organic dye clouding, growth-mark ripples, and open pores.",
    ),
    LeatherStyleSpec(
        10,
        "leather-pullup-oiled-olive",
        "Oiled Pull-Up Olive",
        "olive_pullup_leather",
        5.2,
        (0.26, 0.78),
        "Waxed utility pull-up hide with stress-lightened flex ridges, micro-scuffs, and waxy satin flats.",
    ),
    LeatherStyleSpec(
        11,
        "leather-croc-embossed-noir",
        "Embossed Croc Noir",
        "noir_croc_embossed",
        6.4,
        (0.16, 0.72),
        "Fashion embossed crocodilian flank scales with glossy convex domes, fine striations, and matte seams.",
    ),
)

STYLE_BY_SLUG: dict[str, LeatherStyleSpec] = {s.slug: s for s in LEATHER_STYLES}


def _warp_uv(
    u: np.ndarray,
    v: np.ndarray,
    freq: int,
    amp: float,
    rng: np.random.Generator,
    octaves: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Smooth periodic UV domain warp keeping exact unit-torus periodicity."""
    du = common.periodic_fbm(u, v, freq, octaves, 0.52, rng, mode="fbm") - 0.5
    dv = common.periodic_fbm(u, v, freq, octaves, 0.52, rng, mode="fbm") - 0.5
    return (u + du * float(amp)) % 1.0, (v + dv * float(amp)) % 1.0


def _dermal_grain(
    u: np.ndarray,
    v: np.ndarray,
    cells: int,
    warp_amp: float,
    lloyd_iters: int,
    valley_sharpness: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Domain-warped R2 Lloyd CVT dermal papillae (crown, valley, cell_id, edge)."""
    uw, vw = _warp_uv(u, v, max(4, cells // 3), warp_amp, rng, octaves=3)
    f1, edge, cid, _, _ = common.periodic_lloyd_voronoi(
        uw, vw, cells, 0.88, rng, lloyd_iters=lloyd_iters
    )
    # Softly domed cushion crown (flatter top than a cone, crisp curved descent into valley)
    edge_ramp = 1.0 - np.exp(-edge * float(valley_sharpness))
    crown = np.power(np.clip(edge_ramp, 0.0, 1.0), 0.62) * (
        0.74 + 0.26 * np.sqrt(np.clip(1.0 - f1 * 0.68, 0.0, 1.0))
    )
    valley = np.exp(-np.power(edge * (float(valley_sharpness) * 0.72), 1.55))
    return (
        crown.astype(np.float32),
        valley.astype(np.float32),
        cid.astype(np.float32),
        edge.astype(np.float32),
    )


def _warp_field(
    u: np.ndarray,
    v: np.ndarray,
    freq: int,
    octaves: int,
    persistence: float,
    rng: np.random.Generator,
    warp_strength: float = 0.08,
    stages: int = 1,
) -> np.ndarray:
    field, _, _ = common.periodic_domain_warp_fbm(
        u, v, freq, octaves, persistence, rng, warp_strength=warp_strength, stages=stages
    )
    return field


def _follicle_pores(
    u: np.ndarray,
    v: np.ndarray,
    edge: np.ndarray,
    freq: int,
    rng: np.random.Generator,
    junction_bias: float = 0.72,
    threshold: float = 0.71,
) -> np.ndarray:
    """Organic pinpoint hair-follicle pores clustered near dermal valley junctions."""
    # Two rotated high-frequency periodic Perlin fields multiplied form sharp, circular,
    # non-grid pinpoint peaks without the cost of a second high-N Voronoi pass.
    p1 = common.periodic_rotated_perlin(
        u, v, freq, 0.46, rng, phase_u=float(rng.uniform()), phase_v=float(rng.uniform())
    )
    p2 = common.periodic_rotated_perlin(
        u, v, int(round(freq * 1.28)), 1.18, rng, phase_u=float(rng.uniform()), phase_v=float(rng.uniform())
    )
    pin = np.sqrt(np.clip(p1 * p2, 0.0, 1.0))
    # Modulate local pore density with broad natural hide patchiness
    cluster = common.periodic_fbm(u, v, 7, 3, 0.55, rng, mode="fbm")
    junc = np.exp(-((edge / 0.16) ** 2))
    gate = (1.0 - junction_bias) + junction_bias * junc
    raw = np.clip((pin - threshold + 0.08 * (cluster - 0.5)) / (1.0 - threshold), 0.0, 1.0)
    pores = np.power(raw, 2.1) * gate
    return np.clip(pores * 1.45, 0.0, 1.0).astype(np.float32)


def _hide_creases(
    u: np.ndarray,
    v: np.ndarray,
    macro_freq: int,
    rng: np.random.Generator,
    warp_amp: float = 0.055,
    anisotropy: float = 1.6,
) -> tuple[np.ndarray, np.ndarray]:
    """Meandering supple leather flex creases and broad growth-mark undulations.

    Returns ``(crease_troughs, macro_undulation)`` in ``[0, 1]`` where ``crease_troughs``
    is high inside sharp sunken fold lines and near zero on flat hide.
    """
    uw, vw = _warp_uv(u, v, max(2, macro_freq), warp_amp, rng, octaves=3)
    fy = max(2, int(round(macro_freq * anisotropy)))
    ridged_main = common.periodic_fbm(
        uw, vw, macro_freq, 4, 0.52, rng, freqy=fy, mode="ridged"
    )
    ridged_fine = common.periodic_fbm(
        uw, vw, macro_freq * 2, 4, 0.50, rng, freqy=fy * 2, mode="ridged"
    )
    # Sharpen ridge crests into narrow natural fold creases and gate with macro tension zones
    tension = common.periodic_fbm(u, v, 3, 3, 0.55, rng, mode="fbm")
    main_line = np.power(np.clip((ridged_main - 0.52) / 0.48, 0.0, 1.0), 1.85)
    fine_line = np.power(np.clip((ridged_fine - 0.58) / 0.42, 0.0, 1.0), 2.1)
    creases = np.clip((0.72 * main_line + 0.28 * fine_line) * (0.45 + 0.85 * tension), 0.0, 1.0)

    # Gentle organic hide growth ripples (slightly elongated along one axis)
    growth = common.periodic_fbm(u, v, 3, 3, 0.50, rng, freqy=max(4, int(round(3 * anisotropy))))
    return creases.astype(np.float32), growth.astype(np.float32)


def _micro_collagen(u: np.ndarray, v: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """High-frequency skin micro-stipple resolving cleanly up to 2048px Nyquist."""
    return common.periodic_fbm(u, v, 42, 4, 0.55, rng, mode="fbm")


def _blend_stops(
    field: np.ndarray,
    micro: np.ndarray | None,
    stops: list[tuple[int, int, int]],
    contrast: float = 1.08,
    gamma: float = 1.0,
) -> np.ndarray:
    return common.apply_palette_5stop(field, micro, stops, contrast=contrast, gamma=gamma)


def _build_full_grain_smooth(
    u: np.ndarray, v: np.ndarray, size: int, rng: np.random.Generator, stops: list[tuple[int, int, int]]
) -> tuple[Image.Image, np.ndarray, np.ndarray]:
    crown, valley, cid, edge = _dermal_grain(u, v, cells=38, warp_amp=0.008, lloyd_iters=1, valley_sharpness=4.6, rng=rng)
    pores = _follicle_pores(u, v, edge, freq=74, rng=rng, junction_bias=0.70, threshold=0.70)
    creases, growth = _hide_creases(u, v, macro_freq=4, rng=rng, warp_amp=0.055, anisotropy=1.55)
    micro = _micro_collagen(u, v, rng)
    dye_mottle = _warp_field(u, v, 3, 4, 0.52, rng, warp_strength=0.07)

    # Smooth calfskin has refined shallow dermal valleys, crisp supple creases, fine pores
    h = common.robust_normalize(
        0.38 * crown * (0.88 + 0.12 * cid)
        + 0.24 * growth
        + 0.16 * micro
        - 0.22 * creases
        - 0.14 * pores,
        low_pct=0.25,
        high_pct=99.75,
        soft=True,
    )
    # Subtle aniline tone variation, slightly darker in creases & pores, warmer burnish on crowns
    c = common.robust_normalize(
        0.42 * dye_mottle + 0.34 * crown + 0.14 * micro - 0.18 * creases - 0.16 * pores - 0.10 * valley,
        low_pct=0.3,
        high_pct=99.7,
        soft=True,
    )
    sheen_patch = common.periodic_fbm(u, v, 3, 3, 0.50, rng)
    rough_raw = (
        0.40 * valley
        + 0.34 * creases
        + 0.20 * pores
        + 0.18 * (1.0 - sheen_patch)
        + 0.10 * micro
    )
    rough = common.match_histogram_range(rough_raw, 0.30, 0.54, target_mean=0.40, contrast=1.10)
    albedo = _blend_stops(c, micro, stops, contrast=1.10, gamma=1.02)
    base = common.shade_pbr_surface(
        albedo, h, roughness=rough, normal_strength=3.6, spec_intensity=0.14, ao_strength=0.20, cavity_strength=0.16
    )
    return base, h, rough


def _build_pebbled_grain(
    u: np.ndarray, v: np.ndarray, size: int, rng: np.random.Generator, stops: list[tuple[int, int, int]]
) -> tuple[Image.Image, np.ndarray, np.ndarray]:
    crown, valley, cid, edge = _dermal_grain(u, v, cells=26, warp_amp=0.012, lloyd_iters=2, valley_sharpness=6.2, rng=rng)
    pores = _follicle_pores(u, v, edge, freq=62, rng=rng, junction_bias=0.80, threshold=0.69)
    creases, growth = _hide_creases(u, v, macro_freq=4, rng=rng, warp_amp=0.06, anisotropy=1.45)
    micro = _micro_collagen(u, v, rng)
    patina = _warp_field(u, v, 3, 4, 0.55, rng, warp_strength=0.06)

    # Plump tumbled pebble domes with varied cell elevation and tucked creases
    pebble_dome = crown * (0.78 + 0.22 * cid)
    h = common.robust_normalize(
        0.58 * pebble_dome
        + 0.14 * growth
        + 0.14 * micro
        - 0.20 * creases
        - 0.16 * pores
        - 0.10 * valley,
        low_pct=0.25,
        high_pct=99.75,
        soft=True,
    )
    # Burnished warm cognac pebble tops contrasting against deeply darkened valleys
    c = common.robust_normalize(
        0.48 * pebble_dome + 0.24 * patina + 0.14 * cid + 0.10 * micro - 0.22 * valley - 0.16 * creases - 0.14 * pores,
        low_pct=0.3,
        high_pct=99.7,
        soft=True,
    )
    rough_raw = 0.55 * valley + 0.24 * creases + 0.22 * pores + 0.16 * (1.0 - patina) + 0.10 * (1.0 - micro)
    rough = common.match_histogram_range(rough_raw, 0.24, 0.68, target_mean=0.43, contrast=1.12)
    albedo = _blend_stops(c, micro, stops, contrast=1.14, gamma=0.98)
    base = common.shade_pbr_surface(
        albedo, h, roughness=rough, normal_strength=5.6, spec_intensity=0.16, ao_strength=0.26, cavity_strength=0.22
    )
    return base, h, rough


def _build_suede(
    u: np.ndarray, v: np.ndarray, size: int, rng: np.random.Generator, stops: list[tuple[int, int, int]]
) -> tuple[Image.Image, np.ndarray, np.ndarray]:
    # Multi-orientation sparse Gabor fibers + Nyquist-resolved high-freq tangled nap
    uw, vw = _warp_uv(u, v, 5, 0.025, rng, octaves=3)
    nap_a = common.periodic_gabor_noise(
        uw, vw, grid_cells=26, f0=2.4, omega0=0.45, bandwidth=1.35, rng=rng, isotropic=False, impulses_per_cell=3
    )
    nap_b = common.periodic_gabor_noise(
        uw, vw, grid_cells=30, f0=2.6, omega0=1.85, bandwidth=1.45, rng=rng, isotropic=False, impulses_per_cell=3
    )
    nap_iso = common.periodic_gabor_noise(
        u, v, grid_cells=34, f0=2.0, omega0=0.0, bandwidth=1.6, rng=rng, isotropic=True, impulses_per_cell=2
    )
    fine_fibers = common.periodic_fbm(u, v, 32, 5, 0.62, rng, mode="fbm")
    micro_fuzz = common.periodic_fbm(u, v, 96, 3, 0.65, rng, mode="turbulence")

    # Macro "writing" / directional finger-brush sheen patches characteristic of real suede
    brush_dir = _warp_field(u, v, 4, 4, 0.54, rng, warp_strength=0.10, stages=2)
    soft_fold = common.periodic_fbm(u, v, 4, 3, 0.48, rng, freqy=6)

    # Blend the two brushed directions locally according to the writing-sheen patch field
    fibers = brush_dir * nap_a + (1.0 - brush_dir) * nap_b
    h = common.robust_normalize(
        0.36 * fibers + 0.26 * nap_iso + 0.28 * fine_fibers + 0.18 * micro_fuzz + 0.08 * soft_fold,
        low_pct=0.3,
        high_pct=99.7,
        soft=True,
    )
    # Real suede albedo balances velvety fine fiber grain with soft directional writing sheen
    c = common.robust_normalize(
        0.28 * brush_dir + 0.34 * fibers + 0.24 * fine_fibers + 0.12 * nap_iso + 0.08 * soft_fold - 0.12 * micro_fuzz,
        low_pct=0.35,
        high_pct=99.65,
        soft=True,
    )
    rough_raw = 0.40 * (1.0 - brush_dir) + 0.34 * micro_fuzz + 0.26 * (1.0 - fibers)
    rough = common.match_histogram_range(rough_raw, 0.80, 0.96, target_mean=0.88, contrast=1.08)
    albedo = _blend_stops(c, micro_fuzz, stops, contrast=1.06, gamma=0.96)
    base = common.shade_pbr_surface(
        albedo, h, roughness=rough, normal_strength=2.5, spec_intensity=0.03, ao_strength=0.14, cavity_strength=0.12
    )
    return base, h, rough


def _build_nubuck(
    u: np.ndarray, v: np.ndarray, size: int, rng: np.random.Generator, stops: list[tuple[int, int, int]]
) -> tuple[Image.Image, np.ndarray, np.ndarray]:
    # Sanded top-grain: ultra-fine velvety micro-nap + faint ghost of original tight grain & pores
    crown, valley, _, edge = _dermal_grain(u, v, cells=40, warp_amp=0.008, lloyd_iters=1, valley_sharpness=4.0, rng=rng)
    ghost_pores = _follicle_pores(u, v, edge, freq=78, rng=rng, junction_bias=0.65, threshold=0.73)
    creases, growth = _hide_creases(u, v, macro_freq=4, rng=rng, warp_amp=0.05, anisotropy=1.5)

    velvet_nap = common.periodic_gabor_noise(
        u, v, grid_cells=38, f0=2.2, omega0=0.0, bandwidth=1.7, rng=rng, isotropic=True, impulses_per_cell=3
    )
    powder_micro = common.periodic_fbm(u, v, 54, 4, 0.62, rng, mode="fbm")
    # Rubbed / burnished clothing wear patches (slightly darker & smoother than untouched powdery nap)
    rub_patches = _warp_field(u, v, 4, 4, 0.55, rng, warp_strength=0.09)
    rub_mask = np.clip((rub_patches - 0.56) * 2.4, 0.0, 1.0)

    # Sanding flattens the top-grain domes heavily, leaving silky micro-nap + ghost creases
    h = common.robust_normalize(
        0.34 * velvet_nap
        + 0.30 * powder_micro
        + 0.16 * crown
        + 0.12 * growth
        - 0.14 * creases
        - 0.10 * ghost_pores,
        low_pct=0.3,
        high_pct=99.7,
        soft=True,
    )
    c = common.robust_normalize(
        0.34 * (1.0 - rub_mask * 0.65)
        + 0.28 * velvet_nap
        + 0.24 * powder_micro
        + 0.12 * growth
        - 0.12 * creases
        - 0.10 * ghost_pores,
        low_pct=0.35,
        high_pct=99.65,
        soft=True,
    )
    rough_raw = 0.48 * (1.0 - rub_mask) + 0.22 * valley + 0.20 * powder_micro + 0.16 * ghost_pores
    rough = common.match_histogram_range(rough_raw, 0.68, 0.91, target_mean=0.81, contrast=1.10)
    albedo = _blend_stops(c, powder_micro, stops, contrast=1.06, gamma=1.0)
    base = common.shade_pbr_surface(
        albedo, h, roughness=rough, normal_strength=2.4, spec_intensity=0.05, ao_strength=0.16, cavity_strength=0.12
    )
    return base, h, rough


def _build_patent(
    u: np.ndarray, v: np.ndarray, size: int, rng: np.random.Generator, stops: list[tuple[int, int, int]]
) -> tuple[Image.Image, np.ndarray, np.ndarray]:
    # Thick glossy polyurethane clearcoat fills pores, leaving damped hide undulations,
    # supple flex billows, microscopic lacquer orange-peel, and subtle hairline flex creases.
    res_scale = size / 512.0
    crown, valley, _, _ = _dermal_grain(u, v, cells=34, warp_amp=0.008, lloyd_iters=1, valley_sharpness=4.2, rng=rng)
    # Smooth underlying grain with isotropic FFT Gaussian to simulate clearcoat leveling
    damped_grain = common.periodic_gaussian_blur(crown, sigma=max(1.0, 1.4 * res_scale))
    flex_billows = _warp_field(u, v, 4, 3, 0.50, rng, warp_strength=0.06)
    flex_billows = common.periodic_gaussian_blur(flex_billows, sigma=max(1.4, 2.0 * res_scale))
    creases, _ = _hide_creases(u, v, macro_freq=4, rng=rng, warp_amp=0.055, anisotropy=1.7)
    hairline_flex = common.periodic_gaussian_blur(creases, sigma=max(0.6, 0.8 * res_scale))
    orange_peel = common.periodic_fbm(u, v, 28, 3, 0.48, rng, mode="fbm")

    h = common.robust_normalize(
        0.46 * flex_billows + 0.28 * damped_grain + 0.18 * orange_peel - 0.22 * hairline_flex,
        low_pct=0.3,
        high_pct=99.7,
        soft=True,
    )
    # Deep jet-black lacquer albedo with subtle sheen depth across flex billows
    c = common.robust_normalize(
        0.46 * flex_billows + 0.28 * damped_grain + 0.16 * orange_peel - 0.20 * hairline_flex,
        low_pct=0.35,
        high_pct=99.65,
        soft=True,
    )
    micro_scuff = common.periodic_fbm(u, v, 16, 4, 0.55, rng, mode="turbulence")
    rough_raw = 0.48 * hairline_flex + 0.22 * valley * 0.40 + 0.20 * orange_peel + 0.22 * micro_scuff
    rough = common.match_histogram_range(rough_raw, 0.05, 0.20, target_mean=0.10, contrast=1.12)
    albedo = _blend_stops(c, orange_peel, stops, contrast=1.10, gamma=1.12)
    base = common.shade_pbr_surface(
        albedo, h, roughness=rough, normal_strength=3.4, spec_intensity=0.26, ao_strength=0.16, cavity_strength=0.10
    )
    return base, h, rough


def _build_cracked_vintage(
    u: np.ndarray, v: np.ndarray, size: int, rng: np.random.Generator, stops: list[tuple[int, int, int]]
) -> tuple[Image.Image, np.ndarray, np.ndarray]:
    # Aged distressed leather: underlying grain + curl-advected craze crack network exposing
    # dry lighter tan raw corium inside fissures + oiled burnished islands.
    crown, valley, cid, edge_g = _dermal_grain(u, v, cells=28, warp_amp=0.010, lloyd_iters=1, valley_sharpness=5.2, rng=rng)
    pores = _follicle_pores(u, v, edge_g, freq=64, rng=rng, junction_bias=0.72, threshold=0.71)
    micro = _micro_collagen(u, v, rng)

    # Curl-advected Voronoi perpendicular bisector distance creates realistic meandering craze fissures
    vx, vy = common.periodic_curl_noise(u, v, 4, 3, 0.55, rng)
    uc = (u + vx * 0.032) % 1.0
    vc = (v + vy * 0.032) % 1.0
    _, crack_e1, crack_id, _, _ = common.periodic_voronoi(uc, vc, 11, 0.92, rng)
    _, crack_e2, _, _, _ = common.periodic_voronoi((uc + 0.41) % 1.0, (vc + 0.29) % 1.0, 23, 0.94, rng)

    # Stress-wear mask so deep cracks cluster organically in flexed zones rather than uniformly
    wear_zone = _warp_field(u, v, 3, 4, 0.54, rng, warp_strength=0.08)
    width_mod = 0.55 + 0.90 * np.clip((wear_zone - 0.25) / 0.75, 0.0, 1.0)
    crack_primary = np.exp(-np.power(crack_e1 / (0.042 * width_mod + 0.008), 2.2))
    crack_secondary = np.exp(-np.power(crack_e2 / (0.024 * width_mod + 0.006), 2.4)) * np.clip(wear_zone * 1.35, 0.0, 1.0)
    cracks = np.clip(np.maximum(crack_primary, 0.72 * crack_secondary), 0.0, 1.0)

    # Slight lifted/curled dry finish lip directly adjacent to major cracks
    lip = np.exp(-np.power((crack_e1 - 0.055) / 0.035, 2.0)) * np.clip(wear_zone * 1.2, 0.0, 1.0)
    # Raw suede fiber texture inside the exposed cracks
    corium_fuzz = common.periodic_fbm(u, v, 48, 4, 0.62, rng)

    intact = 1.0 - cracks
    h = common.robust_normalize(
        intact * (0.46 * crown + 0.14 * crack_id + 0.14 * micro - 0.12 * pores)
        + 0.16 * lip
        - 0.52 * cracks
        + 0.08 * cracks * corium_fuzz,
        low_pct=0.25,
        high_pct=99.75,
        soft=True,
    )

    # Base aged chestnut islands vs exposed warm light-tan raw corium in cracks & abraded lips
    island_tone = common.robust_normalize(
        0.38 * wear_zone + 0.30 * crown + 0.18 * crack_id + 0.14 * micro - 0.16 * valley - 0.12 * pores,
        low_pct=0.3,
        high_pct=99.7,
        soft=True,
    )
    albedo_island = _blend_stops(island_tone, micro, stops[:4] + [stops[3]], contrast=1.12, gamma=1.08)
    # Exposed dry corium color (warm light tan from palette top stop, modulated by corium fibers)
    corium_rgb = np.array(stops[4], dtype=np.float32)[None, None, :] * (0.74 + 0.34 * corium_fuzz[..., None])
    # Dark split-edge shadow right at the rim of the crack, lighter dry suede in center
    crack_floor = np.clip((cracks - 0.35) / 0.65, 0.0, 1.0)
    rim_dark = np.clip(cracks * (1.0 - crack_floor) * 2.2, 0.0, 1.0)
    albedo = albedo_island * (1.0 - 0.75 * crack_floor[..., None]) + corium_rgb * (0.75 * crack_floor[..., None])
    albedo = albedo * (1.0 - 0.34 * rim_dark[..., None]) + lip[..., None] * 14.0

    # Wide authentic roughness contrast: oiled islands 0.26..0.46, dry cracked corium 0.76..0.90
    rough_island = 0.26 + 0.22 * valley + 0.14 * (1.0 - wear_zone) + 0.10 * pores + 0.14 * lip
    rough_crack = 0.76 + 0.14 * corium_fuzz
    rough = np.clip(intact * rough_island + cracks * rough_crack, 0.26, 0.90).astype(np.float32)

    base = common.shade_pbr_surface(
        np.clip(albedo, 0.0, 255.0),
        h,
        roughness=rough,
        normal_strength=6.4,
        spec_intensity=0.14,
        ao_strength=0.24,
        cavity_strength=0.22,
    )
    return base, h, rough


def _build_perforated(
    u: np.ndarray, v: np.ndarray, size: int, rng: np.random.Generator, stops: list[tuple[int, int, int]]
) -> tuple[Image.Image, np.ndarray, np.ndarray]:
    # Continuous smooth-to-fine-pebbled moto calfskin beneath staggered 45-deg circular punch grid
    crown, valley, cid, edge = _dermal_grain(u, v, cells=34, warp_amp=0.008, lloyd_iters=1, valley_sharpness=4.8, rng=rng)
    pores = _follicle_pores(u, v, edge, freq=70, rng=rng, junction_bias=0.72, threshold=0.72)
    creases, growth = _hide_creases(u, v, macro_freq=4, rng=rng, warp_amp=0.045, anisotropy=1.5)
    micro = _micro_collagen(u, v, rng)

    # Integer periodic staggered diamond punch lattice (24x24 base -> 1152 crisp holes at 2048px)
    n_grid = 24
    # Subtle continuous hide tension warp so perforations ride on supple leather, not rigid steel
    u_p = (u + (growth - 0.5) * 0.004) % 1.0
    v_p = (v + (creases - 0.5) * 0.002) % 1.0

    def _lattice_dist(uu: np.ndarray, vv: np.ndarray, shift: float) -> np.ndarray:
        gx = (uu * n_grid + shift) % n_grid
        gy = (vv * n_grid + shift) % n_grid
        ix = np.floor(gx).astype(np.int32) % n_grid
        iy = np.floor(gy).astype(np.int32) % n_grid
        fx = gx - np.floor(gx) - 0.5
        fy = gy - np.floor(gy) - 0.5
        # Tiny deterministic punch-die micro-jitter per hole (<= 2.5% of pitch)
        jx = (rng.uniform(-0.025, 0.025, size=(n_grid, n_grid)).astype(np.float32))[iy, ix]
        jy = (rng.uniform(-0.025, 0.025, size=(n_grid, n_grid)).astype(np.float32))[iy, ix]
        return np.sqrt((fx - jx) ** 2 + (fy - jy) ** 2)

    d_a = _lattice_dist(u_p, v_p, 0.0)
    d_b = _lattice_dist(u_p, v_p, 0.5)
    d_hole = np.minimum(d_a, d_b).astype(np.float32)

    r_bore = 0.165
    r_chamfer = 0.225
    r_rim = 0.275

    # Smooth punch profile: slight compacted ridge outside rim, tapered chamfer, deep dark bore
    hole_depth = np.clip((r_chamfer - d_hole) / (r_chamfer - r_bore), 0.0, 1.0)
    hole_smooth = hole_depth * hole_depth * (3.0 - 2.0 * hole_depth)
    bore_mask = np.clip((r_bore + 0.02 - d_hole) / 0.035, 0.0, 1.0)
    chamfer_ring = np.clip(hole_smooth * (1.0 - bore_mask) * 1.8, 0.0, 1.0)
    punch_rim = np.exp(-np.power((d_hole - r_rim) / 0.032, 2.0)) * 0.09

    leather_h = 0.44 * crown + 0.16 * growth + 0.14 * micro - 0.18 * creases - 0.12 * pores
    leather_h = common.robust_normalize(leather_h, low_pct=0.3, high_pct=99.7, soft=True)
    h = common.robust_normalize(
        (0.58 + 0.42 * leather_h + punch_rim) * (1.0 - 0.88 * hole_smooth),
        low_pct=0.1,
        high_pct=99.9,
        soft=False,
    )

    c_leather = common.robust_normalize(
        0.44 * crown + 0.22 * growth + 0.16 * micro + 0.12 * cid - 0.18 * creases - 0.14 * valley - 0.12 * pores,
        low_pct=0.3,
        high_pct=99.7,
        soft=True,
    )
    albedo_leather = _blend_stops(c_leather, micro, stops, contrast=1.12, gamma=1.06)
    # Darken cut chamfer slightly and make recessed bore deep shadowed suede black
    bore_rgb = np.array([5.0, 5.5, 6.5], dtype=np.float32)[None, None, :]
    albedo = albedo_leather * (1.0 - 0.28 * chamfer_ring[..., None])
    albedo = albedo * (1.0 - 0.92 * bore_mask[..., None]) + bore_rgb * (0.92 * bore_mask[..., None])

    rough_leather = common.match_histogram_range(
        0.42 * valley + 0.28 * creases + 0.22 * pores + 0.16 * (1.0 - growth), 0.30, 0.50
    )
    rough = np.clip(
        rough_leather * (1.0 - hole_smooth)
        + 0.68 * chamfer_ring
        + 0.90 * bore_mask,
        0.30,
        0.92,
    ).astype(np.float32)

    base = common.shade_pbr_surface(
        np.clip(albedo, 0.0, 255.0),
        h,
        roughness=rough,
        normal_strength=6.2,
        spec_intensity=0.15,
        ao_strength=0.22,
        cavity_strength=0.18,
    )
    return base, h, rough


def _build_woven_strips(
    u: np.ndarray, v: np.ndarray, size: int, rng: np.random.Generator, stops: list[tuple[int, int, int]]
) -> tuple[Image.Image, np.ndarray, np.ndarray]:
    # Intrecciato woven leather strips (8x8 periodic basketweave) with beveled skived edges,
    # slight organic hand-woven centerline wander, per-strip tonal shift, and real fine leather grain.
    n_strips = 8
    # Gentle organic hand-woven wander (periodic at tile scale)
    wu = (u + (common.periodic_fbm(u, v, 3, 3, 0.5, rng) - 0.5) * 0.010) % 1.0
    wv = (v + (common.periodic_fbm(u, v, 3, 3, 0.5, rng) - 0.5) * 0.010) % 1.0

    su = wu * n_strips
    sv = wv * n_strips
    iu = np.floor(su).astype(np.int32) % n_strips
    iv = np.floor(sv).astype(np.int32) % n_strips
    fu = su - np.floor(su)  # [0, 1) across vertical strip width
    fv = sv - np.floor(sv)  # [0, 1) across horizontal strip width

    # Alternating 1x1 over-under parity: (iu + iv) % 2 == 0 -> vertical strip on top
    vert_on_top = ((iu + iv) & 1) == 0

    def _strip_profile(across: np.ndarray, along_phase: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Flat leather strap cross-section with skived beveled edges & smooth over-under arch."""
        # Gap of 3% between adjacent strips for deep tuck shadow
        d_edge = np.minimum(across, 1.0 - across)
        # Beveled skived edge ramp from 0.018 to 0.145
        edge_t = np.clip((d_edge - 0.018) / 0.125, 0.0, 1.0)
        skive = edge_t * edge_t * (3.0 - 2.0 * edge_t)
        # Gentle pillowed strap top crown
        crown_arch = 0.88 + 0.12 * np.sin(np.clip(across, 0.0, 1.0) * math.pi)
        # Over-under elevation arch along strip length: peak = 1.0 when over, 0.24 when diving under
        over_arch = 0.58 + 0.42 * np.cos(along_phase * math.pi)
        profile = skive * crown_arch * over_arch
        # Cut-edge bevel indicator (high along the skived sides of each strap)
        bevel_mask = np.clip(skive * (1.0 - skive) * 4.0, 0.0, 1.0)
        return profile.astype(np.float32), skive.astype(np.float32), bevel_mask.astype(np.float32)

    # Along-strip phase in [-1, 1]: 0 at the center of an "over" cell, +/-1 at center of an "under" cell
    # For vertical strip iu: over when (iu+iv)%2==0 -> phase = fv - 0.5 when over, +/-0.5..1 when under
    v_over = vert_on_top
    phase_v = np.where(v_over, (fv - 0.5), np.where(fv < 0.5, 0.5 + fv, -1.5 + fv))
    phase_h = np.where(~v_over, (fu - 0.5), np.where(fu < 0.5, 0.5 + fu, -1.5 + fu))

    h_vert, mask_v, bevel_v = _strip_profile(fu, phase_v)
    h_horiz, mask_h, bevel_h = _strip_profile(fv, phase_h)

    # Which strip is visible at each pixel (highest physical elevation wins cleanly)
    show_vert = h_vert >= h_horiz
    weave_h = np.maximum(h_vert, h_horiz)
    bevel_edge = np.where(show_vert, bevel_v, bevel_h)
    gap_mask = np.clip(1.0 - np.maximum(mask_v, mask_h), 0.0, 1.0)

    # Tuck shadow where the visible strip dives underneath the cross strip
    tuck = np.clip(1.0 - np.where(show_vert, np.abs(np.cos(phase_v * math.pi * 0.5)), np.abs(np.cos(phase_h * math.pi * 0.5))), 0.0, 1.0)
    tuck = np.power(tuck, 1.4)

    # Fine calfskin grain + subtle creases mapped along each strip's own coordinate space
    grain_crown, grain_val, _, grain_edge = _dermal_grain(u, v, cells=42, warp_amp=0.007, lloyd_iters=1, valley_sharpness=4.6, rng=rng)
    pores = _follicle_pores(u, v, grain_edge, freq=80, rng=rng, junction_bias=0.70, threshold=0.72)
    micro = _micro_collagen(u, v, rng)
    # Subtle lengthwise flex creases along each strip's bend axis
    crease_v = common.periodic_fbm(u, v, 16, 3, 0.50, rng, freqy=6, mode="ridged")
    crease_h = common.periodic_fbm(u, v, 6, 3, 0.50, rng, freqy=16, mode="ridged")
    strip_crease = np.where(show_vert, crease_v, crease_h)
    strip_crease = np.power(np.clip((strip_crease - 0.58) / 0.42, 0.0, 1.0), 1.8)

    # Subtle per-strip natural hide dye variation (each cut strip from a slightly different hide spot)
    tone_v = rng.uniform(-0.11, 0.11, size=n_strips).astype(np.float32)[iu]
    tone_h = rng.uniform(-0.11, 0.11, size=n_strips).astype(np.float32)[iv]
    strip_tone = np.where(show_vert, tone_v, tone_h)
    strip_patina = _warp_field(u, v, 3, 4, 0.52, rng, warp_strength=0.07)

    h = common.robust_normalize(
        0.70 * weave_h
        + 0.22 * (1.0 - gap_mask) * grain_crown
        + 0.10 * (1.0 - gap_mask) * micro
        - 0.10 * strip_crease * (1.0 - gap_mask)
        - 0.08 * pores * (1.0 - gap_mask),
        low_pct=0.15,
        high_pct=99.85,
        soft=False,
    )

    c = common.robust_normalize(
        0.46 * weave_h
        + 0.24 * grain_crown
        + 0.16 * strip_patina
        + 0.10 * micro
        + strip_tone
        - 0.14 * bevel_edge
        - 0.24 * tuck
        - 0.30 * gap_mask
        - 0.12 * strip_crease
        - 0.08 * pores,
        low_pct=0.25,
        high_pct=99.75,
        soft=True,
    )
    rough_raw = (
        0.48 * tuck
        + 0.36 * gap_mask
        + 0.22 * bevel_edge
        + 0.24 * grain_val
        + 0.18 * strip_crease
        + 0.16 * pores
    )
    rough = common.match_histogram_range(rough_raw, 0.26, 0.78, target_mean=0.45, contrast=1.12)
    albedo = _blend_stops(c, micro, stops, contrast=1.14, gamma=1.02)
    base = common.shade_pbr_surface(
        albedo, h, roughness=rough, normal_strength=6.8, spec_intensity=0.16, ao_strength=0.28, cavity_strength=0.24
    )
    return base, h, rough


def _build_tooled_embossed(
    u: np.ndarray, v: np.ndarray, size: int, rng: np.random.Generator, stops: list[tuple[int, int, int]]
) -> tuple[Image.Image, np.ndarray, np.ndarray]:
    # Artisan hand-tooled vegetable-tanned saddle leather:
    # Classic periodic 4x4 diamond-strapwork + floral-rosette medallion tooling with:
    #   - Per-cell hand-strike depth & antique-burnish variation so stamps feel hand-struck
    #   - Raised burnished smooth carved scrollwork retaining real cowhide grain & soft creases
    #   - Crisp swivel-knife border grooves + beveled drop-off holding dark antique paste
    #   - Densely stippled recessed "backgrounder" pebble-punch pockets
    tiles = 4
    uw, vw = _warp_uv(u, v, 4, 0.008, rng, octaves=3)
    gu = uw * tiles
    gv = vw * tiles
    cell_x = np.floor(gu).astype(np.int32) % tiles
    cell_y = np.floor(gv).astype(np.int32) % tiles
    cx = gu - np.floor(gu) - 0.5
    cy = gv - np.floor(gv) - 0.5
    ax = np.abs(cx)
    ay = np.abs(cy)

    # Hand-strike depth and per-medallion antique patina variation across the 4x4 lattice
    strike_map = rng.uniform(0.84, 1.04, size=(tiles, tiles)).astype(np.float32)
    strike_var = common.periodic_gaussian_blur(
        strike_map[cell_y, cell_x], sigma=max(2.0, (size / 512.0) * 6.0)
    )

    # 1. Interlocking diamond strapwork trellis
    d_diag = np.abs((ax + ay) - 0.50)
    strap_half = 0.058
    strap_dist = strap_half - d_diag
    strap_mask = np.clip((strap_dist + 0.012) / 0.024, 0.0, 1.0)
    strap_dome = strap_mask * np.sqrt(np.clip(1.0 - np.clip(d_diag / strap_half, 0.0, 1.0) ** 2, 0.0, 1.0))

    # 2. Central 8-petal floral rosette medallion inside each diamond (and shifted half-cell)
    def _rosette_sdf(px: np.ndarray, py: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        r = np.sqrt(px * px + py * py + 1e-10)
        ang = np.arctan2(py, px)
        petal_r = 0.215 + 0.048 * np.cos(8.0 * ang) + 0.016 * np.cos(4.0 * ang)
        dist = petal_r - r
        flower_m = np.clip((dist + 0.012) / 0.024, 0.0, 1.0)
        petal_crease = np.power(np.clip(0.5 - 0.5 * np.cos(8.0 * ang), 0.0, 1.0), 3.0) * np.clip((r - 0.065) / 0.14, 0.0, 1.0)
        dome = flower_m * (0.68 + 0.32 * np.sqrt(np.clip(dist / 0.24, 0.0, 1.0)) - 0.28 * petal_crease)
        pod = np.clip((0.064 - r) / 0.018, 0.0, 1.0)
        pod_ring = np.exp(-np.power((r - 0.068) / 0.012, 2.0))
        dome = np.maximum(dome, pod * 0.96) - 0.26 * pod_ring * flower_m
        bevel_groove = np.exp(-np.power((r - (petal_r + 0.014)) / 0.016, 2.0))
        return np.clip(dome, 0.0, 1.0).astype(np.float32), bevel_groove.astype(np.float32)

    ros_a, grv_a = _rosette_sdf(cx, cy)
    cx2 = ((uw * tiles + 0.5) % 1.0) - 0.5
    cy2 = ((vw * tiles + 0.5) % 1.0) - 0.5
    ros_b, grv_b = _rosette_sdf(cx2, cy2)

    strap_groove = np.exp(-np.power((d_diag - (strap_half + 0.014)) / 0.015, 2.0))
    strap_stitch = np.exp(-np.power((d_diag - 0.036) / 0.006, 2.0)) * strap_mask

    motif_h = np.maximum(strap_dome * 0.92, np.maximum(ros_a, ros_b)) * strike_var
    motif_mask = np.clip(motif_h * 3.0, 0.0, 1.0)
    grooves = np.clip(np.maximum(strap_groove, np.maximum(grv_a, grv_b)) + 0.45 * strap_stitch, 0.0, 1.0)

    # Backgrounder stippling punch texture inside the recessed negative pockets
    stipple_a = common.periodic_rotated_perlin(
        u, v, 92, 0.52, rng, phase_u=float(rng.uniform()), phase_v=float(rng.uniform())
    )
    stipple_b = common.periodic_rotated_perlin(
        u, v, 118, 1.24, rng, phase_u=float(rng.uniform()), phase_v=float(rng.uniform())
    )
    stipple = common.robust_normalize(np.power(np.clip(stipple_a * stipple_b, 0.0, 1.0), 0.7), low_pct=1.0, high_pct=99.0)
    bg_mask = 1.0 - motif_mask

    # Real cowhide grain, supple hide creases & hand-applied antique stain mottling across the hide
    grain_crown, grain_val, _, grain_edge = _dermal_grain(u, v, cells=34, warp_amp=0.009, lloyd_iters=1, valley_sharpness=4.6, rng=rng)
    pores = _follicle_pores(u, v, grain_edge, freq=68, rng=rng, junction_bias=0.70, threshold=0.72)
    creases, _ = _hide_creases(u, v, macro_freq=3, rng=rng, warp_amp=0.06, anisotropy=1.5)
    micro = _micro_collagen(u, v, rng)
    hide_patina = _warp_field(u, v, 2, 4, 0.55, rng, warp_strength=0.12, stages=2)

    # Physical elevation: raised burnished leather motifs + crisp stippled pockets + swivel grooves + hide grain
    h = common.robust_normalize(
        0.64 * motif_h
        + 0.18 * motif_mask * grain_crown
        + 0.22 * bg_mask * (0.20 + 0.80 * stipple)
        + 0.10 * micro
        - 0.34 * grooves
        - 0.12 * creases
        - 0.08 * pores * motif_mask,
        low_pct=0.2,
        high_pct=99.8,
        soft=False,
    )

    # Hand-rubbed antique gel varies broadly across the hide so adjacent medallions never look cloned
    c = common.robust_normalize(
        0.40 * motif_h
        + 0.38 * hide_patina
        + 0.14 * grain_crown * motif_mask
        + 0.10 * micro
        - 0.36 * grooves
        - 0.24 * bg_mask * (1.0 - 0.50 * stipple)
        - 0.12 * creases
        - 0.08 * pores,
        low_pct=0.3,
        high_pct=99.7,
        soft=True,
    )
    rough_raw = (
        0.54 * bg_mask
        + 0.24 * bg_mask * (1.0 - stipple)
        + 0.26 * grooves
        + 0.20 * grain_val * motif_mask
        + 0.18 * (1.0 - hide_patina)
    )
    rough = common.match_histogram_range(rough_raw, 0.28, 0.80, target_mean=0.52, contrast=1.14)
    albedo = _blend_stops(c, micro, stops, contrast=1.16, gamma=1.0)
    base = common.shade_pbr_surface(
        albedo, h, roughness=rough, normal_strength=7.0, spec_intensity=0.15, ao_strength=0.28, cavity_strength=0.24
    )
    return base, h, rough


def _build_matte_aniline(
    u: np.ndarray, v: np.ndarray, size: int, rng: np.random.Generator, stops: list[tuple[int, int, int]]
) -> tuple[Image.Image, np.ndarray, np.ndarray]:
    # Naked drum-dyed aniline cowhide: translucent organic dye clouding, elongated neck/growth-mark
    # undulations, gentle fine dermal grain, crisp open pores, subtle matte-satin surface.
    crown, valley, cid, edge = _dermal_grain(u, v, cells=34, warp_amp=0.009, lloyd_iters=1, valley_sharpness=4.4, rng=rng)
    pores = _follicle_pores(u, v, edge, freq=72, rng=rng, junction_bias=0.74, threshold=0.68)
    creases, growth = _hide_creases(u, v, macro_freq=3, rng=rng, warp_amp=0.065, anisotropy=2.1)
    micro = _micro_collagen(u, v, rng)

    # Rich translucent aniline dye absorption clouding (2-stage domain warp)
    dye_cloud = _warp_field(u, v, 3, 4, 0.56, rng, warp_strength=0.10, stages=2)
    # Subtle natural healed range/neck growth ripples
    neck_waves = common.periodic_fbm(u, v, 3, 3, 0.48, rng, freqy=9, mode="fbm")

    h = common.robust_normalize(
        0.34 * crown
        + 0.26 * neck_waves
        + 0.16 * growth
        + 0.16 * micro
        - 0.20 * creases
        - 0.18 * pores,
        low_pct=0.25,
        high_pct=99.75,
        soft=True,
    )
    # Aniline albedo shines through with deep oxblood tonal clouds & darkened open pores
    c = common.robust_normalize(
        0.48 * dye_cloud
        + 0.22 * crown
        + 0.14 * neck_waves
        + 0.10 * cid
        - 0.18 * creases
        - 0.18 * pores
        - 0.10 * valley,
        low_pct=0.3,
        high_pct=99.7,
        soft=True,
    )
    rough_raw = (
        0.34 * valley
        + 0.30 * pores
        + 0.26 * creases
        + 0.22 * (1.0 - dye_cloud)
        + 0.12 * micro
    )
    rough = common.match_histogram_range(rough_raw, 0.42, 0.70, target_mean=0.55, contrast=1.08)
    albedo = _blend_stops(c, micro, stops, contrast=1.12, gamma=1.02)
    base = common.shade_pbr_surface(
        albedo, h, roughness=rough, normal_strength=4.4, spec_intensity=0.10, ao_strength=0.22, cavity_strength=0.16
    )
    return base, h, rough


def _build_pullup_oiled(
    u: np.ndarray, v: np.ndarray, size: int, rng: np.random.Generator, stops: list[tuple[int, int, int]]
) -> tuple[Image.Image, np.ndarray, np.ndarray]:
    # Waxed "Crazy Horse" utility pull-up hide: flexed ridges & micro-scratches lighten
    # noticeably toward warm amber-olive as oil/wax migrates under tension.
    crown, valley, cid, edge = _dermal_grain(u, v, cells=30, warp_amp=0.010, lloyd_iters=1, valley_sharpness=5.0, rng=rng)
    pores = _follicle_pores(u, v, edge, freq=66, rng=rng, junction_bias=0.72, threshold=0.70)
    creases, growth = _hide_creases(u, v, macro_freq=4, rng=rng, warp_amp=0.075, anisotropy=1.4)
    micro = _micro_collagen(u, v, rng)

    # Broad pull-up fold ridges (swiss multifractal gives crisp angular leather bend crests)
    uw, vw = _warp_uv(u, v, 3, 0.06, rng, octaves=3)
    pullup_folds = common.periodic_fbm(uw, vw, 4, 4, 0.54, rng, mode="swiss")
    # Fine multi-angle wear scratches/scuffs (anisotropic ridged Perlin streaks)
    scuff_a = common.periodic_fbm(u, v, 5, 3, 0.52, rng, freqy=22, mode="ridged")
    scuff_b = common.periodic_fbm(u, v, 20, 3, 0.52, rng, freqy=5, mode="ridged")
    scuff_gate = common.periodic_fbm(u, v, 4, 3, 0.55, rng)
    scuffs = np.clip(
        (np.power(np.clip((scuff_a - 0.72) / 0.28, 0.0, 1.0), 2.2) + np.power(np.clip((scuff_b - 0.74) / 0.26, 0.0, 1.0), 2.2))
        * np.clip(scuff_gate * 1.4, 0.0, 1.0),
        0.0,
        1.0,
    )
    stress = np.clip(0.58 * np.power(pullup_folds, 1.5) + 0.34 * creases + 0.42 * scuffs, 0.0, 1.0)

    h = common.robust_normalize(
        0.42 * crown
        + 0.22 * pullup_folds
        + 0.14 * growth
        + 0.14 * micro
        - 0.22 * creases
        - 0.14 * pores
        - 0.10 * scuffs,
        low_pct=0.25,
        high_pct=99.75,
        soft=True,
    )
    # Key pull-up visual signature: stressed ridges, creases & scuffs LIGHTEN markedly!
    c = common.robust_normalize(
        0.46 * stress + 0.24 * crown + 0.16 * growth + 0.10 * cid - 0.14 * valley - 0.10 * pores,
        low_pct=0.3,
        high_pct=99.7,
        soft=True,
    )
    # Waxed unflexed flats stay glossy-satin; pulled-up stress ridges & scuffs turn dry matte
    rough_raw = 0.52 * stress + 0.26 * valley + 0.18 * pores + 0.14 * (1.0 - growth)
    rough = common.match_histogram_range(rough_raw, 0.26, 0.78, target_mean=0.48, contrast=1.12)
    albedo = _blend_stops(c, micro, stops, contrast=1.15, gamma=1.02)
    base = common.shade_pbr_surface(
        albedo, h, roughness=rough, normal_strength=5.2, spec_intensity=0.14, ao_strength=0.22, cavity_strength=0.18
    )
    return base, h, rough


def _build_croc_embossed(
    u: np.ndarray, v: np.ndarray, size: int, rng: np.random.Generator, stops: list[tuple[int, int, int]]
) -> tuple[Image.Image, np.ndarray, np.ndarray]:
    # Fashion embossed crocodile/alligator flank leather:
    # Organic domain-warped staggered rounded-rectangular scales (warped before cell lookup
    # for exact seam continuity) + micro-striations + umbilical sensor pores + hide micro-grain.
    rows = 12
    cols = 10
    # Continuous two-scale organic coordinate warp BEFORE cell indexing (exact T^2 continuity)
    wu = (
        u
        + (common.periodic_fbm(u, v, 3, 3, 0.52, rng) - 0.5) * 0.042
        + (common.periodic_fbm(u, v, 12, 2, 0.50, rng) - 0.5) * 0.010
    ) % 1.0
    wv = (
        v
        + (common.periodic_fbm(u, v, 3, 3, 0.52, rng) - 0.5) * 0.036
        + (common.periodic_fbm(u, v, 12, 2, 0.50, rng) - 0.5) * 0.008
    ) % 1.0

    # Smooth belly-to-flank scale width modulation (periodic cosine)
    u_mod = (wu + 0.055 * np.sin(2.0 * math.pi * wu) / (2.0 * math.pi)) % 1.0
    gv = wv * rows
    row_idx = np.floor(gv).astype(np.int32) % rows
    fv = gv - np.floor(gv)

    # Stagger alternate rows smoothly
    row_shift = rng.uniform(0.30, 0.70, size=rows).astype(np.float32)[row_idx]
    gu = (u_mod * cols + row_shift) % cols
    col_idx = np.floor(gu).astype(np.int32) % cols
    fu = gu - np.floor(gu)

    scale_id = rng.uniform(0.0, 1.0, size=(rows, cols)).astype(np.float32)[row_idx, col_idx]
    ax = np.abs(fu - 0.5) * 2.0
    ay = np.abs(fv - 0.5) * 2.0

    # Superellipse (squircle p=3.0) distance for rounded-rectangular crocodilian plates
    p_exp = 3.0
    squircle_r = np.power(np.power(ax, p_exp) + np.power(ay, p_exp) + 1e-8, 1.0 / p_exp)
    d_seam = np.clip(0.97 - squircle_r, 0.0, 1.0)

    dome = np.power(np.clip(d_seam / 0.34, 0.0, 1.0), 0.55) * (
        0.80 + 0.20 * np.sqrt(np.clip(1.0 - 0.65 * squircle_r * squircle_r, 0.0, 1.0))
    )
    hinge = np.exp(-np.power(d_seam / 0.11, 1.7))

    # Crocodilian integumentary sensory organ (ISO) tiny pit near one end of each scale
    iso_r = np.sqrt((fu - 0.5) ** 2 + (fv - 0.78) ** 2)
    iso_pit = np.exp(-np.power(iso_r / 0.055, 2.0)) * (1.0 - hinge)

    # Fine scale striations + underlying embossed leather micro-grain & subtle creases
    striations = common.periodic_fbm(u, v, 24, 4, 0.58, rng, freqy=72, mode="fbm")
    micro = _micro_collagen(u, v, rng)
    seam_wrinkle = common.periodic_fbm(u, v, 32, 4, 0.58, rng, mode="ridged")
    hide_patina = _warp_field(u, v, 3, 4, 0.55, rng, warp_strength=0.08)

    h = common.robust_normalize(
        0.66 * dome * (0.86 + 0.14 * scale_id)
        + 0.16 * striations * (1.0 - hinge)
        + 0.14 * micro
        - 0.24 * hinge
        - 0.12 * hinge * seam_wrinkle
        - 0.14 * iso_pit,
        low_pct=0.2,
        high_pct=99.8,
        soft=False,
    )
    c = common.robust_normalize(
        0.48 * dome
        + 0.22 * hide_patina
        + 0.16 * scale_id
        + 0.14 * striations
        + 0.10 * micro
        - 0.34 * hinge
        - 0.12 * iso_pit,
        low_pct=0.3,
        high_pct=99.7,
        soft=True,
    )
    rough_raw = 0.60 * hinge + 0.18 * (1.0 - dome) + 0.14 * micro + 0.14 * (1.0 - hide_patina)
    rough = common.match_histogram_range(rough_raw, 0.16, 0.72, target_mean=0.34, contrast=1.15)
    albedo = _blend_stops(c, micro, stops, contrast=1.14, gamma=1.08)
    base = common.shade_pbr_surface(
        albedo, h, roughness=rough, normal_strength=6.4, spec_intensity=0.18, ao_strength=0.28, cavity_strength=0.22
    )
    return base, h, rough


_BUILDERS = (
    _build_full_grain_smooth,
    _build_pebbled_grain,
    _build_suede,
    _build_nubuck,
    _build_patent,
    _build_cracked_vintage,
    _build_perforated,
    _build_woven_strips,
    _build_tooled_embossed,
    _build_matte_aniline,
    _build_pullup_oiled,
    _build_croc_embossed,
)


def _compute_bundle(
    seed: int = 1,
    palette: PaletteArg = None,
    size: int = 512,
    variant: int = 0,
) -> tuple[Image.Image, np.ndarray, np.ndarray, LeatherStyleSpec]:
    style = LEATHER_STYLES[int(variant) % len(LEATHER_STYLES)]
    rng = np.random.default_rng(int(seed))
    u, v = common.make_uv_grid(int(size))
    stops = common.resolve_palette(palette, style.default_palette, rng)
    base, h, rough = _BUILDERS[style.variant](u, v, int(size), rng, stops)
    return base, np.asarray(h, dtype=np.float32), np.asarray(rough, dtype=np.float32), style


def render_leather_bundle(
    style_or_variant: str | int,
    seed: int = 1,
    size: int = 2048,
    palette: PaletteArg = None,
) -> tuple[Image.Image, Image.Image, Image.Image, np.ndarray, np.ndarray, LeatherStyleSpec]:
    """Single-pass PBR bundle renderer returning (base, normal, roughness, height, rough_field, style)."""
    if isinstance(style_or_variant, str):
        style = STYLE_BY_SLUG[style_or_variant]
        variant = style.variant
    else:
        variant = int(style_or_variant) % len(LEATHER_STYLES)
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
    name="leather-suede",
    kind="texture",
    tileable=True,
    origin="tools/generators/types/leather_suede.py",
    default_palette="espresso_fullgrain_leather",
    description=(
        "Apparel leather & suede suite (12 variants: full-grain smooth, pebbled, suede, nubuck, "
        "patent, cracked vintage, perforated, woven strips, tooled saddle, matte aniline, "
        "oiled pull-up, and croc-embossed) with calibrated PBR height, normal, and roughness."
    ),
    render_fn=generate,
    normal_source="native",
    normal_rule="Native per-style leather relief height field (h) converted with calibrated Scharr normal strength.",
    normal_strength=5.2,
    height_fn=generate_height,
    normal_fn=generate_normal,
    roughness_fn=generate_roughness,
    default_roughness=0.45,
)
