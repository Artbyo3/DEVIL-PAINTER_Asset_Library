"""Procedural Glass, Crystal & Ice generator suite (10 tileable PBR materials).

Built strictly on the committed periodic math primitives in ``tools/generators/common.py``
with all palettes defined inline for self-contained operation:
  - frosted-glass: acid-etched pale glass with gentle float-glass wave and dense micro-pitting
  - crystal-facets-clear: CSG-interpenetrating clear rock-crystal polyhedral facets
  - crystal-facets-smoky: tabular smoky crystal blocks with internal depth zoning and cleavage steps
  - cracked-ice: domain-warped multi-scale thermal fractures with subsurface crack echoes
  - ice-sheet-blue: smooth glacial blue ice sheet with curl-flow undulations and subtle rime wisps
  - window-frost: feathery dendritic crystalline frost fronds growing over smooth cold glass
  - stained-glass-amber: warm rippled amber cathedral glass panes framed by half-round lead came
  - glass-pebble: tightly packed translucent rounded glass pebble domes over frosted crevices
  - mirror-flake: densely packed angular tilted mirror shards with near-zero face roughness
  - obsidian-glass: near-black volcanic glass with sweeping conchoidal Wallner fracture ripples
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from PIL import Image

from .. import common
from ..interface import GeneratorSpec, PaletteArg


@dataclass(frozen=True)
class GlassStyleSpec:
    variant: int
    slug: str
    default_palette: str
    normal_strength: float
    rough_range: tuple[float, float]
    description: str


GLASS_PALETTES: dict[str, tuple[tuple[int, int, int], ...]] = {
    "frosted-glass": (
        (194, 211, 214),
        (213, 226, 229),
        (228, 238, 240),
        (240, 246, 247),
        (249, 252, 253),
    ),
    "crystal-facets-clear": (
        (170, 190, 206),
        (199, 217, 230),
        (222, 235, 244),
        (239, 246, 251),
        (250, 253, 255),
    ),
    "crystal-facets-smoky": (
        (30, 28, 30),
        (54, 50, 52),
        (84, 78, 79),
        (122, 115, 114),
        (168, 161, 158),
    ),
    "cracked-ice": (
        (128, 168, 194),
        (164, 199, 220),
        (198, 223, 238),
        (228, 241, 249),
        (248, 252, 255),
    ),
    "ice-sheet-blue": (
        (72, 130, 166),
        (108, 166, 198),
        (148, 196, 222),
        (192, 224, 240),
        (230, 246, 253),
    ),
    "window-frost": (
        (144, 174, 194),
        (180, 205, 220),
        (212, 229, 239),
        (235, 244, 250),
        (250, 253, 255),
    ),
    "stained-glass-amber": (
        (82, 38, 8),
        (146, 74, 14),
        (202, 118, 24),
        (234, 168, 52),
        (249, 214, 110),
    ),
    "glass-pebble": (
        (126, 166, 170),
        (162, 198, 200),
        (194, 222, 223),
        (221, 239, 240),
        (243, 251, 251),
    ),
    "mirror-flake": (
        (132, 142, 152),
        (168, 178, 188),
        (202, 211, 219),
        (228, 234, 240),
        (247, 250, 253),
    ),
    "obsidian-glass": (
        (8, 9, 12),
        (16, 18, 23),
        (26, 30, 38),
        (42, 47, 58),
        (68, 75, 90),
    ),
}


GLASS_STYLES: tuple[GlassStyleSpec, ...] = (
    GlassStyleSpec(
        0,
        "frosted-glass",
        "frosted-glass",
        3.2,
        (0.58, 0.84),
        "Acid-etched pale glass with gentle float-glass waviness and dense grainy micro-pitting.",
    ),
    GlassStyleSpec(
        1,
        "crystal-facets-clear",
        "crystal-facets-clear",
        22.0,
        (0.02, 0.14),
        "Interpenetrating clear polyhedral crystal facets with sharp dihedral ridges and polished faces.",
    ),
    GlassStyleSpec(
        2,
        "crystal-facets-smoky",
        "crystal-facets-smoky",
        24.0,
        (0.02, 0.16),
        "Tabular dark-tinted crystal blocks with truncated crowns, growth striations, and glossy facets.",
    ),
    GlassStyleSpec(
        3,
        "cracked-ice",
        "cracked-ice",
        8.2,
        (0.04, 0.78),
        "Glassy fractured ice plates with organic multi-scale thermal cracks, frosted edges, and trapped bubbles.",
    ),
    GlassStyleSpec(
        4,
        "ice-sheet-blue",
        "ice-sheet-blue",
        9.5,
        (0.04, 0.34),
        "Smooth translucent glacial blue ice sheet with gentle meltwater swells and delicate rime wisps.",
    ),
    GlassStyleSpec(
        5,
        "window-frost",
        "window-frost",
        5.8,
        (0.04, 0.85),
        "Branching feathery dendritic frost crystals spreading across smooth cold window glass.",
    ),
    GlassStyleSpec(
        6,
        "stained-glass-amber",
        "stained-glass-amber",
        9.5,
        (0.05, 0.78),
        "Warm rippled amber cathedral glass panels joined by raised matte dark lead came borders.",
    ),
    GlassStyleSpec(
        7,
        "glass-pebble",
        "glass-pebble",
        10.5,
        (0.06, 0.68),
        "Tightly packed smooth convex translucent glass pebble domes nestled in slightly frosted crevices.",
    ),
    GlassStyleSpec(
        8,
        "mirror-flake",
        "mirror-flake",
        14.0,
        (0.02, 0.58),
        "Angular mosaic of independently tilted mirror-polished shards separated by crisp thin seams.",
    ),
    GlassStyleSpec(
        9,
        "obsidian-glass",
        "obsidian-glass",
        8.2,
        (0.02, 0.16),
        "Near-black glossy volcanic glass marked by sweeping conchoidal fracture arcs and sharp cleavage crests.",
    ),
)

STYLE_BY_SLUG: dict[str, GlassStyleSpec] = {s.slug: s for s in GLASS_STYLES}


# ---------------------------------------------------------------------------
# Internal helpers (built strictly on common.py periodic primitives)
# ---------------------------------------------------------------------------


def _wrap01(arr: np.ndarray) -> np.ndarray:
    """Safe float32 modulo [0, 1) avoiding float32 (-eps % 1.0 == 1.0)."""
    out = np.mod(arr.astype(np.float64), 1.0).astype(np.float32)
    return np.where(out >= 1.0, np.float32(0.0), out)


def _smoothstep(edge0: float, edge1: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - np.float32(edge0)) / np.float32(max(edge1 - edge0, 1e-6)), 0.0, 1.0)
    return (t * t * (3.0 - 2.0 * t)).astype(np.float32)


def _resolve_glass_palette(
    palette: PaletteArg,
    default_name: str,
    rng: np.random.Generator,
) -> list[tuple[int, int, int]]:
    if palette is None:
        return list(GLASS_PALETTES[default_name])
    if isinstance(palette, str):
        key = palette.strip().lower().replace("_", "-")
        if key in GLASS_PALETTES:
            return list(GLASS_PALETTES[key])
    return list(common.resolve_palette(palette, "bismuth_iridescent", rng))


def _interp_stops(field: np.ndarray, stops: list[tuple[int, int, int]]) -> np.ndarray:
    f = np.clip(field, 0.0, 1.0).astype(np.float32)
    pos = np.linspace(0.0, 1.0, len(stops), dtype=np.float32)
    out = np.empty((*f.shape, 3), dtype=np.float32)
    for ch in range(3):
        vals = np.array([float(s[ch]) for s in stops], dtype=np.float32)
        out[..., ch] = np.interp(f, pos, vals)
    return out


def _hash_from_cid(cid: np.ndarray, salt: float) -> np.ndarray:
    """Deterministic uniform [0, 1) per-cell hash derived from Voronoi cid."""
    x = np.sin(cid * np.float32(127.1 + salt * 31.7) + np.float32(salt * 19.19)) * np.float32(43758.5453)
    return (x - np.floor(x)).astype(np.float32)


def _polyhedral_crystal_layer(
    u: np.ndarray,
    v: np.ndarray,
    cells: int,
    rng: np.random.Generator,
    n_faces: int = 6,
    slope_base: float = 1.62,
    cap_range: tuple[float, float] = (0.76, 1.00),
    truncate_prob: float = 0.0,
    striation_amp: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a continuous piecewise-linear polyhedral crystal pyramid layer on T^2.

    Every crystal terminates cleanly before its Voronoi boundary so CSG ``np.maximum``
    across multiple layers yields sharp planar facets with zero artificial cliff spikes.
    Returns ``(height, cell_id, facet_shade)``.
    """
    _, _, cid, rx, ry = common.periodic_lloyd_voronoi(u, v, cells, 0.90, rng, lloyd_iters=1)

    theta0 = _hash_from_cid(cid, 1.3) * np.float32(2.0 * math.pi)
    peak_h = np.float32(cap_range[0]) + _hash_from_cid(cid, 2.7) * np.float32(cap_range[1] - cap_range[0])
    slope_var = np.float32(slope_base) * (np.float32(0.84) + np.float32(0.32) * _hash_from_cid(cid, 4.1))
    # Slight per-cell asymmetry produces realistic quartz r/z alternating face widths
    aniso = np.float32(0.86) + np.float32(0.28) * _hash_from_cid(cid, 5.9)

    # Slight apex offset inside cell makes crystal terminations naturally asymmetric
    ox = (_hash_from_cid(cid, 7.2) - np.float32(0.5)) * np.float32(0.18)
    oy = (_hash_from_cid(cid, 8.6) - np.float32(0.5)) * np.float32(0.18)
    dx = rx - ox
    dy = ry - oy

    max_proj = np.full_like(u, -1e6, dtype=np.float32)
    face_idx = np.zeros_like(u, dtype=np.float32)

    for k in range(n_faces):
        ang = theta0 + np.float32(k * (2.0 * math.pi / n_faces))
        # Alternate slightly steeper/shallower prism/rhombohedral slopes
        face_slope = np.float32(1.0) if (k % 2 == 0) else aniso
        proj = (dx * np.cos(ang) + dy * np.sin(ang)) * face_slope
        if striation_amp > 0.0:
            # Fine horizontal growth steps running parallel to each crystal face
            proj = proj + np.float32(striation_amp) * np.sin(proj * np.float32(42.0) + cid * np.float32(11.0))
        is_dom = proj > max_proj
        max_proj = np.where(is_dom, proj, max_proj)
        face_idx = np.where(is_dom, np.float32(k / max(1, n_faces)), face_idx)

    h = np.clip(peak_h - slope_var * np.maximum(max_proj, np.float32(0.0)), 0.0, 1.0)

    if truncate_prob > 0.0:
        # Optional tilted tabular crown plane (authentic smoky quartz / tabular crystal faces)
        has_cap = _hash_from_cid(cid, 11.4) < np.float32(truncate_prob)
        cap_ang = _hash_from_cid(cid, 12.8) * np.float32(2.0 * math.pi)
        cap_tilt = np.float32(0.32) + np.float32(0.28) * _hash_from_cid(cid, 14.2)
        cap_lvl = peak_h - np.float32(0.16) - np.float32(0.14) * _hash_from_cid(cid, 15.6)
        cap_plane = cap_lvl + cap_tilt * (dx * np.cos(cap_ang) + dy * np.sin(cap_ang))
        capped = np.minimum(h, np.clip(cap_plane, 0.0, 1.0))
        use_cap = has_cap & (capped < h)
        h = np.where(has_cap, capped, h)
        face_idx = np.where(use_cap, np.float32(0.95), face_idx)

    # Per-face angular luminance cue simulating internal facet reflection
    facet_ang = theta0 + face_idx * np.float32(2.0 * math.pi)
    facet_shade = np.clip(
        0.52 + 0.34 * np.cos(facet_ang - np.float32(0.85)) + 0.14 * (cid - np.float32(0.5)),
        0.0,
        1.0,
    )
    return h.astype(np.float32), cid.astype(np.float32), facet_shade.astype(np.float32)


# ---------------------------------------------------------------------------
# 10 Material Builders
# ---------------------------------------------------------------------------


def _build_frosted_glass(
    u: np.ndarray, v: np.ndarray, size: int, rng: np.random.Generator, stops: list[tuple[int, int, int]]
) -> tuple[Image.Image, np.ndarray, np.ndarray]:
    """Acid-etched pale privacy glass: gentle float wave + dense grainy micro-pitting."""
    _ = size
    # Broad subtle architectural float-glass optical wave
    wave = common.periodic_fbm(u, v, 3, 3, 0.45, rng, mode="fbm")
    etch_cloud = common.periodic_domain_warp_fbm(
        u, v, freq=6, octaves=4, persistence=0.52, rng=rng, warp_strength=0.04, stages=1, mode="fbm"
    )[0]

    # Dense acid-etched granular micro-pitting (multi-scale crisp stipple at 2048px)
    pit_mid = common.periodic_fbm(u, v, 38, 3, 0.58, rng, mode="fbm")
    pit_fine = common.periodic_fbm(u, v, 112, 3, 0.62, rng, mode="fbm")
    pit_sharp = common.periodic_fbm(u, v, 196, 2, 0.64, rng, mode="turbulence")

    # Subtle micro-cellular frosted sparkle grain
    f1_micro, _, _, _, _ = common.periodic_voronoi(u, v, 58, 0.92, rng)
    micro_pits = np.clip(1.0 - f1_micro, 0.0, 1.0)

    h = common.robust_normalize(
        0.24 * wave + 0.14 * etch_cloud + 0.24 * pit_mid + 0.24 * pit_fine + 0.08 * pit_sharp + 0.10 * micro_pits,
        low_pct=0.2,
        high_pct=99.8,
        soft=True,
    )

    # High and grainy PBR roughness (0.58 - 0.84)
    rough_raw = 0.34 * etch_cloud + 0.28 * pit_fine + 0.20 * pit_sharp + 0.18 * (1.0 - micro_pits)
    rough = common.match_histogram_range(rough_raw, 0.58, 0.84, target_mean=0.72, contrast=1.12)

    # Pale cool-white / faint soda-lime glass tint with cloudy frosted translucency
    tone = common.robust_normalize(
        0.42 * etch_cloud + 0.24 * wave + 0.20 * pit_mid + 0.14 * pit_fine,
        low_pct=0.5,
        high_pct=99.5,
        soft=True,
    )
    albedo = _interp_stops(tone, stops)
    base = common.shade_pbr_surface(
        albedo, h, roughness=rough, normal_strength=3.2, spec_intensity=0.08, ao_strength=0.08, cavity_strength=0.06
    )
    return base, h, rough


def _build_crystal_facets_clear(
    u: np.ndarray, v: np.ndarray, size: int, rng: np.random.Generator, stops: list[tuple[int, int, int]]
) -> tuple[Image.Image, np.ndarray, np.ndarray]:
    """Interpenetrating clear rock-crystal polyhedral facets with razor-sharp ridges."""
    _ = size
    # Three overlapping polyhedral crystal lattices combined via exact CSG union (max)
    h1, cid1, shade1 = _polyhedral_crystal_layer(
        u, v, cells=6, rng=rng, n_faces=6, slope_base=1.55, cap_range=(0.76, 1.00), truncate_prob=0.18
    )
    h2, cid2, shade2 = _polyhedral_crystal_layer(
        _wrap01(u + 0.37),
        _wrap01(v + 0.61),
        cells=9,
        rng=rng,
        n_faces=5,
        slope_base=1.68,
        cap_range=(0.70, 0.96),
        truncate_prob=0.22,
    )
    h3, cid3, shade3 = _polyhedral_crystal_layer(
        _wrap01(u + 0.71),
        _wrap01(v + 0.23),
        cells=13,
        rng=rng,
        n_faces=6,
        slope_base=1.78,
        cap_range=(0.64, 0.90),
        truncate_prob=0.15,
    )

    h12 = np.maximum(h1, h2)
    win2 = h2 > h1
    cid12 = np.where(win2, cid2, cid1)
    shade12 = np.where(win2, shade2, shade1)

    h_raw = np.maximum(h12, h3)
    win3 = h3 > h12
    cid = np.where(win3, cid3, cid12)
    f_shade = np.where(win3, shade3, shade12)

    # Detect sharp re-entrant junction creases between intersecting crystals
    gap12 = np.abs(h1 - h2)
    gap13 = np.abs(h12 - h3)
    seam = np.maximum(np.exp(-gap12 / np.float32(0.022)), np.exp(-gap13 / np.float32(0.022)))

    # Extremely faint micro-polish crystal clarity variation (keeps faces razor-flat)
    micro = common.periodic_fbm(u, v, 18, 3, 0.45, rng, mode="fbm")
    h = common.normalize_field(0.985 * h_raw + 0.015 * micro)

    # Polished crystal = near-zero roughness (0.02 - 0.14), slightly higher only in micro-seams
    rough_raw = 0.62 * seam + 0.24 * (1.0 - h) + 0.14 * micro
    rough = common.match_histogram_range(rough_raw, 0.02, 0.14, target_mean=0.05, contrast=1.35)

    # Pale ice-clear tint with strong facet-orientation luminance contrast and crisp bright edges
    tone = common.robust_normalize(
        0.46 * f_shade + 0.34 * h + 0.14 * cid + 0.10 * seam * 0.45 + 0.06 * micro,
        low_pct=0.3,
        high_pct=99.7,
        soft=True,
    )
    albedo = _interp_stops(tone, stops)
    # Subtle prismatic ice-white glint right on crystal facet intersections
    ridge_glint = np.clip(seam * 0.22, 0.0, 0.24)[..., None]
    albedo = albedo * (1.0 - ridge_glint) + np.array([250.0, 253.0, 255.0], dtype=np.float32) * ridge_glint

    base = common.shade_pbr_surface(
        albedo, h, roughness=rough, normal_strength=22.0, spec_intensity=0.34, ao_strength=0.14, cavity_strength=0.10
    )
    return base, h, rough


def _build_crystal_facets_smoky(
    u: np.ndarray, v: np.ndarray, size: int, rng: np.random.Generator, stops: list[tuple[int, int, int]]
) -> tuple[Image.Image, np.ndarray, np.ndarray]:
    """Tabular smoky crystal blocks with truncated crowns, growth striations, and depth zoning."""
    _ = size
    # Larger, bolder tabular rhombohedral blocks with truncated flat crowns and subtle growth striations
    h1, cid1, shade1 = _polyhedral_crystal_layer(
        u,
        v,
        cells=5,
        rng=rng,
        n_faces=5,
        slope_base=1.42,
        cap_range=(0.78, 1.00),
        truncate_prob=0.65,
        striation_amp=0.008,
    )
    h2, cid2, shade2 = _polyhedral_crystal_layer(
        _wrap01(u + 0.43),
        _wrap01(v + 0.57),
        cells=8,
        rng=rng,
        n_faces=6,
        slope_base=1.58,
        cap_range=(0.72, 0.96),
        truncate_prob=0.55,
        striation_amp=0.006,
    )
    h3, cid3, shade3 = _polyhedral_crystal_layer(
        _wrap01(u + 0.79),
        _wrap01(v + 0.19),
        cells=11,
        rng=rng,
        n_faces=4,
        slope_base=1.72,
        cap_range=(0.62, 0.88),
        truncate_prob=0.45,
        striation_amp=0.005,
    )

    h12 = np.maximum(h1, h2)
    win2 = h2 > h1
    cid12 = np.where(win2, cid2, cid1)
    shade12 = np.where(win2, shade2, shade1)

    h_raw = np.maximum(h12, h3)
    win3 = h3 > h12
    cid = np.where(win3, cid3, cid12)
    f_shade = np.where(win3, shade3, shade12)

    gap12 = np.abs(h1 - h2)
    gap13 = np.abs(h12 - h3)
    seam = np.maximum(np.exp(-gap12 / np.float32(0.024)), np.exp(-gap13 / np.float32(0.024)))

    # Internal smoky quartz phantom zoning bands running parallel to crystal contours
    smoke_cloud = common.periodic_domain_warp_fbm(
        u, v, freq=5, octaves=4, persistence=0.50, rng=rng, warp_strength=0.045, stages=1, mode="fbm"
    )[0]
    phantom = 0.5 + 0.5 * np.sin((h_raw * 6.5 + 0.25 * smoke_cloud) * np.float32(2.0 * math.pi))

    h = common.normalize_field(0.982 * h_raw + 0.018 * smoke_cloud)

    # Polished smoky crystal = near-zero roughness (0.02 - 0.16)
    rough_raw = 0.58 * seam + 0.26 * (1.0 - h) + 0.16 * (1.0 - phantom)
    rough = common.match_histogram_range(rough_raw, 0.02, 0.16, target_mean=0.06, contrast=1.32)

    # Dark translucent smoky taupe-charcoal tones with brighter quartz facet faces and faint pale edges
    tone = common.robust_normalize(
        0.44 * f_shade + 0.26 * h + 0.14 * phantom + 0.10 * smoke_cloud + 0.06 * cid,
        low_pct=0.4,
        high_pct=99.6,
        soft=True,
    )
    albedo = _interp_stops(tone, stops)
    edge_catch = np.clip(seam * 0.24, 0.0, 0.26)[..., None]
    albedo = albedo * (1.0 - edge_catch) + np.array([186.0, 178.0, 174.0], dtype=np.float32) * edge_catch

    base = common.shade_pbr_surface(
        albedo, h, roughness=rough, normal_strength=24.0, spec_intensity=0.38, ao_strength=0.18, cavity_strength=0.14
    )
    return base, h, rough


def _build_cracked_ice(
    u: np.ndarray, v: np.ndarray, size: int, rng: np.random.Generator, stops: list[tuple[int, int, int]]
) -> tuple[Image.Image, np.ndarray, np.ndarray]:
    """Fractured ice plates with organic multi-scale thermal cracks and subsurface depth echoes."""
    scale = float(size) / 512.0

    # Organic thermal-stress UV warps so ice cracks curve and zig naturally
    du1 = common.periodic_fbm(u, v, 4, 3, 0.52, rng, mode="fbm") - np.float32(0.5)
    dv1 = common.periodic_fbm(u, v, 4, 3, 0.52, rng, mode="fbm") - np.float32(0.5)
    uw1 = _wrap01(u + du1 * np.float32(0.065))
    vw1 = _wrap01(v + dv1 * np.float32(0.065))

    du2 = common.periodic_fbm(u, v, 7, 3, 0.55, rng, mode="fbm") - np.float32(0.5)
    dv2 = common.periodic_fbm(u, v, 7, 3, 0.55, rng, mode="fbm") - np.float32(0.5)
    uw2 = _wrap01(u + du2 * np.float32(0.045))
    vw2 = _wrap01(v + dv2 * np.float32(0.045))

    # Primary deep fracture network
    _, edge1, cid1, rx1, ry1 = common.periodic_voronoi(uw1, vw1, 5, 0.92, rng)
    # Secondary branching shatter cracks (stronger in localized stress pockets)
    _, edge2, _, _, _ = common.periodic_voronoi(uw2, vw2, 11, 0.92, rng)
    # Tertiary hairline crazing
    _, edge3, _, _, _ = common.periodic_voronoi(_wrap01(uw2 + 0.31), _wrap01(vw2 + 0.67), 21, 0.90, rng)

    stress_zone = common.periodic_fbm(u, v, 3, 3, 0.55, rng, mode="fbm")
    stress_mask = _smoothstep(0.26, 0.72, stress_zone)
    stress_hi = _smoothstep(0.46, 0.84, stress_zone)

    c1 = np.exp(-edge1 / np.float32(0.017)).astype(np.float32)
    c2 = (np.exp(-edge2 / np.float32(0.014)) * (np.float32(0.32) + np.float32(0.68) * stress_mask)).astype(np.float32)
    c3 = (np.exp(-edge3 / np.float32(0.010)) * stress_hi * np.float32(0.65)).astype(np.float32)
    cracks = np.clip(np.maximum(c1, np.maximum(c2 * np.float32(0.85), c3 * np.float32(0.60))), 0.0, 1.0)

    # Frosted chipped halo flanking primary fractures
    crack_halo = np.clip(np.exp(-edge1 / np.float32(0.055)) * 0.75 + np.exp(-edge2 / np.float32(0.036)) * 0.40 * stress_mask, 0.0, 1.0)

    # Subsurface angled fracture plane echo (purely periodic roll + Gaussian blur)
    shift_x = max(2, int(round(9.0 * scale)))
    shift_y = max(2, int(round(13.0 * scale)))
    echo_raw = np.roll(c1, shift=(shift_y, shift_x), axis=(0, 1))
    subsurface_echo = common.periodic_gaussian_blur(echo_raw, max(1.0, 2.8 * scale))
    subsurface_echo2 = common.periodic_gaussian_blur(
        np.roll(c1, shift=(shift_y * 2, shift_x * 2), axis=(0, 1)), max(2.0, 5.5 * scale)
    )

    # Subtle trapped air bubbles frozen inside the ice sheet
    f1_bub, _, cid_bub, _, _ = common.periodic_voronoi(u, v, 34, 0.94, rng)
    bub_cluster = _smoothstep(0.42, 0.78, common.periodic_fbm(u, v, 6, 3, 0.5, rng))
    bubbles = _smoothstep(0.11, 0.03, f1_bub) * _smoothstep(0.52, 0.76, cid_bub) * bub_cluster

    # Per-plate gentle tilt & slight elevation offset across major fractures
    plate_ang = _hash_from_cid(cid1, 3.1) * np.float32(2.0 * math.pi)
    plate_tilt = (rx1 * np.cos(plate_ang) + ry1 * np.sin(plate_ang)) * np.float32(0.14)
    plate_step = (cid1 - np.float32(0.5)) * np.float32(0.08)
    swell = common.periodic_fbm(u, v, 4, 3, 0.48, rng, mode="fbm")
    micro_ice = common.periodic_fbm(u, v, 28, 3, 0.52, rng, mode="fbm")

    # Crisp V-groove cracks carved into glossy slightly tilted ice plates
    h_raw = (
        0.64
        + plate_tilt
        + plate_step
        + 0.10 * (swell - 0.5)
        + 0.04 * (micro_ice - 0.5)
        - 0.44 * c1
        - 0.24 * c2
        - 0.11 * c3
        - 0.04 * bubbles
    )
    h = common.robust_normalize(h_raw, low_pct=0.1, high_pct=99.9, soft=False)

    # Glassy smooth ice plates (0.04 - 0.14) vs frosted/crushed fracture lips (up to 0.78)
    plate_rough = 0.04 + 0.09 * swell + 0.05 * micro_ice
    frost_rough = 0.52 * cracks + 0.28 * crack_halo * (0.55 + 0.45 * micro_ice) + 0.16 * bubbles
    rough = np.clip(plate_rough + frost_rough, 0.04, 0.78).astype(np.float32)

    # Pale ice-blue base with bright frosty white fracture planes and subtle subsurface echo ribbons
    depth_field = common.robust_normalize(
        0.38 * swell + 0.22 * cid1 + 0.18 * (1.0 - subsurface_echo2 * 0.65) + 0.12 * micro_ice + 0.10 * h,
        low_pct=0.5,
        high_pct=99.5,
        soft=True,
    )
    albedo = _interp_stops(depth_field * np.float32(0.68), stops)

    # Subsurface bright-cyan/white fracture plane glow
    echo_mix = np.clip(subsurface_echo * 0.42 + subsurface_echo2 * 0.22, 0.0, 0.52)[..., None]
    albedo = albedo * (1.0 - echo_mix) + np.array([224.0, 240.0, 249.0], dtype=np.float32) * echo_mix

    # Sharp bright white fractured ice lines & bubbles
    white_ice = np.clip(cracks * 0.92 + crack_halo * 0.36 * (0.6 + 0.4 * micro_ice) + bubbles * 0.68, 0.0, 0.95)[..., None]
    albedo = albedo * (1.0 - white_ice) + np.array([248.0, 253.0, 255.0], dtype=np.float32) * white_ice

    base = common.shade_pbr_surface(
        albedo, h, roughness=rough, normal_strength=8.2, spec_intensity=0.28, ao_strength=0.12, cavity_strength=0.10
    )
    return base, h, rough


def _build_ice_sheet_blue(
    u: np.ndarray, v: np.ndarray, size: int, rng: np.random.Generator, stops: list[tuple[int, int, int]]
) -> tuple[Image.Image, np.ndarray, np.ndarray]:
    """Smooth glacial blue ice sheet with curl-flow undulations, trapped layers, and rime wisps."""
    _ = size
    # Broad smooth frozen meltwater swells via curl-advected domain warp
    flow_broad, wu, wv = common.periodic_domain_warp_fbm(
        u, v, freq=3, octaves=4, persistence=0.48, rng=rng, warp_strength=0.075, stages=2, mode="fbm", use_curl=True
    )
    uw = _wrap01(u + wu * np.float32(0.45))
    vw = _wrap01(v + wv * np.float32(0.45))

    # Subtle directional wind-glazed frozen ripples
    glaze_ripples = common.periodic_fbm(uw, vw, freqx=6, freqy=14, octaves=3, persistence=0.48, rng=rng, mode="fbm", rotate_octaves=False)
    # Delicate internal thermal veins (sparse, soft compared to cracked-ice)
    veins = common.periodic_domain_warp_fbm(
        u, v, freq=5, octaves=4, persistence=0.52, rng=rng, warp_strength=0.055, stages=1, mode="ridged", use_curl=True
    )[0]
    fine_veins = np.power(np.clip(veins, 0.0, 1.0), 4.2)

    # Delicate wind-blown rime / snow-dust wisps resting locally on smooth ice
    rime_base = common.periodic_domain_warp_fbm(
        u, v, freq=7, octaves=4, persistence=0.54, rng=rng, warp_strength=0.06, stages=1, mode="fbm", use_curl=True
    )[0]
    rime_wisp = _smoothstep(0.58, 0.86, rime_base)

    # Fine trapped bubble streaks along flow lines
    micro = common.periodic_fbm(uw, vw, 32, 3, 0.52, rng, mode="fbm")
    f1_b, _, cid_b, _, _ = common.periodic_voronoi(uw, vw, 28, 0.92, rng)
    bubbles = _smoothstep(0.09, 0.025, f1_b) * _smoothstep(0.62, 0.82, cid_b) * _smoothstep(0.35, 0.72, flow_broad)

    h = common.robust_normalize(
        0.50 * flow_broad + 0.26 * glaze_ripples + 0.12 * fine_veins + 0.07 * rime_wisp + 0.05 * micro,
        low_pct=0.25,
        high_pct=99.75,
        soft=True,
    )

    # Smooth glossy glacial ice (0.04 - 0.14) with localized rime wisps (up to 0.34)
    rough_raw = 0.22 * (1.0 - flow_broad) + 0.18 * glaze_ripples + 0.44 * rime_wisp + 0.16 * micro
    rough = common.match_histogram_range(rough_raw, 0.04, 0.34, target_mean=0.12, contrast=1.18)

    # Translucent deep-to-pale glacial blue depth variation + pale internal veins and rime
    depth = common.robust_normalize(
        0.46 * flow_broad + 0.22 * glaze_ripples + 0.16 * fine_veins + 0.12 * rime_wisp + 0.08 * micro,
        low_pct=0.4,
        high_pct=99.6,
        soft=True,
    )
    albedo = _interp_stops(depth, stops)
    highlight = np.clip(fine_veins * 0.38 + rime_wisp * 0.28 + bubbles * 0.55, 0.0, 0.70)[..., None]
    albedo = albedo * (1.0 - highlight) + np.array([234.0, 247.0, 254.0], dtype=np.float32) * highlight

    base = common.shade_pbr_surface(
        albedo, h, roughness=rough, normal_strength=9.5, spec_intensity=0.30, ao_strength=0.10, cavity_strength=0.07
    )
    return base, h, rough


def _build_window_frost(
    u: np.ndarray, v: np.ndarray, size: int, rng: np.random.Generator, stops: list[tuple[int, int, int]]
) -> tuple[Image.Image, np.ndarray, np.ndarray]:
    """Feathery dendritic ice-crystal frost fronds growing over smooth cold window glass."""
    _ = size
    # Curl-advected UV for graceful sweeping frost plumes
    vx, vy = common.periodic_curl_noise(u, v, 3, 3, 0.52, rng)
    uw = _wrap01(u + vx * np.float32(0.055))
    vw = _wrap01(v + vy * np.float32(0.055))

    # Branching frost colonies from toroidal Lloyd seeds
    f1_col, edge_col, _, rx_col, ry_col = common.periodic_lloyd_voronoi(uw, vw, 4, 0.90, rng, lloyd_iters=1)
    colony_ang = np.arctan2(ry_col, rx_col)
    radial_rays = np.float32(0.5) + np.float32(0.5) * np.cos(colony_ang * np.float32(14.0) + f1_col * np.float32(9.0))

    # Macro dendritic frost stems using analytic Swiss slope advection
    swiss_stems = common.periodic_fbm(uw, vw, 5, 5, 0.56, rng, mode="swiss")
    # Secondary curling feathery ridges
    ridged_fronds, _, _ = common.periodic_domain_warp_fbm(
        u, v, freq=11, octaves=5, persistence=0.58, rng=rng, warp_strength=0.052, stages=2, mode="ridged", use_curl=True
    )
    # Fine crystalline frost needle spikes (crossed directional Gabor + high-freq ridged)
    gabor_a = common.periodic_gabor_noise(uw, vw, grid_cells=18, f0=3.2, omega0=0.62, bandwidth=1.15, rng=rng, impulses_per_cell=2)
    gabor_b = common.periodic_gabor_noise(uw, vw, grid_cells=18, f0=3.2, omega0=-0.95, bandwidth=1.15, rng=rng, impulses_per_cell=2)
    needles = np.maximum(np.abs(gabor_a - np.float32(0.5)) * np.float32(2.0), np.abs(gabor_b - np.float32(0.5)) * np.float32(2.0))
    fine_crystal = common.periodic_fbm(uw, vw, 34, 4, 0.60, rng, mode="ridged")

    # Dendritic frond body combining macro stems, feathery branches, and needle spurs
    dendrite = common.robust_normalize(
        0.34 * swiss_stems
        + 0.30 * np.power(ridged_fronds, 1.5)
        + 0.18 * needles
        + 0.14 * fine_crystal
        + 0.08 * radial_rays * (1.0 - edge_col * 0.5),
        low_pct=0.4,
        high_pct=99.6,
        soft=True,
    )

    # Frost coverage mask leaving ~25% smooth clear cold glass patches peeking between frost fans
    growth_field = common.robust_normalize(
        0.48 * (1.0 - f1_col) + 0.34 * swiss_stems + 0.24 * ridged_fronds,
        low_pct=0.5,
        high_pct=99.5,
        soft=True,
    )
    frost_mask = _smoothstep(0.32, 0.62, growth_field)
    thin_rime = _smoothstep(0.22, 0.48, growth_field) * (1.0 - frost_mask * 0.5)

    # Flat cold glass baseline (0.04) under raised crisp dendritic frost crystals
    glass_wave = common.periodic_fbm(u, v, 3, 2, 0.45, rng, mode="fbm")
    frost_relief = frost_mask * (0.28 + 0.72 * dendrite) + thin_rime * 0.14 * fine_crystal
    h = common.robust_normalize(
        0.04 * glass_wave + 0.96 * frost_relief,
        low_pct=0.1,
        high_pct=99.9,
        soft=False,
    )

    # Stark physical roughness contrast: clear cold glass (0.04 - 0.10) vs matte crystalline frost (0.58 - 0.85)
    glass_r = 0.04 + 0.04 * glass_wave
    frost_r = 0.56 + 0.24 * dendrite + 0.08 * fine_crystal
    rime_r = 0.30 + 0.18 * fine_crystal
    rough = np.clip(
        glass_r * (1.0 - np.maximum(frost_mask, thin_rime))
        + rime_r * thin_rime * (1.0 - frost_mask)
        + frost_r * frost_mask,
        0.04,
        0.85,
    ).astype(np.float32)

    # Cool pale blue-tinted glass patches transitioning to crisp frosted ice-white fronds
    tone = np.clip(
        0.10 * glass_wave * (1.0 - frost_mask)
        + 0.36 * thin_rime
        + frost_mask * (0.48 + 0.52 * dendrite),
        0.0,
        1.0,
    )
    albedo = _interp_stops(tone, stops)
    base = common.shade_pbr_surface(
        albedo, h, roughness=rough, normal_strength=5.8, spec_intensity=0.22, ao_strength=0.10, cavity_strength=0.08
    )
    return base, h, rough


def _build_stained_glass_amber(
    u: np.ndarray, v: np.ndarray, size: int, rng: np.random.Generator, stops: list[tuple[int, int, int]]
) -> tuple[Image.Image, np.ndarray, np.ndarray]:
    """Warm rippled amber cathedral glass panels framed by raised dark lead came borders."""
    _ = size
    # Very slight hand-cut waviness on leaded pane contours
    du = (common.periodic_fbm(u, v, 5, 2, 0.48, rng) - np.float32(0.5)) * np.float32(0.014)
    dv = (common.periodic_fbm(u, v, 5, 2, 0.48, rng) - np.float32(0.5)) * np.float32(0.014)
    uw = _wrap01(u + du)
    vw = _wrap01(v + dv)

    f1, edge, cid, rx, ry = common.periodic_lloyd_voronoi(uw, vw, 7, 0.84, rng, lloyd_iters=1)

    # Raised half-round lead H-came cross section sqrt(1 - (edge/w)^2)
    came_w = np.float32(0.062)
    lead_t = np.clip(1.0 - edge / came_w, 0.0, 1.0)
    lead_arch = np.sqrt(np.clip(lead_t * (2.0 - lead_t), 0.0, 1.0)).astype(np.float32)
    lead_mask = _smoothstep(0.08, 0.32, lead_t)
    lead_bump = common.periodic_fbm(u, v, 32, 3, 0.55, rng, mode="fbm")

    # Mouth-blown antique cathedral glass ripples + subtle kiln-seeded bubbles inside each pane
    ripple, _, _ = common.periodic_domain_warp_fbm(
        u, v, freq=9, octaves=4, persistence=0.50, rng=rng, warp_strength=0.048, stages=1, mode="fbm", use_curl=True
    )
    fine_ripple = common.periodic_fbm(u, v, 24, 3, 0.48, rng, mode="fbm")
    f1_seed, _, cid_seed, _, _ = common.periodic_voronoi(u, v, 42, 0.94, rng)
    seed_bubbles = _smoothstep(0.11, 0.03, f1_seed) * _smoothstep(0.60, 0.82, cid_seed) * (1.0 - lead_mask)

    # Slight pane-to-pane glass tilt + pillowy thickness toward pane center
    pane_ang = _hash_from_cid(cid, 2.3) * np.float32(2.0 * math.pi)
    pane_tilt = (rx * np.cos(pane_ang) + ry * np.sin(pane_ang)) * np.float32(0.14)
    glass_h = np.clip(0.24 + 0.48 * ripple + 0.14 * fine_ripple + pane_tilt - 0.10 * f1 - 0.06 * seed_bubbles, 0.0, 1.0)

    # Combine raised soldered lead came and recessed rippled amber glass
    h = common.normalize_field(
        (1.0 - lead_mask) * (0.68 * glass_h) + lead_mask * (0.54 + 0.42 * lead_arch + 0.04 * lead_bump)
    )

    # Glossy rippled amber glass (0.05 - 0.16) vs matte oxidized pewter/lead came (0.58 - 0.78)
    glass_r = np.clip(0.05 + 0.07 * ripple + 0.04 * fine_ripple + 0.05 * seed_bubbles, 0.05, 0.18)
    lead_r = np.clip(0.58 + 0.18 * lead_bump + 0.04 * (1.0 - lead_arch), 0.56, 0.78)
    rough = ((1.0 - lead_mask) * glass_r + lead_mask * lead_r).astype(np.float32)

    # Rich amber/honey/cognac per-pane color variation + bright cathedral glass center glow
    pane_tint = _hash_from_cid(cid, 6.7)
    amber_t = common.robust_normalize(
        0.44 * pane_tint + 0.28 * ripple + 0.16 * (1.0 - f1) + 0.12 * fine_ripple,
        low_pct=0.4,
        high_pct=99.6,
        soft=True,
    )
    glass_rgb = _interp_stops(amber_t, stops)
    # Subtle golden glint on tiny kiln-seeded bubbles
    sb = (seed_bubbles * 0.35)[..., None]
    glass_rgb = glass_rgb * (1.0 - sb) + np.array([252.0, 228.0, 146.0], dtype=np.float32) * sb

    # Dark charcoal-pewter soldered lead came borders
    lead_val = np.clip(28.0 + 24.0 * lead_arch + 12.0 * (lead_bump - 0.5), 18.0, 66.0)[..., None]
    lead_rgb = lead_val * np.array([0.98, 0.96, 0.93], dtype=np.float32)

    lm = lead_mask[..., None]
    albedo = glass_rgb * (1.0 - lm) + lead_rgb * lm
    base = common.shade_pbr_surface(
        albedo, h, roughness=rough, normal_strength=9.5, spec_intensity=0.34, ao_strength=0.16, cavity_strength=0.14
    )
    return base, h, rough


def _build_glass_pebble(
    u: np.ndarray, v: np.ndarray, size: int, rng: np.random.Generator, stops: list[tuple[int, int, int]]
) -> tuple[Image.Image, np.ndarray, np.ndarray]:
    """Tightly packed smooth convex translucent glass pebbles over slightly frosted crevices."""
    _ = size
    # Subtle organic oval wobble so tumbled glass pebbles aren't rigidly polygonal
    du = (common.periodic_fbm(u, v, 5, 3, 0.50, rng) - np.float32(0.5)) * np.float32(0.024)
    dv = (common.periodic_fbm(u, v, 5, 3, 0.50, rng) - np.float32(0.5)) * np.float32(0.024)
    uw = _wrap01(u + du)
    vw = _wrap01(v + dv)

    # Primary packed rounded glass pebbles
    f1, edge, cid, _, _ = common.periodic_lloyd_voronoi(uw, vw, 8, 0.88, rng, lloyd_iters=2)
    # Smaller nested glass pebbles filling interstitial corners
    f1_s, edge_s, cid_s, _, _ = common.periodic_lloyd_voronoi(
        _wrap01(uw + 0.41), _wrap01(vw + 0.63), 16, 0.88, rng, lloyd_iters=1
    )

    # True continuous convex cabochon glass domes from rim to crown
    peb_mask = _smoothstep(0.026, 0.068, edge)
    e_norm = np.clip((edge - np.float32(0.020)) / np.float32(0.58), 0.0, 1.0)
    dome = np.sqrt(np.clip(e_norm * (2.0 - e_norm), 0.0, 1.0)) * (1.0 - 0.36 * f1 * f1)
    peb_scale = np.float32(0.84) + np.float32(0.16) * _hash_from_cid(cid, 4.4)
    h_primary = peb_mask * (0.20 + 0.80 * dome) * peb_scale

    sub_mask = _smoothstep(0.040, 0.084, edge_s) * (1.0 - peb_mask)
    es_norm = np.clip((edge_s - np.float32(0.036)) / np.float32(0.45), 0.0, 1.0)
    sub_dome = np.sqrt(np.clip(es_norm * (2.0 - es_norm), 0.0, 1.0)) * (1.0 - 0.32 * f1_s * f1_s)
    h_sub = sub_mask * (0.14 + 0.46 * sub_dome)

    # Subtle tumbled glass surface sheen & frosted interstitial grit
    glass_swirl = common.periodic_fbm(u, v, 14, 3, 0.48, rng, mode="fbm")
    crevice_grit = common.periodic_fbm(u, v, 54, 3, 0.58, rng, mode="fbm")
    any_peb = np.clip(peb_mask + sub_mask, 0.0, 1.0)

    h = common.normalize_field(
        np.maximum(h_primary, h_sub)
        + any_peb * 0.020 * glass_swirl
        + (1.0 - any_peb) * 0.07 * crevice_grit
    )

    # Smooth glossy glass crowns (0.06 - 0.18), scuffed pebble edges & frosted crevices (up to 0.68)
    edge_scuff = _smoothstep(0.24, 0.04, edge) * peb_mask
    crown_r = 0.06 + 0.08 * glass_swirl + 0.14 * edge_scuff
    crevice_r = 0.48 + 0.20 * crevice_grit
    rough = np.clip(any_peb * crown_r + (1.0 - any_peb) * crevice_r, 0.06, 0.68).astype(np.float32)

    # Pale translucent sea-glass / aqua-white tint per pebble with deeper cool recessed gaps
    p_tone = (
        peb_mask * (0.34 + 0.46 * _hash_from_cid(cid, 8.1) + 0.14 * dome + 0.08 * glass_swirl)
        + sub_mask * (0.28 + 0.42 * _hash_from_cid(cid_s, 9.5) + 0.12 * sub_dome) * (1.0 - peb_mask)
        + (1.0 - any_peb) * (0.06 + 0.14 * crevice_grit)
    )
    albedo = _interp_stops(np.clip(p_tone, 0.0, 1.0), stops)
    base = common.shade_pbr_surface(
        albedo, h, roughness=rough, normal_strength=10.5, spec_intensity=0.34, ao_strength=0.18, cavity_strength=0.14
    )
    return base, h, rough


def _build_mirror_flake(
    u: np.ndarray, v: np.ndarray, size: int, rng: np.random.Generator, stops: list[tuple[int, int, int]]
) -> tuple[Image.Image, np.ndarray, np.ndarray]:
    """Densely packed angular mirror shards tilted at independent planes with crisp thin seams."""
    _ = size
    # Angular Voronoi mirror shards (zero Lloyd relaxation preserves sharp irregular broken glass polygons)
    _, edge1, cid1, rx1, ry1 = common.periodic_voronoi(u, v, 13, 0.94, rng)
    # Secondary smaller splinter shards overlaid selectively for multi-scale flake sparkle
    _, edge2, cid2, rx2, ry2 = common.periodic_voronoi(_wrap01(u + 0.37), _wrap01(v + 0.59), 24, 0.94, rng)

    # Independent strongly tilted flat plane inside each primary mirror shard
    ang1 = _hash_from_cid(cid1, 1.7) * np.float32(2.0 * math.pi)
    mag1 = np.float32(0.42) + np.float32(0.46) * _hash_from_cid(cid1, 3.3)
    z0_1 = np.float32(0.56) + np.float32(0.14) * _hash_from_cid(cid1, 5.1)
    plane1 = np.clip(z0_1 + (rx1 * np.cos(ang1) + ry1 * np.sin(ang1)) * mag1, 0.18, 0.98)
    bevel1 = _smoothstep(0.008, 0.048, edge1)
    h1 = bevel1 * plane1 + (1.0 - bevel1) * (plane1 * np.float32(0.68))

    # Secondary smaller flakes active in ~38% of primary zones
    use_small = _hash_from_cid(cid1, 7.9) > np.float32(0.62)
    ang2 = _hash_from_cid(cid2, 2.5) * np.float32(2.0 * math.pi)
    mag2 = np.float32(0.44) + np.float32(0.46) * _hash_from_cid(cid2, 4.7)
    z0_2 = np.float32(0.56) + np.float32(0.14) * _hash_from_cid(cid2, 6.3)
    plane2 = np.clip(z0_2 + (rx2 * np.cos(ang2) + ry2 * np.sin(ang2)) * mag2, 0.18, 0.98)
    bevel2 = _smoothstep(0.010, 0.056, edge2)

    # Keep primary seam continuous even where sub-flakes subdivide a region
    b_sub = np.minimum(bevel1, bevel2)
    h_sub = b_sub * plane2 + (1.0 - b_sub) * (plane2 * np.float32(0.68))
    h_raw = np.where(use_small, h_sub, h1)
    bevel = np.where(use_small, b_sub, bevel1)
    ang = np.where(use_small, ang2, ang1)
    mag = np.where(use_small, mag2, mag1)
    cid = np.where(use_small, cid2, cid1)

    micro = common.periodic_fbm(u, v, 36, 2, 0.45, rng, mode="fbm")
    h = common.normalize_field(0.994 * h_raw + 0.006 * micro)

    # Near-zero mirror face roughness (0.02 - 0.07) vs crisp matte bevel/seam grooves (0.38 - 0.58)
    seam_mask = 1.0 - bevel
    face_r = 0.02 + 0.04 * _hash_from_cid(cid, 9.1) + 0.015 * micro
    seam_r = 0.40 + 0.18 * micro
    rough = np.clip(bevel * face_r + seam_mask * seam_r, 0.02, 0.58).astype(np.float32)

    # Directional mirror reflection glint per flake + dark hairline joints
    refl = np.clip(
        0.52 + 0.42 * np.cos(ang - np.float32(0.75)) * (0.45 + 0.65 * mag) + 0.10 * (cid - np.float32(0.5)),
        0.04,
        1.0,
    )
    flake_rgb = _interp_stops(refl, stops)
    seam_rgb = np.array([34.0, 38.0, 44.0], dtype=np.float32)[None, None, :]
    bm = bevel[..., None]
    albedo = flake_rgb * bm + seam_rgb * (1.0 - bm)

    base = common.shade_pbr_surface(
        albedo, h, roughness=rough, normal_strength=14.0, spec_intensity=0.40, ao_strength=0.14, cavity_strength=0.12
    )
    return base, h, rough


def _build_obsidian_glass(
    u: np.ndarray, v: np.ndarray, size: int, rng: np.random.Generator, stops: list[tuple[int, int, int]]
) -> tuple[Image.Image, np.ndarray, np.ndarray]:
    """Near-black volcanic glass with sweeping conchoidal Wallner fracture ripples and sharp ridges."""
    _ = size
    # Gentle organic warp for curved natural volcanic fracture boundaries
    du = (common.periodic_fbm(u, v, 3, 3, 0.50, rng) - np.float32(0.5)) * np.float32(0.055)
    dv = (common.periodic_fbm(u, v, 3, 3, 0.50, rng) - np.float32(0.5)) * np.float32(0.055)
    uw = _wrap01(u + du)
    vw = _wrap01(v + dv)

    # Primary bold conchoidal fracture scoops
    f1, edge, cid, rx, ry = common.periodic_lloyd_voronoi(uw, vw, 4, 0.90, rng, lloyd_iters=1)
    # Secondary overlapping chip scars
    f1_2, edge2, cid2, _, _ = common.periodic_lloyd_voronoi(
        _wrap01(uw + 0.43), _wrap01(vw + 0.61), 7, 0.92, rng, lloyd_iters=1
    )

    # Off-center conchoidal origin per fracture basin -> sweeping shell-like Wallner arcs
    ox = np.cos(_hash_from_cid(cid, 1.9) * np.float32(2.0 * math.pi)) * np.float32(0.26)
    oy = np.sin(_hash_from_cid(cid, 1.9) * np.float32(2.0 * math.pi)) * np.float32(0.26)
    dx = rx - ox
    dy = ry - oy
    r_loc = np.sqrt(dx * dx + dy * dy + np.float32(1e-6))
    phi_loc = np.arctan2(dy, dx)

    warp_arc = common.periodic_fbm(u, v, 5, 3, 0.48, rng, mode="fbm")
    # Smooth concave glassy scoop rising toward razor-sharp perimeter ridge
    scoop = np.clip(0.24 + 0.76 * np.power(np.clip(r_loc * 1.15, 0.0, 1.0), 1.35), 0.0, 1.0)
    ridge = np.exp(-edge / np.float32(0.045)).astype(np.float32)

    # Sweeping nested conchoidal Wallner ripples (shell-like curved wave steps)
    arc_freq = np.float32(11.0) + np.float32(7.0) * _hash_from_cid(cid, 4.3)
    phase = (r_loc * arc_freq + warp_arc * np.float32(0.45)) * np.float32(2.0 * math.pi)
    wallner_wave = np.float32(0.5) + np.float32(0.38) * np.sin(phase) + np.float32(0.12) * np.sin(phase * np.float32(2.0) + np.float32(0.7))
    # Attenuate ripples smoothly right at the center and right across the sharp boundary
    wallner_env = _smoothstep(0.10, 0.55, r_loc) * _smoothstep(0.015, 0.14, edge)
    wallner = wallner_wave * wallner_env

    # Subtle radiating hackle micro-striations near steep fracture margins
    hackle_env = _smoothstep(0.20, 0.03, edge) * _smoothstep(0.005, 0.035, edge)
    hackle = (np.float32(0.5) + np.float32(0.5) * np.sin(phi_loc * np.float32(28.0) + cid * np.float32(13.0))) * hackle_env

    # Secondary smaller conchoidal chip scars
    scar_mask = _smoothstep(0.58, 0.76, _hash_from_cid(cid2, 8.2))
    scar_ridge = np.exp(-edge2 / np.float32(0.038)).astype(np.float32) * scar_mask
    scar_bowl = (0.35 + 0.65 * f1_2) * scar_mask

    h_raw = (
        0.46 * scoop
        + 0.24 * ridge
        + 0.19 * wallner
        + 0.06 * hackle
        + 0.08 * scar_ridge
        - 0.05 * scar_bowl
        + 0.04 * (cid - np.float32(0.5))
    )
    h = common.robust_normalize(h_raw, low_pct=0.15, high_pct=99.85, soft=True)

    # Mirror-slick volcanic glass = near-zero roughness (0.02 - 0.16), slight micro-chip lift on razor ridges
    micro_dust = common.periodic_fbm(u, v, 42, 2, 0.50, rng, mode="fbm")
    rough_raw = 0.58 * ridge + 0.22 * scar_ridge + 0.14 * hackle + 0.06 * micro_dust
    rough = common.match_histogram_range(rough_raw, 0.02, 0.16, target_mean=0.05, contrast=1.40)

    # Near-black obsidian tint with crisp cool charcoal-slate highlights along fracture edges and Wallner crests
    tone = common.robust_normalize(
        0.38 * ridge + 0.26 * wallner + 0.20 * scoop + 0.10 * scar_ridge + 0.06 * hackle,
        low_pct=0.5,
        high_pct=99.5,
        soft=True,
    )
    albedo = _interp_stops(tone, stops)
    base = common.shade_pbr_surface(
        albedo, h, roughness=rough, normal_strength=8.2, spec_intensity=0.38, ao_strength=0.18, cavity_strength=0.14
    )
    return base, h, rough


_BUILDERS = (
    _build_frosted_glass,
    _build_crystal_facets_clear,
    _build_crystal_facets_smoky,
    _build_cracked_ice,
    _build_ice_sheet_blue,
    _build_window_frost,
    _build_stained_glass_amber,
    _build_glass_pebble,
    _build_mirror_flake,
    _build_obsidian_glass,
)


def _compute_bundle(
    seed: int = 1,
    palette: PaletteArg = None,
    size: int = 512,
    variant: int = 0,
) -> tuple[Image.Image, np.ndarray, np.ndarray, GlassStyleSpec]:
    style = GLASS_STYLES[int(variant) % len(GLASS_STYLES)]
    rng = np.random.default_rng(int(seed))
    u, v = common.make_uv_grid(int(size))
    stops = _resolve_glass_palette(palette, style.default_palette, rng)
    base, h, rough = _BUILDERS[style.variant](u, v, int(size), rng, stops)
    return base, np.asarray(h, dtype=np.float32), np.asarray(rough, dtype=np.float32), style


def render_glass_bundle(
    style_or_variant: str | int,
    seed: int = 1,
    size: int = 2048,
    palette: PaletteArg = None,
) -> tuple[Image.Image, Image.Image, Image.Image, np.ndarray, np.ndarray, GlassStyleSpec]:
    """Single-pass PBR bundle renderer returning (base, normal, roughness, height, rough_field, style)."""
    if isinstance(style_or_variant, str):
        style = STYLE_BY_SLUG[style_or_variant]
        variant = style.variant
    else:
        variant = int(style_or_variant) % len(GLASS_STYLES)
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
    name="glass-pack",
    kind="texture",
    tileable=True,
    origin="tools/generators/types/glass_pack.py",
    default_palette="frosted-glass",
    description=(
        "Glass, crystal, and ice surface suite (10 variants: frosted-glass, crystal-facets-clear, "
        "crystal-facets-smoky, cracked-ice, ice-sheet-blue, window-frost, stained-glass-amber, "
        "glass-pebble, mirror-flake, and obsidian-glass) with calibrated PBR normal and roughness."
    ),
    render_fn=generate,
    normal_source="native",
    normal_rule="Native per-style glass, crystal, and ice height field converted with calibrated Scharr normal strength.",
    normal_strength=6.0,
    height_fn=generate_height,
    normal_fn=generate_normal,
    roughness_fn=generate_roughness,
    default_roughness=0.18,
)
