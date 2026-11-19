"""Procedural neoprene, latex, vinyl & vulcanized rubber generator suite (10 styles).

Self-contained module built strictly on the committed periodic math primitives in
``tools/generators/common.py`` with all material palettes defined locally:
  - neoprene-smooth-black: closed-cell elastomer foam laminated with matte black micro-knit
  - neoprene-navy: closed-cell elastomer foam laminated with deep marine-navy micro-knit
  - neoprene-perforated-charcoal: staggered circular ventilation punches through charcoal neoprene
  - neoprene-heather-grey: marled heather-grey synthetic micro-knit laminated over elastomer foam
  - latex-gloss-black: mirror-gloss dipped black elastomer with liquid-smooth elastic billows
  - latex-gloss-red: mirror-gloss dipped crimson elastomer with supple drape ripples
  - rubber-matte-soft-touch: velvety bead-blasted matte vulcanized soft-touch elastomer
  - vinyl-gloss-clear: calendered optical-gloss clear PVC sheet with smooth roll ripples
  - rubber-grip-tread: molded alternating twin-bar anti-slip vulcanized rubber traction tread
  - rubberized-canvas-olive: elastomer-coated plain-weave olive canvas with flex creases

Roughness calibration defines the physical identity of this family:
  - Gloss latex and clear vinyl sit near-zero roughness (0.02-0.12).
  - Matte soft-touch rubber and laminated neoprene sit high and even (0.68-0.88).
  - Molded grip tread and rubberized canvas vary strongly by surface feature.
  - Normals stay smooth and subtle except on perforated, tread, and coated-weave variants.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from PIL import Image

from .. import common
from ..interface import GeneratorSpec, PaletteArg


# ---------------------------------------------------------------------------
# Self-contained 5-stop RGB palettes (darkest shadow -> brightest highlight)
# ---------------------------------------------------------------------------

RUBBER_PALETTES: dict[str, list[tuple[int, int, int]]] = {
    "neoprene_smooth_black": [
        (12, 13, 15),
        (20, 22, 26),
        (30, 33, 38),
        (42, 46, 52),
        (56, 61, 68),
    ],
    "neoprene_navy": [
        (10, 18, 34),
        (18, 32, 58),
        (28, 48, 84),
        (42, 68, 112),
        (62, 92, 138),
    ],
    "neoprene_perforated_charcoal": [
        (20, 22, 26),
        (36, 40, 46),
        (54, 59, 67),
        (74, 80, 90),
        (96, 103, 114),
    ],
    "neoprene_heather_grey": [
        (58, 62, 68),
        (94, 99, 106),
        (132, 138, 146),
        (168, 174, 182),
        (202, 207, 214),
    ],
    "latex_gloss_black": [
        (6, 7, 9),
        (12, 14, 18),
        (20, 23, 28),
        (30, 34, 42),
        (44, 50, 60),
    ],
    "latex_gloss_red": [
        (52, 6, 10),
        (102, 12, 18),
        (154, 18, 26),
        (196, 28, 38),
        (228, 52, 62),
    ],
    "rubber_matte_soft_touch": [
        (22, 24, 27),
        (34, 37, 41),
        (46, 50, 56),
        (60, 65, 72),
        (76, 82, 90),
    ],
    "vinyl_gloss_clear": [
        (148, 164, 178),
        (172, 188, 202),
        (196, 210, 222),
        (216, 228, 238),
        (234, 243, 250),
    ],
    "rubber_grip_tread": [
        (14, 15, 17),
        (26, 28, 32),
        (40, 44, 49),
        (58, 63, 70),
        (80, 86, 94),
    ],
    "rubberized_canvas_olive": [
        (26, 32, 22),
        (44, 54, 36),
        (66, 80, 52),
        (92, 108, 72),
        (122, 138, 96),
    ],
}


@dataclass(frozen=True)
class RubberStyleSpec:
    variant: int
    slug: str
    label: str
    default_palette: str
    normal_strength: float
    rough_range: tuple[float, float]
    description: str


RUBBER_STYLES: tuple[RubberStyleSpec, ...] = (
    RubberStyleSpec(
        0,
        "neoprene-smooth-black",
        "neoprene-smooth-black",
        "neoprene_smooth_black",
        0.55,
        (0.72, 0.84),
        "Closed-cell elastomer foam laminated with dense matte black micro-knit face, subtle spongy undulation, and even high roughness.",
    ),
    RubberStyleSpec(
        1,
        "neoprene-navy",
        "neoprene-navy",
        "neoprene_navy",
        0.58,
        (0.68, 0.82),
        "Closed-cell elastomer foam laminated with deep marine-navy synthetic micro-knit, smooth cushion billows, and even matte-satin roughness.",
    ),
    RubberStyleSpec(
        2,
        "neoprene-perforated-charcoal",
        "neoprene-perforated-charcoal",
        "neoprene_perforated_charcoal",
        5.2,
        (0.72, 0.95),
        "Charcoal laminated elastomer sheet punched with a staggered circular ventilation perforation grid, beveled foam rims, and deep matte bores.",
    ),
    RubberStyleSpec(
        3,
        "neoprene-heather-grey",
        "neoprene-heather-grey",
        "neoprene_heather_grey",
        0.58,
        (0.74, 0.88),
        "Closed-cell elastomer sheet laminated with marled heather-grey synthetic micro-knit showing multi-tone fiber melange and even matte roughness.",
    ),
    RubberStyleSpec(
        4,
        "latex-gloss-black",
        "latex-gloss-black",
        "latex_gloss_black",
        6.8,
        (0.03, 0.11),
        "High-gloss dipped black elastomer film with liquid-smooth elastic stretch billows, subtle leveled normals, and near-zero specular roughness.",
    ),
    RubberStyleSpec(
        5,
        "latex-gloss-red",
        "latex-gloss-red",
        "latex_gloss_red",
        6.8,
        (0.03, 0.12),
        "High-gloss dipped crimson elastomer sheet with supple liquid-smooth drape undulations, delicate normals, and near-zero roughness.",
    ),
    RubberStyleSpec(
        6,
        "rubber-matte-soft-touch",
        "rubber-matte-soft-touch",
        "rubber_matte_soft_touch",
        1.2,
        (0.76, 0.88),
        "Velvety bead-blasted matte vulcanized elastomer with ultra-fine powdery micro-grain, smooth normals, and tightly even high roughness.",
    ),
    RubberStyleSpec(
        7,
        "vinyl-gloss-clear",
        "vinyl-gloss-clear",
        "vinyl_gloss_clear",
        5.8,
        (0.02, 0.09),
        "Calendered high-gloss clear PVC sheet with smooth directional roll ripples, translucent cool optical sheen, and near-zero roughness.",
    ),
    RubberStyleSpec(
        8,
        "rubber-grip-tread",
        "rubber-grip-tread",
        "rubber_grip_tread",
        6.2,
        (0.40, 0.92),
        "Molded vulcanized rubber anti-slip sheet with alternating orthogonal twin traction lugs, burnished micro-stippled crowns, and matte drainage channels.",
    ),
    RubberStyleSpec(
        9,
        "rubberized-canvas-olive",
        "rubberized-canvas-olive",
        "rubberized_canvas_olive",
        1.45,
        (0.46, 0.80),
        "Heavy plain-weave olive canvas sealed under a matte-satin vulcanized elastomer coating with leveled weave relief and pale stress creases.",
    ),
)

STYLE_BY_SLUG: dict[str, RubberStyleSpec] = {s.slug: s for s in RUBBER_STYLES}


# ---------------------------------------------------------------------------
# Local palette resolution & shared periodic primitives (strictly on common.py)
# ---------------------------------------------------------------------------


def _resolve_local_palette(
    palette: PaletteArg,
    default_name: str,
    rng: np.random.Generator,
) -> list[tuple[int, int, int]]:
    """Resolve palettes from RUBBER_PALETTES first, falling back to common.resolve_palette."""
    if palette is None:
        return list(RUBBER_PALETTES[default_name])
    if isinstance(palette, str):
        key = palette.strip().lower().replace("-", "_")
        if key in RUBBER_PALETTES:
            return list(RUBBER_PALETTES[key])
        if key == "auto":
            names = sorted(RUBBER_PALETTES.keys())
            return list(RUBBER_PALETTES[names[int(rng.integers(0, len(names)))]])
    return common.resolve_palette(palette, "charcoal_flannel", rng)


def _wrap01(arr: np.ndarray) -> np.ndarray:
    """Safe float32 modulo [0, 1) avoiding float32 (-eps % 1.0 == 1.0)."""
    out = np.mod(arr.astype(np.float64), 1.0).astype(np.float32)
    return np.where(out >= 1.0, np.float32(0.0), out)


def _warp_uv(
    u: np.ndarray,
    v: np.ndarray,
    freq: int,
    amp: float,
    rng: np.random.Generator,
    octaves: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Gentle periodic sheet-tension UV warp preserving exact T^2 tiling."""
    du = common.periodic_fbm(u, v, freq, octaves, 0.50, rng, mode="fbm") - 0.5
    dv = common.periodic_fbm(u, v, freq, octaves, 0.50, rng, mode="fbm") - 0.5
    return _wrap01(u + du * float(amp)), _wrap01(v + dv * float(amp))


def _smooth_1d_periodic(n: int, rng: np.random.Generator, amp: float = 1.0) -> np.ndarray:
    """1D periodic per-row/per-column micro-variation with gentle neighbor correlation."""
    raw = rng.normal(0.0, 1.0, size=n).astype(np.float32)
    smooth = 0.72 * raw + 0.14 * np.roll(raw, 1) + 0.14 * np.roll(raw, -1)
    mx = float(np.percentile(np.abs(smooth), 98)) + 1e-6
    return np.clip(smooth / mx, -1.0, 1.0).astype(np.float32) * float(amp)


def _neoprene_micro_knit(
    u: np.ndarray,
    v: np.ndarray,
    size: int,
    wales: int,
    courses: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Ultra-fine laminated synthetic jersey face bonded to closed-cell elastomer foam.

    Returns ``(knit_height, stitch_tone)`` both on [0, 1] with exact T^2 periodicity.
    """
    uw, vw = _warp_uv(u, v, freq=3, amp=0.006, rng=rng, octaves=2)
    gx = uw * wales
    gy = vw * courses
    iw = np.floor(gx).astype(np.int32) % wales
    ic = np.floor(gy).astype(np.int32) % courses
    fx = (gx - np.floor(gx)).astype(np.float32)
    fy = (gy - np.floor(gy)).astype(np.float32)

    # Subtle per-wale and per-course tension/dye variation
    wale_var = _smooth_1d_periodic(wales, rng, amp=0.045)[iw]
    course_var = _smooth_1d_periodic(courses, rng, amp=0.045)[ic]
    rad = np.float32(0.215) * (1.0 + 0.25 * wale_var)

    # Fine interlocking synthetic V-loop legs over dense opaque foam backing
    xl = 0.245 + 0.115 * np.sin(fy * math.pi)
    xr = 1.0 - xl
    dl = (fx - xl) / rad
    dr = (fx - xr) / rad
    leg_l = np.sqrt(np.clip(1.0 - dl * dl, 0.0, 1.0))
    leg_r = np.sqrt(np.clip(1.0 - dr * dr, 0.0, 1.0))

    # Longitudinal arch plus smooth interlocking loop bridge so no raw gaps show through
    arch = 0.72 + 0.28 * np.sin(fy * math.pi)
    bridge = 0.42 + 0.14 * np.cos((fx - 0.5) * 2.0 * math.pi) * np.cos(fy * 2.0 * math.pi)
    knit_raw = np.maximum(np.maximum(leg_l, leg_r) * arch, bridge)

    # Soft bonding damping from elastic foam core lamination
    res_scale = float(size) / 512.0
    knit_h = common.periodic_gaussian_blur(knit_raw, sigma=max(0.35, 0.42 * res_scale))
    knit_h = common.robust_normalize(knit_h, low_pct=0.25, high_pct=99.75, soft=True)
    stitch_tone = np.clip(0.5 + wale_var * 2.2 + course_var * 2.0, 0.0, 1.0).astype(np.float32)
    return knit_h, stitch_tone


# ---------------------------------------------------------------------------
# Per-material procedural builders (exact 10 materials)
# ---------------------------------------------------------------------------


def _build_neoprene_smooth_black(
    u: np.ndarray,
    v: np.ndarray,
    size: int,
    rng: np.random.Generator,
    stops: Sequence[tuple[int, int, int]],
) -> tuple[Image.Image, np.ndarray, np.ndarray]:
    """Closed-cell elastomer foam laminated with dense matte black synthetic micro-knit."""
    res_scale = float(size) / 512.0
    knit_h, stitch_tone = _neoprene_micro_knit(u, v, size, wales=121, courses=157, rng=rng)
    cushion, _, _ = common.periodic_domain_warp_fbm(
        u, v, freq=3, octaves=3, persistence=0.50, rng=rng, warp_strength=0.055
    )
    cushion = common.periodic_gaussian_blur(cushion, sigma=max(1.2, 1.8 * res_scale))
    micro = common.periodic_fbm(u, v, 84, 3, 0.56, rng, mode="fbm")

    h = common.robust_normalize(
        0.28 * knit_h + 0.54 * cushion + 0.18 * micro,
        low_pct=0.25,
        high_pct=99.75,
        soft=True,
    )
    c = common.robust_normalize(
        0.38 * knit_h + 0.36 * cushion + 0.16 * micro + 0.10 * stitch_tone,
        low_pct=0.35,
        high_pct=99.65,
        soft=True,
    )
    rough_raw = 0.44 * (1.0 - knit_h) + 0.32 * (1.0 - cushion) + 0.24 * micro
    rough = common.match_histogram_range(rough_raw, 0.72, 0.84, target_mean=0.78, contrast=1.04)
    albedo = common.apply_palette_5stop(c, micro, stops, contrast=1.06, gamma=1.04)
    base = common.shade_pbr_surface(
        albedo, h, roughness=rough, normal_strength=0.55, spec_intensity=0.04, ao_strength=0.14, cavity_strength=0.10
    )
    return base, h, rough


def _build_neoprene_navy(
    u: np.ndarray,
    v: np.ndarray,
    size: int,
    rng: np.random.Generator,
    stops: Sequence[tuple[int, int, int]],
) -> tuple[Image.Image, np.ndarray, np.ndarray]:
    """Closed-cell elastomer foam laminated with deep marine-navy synthetic micro-knit."""
    res_scale = float(size) / 512.0
    knit_h, stitch_tone = _neoprene_micro_knit(u, v, size, wales=113, courses=147, rng=rng)
    cushion, _, _ = common.periodic_domain_warp_fbm(
        u, v, freq=3, octaves=3, persistence=0.52, rng=rng, warp_strength=0.06
    )
    cushion = common.periodic_gaussian_blur(cushion, sigma=max(1.2, 1.8 * res_scale))
    sheen_band = common.periodic_fbm(u, v, 5, 3, 0.50, rng, freqy=14, mode="fbm")
    micro = common.periodic_fbm(u, v, 76, 3, 0.55, rng, mode="fbm")

    h = common.robust_normalize(
        0.30 * knit_h + 0.52 * cushion + 0.18 * micro,
        low_pct=0.25,
        high_pct=99.75,
        soft=True,
    )
    c = common.robust_normalize(
        0.36 * knit_h + 0.32 * cushion + 0.14 * sheen_band + 0.10 * micro + 0.08 * stitch_tone,
        low_pct=0.35,
        high_pct=99.65,
        soft=True,
    )
    rough_raw = 0.42 * (1.0 - knit_h) + 0.30 * (1.0 - cushion) + 0.16 * (1.0 - sheen_band) + 0.12 * micro
    rough = common.match_histogram_range(rough_raw, 0.68, 0.82, target_mean=0.75, contrast=1.05)
    albedo = common.apply_palette_5stop(c, micro, stops, contrast=1.08, gamma=1.02)
    base = common.shade_pbr_surface(
        albedo, h, roughness=rough, normal_strength=0.58, spec_intensity=0.06, ao_strength=0.16, cavity_strength=0.12
    )
    return base, h, rough


def _build_neoprene_perforated_charcoal(
    u: np.ndarray,
    v: np.ndarray,
    size: int,
    rng: np.random.Generator,
    stops: Sequence[tuple[int, int, int]],
) -> tuple[Image.Image, np.ndarray, np.ndarray]:
    """Perforated charcoal elastomer sheet with staggered circular ventilation bores."""
    res_scale = float(size) / 512.0
    knit_h, stitch_tone = _neoprene_micro_knit(u, v, size, wales=117, courses=151, rng=rng)
    cushion, _, _ = common.periodic_domain_warp_fbm(
        u, v, freq=3, octaves=3, persistence=0.50, rng=rng, warp_strength=0.05
    )
    cushion = common.periodic_gaussian_blur(cushion, sigma=max(1.2, 1.8 * res_scale))
    micro = common.periodic_fbm(u, v, 80, 3, 0.58, rng, mode="fbm")

    # Staggered diamond punch lattice (21x21 odd base -> 882 circular holes at 2048px)
    n_grid = 21
    u_p = _wrap01(u + (cushion - 0.5) * 0.004)
    v_p = _wrap01(v + (cushion - 0.5) * 0.003)

    def _lattice_dist(uu: np.ndarray, vv: np.ndarray, shift: float) -> np.ndarray:
        gx = (uu * n_grid + shift) % n_grid
        gy = (vv * n_grid + shift) % n_grid
        ix = np.floor(gx).astype(np.int32) % n_grid
        iy = np.floor(gy).astype(np.int32) % n_grid
        fx = (gx - np.floor(gx) - 0.5).astype(np.float32)
        fy = (gy - np.floor(gy) - 0.5).astype(np.float32)
        jx = rng.uniform(-0.016, 0.016, size=(n_grid, n_grid)).astype(np.float32)[iy, ix]
        jy = rng.uniform(-0.016, 0.016, size=(n_grid, n_grid)).astype(np.float32)[iy, ix]
        return np.sqrt((fx - jx) ** 2 + (fy - jy) ** 2)

    d_a = _lattice_dist(u_p, v_p, 0.0)
    d_b = _lattice_dist(u_p, v_p, 0.5)
    d_hole = np.minimum(d_a, d_b).astype(np.float32)

    r_bore = 0.175
    r_chamfer = 0.240
    r_rim = 0.290

    hole_depth = np.clip((r_chamfer - d_hole) / (r_chamfer - r_bore), 0.0, 1.0)
    hole_smooth = hole_depth * hole_depth * (3.0 - 2.0 * hole_depth)
    bore_mask = np.clip((r_bore + 0.018 - d_hole) / 0.032, 0.0, 1.0)
    chamfer_ring = np.clip(hole_smooth * (1.0 - bore_mask) * 2.0, 0.0, 1.0)
    punch_rim = np.exp(-np.power((d_hole - r_rim) / 0.036, 2.0)) * 0.06

    face_h = common.robust_normalize(
        0.30 * knit_h + 0.52 * cushion + 0.18 * micro,
        low_pct=0.3,
        high_pct=99.7,
        soft=True,
    )
    h = common.robust_normalize(
        (0.80 + 0.20 * face_h + punch_rim) * (1.0 - 0.90 * hole_smooth),
        low_pct=0.1,
        high_pct=99.9,
        soft=False,
    )

    c_face = common.robust_normalize(
        0.38 * knit_h + 0.34 * cushion + 0.16 * micro + 0.12 * stitch_tone,
        low_pct=0.35,
        high_pct=99.65,
        soft=True,
    )
    albedo_face = common.apply_palette_5stop(c_face, micro, stops, contrast=1.08, gamma=1.04)
    bore_rgb = np.array([8.0, 9.0, 11.0], dtype=np.float32)[None, None, :]
    albedo = albedo_face * (1.0 - 0.26 * chamfer_ring[..., None])
    albedo = albedo * (1.0 - 0.92 * bore_mask[..., None]) + bore_rgb * (0.92 * bore_mask[..., None])

    rough_face = common.match_histogram_range(
        0.44 * (1.0 - knit_h) + 0.32 * (1.0 - cushion) + 0.24 * micro,
        0.72,
        0.82,
        target_mean=0.77,
        contrast=1.04,
    )
    rough = np.clip(
        rough_face * (1.0 - hole_smooth) + 0.87 * chamfer_ring + 0.94 * bore_mask,
        0.72,
        0.95,
    ).astype(np.float32)

    base = common.shade_pbr_surface(
        np.clip(albedo, 0.0, 255.0),
        h,
        roughness=rough,
        normal_strength=5.2,
        spec_intensity=0.05,
        ao_strength=0.22,
        cavity_strength=0.18,
    )
    return base, h, rough


def _build_neoprene_heather_grey(
    u: np.ndarray,
    v: np.ndarray,
    size: int,
    rng: np.random.Generator,
    stops: Sequence[tuple[int, int, int]],
) -> tuple[Image.Image, np.ndarray, np.ndarray]:
    """Closed-cell elastomer laminated with marled heather-grey synthetic micro-knit."""
    res_scale = float(size) / 512.0
    knit_h, stitch_tone = _neoprene_micro_knit(u, v, size, wales=119, courses=153, rng=rng)
    cushion, _, _ = common.periodic_domain_warp_fbm(
        u, v, freq=3, octaves=3, persistence=0.52, rng=rng, warp_strength=0.06
    )
    cushion = common.periodic_gaussian_blur(cushion, sigma=max(1.2, 1.8 * res_scale))
    micro = common.periodic_fbm(u, v, 86, 3, 0.58, rng, mode="fbm")

    # Authentic multi-tone heather melange fiber specks aligned softly along knit courses
    p1 = common.periodic_rotated_perlin(
        u, v, 58, 0.38, rng, phase_u=float(rng.uniform()), phase_v=float(rng.uniform())
    )
    p2 = common.periodic_rotated_perlin(
        u, v, 92, 1.22, rng, phase_u=float(rng.uniform()), phase_v=float(rng.uniform())
    )
    course_heather = common.periodic_fbm(u, v, 18, 3, 0.56, rng, freqy=124, mode="fbm")
    melange = common.robust_normalize(
        0.34 * p1 + 0.30 * p2 + 0.24 * course_heather + 0.12 * cushion,
        low_pct=0.4,
        high_pct=99.6,
        soft=True,
    )

    h = common.robust_normalize(
        0.30 * knit_h + 0.52 * cushion + 0.18 * micro,
        low_pct=0.25,
        high_pct=99.75,
        soft=True,
    )
    c = common.robust_normalize(
        0.50 * melange + 0.24 * knit_h + 0.14 * cushion + 0.12 * stitch_tone,
        low_pct=0.35,
        high_pct=99.65,
        soft=True,
    )
    rough_raw = 0.42 * (1.0 - knit_h) + 0.30 * (1.0 - cushion) + 0.16 * micro + 0.12 * (1.0 - melange)
    rough = common.match_histogram_range(rough_raw, 0.74, 0.88, target_mean=0.81, contrast=1.05)
    albedo = common.apply_palette_5stop(c, micro, stops, contrast=1.08, gamma=0.98)
    base = common.shade_pbr_surface(
        albedo, h, roughness=rough, normal_strength=0.58, spec_intensity=0.035, ao_strength=0.14, cavity_strength=0.10
    )
    return base, h, rough


def _build_latex_gloss_black(
    u: np.ndarray,
    v: np.ndarray,
    size: int,
    rng: np.random.Generator,
    stops: Sequence[tuple[int, int, int]],
) -> tuple[Image.Image, np.ndarray, np.ndarray]:
    """High-gloss dipped jet-black elastomer sheet with liquid-smooth elastic billows."""
    res_scale = float(size) / 512.0
    billow, _, _ = common.periodic_domain_warp_fbm(
        u, v, freq=3, octaves=3, persistence=0.48, rng=rng, warp_strength=0.08
    )
    stretch = common.periodic_fbm(u, v, 2, 3, 0.46, rng, freqy=5, mode="fbm")
    billow_s = common.periodic_gaussian_blur(billow, sigma=max(2.2, 3.4 * res_scale))
    stretch_s = common.periodic_gaussian_blur(stretch, sigma=max(1.8, 2.8 * res_scale))
    micro_sheen = common.periodic_gaussian_blur(
        common.periodic_fbm(u, v, 18, 2, 0.42, rng, mode="fbm"),
        sigma=max(1.0, 1.5 * res_scale),
    )

    h = common.robust_normalize(
        0.56 * billow_s + 0.34 * stretch_s + 0.10 * micro_sheen,
        low_pct=0.25,
        high_pct=99.75,
        soft=True,
    )
    c = common.robust_normalize(
        0.54 * billow_s + 0.36 * stretch_s + 0.10 * micro_sheen,
        low_pct=0.35,
        high_pct=99.65,
        soft=True,
    )
    smudge = common.periodic_gaussian_blur(
        common.periodic_fbm(u, v, 7, 3, 0.50, rng, mode="fbm"),
        sigma=max(1.4, 2.0 * res_scale),
    )
    rough_raw = 0.56 * smudge + 0.24 * (1.0 - billow_s) + 0.20 * micro_sheen
    rough = common.match_histogram_range(rough_raw, 0.03, 0.11, target_mean=0.06, contrast=1.08)
    albedo = common.apply_palette_5stop(c, None, stops, contrast=1.08, gamma=1.12)
    base = common.shade_pbr_surface(
        albedo, h, roughness=rough, normal_strength=6.8, spec_intensity=0.58, ao_strength=0.08, cavity_strength=0.05
    )
    return base, h, rough


def _build_latex_gloss_red(
    u: np.ndarray,
    v: np.ndarray,
    size: int,
    rng: np.random.Generator,
    stops: Sequence[tuple[int, int, int]],
) -> tuple[Image.Image, np.ndarray, np.ndarray]:
    """High-gloss dipped crimson elastomer sheet with supple liquid-smooth drape waves."""
    res_scale = float(size) / 512.0
    billow, _, _ = common.periodic_domain_warp_fbm(
        u, v, freq=3, octaves=3, persistence=0.48, rng=rng, warp_strength=0.085
    )
    stretch = common.periodic_fbm(u, v, 5, 3, 0.46, rng, freqy=2, mode="fbm")
    billow_s = common.periodic_gaussian_blur(billow, sigma=max(2.2, 3.2 * res_scale))
    stretch_s = common.periodic_gaussian_blur(stretch, sigma=max(1.8, 2.6 * res_scale))
    micro_sheen = common.periodic_gaussian_blur(
        common.periodic_fbm(u, v, 18, 2, 0.42, rng, mode="fbm"),
        sigma=max(1.0, 1.5 * res_scale),
    )
    body_depth = common.periodic_gaussian_blur(
        common.periodic_fbm(u, v, 4, 3, 0.50, rng, mode="fbm"),
        sigma=max(1.8, 2.6 * res_scale),
    )

    h = common.robust_normalize(
        0.54 * billow_s + 0.36 * stretch_s + 0.10 * micro_sheen,
        low_pct=0.25,
        high_pct=99.75,
        soft=True,
    )
    c = common.robust_normalize(
        0.44 * billow_s + 0.28 * stretch_s + 0.20 * body_depth + 0.08 * micro_sheen,
        low_pct=0.35,
        high_pct=99.65,
        soft=True,
    )
    smudge = common.periodic_gaussian_blur(
        common.periodic_fbm(u, v, 7, 3, 0.50, rng, mode="fbm"),
        sigma=max(1.4, 2.0 * res_scale),
    )
    rough_raw = 0.54 * smudge + 0.26 * (1.0 - billow_s) + 0.20 * micro_sheen
    rough = common.match_histogram_range(rough_raw, 0.03, 0.12, target_mean=0.065, contrast=1.08)
    albedo = common.apply_palette_5stop(c, None, stops, contrast=1.10, gamma=1.04)
    base = common.shade_pbr_surface(
        albedo, h, roughness=rough, normal_strength=6.8, spec_intensity=0.52, ao_strength=0.09, cavity_strength=0.05
    )
    return base, h, rough


def _build_rubber_matte_soft_touch(
    u: np.ndarray,
    v: np.ndarray,
    size: int,
    rng: np.random.Generator,
    stops: Sequence[tuple[int, int, int]],
) -> tuple[Image.Image, np.ndarray, np.ndarray]:
    """Velvety bead-blasted matte vulcanized elastomer with soft-touch surface grain."""
    res_scale = float(size) / 512.0
    grain_a = common.periodic_rotated_perlin(
        u, v, 96, 0.42, rng, phase_u=float(rng.uniform()), phase_v=float(rng.uniform())
    )
    grain_b = common.periodic_rotated_perlin(
        u, v, 134, 1.19, rng, phase_u=float(rng.uniform()), phase_v=float(rng.uniform())
    )
    micro_pebble = common.periodic_fbm(u, v, 64, 3, 0.56, rng, mode="fbm")
    velvet = common.robust_normalize(
        0.40 * grain_a + 0.35 * grain_b + 0.25 * micro_pebble,
        low_pct=0.3,
        high_pct=99.7,
        soft=True,
    )
    velvet_soft = common.periodic_gaussian_blur(velvet, sigma=max(0.35, 0.45 * res_scale))

    mold_cloud, _, _ = common.periodic_domain_warp_fbm(
        u, v, freq=3, octaves=3, persistence=0.50, rng=rng, warp_strength=0.06
    )
    mold_cloud = common.periodic_gaussian_blur(mold_cloud, sigma=max(1.5, 2.2 * res_scale))

    h = common.robust_normalize(
        0.56 * mold_cloud + 0.44 * velvet_soft,
        low_pct=0.3,
        high_pct=99.7,
        soft=True,
    )
    c = common.robust_normalize(
        0.54 * mold_cloud + 0.34 * velvet_soft + 0.12 * micro_pebble,
        low_pct=0.35,
        high_pct=99.65,
        soft=True,
    )
    rough_raw = 0.45 * (1.0 - velvet_soft) + 0.35 * (1.0 - mold_cloud) + 0.20 * micro_pebble
    rough = common.match_histogram_range(rough_raw, 0.76, 0.88, target_mean=0.82, contrast=1.02)
    albedo = common.apply_palette_5stop(c, micro_pebble, stops, contrast=1.04, gamma=1.02)
    base = common.shade_pbr_surface(
        albedo, h, roughness=rough, normal_strength=1.2, spec_intensity=0.04, ao_strength=0.12, cavity_strength=0.08
    )
    return base, h, rough


def _build_vinyl_gloss_clear(
    u: np.ndarray,
    v: np.ndarray,
    size: int,
    rng: np.random.Generator,
    stops: Sequence[tuple[int, int, int]],
) -> tuple[Image.Image, np.ndarray, np.ndarray]:
    """Calendered optical-gloss clear PVC vinyl sheet with smooth extrusion ripples."""
    res_scale = float(size) / 512.0
    billow, _, _ = common.periodic_domain_warp_fbm(
        u, v, freq=2, octaves=3, persistence=0.45, rng=rng, warp_strength=0.07
    )
    calender = common.periodic_fbm(u, v, 3, 3, 0.44, rng, freqy=8, mode="fbm")
    billow_s = common.periodic_gaussian_blur(billow, sigma=max(2.4, 3.8 * res_scale))
    calender_s = common.periodic_gaussian_blur(calender, sigma=max(1.8, 2.8 * res_scale))

    curl_u, curl_v = common.periodic_curl_noise(u, v, 3, 2, 0.48, rng)
    u_c = _wrap01(u + curl_u * 0.035)
    v_c = _wrap01(v + curl_v * 0.035)
    optical = common.periodic_gaussian_blur(
        common.periodic_fbm(u_c, v_c, 4, 3, 0.48, rng, mode="fbm"),
        sigma=max(1.8, 2.6 * res_scale),
    )

    h = common.robust_normalize(
        0.50 * billow_s + 0.34 * calender_s + 0.16 * optical,
        low_pct=0.25,
        high_pct=99.75,
        soft=True,
    )
    c = common.robust_normalize(
        0.40 * optical + 0.35 * billow_s + 0.25 * calender_s,
        low_pct=0.35,
        high_pct=99.65,
        soft=True,
    )
    haze = common.periodic_gaussian_blur(
        common.periodic_fbm(u, v, 6, 3, 0.48, rng, mode="fbm"),
        sigma=max(1.5, 2.2 * res_scale),
    )
    micro_streak = common.periodic_fbm(u, v, 14, 2, 0.45, rng, freqy=4, mode="fbm")
    rough_raw = 0.62 * haze + 0.22 * micro_streak + 0.16 * (1.0 - optical)
    rough = common.match_histogram_range(rough_raw, 0.02, 0.09, target_mean=0.045, contrast=1.06)
    albedo = common.apply_palette_5stop(c, None, stops, contrast=1.06, gamma=0.98)
    base = common.shade_pbr_surface(
        albedo, h, roughness=rough, normal_strength=5.8, spec_intensity=0.40, ao_strength=0.05, cavity_strength=0.03
    )
    return base, h, rough


def _build_rubber_grip_tread(
    u: np.ndarray,
    v: np.ndarray,
    size: int,
    rng: np.random.Generator,
    stops: Sequence[tuple[int, int, int]],
) -> tuple[Image.Image, np.ndarray, np.ndarray]:
    """Molded vulcanized rubber anti-slip grip tread with alternating orthogonal twin lugs."""
    # Even cell count (14 = 2 * 7) guarantees exact checkerboard parity across the T^2 wrap
    # while odd supercell count (7) avoids half-tile quadrant phase lock.
    n_cells = 14
    gx = u * n_cells
    gy = v * n_cells
    ix = np.floor(gx).astype(np.int32) % n_cells
    iy = np.floor(gy).astype(np.int32) % n_cells
    lx = (gx - np.floor(gx) - 0.5).astype(np.float32)
    ly = (gy - np.floor(gy) - 0.5).astype(np.float32)

    horiz = ((ix + iy) & 1) == 0
    along = np.where(horiz, np.abs(lx), np.abs(ly))
    across = np.where(horiz, np.abs(ly), np.abs(lx))

    # Paired parallel molded traction ribs per cell with tapered lozenge ends
    across_twin = np.abs(across - 0.19)
    u_n = along / 0.41
    v_n = across_twin / 0.115
    p_exp = 2.2
    r_lug = np.power(np.power(u_n, p_exp) + np.power(v_n, p_exp) + 1e-8, 1.0 / p_exp)

    # Smooth beveled sidewalls + flat micro-stippled lug crowns
    d_in = 1.0 - r_lug
    lug_t = np.clip((d_in + 0.06) / 0.26, 0.0, 1.0)
    lug_h = (lug_t * lug_t * (3.0 - 2.0 * lug_t)).astype(np.float32)
    crown_mask = np.clip((d_in - 0.14) / 0.10, 0.0, 1.0).astype(np.float32)
    bevel_mask = np.clip(lug_h * (1.0 - lug_h) * 4.0, 0.0, 1.0).astype(np.float32)

    # Subtle per-cell mold height & scuff variation
    cell_var = rng.uniform(0.95, 1.04, size=(n_cells, n_cells)).astype(np.float32)[iy, ix]
    micro_knurl = common.periodic_fbm(u, v, 96, 3, 0.58, rng, mode="fbm")
    floor_grain = common.periodic_fbm(u, v, 56, 3, 0.60, rng, mode="turbulence")
    wear, _, _ = common.periodic_domain_warp_fbm(
        u, v, freq=3, octaves=4, persistence=0.54, rng=rng, warp_strength=0.08
    )

    h = common.robust_normalize(
        0.78 * lug_h * cell_var
        + 0.08 * crown_mask * micro_knurl
        + 0.08 * (1.0 - lug_h) * floor_grain
        + 0.05 * wear,
        low_pct=0.1,
        high_pct=99.9,
        soft=False,
    )
    c = common.robust_normalize(
        0.54 * lug_h
        + 0.18 * crown_mask * wear
        + 0.12 * micro_knurl * lug_h
        + 0.10 * (1.0 - lug_h) * floor_grain
        + 0.06 * bevel_mask,
        low_pct=0.25,
        high_pct=99.75,
        soft=True,
    )

    # Feature-driven roughness: burnished satin lug crowns, mid-matte bevels, matte floor
    rough_crown = 0.42 + 0.14 * (0.55 * (1.0 - wear) + 0.45 * micro_knurl)
    rough_floor = 0.80 + 0.12 * (0.55 * floor_grain + 0.45 * (1.0 - wear))
    rough = rough_floor * (1.0 - lug_h) + rough_crown * lug_h + 0.12 * bevel_mask
    rough = np.clip(rough, 0.40, 0.92).astype(np.float32)

    albedo = common.apply_palette_5stop(c, micro_knurl, stops, contrast=1.12, gamma=1.06)
    base = common.shade_pbr_surface(
        albedo, h, roughness=rough, normal_strength=6.2, spec_intensity=0.14, ao_strength=0.30, cavity_strength=0.24
    )
    return base, h, rough


def _build_rubberized_canvas_olive(
    u: np.ndarray,
    v: np.ndarray,
    size: int,
    rng: np.random.Generator,
    stops: Sequence[tuple[int, int, int]],
) -> tuple[Image.Image, np.ndarray, np.ndarray]:
    """Elastomer-coated plain-weave olive canvas with leveled weave relief & flex creases."""
    res_scale = float(size) / 512.0
    uw, vw = _warp_uv(u, v, freq=3, amp=0.009, rng=rng, octaves=3)
    plain_1x1 = np.array([[1, 0], [0, 1]], dtype=np.int32)
    tu, tv = 96, 92

    warp_slub = common.periodic_fbm(uw, vw, tu, 3, 0.54, rng, mode="fbm", freqy=6)
    weft_slub = common.periodic_fbm(uw, vw, 6, 2, 0.54, rng, mode="fbm", freqy=tv)
    warp_rad = np.clip(0.92 + 0.09 * (warp_slub - 0.5), 0.80, 1.00).astype(np.float32)
    weft_rad = np.clip(0.92 + 0.09 * (weft_slub - 0.5), 0.80, 1.00).astype(np.float32)

    weave_h, _, _, _ = common.woven_fabric_field(
        uw, vw, tu, tv, plain_1x1, float_arch=0.38, ply_twist=0.06, warp_radius=warp_rad, weft_radius=weft_rad
    )

    # Vulcanized elastomer coating bridges deep inter-yarn pores while preserving woven texture
    coated_fill = common.periodic_gaussian_blur(weave_h, sigma=max(0.8, 1.35 * res_scale))
    coated_weave = (0.42 * weave_h + 0.58 * coated_fill).astype(np.float32)

    rubber_micro = common.periodic_fbm(u, v, 72, 3, 0.56, rng, mode="fbm")
    patina, _, _ = common.periodic_domain_warp_fbm(
        u, v, freq=3, octaves=4, persistence=0.54, rng=rng, warp_strength=0.08
    )

    # Curl-advected pale-olive elastomer flex creases
    cu, cv = common.periodic_curl_noise(u, v, 3, 3, 0.52, rng)
    uc = _wrap01(u + cu * 0.042)
    vc = _wrap01(v + cv * 0.042)
    crease_broad = common.periodic_fbm(uc, vc, 4, 3, 0.52, rng, mode="ridged")
    crease_fine = common.periodic_fbm(uc, vc, 9, 3, 0.54, rng, mode="ridged")
    crease_lines = np.clip(
        0.60 * np.power(crease_broad, 3.6) + 0.40 * np.power(crease_fine, 4.0),
        0.0,
        1.0,
    ).astype(np.float32)

    h = common.robust_normalize(
        0.54 * coated_weave + 0.14 * rubber_micro + 0.22 * patina - 0.14 * crease_lines,
        low_pct=0.25,
        high_pct=99.75,
        soft=True,
    )
    c = common.robust_normalize(
        0.36 * coated_weave + 0.28 * patina + 0.24 * crease_lines + 0.12 * rubber_micro,
        low_pct=0.35,
        high_pct=99.65,
        soft=True,
    )
    rough_raw = (
        0.36 * (1.0 - coated_weave)
        + 0.28 * crease_lines
        + 0.20 * (1.0 - patina)
        + 0.16 * rubber_micro
    )
    rough = common.match_histogram_range(rough_raw, 0.46, 0.80, target_mean=0.62, contrast=1.10)
    albedo = common.apply_palette_5stop(c, rubber_micro, stops, contrast=1.10, gamma=1.02)
    base = common.shade_pbr_surface(
        albedo, h, roughness=rough, normal_strength=1.45, spec_intensity=0.14, ao_strength=0.22, cavity_strength=0.18
    )
    return base, h, rough


_BUILDERS = (
    _build_neoprene_smooth_black,
    _build_neoprene_navy,
    _build_neoprene_perforated_charcoal,
    _build_neoprene_heather_grey,
    _build_latex_gloss_black,
    _build_latex_gloss_red,
    _build_rubber_matte_soft_touch,
    _build_vinyl_gloss_clear,
    _build_rubber_grip_tread,
    _build_rubberized_canvas_olive,
)


def _compute_bundle(
    seed: int = 1,
    palette: PaletteArg = None,
    size: int = 512,
    variant: int = 0,
) -> tuple[Image.Image, np.ndarray, np.ndarray, RubberStyleSpec]:
    style = RUBBER_STYLES[int(variant) % len(RUBBER_STYLES)]
    rng = np.random.default_rng(int(seed))
    u, v = common.make_uv_grid(int(size))
    stops = _resolve_local_palette(palette, style.default_palette, rng)
    base, h, rough = _BUILDERS[style.variant](u, v, int(size), rng, stops)
    return base, np.asarray(h, dtype=np.float32), np.asarray(rough, dtype=np.float32), style


def render_rubber_bundle(
    style_or_variant: str | int,
    seed: int = 1,
    size: int = 2048,
    palette: PaletteArg = None,
) -> tuple[Image.Image, Image.Image, Image.Image, np.ndarray, np.ndarray, RubberStyleSpec]:
    """Single-pass PBR bundle renderer returning (base, normal, roughness, height, rough_field, style)."""
    if isinstance(style_or_variant, str):
        style = STYLE_BY_SLUG[style_or_variant]
        variant = style.variant
    else:
        variant = int(style_or_variant) % len(RUBBER_STYLES)
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
    name="rubber-pack",
    kind="texture",
    tileable=True,
    origin="tools/generators/types/rubber_pack.py",
    default_palette="neoprene_smooth_black",
    description=(
        "Neoprene & Rubber procedural suite (10 styles: neoprene-smooth-black, neoprene-navy, "
        "neoprene-perforated-charcoal, neoprene-heather-grey, latex-gloss-black, latex-gloss-red, "
        "rubber-matte-soft-touch, vinyl-gloss-clear, rubber-grip-tread, and rubberized-canvas-olive) "
        "with calibrated PBR height, normal, and roughness."
    ),
    render_fn=generate,
    normal_source="native",
    normal_rule="Native per-style elastomer, micro-knit, perforation, and molded tread height field converted via Scharr normals.",
    normal_strength=2.2,
    height_fn=generate_height,
    normal_fn=generate_normal,
    roughness_fn=generate_roughness,
    default_roughness=0.55,
)
