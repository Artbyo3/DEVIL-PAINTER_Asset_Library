"""Procedural clothing knits & fleece generator suite (12 physically grounded styles).

Built strictly on the committed periodic math primitives in ``tools/generators/common.py``:
  - Interlocking Chamberlain/Leaf-Glaskin V-loop knit legs, horizontal semicircular purl
    arches, braided Aran rope cables with curve-aligned sub-wales, loopback French terry
    pile curls, thermal waffle & hexagonal honeycomb tuck-stitch pockets, and staggered
    shell-cluster crochet.
  - Multi-orientation sparse Gabor fibers, curl-advected brushed mohair halos, and
    domain-warped R2 Lloyd CVT curly sherpa / anti-pill polar fleece tufts.
  - 2D per-stitch organic tension/slub irregularity, gentle low-frequency garment drape
    warp, Nyquist-aware rotated fBm fiber grain, and fine micro-fuzz so 2048px renders
    hold authentic textile fiber scale without synthetic plastic smoothing.
  - Calibrated per-style matte textile PBR roughness via ``match_histogram_range`` and
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
class KnitStyleSpec:
    variant: int
    slug: str
    label: str
    default_palette: str
    normal_strength: float
    rough_range: tuple[float, float]
    description: str


KNIT_STYLES: tuple[KnitStyleSpec, ...] = (
    KnitStyleSpec(
        0,
        "knit-jersey-heather-grey",
        "Combed Jersey Heather Grey",
        "heather_cotton_jersey",
        3.8,
        (0.76, 0.92),
        "Fine-gauge combed cotton single-jersey knit with interlocking V-loops, heather melange specks, and soft yarn slub.",
    ),
    KnitStyleSpec(
        1,
        "knit-rib-2x2-forest",
        "2x2 Rib Knit Forest",
        "forest_pine_ribknit",
        5.0,
        (0.74, 0.93),
        "Elastic 2x2 sweater rib alternating raised twin V-loop wales and deeply recessed horizontal purl-bump troughs.",
    ),
    KnitStyleSpec(
        2,
        "knit-cable-aran-cream",
        "Aran Cable Knit Cream",
        "aran_natural_wool",
        5.6,
        (0.76, 0.94),
        "Heavy fisherman Aran wool cable knit pairing braided rope plaits with curve-aligned sub-wales and purl seed gutters.",
    ),
    KnitStyleSpec(
        3,
        "knit-stockinette-merino-navy",
        "Merino Stockinette Navy",
        "merino_midnight_navy",
        4.6,
        (0.70, 0.90),
        "Smooth medium-gauge merino sweater stockinette with plush twisted yarn legs, subtle course tension drift, and wool sheen.",
    ),
    KnitStyleSpec(
        4,
        "knit-polar-fleece-charcoal",
        "Anti-Pill Polar Fleece Charcoal",
        "charcoal_polar_fleece",
        3.0,
        (0.84, 0.98),
        "Plush brushed outdoor polar fleece built from matted multi-angle Gabor fibers and soft micro-tuft pile.",
    ),
    KnitStyleSpec(
        5,
        "knit-french-terry-oatmeal",
        "Loopback French Terry Oatmeal",
        "oatmeal_french_terry",
        5.0,
        (0.78, 0.95),
        "Unbrushed loopback sweatshirt French terry with staggered curved cotton yarn loops over a fine knit backing.",
    ),
    KnitStyleSpec(
        6,
        "knit-waffle-sage",
        "Thermal Waffle Knit Sage",
        "sage_waffle_thermal",
        5.4,
        (0.76, 0.94),
        "Heavy henley waffle knit with sunken cellular pockets, ribbed wale walls, purl bridges, and washed cotton slub.",
    ),
    KnitStyleSpec(
        7,
        "knit-crochet-rust",
        "Hand Crochet Terracotta Rust",
        "terracotta_rust_crochet",
        5.6,
        (0.76, 0.95),
        "Chunky hand-crocheted staggered fan-shell clusters and horizontal chain ridges with open eyelet pockets.",
    ),
    KnitStyleSpec(
        8,
        "knit-brushed-mohair-blush",
        "Brushed Mohair Knit Blush",
        "blush_brushed_mohair",
        3.4,
        (0.82, 0.97),
        "Fuzzy brushed mohair sweater knit with curl-swept long Gabor halo fibers veiling soft stockinette stitches.",
    ),
    KnitStyleSpec(
        9,
        "knit-thermal-honeycomb-slate",
        "Thermal Honeycomb Knit Slate",
        "slate_thermal_honeycomb",
        5.2,
        (0.76, 0.94),
        "Staggered hexagonal brioche thermal honeycomb with diagonal tuck-stitch float arches and recessed purl cells.",
    ),
    KnitStyleSpec(
        10,
        "knit-sherpa-fleece-ivory",
        "Curly Sherpa Fleece Ivory",
        "ivory_sherpa_fleece",
        4.8,
        (0.84, 0.98),
        "High-pile curly lambskin sherpa fleece with two-scale swirled wool locks, matte interstices, and lofty fibers.",
    ),
    KnitStyleSpec(
        11,
        "knit-seed-stitch-burgundy",
        "Seed Stitch Knit Burgundy",
        "burgundy_seed_wool",
        5.4,
        (0.76, 0.94),
        "Chunky hand-knitted moss/seed stitch alternating knit V-loops and protruding purl nubs in marled wool.",
    ),
)

STYLE_BY_SLUG: dict[str, KnitStyleSpec] = {s.slug: s for s in KNIT_STYLES}


def _warp_uv(
    u: np.ndarray,
    v: np.ndarray,
    freq: int,
    amp: float,
    rng: np.random.Generator,
    octaves: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Smooth periodic fabric drape/tension UV warp keeping exact torus periodicity."""
    du = common.periodic_fbm(u, v, freq, octaves, 0.52, rng, mode="fbm") - 0.5
    dv = common.periodic_fbm(u, v, freq, octaves, 0.52, rng, mode="fbm") - 0.5
    return (u + du * float(amp)) % 1.0, (v + dv * float(amp)) % 1.0


def _yarn_micro(u: np.ndarray, v: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Return `(fine_fiber, micro_fuzz)` resolving crisp textile fiber texture at 2048px."""
    fine = common.periodic_fbm(u, v, 52, 4, 0.60, rng, mode="fbm")
    fuzz = common.periodic_fbm(u, v, 118, 3, 0.64, rng, mode="turbulence")
    return fine.astype(np.float32), fuzz.astype(np.float32)


def _course_slub(u: np.ndarray, v: np.ndarray, rows: int, rng: np.random.Generator) -> np.ndarray:
    """Yarn thickness & dye slub stretched along weft-knitted course direction."""
    return common.periodic_fbm(u, v, max(4, rows // 8), 3, 0.54, rng, freqy=min(128, max(8, rows))).astype(np.float32)


def _heather_melange(u: np.ndarray, v: np.ndarray, rng: np.random.Generator, scale: int = 48) -> np.ndarray:
    """Authentic marled/heather fiber flecks + gentle dye-bath clouding."""
    p1 = common.periodic_rotated_perlin(
        u, v, scale, 0.35, rng, phase_u=float(rng.uniform()), phase_v=float(rng.uniform())
    )
    p2 = common.periodic_rotated_perlin(
        u, v, int(round(scale * 1.48)), 1.25, rng, phase_u=float(rng.uniform()), phase_v=float(rng.uniform())
    )
    broad, _, _ = common.periodic_domain_warp_fbm(u, v, 3, 4, 0.54, rng, warp_strength=0.09)
    speck = np.clip(0.55 * p1 + 0.45 * p2, 0.0, 1.0)
    return common.robust_normalize(0.54 * speck + 0.46 * broad, low_pct=0.4, high_pct=99.6, soft=True)


def _blend_stops(
    field: np.ndarray,
    micro: np.ndarray | None,
    stops: list[tuple[int, int, int]],
    contrast: float = 1.08,
    gamma: float = 1.0,
) -> np.ndarray:
    return common.apply_palette_5stop(field, micro, stops, contrast=contrast, gamma=gamma)


def _knit_vloop_field(
    uw: np.ndarray,
    vw: np.ndarray,
    cols: int,
    rows: int,
    slub: np.ndarray,
    rng: np.random.Generator,
    r_yarn: float = 0.22,
    ply_freq: float = 4.6,
    tension_jitter: float = 0.09,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Interlocking Chamberlain/Leaf-Glaskin V-loop knit field with 2D per-stitch irregularity.

    Returns ``(height, fuzz_halo, twist_tex, stitch_tone)`` all in ``[0, 1]``.
    """
    gx = uw * cols
    gy = vw * rows
    ix_base = np.floor(gx).astype(np.int32)
    iy_base = np.floor(gy).astype(np.int32)
    fx = gx - ix_base
    cx_idx = ix_base % cols

    # 2D periodic per-stitch organic tension, sway, thickness, and yarn-lot tone tables
    st_shift_y = (
        rng.uniform(-tension_jitter * 0.65, tension_jitter * 0.65, size=(rows, cols))
        + rng.uniform(-tension_jitter * 0.5, tension_jitter * 0.5, size=(rows, 1))
    ).astype(np.float32)
    st_shift_x = rng.uniform(-0.055, 0.055, size=(rows, cols)).astype(np.float32)
    st_rad = rng.uniform(0.88, 1.12, size=(rows, cols)).astype(np.float32)
    st_tone = (
        rng.uniform(-0.09, 0.09, size=(rows, cols))
        + rng.uniform(-0.06, 0.06, size=(rows, 1))
    ).astype(np.float32)

    best_h = np.zeros_like(uw, dtype=np.float32)
    best_halo = np.zeros_like(uw, dtype=np.float32)
    best_twist = np.full_like(uw, 0.5, dtype=np.float32)
    best_tone = np.zeros_like(uw, dtype=np.float32)

    for dy in (-1, 0, 1):
        cy_idx = (iy_base + dy) % rows
        fy = gy - (iy_base + dy) + st_shift_y[cy_idx, cx_idx]
        wx = st_shift_x[cy_idx, cx_idx]
        rad = np.float32(r_yarn) * (0.88 + 0.24 * slub) * st_rad[cy_idx, cx_idx]

        t_raw = (fy + 0.24) / 1.40
        active = (t_raw > 0.0) & (t_raw < 1.0)
        t = np.clip(t_raw, 0.0, 1.0)
        tuck = np.power(np.sin(t * math.pi), 0.42) * active.astype(np.float32)

        x_l = 0.20 + 0.145 * np.sin(t * math.pi * 0.88) + 0.042 * (1.0 - np.cos(t * math.pi)) + wx
        x_r = 1.0 - (0.20 + 0.145 * np.sin(t * math.pi * 0.88) + 0.042 * (1.0 - np.cos(t * math.pi))) + wx
        z_leg = (0.34 + 0.66 * np.sin(np.power(t, 0.88) * math.pi)) * tuck

        dl = (fx - x_l) / np.maximum(rad, 0.05)
        dr = (fx - x_r) / np.maximum(rad, 0.05)
        pl = np.sqrt(np.clip(1.0 - dl * dl, 0.0, 1.0))
        pr = np.sqrt(np.clip(1.0 - dr * dr, 0.0, 1.0))

        hl = np.exp(-1.75 * dl * dl) * tuck
        hr = np.exp(-1.75 * dr * dr) * tuck

        # 3-ply twisted yarn strand ridges + secondary fine spun-fiber groove
        ph_l = (t * ply_freq + dl * 0.52) * 2.0 * math.pi
        ph_r = (t * ply_freq - dr * 0.52) * 2.0 * math.pi
        tw_l = 0.78 + 0.16 * np.power(0.5 + 0.5 * np.cos(ph_l), 0.72) + 0.06 * np.cos(ph_l * 3.0)
        tw_r = 0.78 + 0.16 * np.power(0.5 + 0.5 * np.cos(ph_r), 0.72) + 0.06 * np.cos(ph_r * 3.0)

        cand_l = pl * tw_l * z_leg
        cand_r = pr * tw_r * z_leg

        crown_y = 1.00 - 0.10 * np.cos((fx - 0.5) * math.pi)
        dc = (fy - crown_y) / (rad * 1.18)
        cx_env = np.sqrt(np.clip(1.0 - ((fx - 0.5) / 0.34) ** 2, 0.0, 1.0))
        tw_c = 0.84 + 0.16 * np.cos((fx * 5.0 + dc * 0.6) * 2.0 * math.pi)
        cand_c = np.sqrt(np.clip(1.0 - dc * dc, 0.0, 1.0)) * cx_env * tw_c * 0.46

        use_l = cand_l >= cand_r
        leg_h = np.where(use_l, cand_l, cand_r)
        leg_tw = np.where(use_l, tw_l, tw_r)

        use_leg = leg_h >= cand_c
        cand_h = np.where(use_leg, leg_h, cand_c)
        cand_tw = np.where(use_leg, leg_tw, tw_c)

        better = cand_h > best_h
        best_h = np.where(better, cand_h, best_h)
        best_twist = np.where(better, cand_tw, best_twist)
        best_tone = np.where(better, st_tone[cy_idx, cx_idx], best_tone)
        best_halo = np.maximum(best_halo, np.maximum(hl, hr) * 0.65)

    return best_h, best_halo, best_twist, best_tone


def _purl_bump_field(
    uw: np.ndarray,
    vw: np.ndarray,
    cols: int,
    rows: int,
    slub: np.ndarray,
    rng: np.random.Generator,
    r_yarn: float = 0.24,
    ply_freq: float = 4.4,
) -> tuple[np.ndarray, np.ndarray]:
    """Authentic interlocking horizontal semicircular purl bump arches on T^2."""
    gx = uw * cols
    gy = vw * rows
    ix_base = np.floor(gx).astype(np.int32)
    iy_base = np.floor(gy).astype(np.int32)
    fx = gx - ix_base
    cx_idx = ix_base % cols

    st_shift = rng.uniform(-0.08, 0.08, size=(rows, cols)).astype(np.float32)
    st_rad = rng.uniform(0.90, 1.12, size=(rows, cols)).astype(np.float32)

    best_h = np.zeros_like(uw, dtype=np.float32)
    best_halo = np.zeros_like(uw, dtype=np.float32)

    for dy in (-1, 0, 1):
        cy_idx = (iy_base + dy) % rows
        fy = gy - (iy_base + dy) + st_shift[cy_idx, cx_idx]
        rad = np.float32(r_yarn) * (0.88 + 0.24 * slub) * st_rad[cy_idx, cx_idx]
        y_center = 0.50 + 0.17 * np.cos((fx - 0.5) * 2.0 * math.pi)
        dv = (fy - y_center) / np.maximum(rad, 0.06)
        cyl = np.sqrt(np.clip(1.0 - dv * dv, 0.0, 1.0))
        z_arch = 0.62 + 0.38 * (0.5 + 0.5 * np.cos((fx - 0.5) * 2.0 * math.pi))
        ph = (fx * ply_freq + dv * 0.55) * 2.0 * math.pi
        tw = 0.80 + 0.15 * np.power(0.5 + 0.5 * np.cos(ph), 0.72) + 0.05 * np.cos(ph * 3.0)

        dx_leg = np.minimum(np.abs(fx - 0.18), np.abs(fx - 0.82)) / (rad * 0.85)
        leg_back = np.sqrt(np.clip(1.0 - dx_leg * dx_leg, 0.0, 1.0)) * 0.34

        cand = np.maximum(cyl * z_arch * tw, leg_back)
        best_h = np.maximum(best_h, cand)
        best_halo = np.maximum(best_halo, np.exp(-1.8 * dv * dv) * z_arch)

    return best_h, best_halo


def _build_jersey_heather(
    u: np.ndarray, v: np.ndarray, size: int, rng: np.random.Generator, stops: list[tuple[int, int, int]]
) -> tuple[Image.Image, np.ndarray, np.ndarray]:
    uw, vw = _warp_uv(u, v, freq=3, amp=0.016, rng=rng, octaves=3)
    cols, rows = 37, 51  # odd stitch counts avoid half-tile quadrant phase lock
    slub = _course_slub(uw, vw, rows, rng)
    fine, fuzz = _yarn_micro(u, v, rng)
    heather = _heather_melange(uw, vw, rng, scale=68)
    drape, _, _ = common.periodic_domain_warp_fbm(u, v, 3, 4, 0.52, rng, warp_strength=0.08)

    loop_h, halo, twist, s_tone = _knit_vloop_field(
        uw, vw, cols, rows, slub, rng, r_yarn=0.228, ply_freq=4.6, tension_jitter=0.08
    )
    h = common.robust_normalize(
        0.66 * loop_h + 0.12 * halo + 0.11 * fine + 0.08 * fuzz + 0.08 * drape,
        low_pct=0.25,
        high_pct=99.75,
        soft=True,
    )
    c = common.robust_normalize(
        0.34 * loop_h + 0.40 * heather + 0.14 * slub + 0.12 * fine + 0.10 * drape + s_tone - 0.10 * (1.0 - halo),
        low_pct=0.35,
        high_pct=99.65,
        soft=True,
    )
    rough_raw = 0.42 * (1.0 - loop_h) + 0.28 * fuzz + 0.18 * (1.0 - twist) + 0.14 * (1.0 - fine)
    rough = common.match_histogram_range(rough_raw, 0.76, 0.92, target_mean=0.85, contrast=1.08)
    albedo = _blend_stops(c, fine, stops, contrast=1.10, gamma=1.00)
    base = common.shade_pbr_surface(
        albedo, h, roughness=rough, normal_strength=3.8, spec_intensity=0.04, ao_strength=0.24, cavity_strength=0.22
    )
    return base, h, rough


def _build_rib_2x2(
    u: np.ndarray, v: np.ndarray, size: int, rng: np.random.Generator, stops: list[tuple[int, int, int]]
) -> tuple[Image.Image, np.ndarray, np.ndarray]:
    uw, vw = _warp_uv(u, v, freq=3, amp=0.017, rng=rng, octaves=3)
    period = 4  # 2 knit wales + 2 purl wales
    wales = 36  # 9 ribs across (odd rib count prevents half-tile phase lock!)
    courses = 49
    slub = _course_slub(uw, vw, courses, rng)
    fine, fuzz = _yarn_micro(u, v, rng)
    heather = _heather_melange(uw, vw, rng, scale=58)
    drape, _, _ = common.periodic_domain_warp_fbm(u, v, 3, 4, 0.54, rng, warp_strength=0.09)

    knit_h, knit_halo, twist, s_tone = _knit_vloop_field(
        uw, vw, wales, courses, slub, rng, r_yarn=0.232, ply_freq=5.0, tension_jitter=0.09
    )
    purl_h, purl_halo = _purl_bump_field(uw, vw, wales, courses, slub, rng, r_yarn=0.235, ply_freq=4.4)

    gx = uw * wales
    phase_rib = (gx % period) / float(period)
    rib_wave = 0.5 + 0.5 * np.tanh(7.0 * np.sin(phase_rib * 2.0 * math.pi))
    rib_dome = np.power(np.clip(np.sin(np.clip(phase_rib * 2.0, 0.0, 1.0) * math.pi), 0.0, 1.0), 0.45)

    # Richer wool fiber nap so rib surface reads tactile & fibrous rather than smooth stripes
    nap = common.periodic_gabor_noise(
        u, v, grid_cells=36, f0=2.3, omega0=0.0, bandwidth=1.65, rng=rng, isotropic=True, impulses_per_cell=3
    )

    rib_h = rib_wave * (0.28 * rib_dome + 0.72 * knit_h) + (1.0 - rib_wave) * (0.06 + 0.48 * purl_h)
    h = common.robust_normalize(
        0.66 * rib_h
        + 0.10 * (rib_wave * knit_halo + (1.0 - rib_wave) * purl_halo)
        + 0.13 * fine
        + 0.12 * nap
        + 0.09 * fuzz
        + 0.06 * drape,
        low_pct=0.25,
        high_pct=99.75,
        soft=True,
    )
    c = common.robust_normalize(
        0.40 * rib_h + 0.14 * rib_wave + 0.24 * heather + 0.14 * fine + 0.12 * nap + 0.10 * drape + 0.08 * slub + s_tone,
        low_pct=0.30,
        high_pct=99.70,
        soft=True,
    )
    rough_raw = 0.40 * (1.0 - rib_h) + 0.22 * (1.0 - rib_wave) + 0.24 * fuzz + 0.14 * (1.0 - twist)
    rough = common.match_histogram_range(rough_raw, 0.74, 0.93, target_mean=0.84, contrast=1.10)
    albedo = _blend_stops(c, fine, stops, contrast=1.12, gamma=1.02)
    base = common.shade_pbr_surface(
        albedo, h, roughness=rough, normal_strength=5.0, spec_intensity=0.04, ao_strength=0.28, cavity_strength=0.26
    )
    return base, h, rough


def _build_cable_aran(
    u: np.ndarray, v: np.ndarray, size: int, rng: np.random.Generator, stops: list[tuple[int, int, int]]
) -> tuple[Image.Image, np.ndarray, np.ndarray]:
    """Fisherman Aran sweater: alternating 3-strand braided cable + purl gutter + 2-strand twist cord."""
    uw, vw = _warp_uv(u, v, freq=3, amp=0.014, rng=rng, octaves=3)
    fine, fuzz = _yarn_micro(u, v, rng)
    wool_nap = common.periodic_gabor_noise(
        u, v, grid_cells=36, f0=2.3, omega0=0.0, bandwidth=1.60, rng=rng, isotropic=True, impulses_per_cell=3
    )
    wool_mottle, _, _ = common.periodic_domain_warp_fbm(u, v, 3, 4, 0.54, rng, warp_strength=0.10)
    slub = _course_slub(uw, vw, 49, rng)

    bays = 3
    bu = (uw * bays) % 1.0
    bv = (vw * 5.0) % 1.0

    # --- Wide 3-strand Aran braid ---
    wide_u = (bu - 0.31) / 0.25
    wide_mask = np.clip(1.0 - np.maximum(np.abs(wide_u) - 0.88, 0.0) / 0.14, 0.0, 1.0)
    phase_w = bv * 2.0 * math.pi
    braid_h = np.zeros_like(u, dtype=np.float32)

    for k in range(3):
        ph = phase_w + k * (2.0 * math.pi / 3.0)
        cx = 0.54 * np.sin(ph)
        dx_dv = 0.54 * 2.0 * math.pi * np.cos(ph)
        z_strand = 0.66 + 0.34 * np.sin(2.0 * ph) * 0.5 + 0.32 * np.cos(ph)
        d = (wide_u - cx) / (0.34 * (0.93 + 0.14 * slub))
        cyl = np.sqrt(np.clip(1.0 - d * d, 0.0, 1.0))

        sub_x = ((d + 1.0) * 1.0) % 1.0
        sub_y = (bv * 8.0 - d * 0.35 * np.sign(dx_dv + 1e-4)) % 1.0
        v_chev = np.abs(sub_x - 0.5) * 2.0
        sub_loop = 0.70 + 0.30 * np.power(0.5 + 0.5 * np.sin((sub_y * 2.0 + v_chev * 0.75) * 2.0 * math.pi), 0.65)
        sub_ply = 0.86 + 0.14 * np.cos((sub_y * 8.0 + sub_x * 4.0) * 2.0 * math.pi)
        strand = cyl * z_strand * sub_loop * sub_ply * wide_mask
        braid_h = np.maximum(braid_h, strand)

    # --- Narrow 2-strand twisted rope cord ---
    nar_u = (bu - 0.81) / 0.13
    nar_mask = np.clip(1.0 - np.maximum(np.abs(nar_u) - 0.86, 0.0) / 0.16, 0.0, 1.0)
    phase_n = (vw * 9.0) % 1.0 * 2.0 * math.pi  # odd cycle count breaks half-tile symmetry
    cord_h = np.zeros_like(u, dtype=np.float32)
    for k in range(2):
        ph = phase_n + k * math.pi
        cx = 0.36 * np.sin(ph)
        z_c = 0.68 + 0.32 * np.cos(ph)
        d = (nar_u - cx) / 0.48
        cyl = np.sqrt(np.clip(1.0 - d * d, 0.0, 1.0))
        sub_st = 0.76 + 0.24 * np.sin(((vw * 38.0) + np.abs(d) * 0.9) * 2.0 * math.pi)
        cand = cyl * z_c * sub_st * nar_mask * 0.86
        cord_h = np.maximum(cord_h, cand)

    # --- Recessed reverse-stockinette purl bumps in gutters ---
    purl_h, purl_halo = _purl_bump_field(uw, vw, cols=37, rows=49, slub=slub, rng=rng, r_yarn=0.23, ply_freq=4.2)
    cable_h = np.maximum(braid_h, cord_h)
    cable_presence = np.clip(cable_h * 3.2, 0.0, 1.0)
    combined = np.maximum(cable_h, purl_h * 0.45 * (1.0 - 0.55 * cable_presence))

    h = common.robust_normalize(
        0.68 * combined + 0.13 * wool_nap + 0.11 * fine + 0.08 * fuzz + 0.05 * purl_halo * (1.0 - cable_presence),
        low_pct=0.25,
        high_pct=99.75,
        soft=True,
    )
    c = common.robust_normalize(
        0.48 * combined + 0.24 * wool_mottle + 0.14 * wool_nap + 0.12 * fine + 0.08 * slub,
        low_pct=0.30,
        high_pct=99.70,
        soft=True,
    )
    rough_raw = 0.44 * (1.0 - combined) + 0.28 * fuzz + 0.18 * wool_nap + 0.12 * (1.0 - wool_mottle)
    rough = common.match_histogram_range(rough_raw, 0.76, 0.94, target_mean=0.86, contrast=1.10)
    albedo = _blend_stops(c, fine, stops, contrast=1.12, gamma=0.98)
    base = common.shade_pbr_surface(
        albedo, h, roughness=rough, normal_strength=5.6, spec_intensity=0.04, ao_strength=0.30, cavity_strength=0.26
    )
    return base, h, rough


def _build_stockinette_merino(
    u: np.ndarray, v: np.ndarray, size: int, rng: np.random.Generator, stops: list[tuple[int, int, int]]
) -> tuple[Image.Image, np.ndarray, np.ndarray]:
    uw, vw = _warp_uv(u, v, freq=3, amp=0.017, rng=rng, octaves=3)
    cols, rows = 27, 37  # odd counts eliminate quadrant lockstep
    slub = _course_slub(uw, vw, rows, rng)
    fine, fuzz = _yarn_micro(u, v, rng)
    heather = _heather_melange(uw, vw, rng, scale=46)
    merino_nap = common.periodic_gabor_noise(
        u, v, grid_cells=38, f0=2.3, omega0=0.0, bandwidth=1.7, rng=rng, isotropic=True, impulses_per_cell=3
    )
    drape, _, _ = common.periodic_domain_warp_fbm(u, v, 3, 4, 0.52, rng, warp_strength=0.09)

    loop_h, halo, twist, s_tone = _knit_vloop_field(
        uw, vw, cols, rows, slub, rng, r_yarn=0.235, ply_freq=5.2, tension_jitter=0.10
    )

    h = common.robust_normalize(
        0.68 * loop_h + 0.10 * halo + 0.11 * fine + 0.09 * merino_nap + 0.07 * fuzz + 0.07 * drape,
        low_pct=0.25,
        high_pct=99.75,
        soft=True,
    )
    c = common.robust_normalize(
        0.42 * loop_h + 0.26 * heather + 0.14 * twist + 0.12 * fine + 0.10 * drape + 0.08 * slub + s_tone * 1.25,
        low_pct=0.30,
        high_pct=99.70,
        soft=True,
    )
    rough_raw = 0.48 * (1.0 - loop_h) + 0.24 * fuzz + 0.18 * (1.0 - twist) + 0.12 * merino_nap
    rough = common.match_histogram_range(rough_raw, 0.70, 0.90, target_mean=0.81, contrast=1.10)
    albedo = _blend_stops(c, fine, stops, contrast=1.14, gamma=1.02)
    base = common.shade_pbr_surface(
        albedo, h, roughness=rough, normal_strength=4.6, spec_intensity=0.05, ao_strength=0.28, cavity_strength=0.24
    )
    return base, h, rough


def _build_polar_fleece(
    u: np.ndarray, v: np.ndarray, size: int, rng: np.random.Generator, stops: list[tuple[int, int, int]]
) -> tuple[Image.Image, np.ndarray, np.ndarray]:
    """Anti-pill brushed polar fleece: zero stitch grid, pure tangled microfiber pile."""
    uw, vw = _warp_uv(u, v, freq=5, amp=0.028, rng=rng, octaves=3)

    f1, edge, cid, _, _ = common.periodic_lloyd_voronoi(uw, vw, 30, 0.90, rng, lloyd_iters=1)
    tuft_soft = common.periodic_gaussian_blur(
        np.power(np.clip(1.0 - f1 * 0.85, 0.0, 1.0), 0.7) * (0.78 + 0.22 * cid),
        sigma=max(1.5, size / 420.0),
    )

    g_a = common.periodic_gabor_noise(
        uw, vw, grid_cells=26, f0=2.5, omega0=0.35, bandwidth=1.35, rng=rng, isotropic=False, impulses_per_cell=3
    )
    g_b = common.periodic_gabor_noise(
        uw, vw, grid_cells=28, f0=2.5, omega0=1.65, bandwidth=1.40, rng=rng, isotropic=False, impulses_per_cell=3
    )
    g_iso = common.periodic_gabor_noise(
        u, v, grid_cells=36, f0=2.2, omega0=0.0, bandwidth=1.55, rng=rng, isotropic=True, impulses_per_cell=3
    )
    brush_patch, _, _ = common.periodic_domain_warp_fbm(u, v, 4, 4, 0.55, rng, warp_strength=0.11, stages=2)
    fine, fuzz = _yarn_micro(u, v, rng)
    micro_pill = common.periodic_fbm(u, v, 68, 4, 0.64, rng, mode="billow")

    fibers = brush_patch * g_a + (1.0 - brush_patch) * g_b
    h = common.robust_normalize(
        0.30 * tuft_soft + 0.28 * fibers + 0.24 * g_iso + 0.16 * micro_pill + 0.14 * fine + 0.10 * fuzz,
        low_pct=0.30,
        high_pct=99.70,
        soft=True,
    )
    c = common.robust_normalize(
        0.28 * brush_patch + 0.26 * tuft_soft + 0.24 * fibers + 0.16 * g_iso + 0.12 * micro_pill - 0.10 * fuzz,
        low_pct=0.35,
        high_pct=99.65,
        soft=True,
    )
    rough_raw = 0.36 * (1.0 - brush_patch) + 0.30 * fuzz + 0.22 * (1.0 - tuft_soft) + 0.16 * (1.0 - fibers)
    rough = common.match_histogram_range(rough_raw, 0.84, 0.98, target_mean=0.92, contrast=1.08)
    albedo = _blend_stops(c, fuzz, stops, contrast=1.08, gamma=1.02)
    base = common.shade_pbr_surface(
        albedo, h, roughness=rough, normal_strength=3.0, spec_intensity=0.025, ao_strength=0.18, cavity_strength=0.16
    )
    return base, h, rough


def _build_french_terry(
    u: np.ndarray, v: np.ndarray, size: int, rng: np.random.Generator, stops: list[tuple[int, int, int]]
) -> tuple[Image.Image, np.ndarray, np.ndarray]:
    """Loopback sweatshirt French terry: staggered curved cotton pile loops over fine knit backing."""
    uw, vw = _warp_uv(u, v, freq=3, amp=0.014, rng=rng, octaves=3)
    fine, fuzz = _yarn_micro(u, v, rng)
    cotton_fleck = _heather_melange(uw, vw, rng, scale=56)

    backing_slub = _course_slub(uw, vw, 44, rng)
    back_h, _ = _purl_bump_field(uw, vw, cols=37, rows=44, slub=backing_slub, rng=rng, r_yarn=0.22, ply_freq=3.8)

    cols_l, rows_l = 23, 28  # cols odd, rows even for half-row stagger periodicity
    gx = uw * cols_l
    gy = vw * rows_l
    iy_base = np.floor(gy).astype(np.int32)

    loop_tilt = rng.uniform(-0.20, 0.20, size=(rows_l, cols_l)).astype(np.float32)
    loop_droop = rng.uniform(0.82, 1.18, size=(rows_l, cols_l)).astype(np.float32)
    loop_z = rng.uniform(0.74, 1.00, size=(rows_l, cols_l)).astype(np.float32)
    loop_off_x = rng.uniform(-0.12, 0.12, size=(rows_l, cols_l)).astype(np.float32)
    loop_off_y = rng.uniform(-0.12, 0.12, size=(rows_l, cols_l)).astype(np.float32)
    loop_tone_lut = rng.uniform(-0.09, 0.09, size=(rows_l, cols_l)).astype(np.float32)

    terry_h = np.zeros_like(u, dtype=np.float32)
    terry_halo = np.zeros_like(u, dtype=np.float32)
    terry_tone = np.zeros_like(u, dtype=np.float32)

    for dy in (-1, 0, 1):
        cy_i = (iy_base + dy) % rows_l
        stagger = (cy_i % 2).astype(np.float32) * 0.5
        gx_s = gx - stagger
        ix_s = np.floor(gx_s).astype(np.int32)
        for dx in (-1, 0, 1):
            cx_i = (ix_s + dx) % cols_l

            lx = gx_s - (ix_s + dx) - 0.5 - loop_off_x[cy_i, cx_i]
            ly = gy - (iy_base + dy) - 0.5 - loop_off_y[cy_i, cx_i]

            ang = loop_tilt[cy_i, cx_i]
            ca, sa = np.cos(ang), np.sin(ang)
            rx = lx * ca - ly * sa
            ry = lx * sa + ly * ca

            rad_x = 0.40
            rad_y = 0.44 * loop_droop[cy_i, cx_i]
            ex = rx / rad_x
            ey = (ry + 0.06) / rad_y
            ell_r = np.sqrt(ex * ex + ey * ey + 1e-6)
            theta = np.arctan2(ey, ex)

            arc_z = np.clip(0.58 + 0.42 * np.sin(theta), 0.0, 1.0)
            arc_mask = np.clip((np.sin(theta) + 0.55) / 0.45, 0.0, 1.0)

            tube_w = 0.245 * (0.90 + 0.20 * backing_slub)
            d_tube = np.abs(ell_r - 1.0) / np.maximum(tube_w, 0.06)
            prof = np.sqrt(np.clip(1.0 - d_tube * d_tube, 0.0, 1.0)) * arc_mask
            ply = 0.82 + 0.18 * np.power(0.5 + 0.5 * np.cos((theta * 5.5 + d_tube * 1.4) * 2.0), 0.75)

            cand = prof * ply * arc_z * loop_z[cy_i, cx_i]
            better = cand > terry_h
            terry_h = np.where(better, cand, terry_h)
            terry_tone = np.where(better, loop_tone_lut[cy_i, cx_i], terry_tone)
            terry_halo = np.maximum(terry_halo, np.exp(-2.0 * d_tube * d_tube) * arc_mask * 0.55)

    combined = np.maximum(terry_h, back_h * 0.36)
    seed_p1 = common.periodic_rotated_perlin(u, v, 62, 0.5, rng, phase_u=float(rng.uniform()), phase_v=float(rng.uniform()))
    seed_p2 = common.periodic_rotated_perlin(u, v, 78, 1.3, rng, phase_u=float(rng.uniform()), phase_v=float(rng.uniform()))
    oat_specks = np.power(np.clip((np.sqrt(seed_p1 * seed_p2) - 0.72) / 0.28, 0.0, 1.0), 1.8)

    h = common.robust_normalize(
        0.72 * combined + 0.10 * terry_halo + 0.10 * fine + 0.08 * fuzz,
        low_pct=0.25,
        high_pct=99.75,
        soft=True,
    )
    c = common.robust_normalize(
        0.46 * combined + 0.24 * cotton_fleck + 0.12 * fine + 0.08 * backing_slub + terry_tone - 0.28 * oat_specks,
        low_pct=0.30,
        high_pct=99.70,
        soft=True,
    )
    rough_raw = 0.44 * (1.0 - combined) + 0.28 * fuzz + 0.18 * (1.0 - fine) + 0.14 * oat_specks
    rough = common.match_histogram_range(rough_raw, 0.78, 0.95, target_mean=0.87, contrast=1.08)
    albedo = _blend_stops(c, fine, stops, contrast=1.10, gamma=0.98)
    base = common.shade_pbr_surface(
        albedo, h, roughness=rough, normal_strength=5.0, spec_intensity=0.035, ao_strength=0.28, cavity_strength=0.25
    )
    return base, h, rough


def _build_waffle_knit(
    u: np.ndarray, v: np.ndarray, size: int, rng: np.random.Generator, stops: list[tuple[int, int, int]]
) -> tuple[Image.Image, np.ndarray, np.ndarray]:
    """Heavy henley thermal waffle knit with true knit sub-stitches on walls and pocket floors."""
    uw, vw = _warp_uv(u, v, freq=3, amp=0.016, rng=rng, octaves=3)
    cells_u, cells_v = 15, 19  # odd cell counts avoid quadrant lockstep
    fine, fuzz = _yarn_micro(u, v, rng)
    slub = _course_slub(uw, vw, cells_v * 3, rng)
    heather = _heather_melange(uw, vw, rng, scale=54)
    drape, _, _ = common.periodic_domain_warp_fbm(u, v, 3, 4, 0.52, rng, warp_strength=0.09)

    sub_knit, sub_halo, twist, s_tone = _knit_vloop_field(
        uw, vw, cols=cells_u * 3, rows=cells_v * 3, slub=slub, rng=rng, r_yarn=0.232, ply_freq=4.2, tension_jitter=0.07
    )
    sub_purl, _ = _purl_bump_field(uw, vw, cols=cells_u * 3, rows=cells_v * 3, slub=slub, rng=rng, r_yarn=0.235, ply_freq=4.0)

    gx = uw * cells_u
    gy = vw * cells_v
    ix = np.floor(gx).astype(np.int32) % cells_u
    iy = np.floor(gy).astype(np.int32) % cells_v
    fu = gx - np.floor(gx)
    fv = gy - np.floor(gy)

    cell_var = rng.uniform(0.84, 1.16, size=(cells_v, cells_u)).astype(np.float32)[iy, ix]
    dx = np.abs(fu - 0.5) * 2.0
    dy = np.abs(fv - 0.5) * 2.0

    wall_u = np.power(np.sin(np.clip((dx - 0.16) / 0.84, 0.0, 1.0) * math.pi * 0.5), 1.65)
    wall_v = np.power(np.sin(np.clip((dy - 0.16) / 0.84, 0.0, 1.0) * math.pi * 0.5), 1.65)
    ridge = np.clip(0.56 * np.maximum(wall_u, wall_v) + 0.44 * (0.5 * (wall_u + wall_v)) + 0.14 * wall_u * wall_v, 0.0, 1.0)
    ridge = ridge * cell_var

    stitch_detail = np.where(wall_u >= wall_v, sub_knit, sub_purl)
    waffle_h = ridge * (0.52 + 0.48 * stitch_detail) + (1.0 - ridge) * (0.08 + 0.30 * sub_purl)

    h = common.robust_normalize(
        0.68 * waffle_h + 0.10 * sub_halo + 0.11 * fine + 0.09 * fuzz + 0.06 * drape,
        low_pct=0.25,
        high_pct=99.75,
        soft=True,
    )
    c = common.robust_normalize(
        0.46 * waffle_h + 0.24 * heather + 0.12 * slub + 0.12 * fine + 0.10 * drape + s_tone,
        low_pct=0.30,
        high_pct=99.70,
        soft=True,
    )
    rough_raw = 0.44 * (1.0 - waffle_h) + 0.26 * fuzz + 0.18 * (1.0 - twist) + 0.14 * (1.0 - fine)
    rough = common.match_histogram_range(rough_raw, 0.76, 0.94, target_mean=0.86, contrast=1.10)
    albedo = _blend_stops(c, fine, stops, contrast=1.12, gamma=1.00)
    base = common.shade_pbr_surface(
        albedo, h, roughness=rough, normal_strength=5.4, spec_intensity=0.04, ao_strength=0.30, cavity_strength=0.26
    )
    return base, h, rough


def _seg_dist_and_t(
    px: np.ndarray, py: np.ndarray, ax: float, ay: float, bx: float, by: float
) -> tuple[np.ndarray, np.ndarray]:
    """2D capsule segment distance & parameter t in [0, 1]."""
    abx = bx - ax
    aby = by - ay
    ab2 = abx * abx + aby * aby + 1e-8
    t = np.clip(((px - ax) * abx + (py - ay) * aby) / ab2, 0.0, 1.0)
    dx = px - (ax + t * abx)
    dy = py - (ay + t * aby)
    return np.sqrt(dx * dx + dy * dy + 1e-10).astype(np.float32), t.astype(np.float32)


def _build_crochet_shell(
    u: np.ndarray, v: np.ndarray, size: int, rng: np.random.Generator, stops: list[tuple[int, int, int]]
) -> tuple[Image.Image, np.ndarray, np.ndarray]:
    """Chunky hand-crocheted staggered shell/fan clusters with twisted posts and top chain arches."""
    uw, vw = _warp_uv(u, v, freq=3, amp=0.018, rng=rng, octaves=3)
    fine, fuzz = _yarn_micro(u, v, rng)
    wool_nap = common.periodic_gabor_noise(
        u, v, grid_cells=36, f0=2.3, omega0=0.0, bandwidth=1.60, rng=rng, isotropic=True, impulses_per_cell=3
    )
    dye_var, _, _ = common.periodic_domain_warp_fbm(u, v, 3, 4, 0.54, rng, warp_strength=0.10)

    cols, rows = 9, 10
    gx = uw * cols
    gy = vw * rows
    iy_base = np.floor(gy).astype(np.int32)

    shell_jitter_x = rng.uniform(-0.075, 0.075, size=(rows, cols)).astype(np.float32)
    shell_jitter_y = rng.uniform(-0.075, 0.075, size=(rows, cols)).astype(np.float32)
    shell_rad_lut = rng.uniform(0.90, 1.10, size=(rows, cols)).astype(np.float32)
    shell_tone_lut = rng.uniform(-0.09, 0.09, size=(rows, cols)).astype(np.float32)

    best_h = np.zeros_like(u, dtype=np.float32)
    best_halo = np.zeros_like(u, dtype=np.float32)
    best_tone = np.zeros_like(u, dtype=np.float32)

    post_specs = (
        (-0.06, -0.34, -0.34, 0.24, 0.90),
        (0.00, -0.36, 0.00, 0.30, 1.00),
        (0.06, -0.34, 0.34, 0.24, 0.92),
    )

    for dy in (-1, 0, 1):
        cy_i = (iy_base + dy) % rows
        stagger = (cy_i % 2).astype(np.float32) * 0.5
        gx_s = gx - stagger
        ix_s = np.floor(gx_s).astype(np.int32)
        for dx in (-1, 0, 1):
            cx_i = (ix_s + dx) % cols
            px = gx_s - (ix_s + dx) - 0.5 - shell_jitter_x[cy_i, cx_i]
            py = gy - (iy_base + dy) - 0.5 - shell_jitter_y[cy_i, cx_i]
            r_sc = shell_rad_lut[cy_i, cx_i]

            shell_h = np.zeros_like(u, dtype=np.float32)

            for ax, ay, bx, by, z_p in post_specs:
                dist, t = _seg_dist_and_t(px, py, ax, ay, bx, by)
                rad = 0.115 * r_sc * (0.92 + 0.16 * np.sin(t * math.pi))
                dn = dist / rad
                cyl = np.sqrt(np.clip(1.0 - dn * dn, 0.0, 1.0))
                ph_w = (t * 5.0 + dn * 0.7) * 2.0 * math.pi
                wraps = 0.74 + 0.20 * np.power(0.5 + 0.5 * np.cos(ph_w), 0.7) + 0.06 * np.cos(ph_w * 3.0)
                z_arch = (0.68 + 0.32 * np.sin(t * math.pi)) * z_p
                cand = cyl * wraps * z_arch
                shell_h = np.maximum(shell_h, cand)
                best_halo = np.maximum(best_halo, np.exp(-1.85 * dn * dn) * 0.55)

            chain_y = 0.34 - 0.38 * (px * px)
            in_chain = np.sqrt(np.clip(1.0 - (px / 0.46) ** 2, 0.0, 1.0))
            dc = (py - chain_y) / (0.105 * r_sc)
            c_cyl = np.sqrt(np.clip(1.0 - dc * dc, 0.0, 1.0)) * in_chain
            ph_c = (px * 11.0 + np.abs(dc) * 1.4) * 2.0 * math.pi
            c_braid = 0.74 + 0.20 * np.cos(ph_c) + 0.06 * np.cos(ph_c * 2.5)
            chain_h = c_cyl * c_braid * 1.04
            shell_h = np.maximum(shell_h, chain_h)
            best_halo = np.maximum(best_halo, np.exp(-1.95 * dc * dc) * in_chain * 0.60)

            r_knot = np.sqrt(px * px + ((py + 0.36) * 1.25) ** 2) / (0.16 * r_sc)
            knot_h = np.sqrt(np.clip(1.0 - r_knot * r_knot, 0.0, 1.0)) * 0.72
            shell_h = np.maximum(shell_h, knot_h)

            better = shell_h > best_h
            best_h = np.where(better, shell_h, best_h)
            best_tone = np.where(better, shell_tone_lut[cy_i, cx_i], best_tone)

    h = common.robust_normalize(
        0.68 * best_h + 0.10 * best_halo + 0.13 * wool_nap + 0.11 * fine + 0.09 * fuzz,
        low_pct=0.25,
        high_pct=99.75,
        soft=True,
    )
    c = common.robust_normalize(
        0.48 * best_h + 0.22 * dye_var + 0.14 * wool_nap + 0.12 * fine + best_tone - 0.16 * (1.0 - np.clip(best_h * 2.5, 0.0, 1.0)),
        low_pct=0.30,
        high_pct=99.70,
        soft=True,
    )
    rough_raw = 0.44 * (1.0 - best_h) + 0.28 * fuzz + 0.18 * wool_nap + 0.12 * (1.0 - dye_var)
    rough = common.match_histogram_range(rough_raw, 0.76, 0.95, target_mean=0.86, contrast=1.10)
    albedo = _blend_stops(c, fine, stops, contrast=1.14, gamma=1.00)
    base = common.shade_pbr_surface(
        albedo, h, roughness=rough, normal_strength=5.6, spec_intensity=0.04, ao_strength=0.34, cavity_strength=0.28
    )
    return base, h, rough


def _build_brushed_mohair(
    u: np.ndarray, v: np.ndarray, size: int, rng: np.random.Generator, stops: list[tuple[int, int, int]]
) -> tuple[Image.Image, np.ndarray, np.ndarray]:
    """Fuzzy brushed mohair/alpaca sweater: softened knit veiled by long curl-swept Gabor halo fibers."""
    uw, vw = _warp_uv(u, v, freq=3, amp=0.018, rng=rng, octaves=3)
    cols, rows = 23, 31  # odd counts avoid quadrant lockstep
    slub = _course_slub(uw, vw, rows, rng)
    fine, fuzz = _yarn_micro(u, v, rng)
    heather = _heather_melange(uw, vw, rng, scale=42)

    loop_h, halo, _, s_tone = _knit_vloop_field(
        uw, vw, cols, rows, slub, rng, r_yarn=0.242, ply_freq=3.8, tension_jitter=0.10
    )
    soft_knit = common.periodic_gaussian_blur(loop_h, sigma=max(2.0, size / 380.0))

    # Curl-advected long brushed mohair wisps sweeping organically across the knit
    vx, vy = common.periodic_curl_noise(u, v, freq=4, octaves=3, persistence=0.52, rng=rng)
    uc = (u + 0.026 * vx) % 1.0
    vc = (v + 0.026 * vy) % 1.0

    brush_main = common.periodic_gabor_noise(
        uc, vc, grid_cells=26, f0=2.8, omega0=1.38, bandwidth=1.15, rng=rng, isotropic=False, impulses_per_cell=3
    )
    brush_cross = common.periodic_gabor_noise(
        uc, vc, grid_cells=30, f0=2.7, omega0=1.82, bandwidth=1.25, rng=rng, isotropic=False, impulses_per_cell=3
    )
    wisps_iso = common.periodic_gabor_noise(
        u, v, grid_cells=38, f0=2.4, omega0=0.0, bandwidth=1.60, rng=rng, isotropic=True, impulses_per_cell=3
    )
    brush_veil, _, _ = common.periodic_domain_warp_fbm(u, v, 4, 4, 0.55, rng, warp_strength=0.11, stages=2)

    fibers = 0.46 * brush_main + 0.28 * brush_cross + 0.26 * wisps_iso
    h = common.robust_normalize(
        0.34 * soft_knit + 0.44 * fibers + 0.14 * fine + 0.12 * fuzz + 0.06 * halo,
        low_pct=0.30,
        high_pct=99.70,
        soft=True,
    )
    c = common.robust_normalize(
        0.24 * soft_knit + 0.34 * fibers + 0.24 * heather + 0.18 * brush_veil + 0.10 * fine + s_tone,
        low_pct=0.35,
        high_pct=99.65,
        soft=True,
    )
    rough_raw = 0.34 * (1.0 - soft_knit) + 0.32 * fuzz + 0.24 * (1.0 - fibers) + 0.14 * brush_veil
    rough = common.match_histogram_range(rough_raw, 0.82, 0.97, target_mean=0.90, contrast=1.08)
    albedo = _blend_stops(c, fine, stops, contrast=1.06, gamma=0.96)
    base = common.shade_pbr_surface(
        albedo, h, roughness=rough, normal_strength=3.4, spec_intensity=0.03, ao_strength=0.20, cavity_strength=0.18
    )
    return base, h, rough


def _build_thermal_honeycomb(
    u: np.ndarray, v: np.ndarray, size: int, rng: np.random.Generator, stops: list[tuple[int, int, int]]
) -> tuple[Image.Image, np.ndarray, np.ndarray]:
    """Staggered hexagonal brioche thermal honeycomb knit with diagonal tuck-stitch arches."""
    uw, vw = _warp_uv(u, v, freq=3, amp=0.016, rng=rng, octaves=3)
    fine, fuzz = _yarn_micro(u, v, rng)
    heather = _heather_melange(uw, vw, rng, scale=56)
    drape, _, _ = common.periodic_domain_warp_fbm(u, v, 3, 4, 0.52, rng, warp_strength=0.09)

    cols, rows = 15, 18  # cols odd, rows even for half-row stagger periodicity
    slub = _course_slub(uw, vw, rows * 2, rng)
    sub_purl, _ = _purl_bump_field(uw, vw, cols=cols * 3, rows=rows * 2, slub=slub, rng=rng, r_yarn=0.23, ply_freq=4.2)

    gx = uw * cols
    gy = vw * rows
    iy_base = np.floor(gy).astype(np.int32)

    f1 = np.full_like(u, 1e9, dtype=np.float32)
    f2 = np.full_like(u, 1e9, dtype=np.float32)
    best_dx = np.zeros_like(u, dtype=np.float32)
    best_dy = np.zeros_like(v, dtype=np.float32)
    cell_var_lut = rng.uniform(0.86, 1.14, size=(rows, cols)).astype(np.float32)
    cell_jx_lut = rng.uniform(-0.07, 0.07, size=(rows, cols)).astype(np.float32)
    cell_jy_lut = rng.uniform(-0.07, 0.07, size=(rows, cols)).astype(np.float32)
    cell_var = np.ones_like(u, dtype=np.float32)

    aspect = 1.12
    for dy in (-1, 0, 1):
        cy_i = (iy_base + dy) % rows
        stagger = (cy_i % 2).astype(np.float32) * 0.5
        gx_s = gx - stagger
        ix_s = np.floor(gx_s).astype(np.int32)
        for dx in (-1, 0, 1):
            cx_i = (ix_s + dx) % cols
            rx = (ix_s + dx + 0.5 + cell_jx_lut[cy_i, cx_i]) - gx_s
            ry = ((iy_base + dy + 0.5 + cell_jy_lut[cy_i, cx_i]) - gy) * aspect
            dist = np.sqrt(rx * rx + ry * ry + 1e-10)
            closer1 = dist < f1
            between = (~closer1) & (dist < f2)
            f2 = np.where(closer1, f1, np.where(between, dist, f2))
            f1 = np.where(closer1, dist, f1)
            best_dx = np.where(closer1, rx, best_dx)
            best_dy = np.where(closer1, ry, best_dy)
            cell_var = np.where(closer1, cell_var_lut[cy_i, cx_i], cell_var)

    edge = np.clip(f2 - f1, 0.0, 1.0)
    rib_w = 0.26 * (0.88 + 0.24 * slub)
    d_rib = np.clip(edge / np.maximum(rib_w, 0.06), 0.0, 1.0)
    rib_arch = np.sqrt(np.clip(1.0 - d_rib * d_rib, 0.0, 1.0))

    ang_cell = np.arctan2(best_dy, best_dx)
    rib_twist = 0.76 + 0.24 * np.power(0.5 + 0.5 * np.cos((ang_cell * 6.0 + d_rib * 2.0) * 2.0), 0.75)
    knot = np.exp(-((f1 - 0.56) / 0.14) ** 2) * (1.0 - d_rib)
    rib_h = (rib_arch * rib_twist + 0.18 * knot) * cell_var

    pocket_mask = np.clip((edge - rib_w * 0.65) / (rib_w * 0.65), 0.0, 1.0)
    comb = np.maximum(rib_h, pocket_mask * (0.12 + 0.36 * sub_purl))

    h = common.robust_normalize(
        0.68 * comb + 0.12 * fine + 0.09 * fuzz + 0.09 * sub_purl * (1.0 - 0.5 * rib_arch) + 0.06 * drape,
        low_pct=0.25,
        high_pct=99.75,
        soft=True,
    )
    c = common.robust_normalize(
        0.46 * comb + 0.26 * heather + 0.12 * slub + 0.12 * fine + 0.10 * drape - 0.14 * pocket_mask,
        low_pct=0.30,
        high_pct=99.70,
        soft=True,
    )
    rough_raw = 0.42 * (1.0 - comb) + 0.24 * pocket_mask + 0.22 * fuzz + 0.14 * (1.0 - fine)
    rough = common.match_histogram_range(rough_raw, 0.76, 0.94, target_mean=0.85, contrast=1.10)
    albedo = _blend_stops(c, fine, stops, contrast=1.12, gamma=1.00)
    base = common.shade_pbr_surface(
        albedo, h, roughness=rough, normal_strength=5.2, spec_intensity=0.04, ao_strength=0.30, cavity_strength=0.26
    )
    return base, h, rough


def _build_sherpa_fleece(
    u: np.ndarray, v: np.ndarray, size: int, rng: np.random.Generator, stops: list[tuple[int, int, int]]
) -> tuple[Image.Image, np.ndarray, np.ndarray]:
    """High-pile curly shearling/sherpa fleece: two-scale swirled curly wool locks."""
    uw, vw = _warp_uv(u, v, freq=5, amp=0.032, rng=rng, octaves=3)
    vx, vy = common.periodic_curl_noise(u, v, freq=6, octaves=3, persistence=0.55, rng=rng)
    uc = (uw + 0.022 * vx) % 1.0
    vc = (vw + 0.022 * vy) % 1.0

    f1_m, edge_m, cid_m, dx_m, dy_m = common.periodic_lloyd_voronoi(uc, vc, 15, 0.90, rng, lloyd_iters=1)
    f1_s, edge_s, cid_s, dx_s, dy_s = common.periodic_lloyd_voronoi(uc, vc, 34, 0.92, rng, lloyd_iters=1)

    lock_m = np.power(np.clip(1.0 - np.exp(-edge_m * 4.2), 0.0, 1.0), 0.55) * (0.76 + 0.24 * cid_m)
    lock_s = np.power(np.clip(1.0 - np.exp(-edge_s * 3.8), 0.0, 1.0), 0.60) * (0.78 + 0.22 * cid_s)
    crevice = np.exp(-np.power(edge_m * 3.4, 1.4))

    ang_m = np.arctan2(dy_m, dx_m)
    ringlet = 0.5 + 0.5 * np.sin(f1_m * 11.0 + ang_m * 2.0 + cid_m * 6.28)
    ang_s = np.arctan2(dy_s, dx_s)
    curl_s = 0.5 + 0.5 * np.sin(f1_s * 9.0 - ang_s * 1.0 + cid_s * 6.28)

    wool_g1 = common.periodic_gabor_noise(
        uc, vc, grid_cells=28, f0=2.4, omega0=0.6, bandwidth=1.35, rng=rng, isotropic=True, impulses_per_cell=3
    )
    wool_g2 = common.periodic_gabor_noise(
        u, v, grid_cells=40, f0=2.2, omega0=0.0, bandwidth=1.60, rng=rng, isotropic=True, impulses_per_cell=2
    )
    fine, fuzz = _yarn_micro(u, v, rng)
    cream_mottle = common.periodic_fbm(u, v, 4, 3, 0.50, rng, mode="fbm")

    tuft = 0.54 * lock_m * (0.74 + 0.26 * ringlet) + 0.46 * lock_s * (0.78 + 0.22 * curl_s)
    h = common.robust_normalize(
        0.60 * tuft + 0.18 * wool_g1 + 0.14 * wool_g2 + 0.10 * fine + 0.08 * fuzz - 0.18 * crevice,
        low_pct=0.25,
        high_pct=99.75,
        soft=True,
    )
    c = common.robust_normalize(
        0.50 * tuft + 0.18 * cream_mottle + 0.14 * wool_g1 + 0.10 * fine - 0.24 * crevice - 0.08 * fuzz,
        low_pct=0.30,
        high_pct=99.70,
        soft=True,
    )
    rough_raw = 0.44 * crevice + 0.28 * fuzz + 0.20 * (1.0 - tuft) + 0.14 * (1.0 - wool_g1)
    rough = common.match_histogram_range(rough_raw, 0.84, 0.98, target_mean=0.92, contrast=1.08)
    albedo = _blend_stops(c, fine, stops, contrast=1.08, gamma=0.96)
    base = common.shade_pbr_surface(
        albedo, h, roughness=rough, normal_strength=4.8, spec_intensity=0.025, ao_strength=0.32, cavity_strength=0.26
    )
    return base, h, rough


def _build_seed_stitch(
    u: np.ndarray, v: np.ndarray, size: int, rng: np.random.Generator, stops: list[tuple[int, int, int]]
) -> tuple[Image.Image, np.ndarray, np.ndarray]:
    """Chunky hand-knitted moss/seed stitch alternating knit V-loops and protruding purl nubs."""
    uw, vw = _warp_uv(u, v, freq=3, amp=0.017, rng=rng, octaves=3)
    cols, rows = 26, 30  # 13x15 two-stitch repeats (odd repeat counts avoid quadrant phase lock!)
    slub = _course_slub(uw, vw, rows, rng)
    fine, fuzz = _yarn_micro(u, v, rng)
    heather = _heather_melange(uw, vw, rng, scale=48)
    wool_nap = common.periodic_gabor_noise(
        u, v, grid_cells=36, f0=2.2, omega0=0.0, bandwidth=1.6, rng=rng, isotropic=True, impulses_per_cell=3
    )
    drape, _, _ = common.periodic_domain_warp_fbm(u, v, 3, 4, 0.52, rng, warp_strength=0.09)

    knit_h, knit_halo, twist, s_tone = _knit_vloop_field(
        uw, vw, cols, rows, slub, rng, r_yarn=0.238, ply_freq=4.6, tension_jitter=0.09
    )
    purl_h, purl_halo = _purl_bump_field(uw, vw, cols, rows, slub, rng, r_yarn=0.258, ply_freq=4.5)

    gx = uw * cols
    gy = vw * rows
    checker_raw = np.sin(gx * math.pi) * np.sin(gy * math.pi)
    is_purl = 0.5 + 0.5 * np.tanh(4.5 * checker_raw)

    seed_h = is_purl * (0.14 + 0.90 * purl_h) + (1.0 - is_purl) * (0.04 + 0.78 * knit_h)
    halo = is_purl * purl_halo + (1.0 - is_purl) * knit_halo

    h = common.robust_normalize(
        0.66 * seed_h + 0.10 * halo + 0.11 * wool_nap + 0.10 * fine + 0.08 * fuzz + 0.06 * drape,
        low_pct=0.25,
        high_pct=99.75,
        soft=True,
    )
    c = common.robust_normalize(
        0.42 * seed_h + 0.26 * heather + 0.12 * slub + 0.12 * fine + 0.10 * wool_nap + 0.08 * drape + s_tone,
        low_pct=0.30,
        high_pct=99.70,
        soft=True,
    )
    rough_raw = 0.42 * (1.0 - seed_h) + 0.26 * fuzz + 0.18 * wool_nap + 0.14 * (1.0 - twist)
    rough = common.match_histogram_range(rough_raw, 0.76, 0.94, target_mean=0.86, contrast=1.10)
    albedo = _blend_stops(c, fine, stops, contrast=1.12, gamma=1.00)
    base = common.shade_pbr_surface(
        albedo, h, roughness=rough, normal_strength=5.4, spec_intensity=0.04, ao_strength=0.30, cavity_strength=0.26
    )
    return base, h, rough


_BUILDERS = (
    _build_jersey_heather,
    _build_rib_2x2,
    _build_cable_aran,
    _build_stockinette_merino,
    _build_polar_fleece,
    _build_french_terry,
    _build_waffle_knit,
    _build_crochet_shell,
    _build_brushed_mohair,
    _build_thermal_honeycomb,
    _build_sherpa_fleece,
    _build_seed_stitch,
)


def _compute_bundle(
    seed: int = 1,
    palette: PaletteArg = None,
    size: int = 512,
    variant: int = 0,
) -> tuple[Image.Image, np.ndarray, np.ndarray, KnitStyleSpec]:
    style = KNIT_STYLES[int(variant) % len(KNIT_STYLES)]
    rng = np.random.default_rng(int(seed))
    u, v = common.make_uv_grid(int(size))
    stops = common.resolve_palette(palette, style.default_palette, rng)
    base, h, rough = _BUILDERS[style.variant](u, v, int(size), rng, stops)
    return base, np.asarray(h, dtype=np.float32), np.asarray(rough, dtype=np.float32), style


def render_knit_bundle(
    style_or_variant: str | int,
    seed: int = 1,
    size: int = 2048,
    palette: PaletteArg = None,
) -> tuple[Image.Image, Image.Image, Image.Image, np.ndarray, np.ndarray, KnitStyleSpec]:
    """Single-pass PBR bundle renderer returning (base, normal, roughness, height, rough_field, style)."""
    if isinstance(style_or_variant, str):
        style = STYLE_BY_SLUG[style_or_variant]
        variant = style.variant
    else:
        variant = int(style_or_variant) % len(KNIT_STYLES)
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
    name="knits-fleece",
    kind="texture",
    tileable=True,
    origin="tools/generators/types/knits_fleece.py",
    default_palette="heather_cotton_jersey",
    description=(
        "Apparel knits & fleece suite (12 variants: combed heather jersey, 2x2 rib knit, "
        "Aran cable knit, merino stockinette, anti-pill polar fleece, loopback French terry, "
        "thermal waffle knit, hand crochet shell, brushed mohair knit, thermal honeycomb, "
        "curly sherpa fleece, and chunky seed stitch) with calibrated PBR height, normal, and roughness."
    ),
    render_fn=generate,
    normal_source="native",
    normal_rule="Native per-style knit & fleece relief height field (h) converted with calibrated Scharr normal strength.",
    normal_strength=4.8,
    height_fn=generate_height,
    normal_fn=generate_normal,
    roughness_fn=generate_roughness,
    default_roughness=0.86,
)
