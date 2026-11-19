"""Procedural clothing denim & workwear generator suite (12 physically grounded styles).

Built strictly on the committed periodic math primitives in ``tools/generators/common.py``:
  - Authentic 3/1 right-hand (Z) & left-hand (S) denim twills, 2/1 workwear twills,
    1/1 shirting chambray, 2x1 firm cotton duck & 2x2 waxed field basketweave canvas,
    exact broken-twill 2/2 herringbone (HBT) with zero reversal float lumps, plush
    wide-wale cut-pile corduroy over woven backing, yarn-dyed railroad hickory stripe,
    and bleach-discharged Stifel Wabash dot stripe.
  - Smooth yarn visibility blending (exposing under-weft yarns cleanly in inter-warp
    gaps), half-yarn phase centering (zero seam edge bias), per-yarn 1D periodic
    dye-lot & slub tables, ring-spun vertical warp slub ("tate-ochi" streaks), pumice
    stone-wash & sun-fade cotton-core abrasion, raw cotton seed-coat botanical neps,
    and curl-warped waxed canvas pull-up creases.
  - Multi-orientation sparse Gabor cut-pile nap, Nyquist-aware rotated fBm cotton
    micro-grain, and gentle periodic drape warp so 2048px renders hold tactile fiber
    realism with seamless toroidal tiling and zero quadrant phase lock.
  - Calibrated per-style PBR roughness via ``match_histogram_range`` and per-style
    Scharr normal strengths matched to physical fabric relief angles (~20-32 deg).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from PIL import Image

from .. import common
from ..interface import GeneratorSpec, PaletteArg


@dataclass(frozen=True)
class DenimStyleSpec:
    variant: int
    slug: str
    label: str
    default_palette: str
    normal_strength: float
    rough_range: tuple[float, float]
    description: str


DENIM_STYLES: tuple[DenimStyleSpec, ...] = (
    DenimStyleSpec(
        0,
        "denim-raw-selvedge-indigo",
        "Raw Selvedge Indigo Denim",
        "raw_indigo_selvedge",
        2.5,
        (0.66, 0.88),
        "Heavy 14oz unwashed 3/1 right-hand twill indigo selvedge denim with ring-spun warp slub, raw cotton neps, and ecru weft show-through.",
    ),
    DenimStyleSpec(
        1,
        "denim-stone-washed-indigo",
        "Stone-Washed Mid Indigo Denim",
        "stonewash_mid_denim",
        2.4,
        (0.74, 0.94),
        "Pumice stone-washed 3/1 right-hand twill denim with mottled indigo-to-white crown abrasion and pronounced vertical slub grain.",
    ),
    DenimStyleSpec(
        2,
        "denim-faded-vintage-sky",
        "Faded Vintage Sky Denim",
        "vintage_faded_sky_denim",
        2.3,
        (0.78, 0.96),
        "Sun-and-wash-faded 3/1 left-hand twill vintage denim in pale sky blue with rain-fade warp streaks and soft brushed cotton nap.",
    ),
    DenimStyleSpec(
        3,
        "denim-black-washed-charcoal",
        "Washed Black Charcoal Denim",
        "washed_black_sulphur_denim",
        2.5,
        (0.72, 0.92),
        "Sulphur-dyed washed black 3/1 twill denim with charcoal salt-and-pepper slub fades and muted ash-grey weft.",
    ),
    DenimStyleSpec(
        4,
        "denim-ecru-natural-bull",
        "Natural Ecru Bull Denim",
        "ecru_natural_bull_denim",
        2.6,
        (0.76, 0.94),
        "Heavy unbleached raw cotton 3/1 bull denim in warm cream with authentic dark botanical cotton-seed hull flecks.",
    ),
    DenimStyleSpec(
        5,
        "workwear-chambray-indigo",
        "Indigo Workwear Chambray",
        "workwear_indigo_chambray",
        2.1,
        (0.74, 0.92),
        "Lightweight 1/1 plain-weave workshirt chambray pairing rope-dyed indigo warp yarns with crisp unbleached white weft.",
    ),
    DenimStyleSpec(
        6,
        "workwear-canvas-duck-brown",
        "Heavy Canvas Duck Brown",
        "carhartt_duck_brown",
        2.7,
        (0.74, 0.94),
        "Rugged 12oz tobacco-brown cotton duck canvas built on paired twin-warp plain duck weave with golden crown wear.",
    ),
    DenimStyleSpec(
        7,
        "workwear-waxed-canvas-olive",
        "Waxed Field Canvas Olive",
        "waxed_field_olive_canvas",
        2.4,
        (0.44, 0.78),
        "Paraffin-waxed olive 2/2 basketweave field duck canvas with semi-matte waxy face sheen and chalky pull-up crease patina.",
    ),
    DenimStyleSpec(
        8,
        "workwear-corduroy-8wale-rust",
        "Wide-Wale Corduroy Rust",
        "rust_widewale_corduroy",
        4.2,
        (0.80, 0.98),
        "Plush 8-wale workwear corduroy in warm rust with directional velveteen cut-pile Gabor tufts over woven ground channels.",
    ),
    DenimStyleSpec(
        9,
        "workwear-herringbone-twill-olive",
        "Chore Herringbone Twill Olive",
        "chore_hbt_olive_drab",
        2.5,
        (0.74, 0.93),
        "Vintage military and chore-coat olive drab 2/2 broken-twill herringbone (HBT) with chevron light reversal and cotton slub.",
    ),
    DenimStyleSpec(
        10,
        "workwear-hickory-stripe-indigo",
        "Railroad Hickory Stripe",
        "railroad_hickory_stripe",
        2.3,
        (0.72, 0.92),
        "Classic yarn-dyed railroad engineer 2/1 twill hickory stripe alternating indigo and natural-ecru warp bands over ecru weft.",
    ),
    DenimStyleSpec(
        11,
        "workwear-wabash-stripe-indigo",
        "Heritage Wabash Dot Stripe",
        "heritage_wabash_indigo",
        2.3,
        (0.70, 0.90),
        "Deep indigo 2/1 workshirt twill with yarn-wicked bleach-discharged vertical oval pin-dot stripes and subtle slub wear.",
    ),
)

STYLE_BY_SLUG: dict[str, DenimStyleSpec] = {s.slug: s for s in DENIM_STYLES}


# ---------------------------------------------------------------------------
# Shared toroidal textile primitives (built strictly on common.py)
# ---------------------------------------------------------------------------


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
    """Gentle periodic fabric tension/drape UV warp preserving exact T^2 tiling."""
    du = common.periodic_fbm(u, v, freq, octaves, 0.52, rng, mode="fbm") - 0.5
    dv = common.periodic_fbm(u, v, freq, octaves, 0.52, rng, mode="fbm") - 0.5
    return _wrap01(u + du * float(amp)), _wrap01(v + dv * float(amp))


def _cotton_micro(u: np.ndarray, v: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(fine_fiber, micro_fuzz)`` resolving crisp spun cotton at 2048px."""
    fine = common.periodic_fbm(u, v, 56, 3, 0.60, rng, mode="fbm")
    fuzz = common.periodic_fbm(u, v, 124, 2, 0.64, rng, mode="turbulence")
    return fine.astype(np.float32), fuzz.astype(np.float32)


def _yarn_slub(
    uw: np.ndarray,
    vw: np.ndarray,
    threads_u: int,
    threads_v: int,
    rng: np.random.Generator,
    warp_freqy: int = 6,
    weft_freqx: int = 6,
) -> tuple[np.ndarray, np.ndarray]:
    """Ring-spun directional slub: vertical along warp ends, horizontal along weft picks."""
    ws = common.periodic_fbm(
        uw, vw, min(116, max(24, threads_u)), 3, 0.55, rng, mode="fbm", freqy=warp_freqy
    )
    fs = common.periodic_fbm(
        uw, vw, weft_freqx, 2, 0.54, rng, mode="fbm", freqy=min(116, max(24, threads_v))
    )
    return ws.astype(np.float32), fs.astype(np.float32)


def _twill_matrix(over: int, under: int, step: int = 1) -> np.ndarray:
    """Build exact periodic (P, P) twill matrix with P = over + under (step=+1 RHT, -1 LHT)."""
    p = over + under
    idx = np.arange(p)
    ii, jj = np.meshgrid(idx, idx)
    return (((ii + step * jj) % p) < over).astype(np.int32)


def _herringbone_matrix(band_w: int = 6) -> np.ndarray:
    """Build exact periodic (4, 2*band_w) broken 2/2 twill herringbone matrix.

    For even ``band_w`` (e.g. 6), every reversal boundary is an exact binary complement,
    eliminating 3-pick float lumps while keeping exact T^2 weave continuity.
    """
    p = 2 * band_w
    m = np.zeros((4, p), dtype=np.int32)
    for col in range(p):
        band_idx = col // band_w
        direction = 1 if (band_idx % 2 == 0) else -1
        break_shift = 0 if (band_w % 4 == 2) else (2 if (band_idx % 2 == 1) else 0)
        for row in range(4):
            m[row, col] = 1 if ((col + direction * row + break_shift) % 4) < 2 else 0
    return m


def _smooth_1d_periodic(n: int, rng: np.random.Generator, amp: float = 1.0) -> np.ndarray:
    """1D periodic per-yarn variation combining independent thread noise and slight neighbor drift."""
    raw = rng.normal(0.0, 1.0, size=n).astype(np.float32)
    smooth = 0.72 * raw + 0.14 * np.roll(raw, 1) + 0.14 * np.roll(raw, -1)
    mx = float(np.percentile(np.abs(smooth), 98)) + 1e-6
    return np.clip(smooth / mx, -1.0, 1.0).astype(np.float32) * float(amp)


def _woven_cloth(
    uw: np.ndarray,
    vw: np.ndarray,
    threads_u: int,
    threads_v: int,
    matrix: np.ndarray,
    warp_slub: np.ndarray,
    weft_slub: np.ndarray,
    rng: np.random.Generator,
    size: int,
    float_arch: float = 0.38,
    ply_twist: float = 0.11,
    warp_rad_base: float = 0.91,
    weft_rad_base: float = 0.89,
    slub_rad_amp: float = 0.13,
) -> dict[str, np.ndarray]:
    """Enhanced woven cloth evaluator wrapping ``common.woven_fabric_field``.

    - Centers tile edges on smooth yarn crowns via a half-thread phase shift (eliminating
      artificially elevated wrap-seam gradients when u=0/v=0 falls on a yarn groove).
    - Adds per-yarn radius/dye variation and smooth physical yarn visibility so exposed
      under-weft threads in inter-warp gaps render with accurate weft albedo.
    - Softens the infinite mathematical derivative at cylinder boundaries (``du = +-1``)
      with a sub-yarn periodic Gaussian micro-bevel so spun cotton yarns read plush and
      tactile rather than sharp machined cylinders.
    """
    m = (np.asarray(matrix) > 0).astype(np.int32)
    q, p = m.shape
    tu_n = max(p, ((int(threads_u) + p - 1) // p) * p)
    tv_n = max(q, ((int(threads_v) + q - 1) // q) * q)

    # Half-thread phase offset places tile boundaries on smooth yarn crowns instead of grooves
    uu = _wrap01(uw + 0.5 / float(tu_n))
    vv = _wrap01(vw + 0.5 / float(tv_n))

    iu = np.floor(uu * tu_n).astype(np.int32) % tu_n
    iv = np.floor(vv * tv_n).astype(np.int32) % tv_n

    warp_lot = _smooth_1d_periodic(tu_n, rng, amp=1.0)[iu]
    weft_lot = _smooth_1d_periodic(tv_n, rng, amp=1.0)[iv]
    warp_rad_j = _smooth_1d_periodic(tu_n, rng, amp=0.05)[iu]
    weft_rad_j = _smooth_1d_periodic(tv_n, rng, amp=0.05)[iv]

    warp_rad = np.clip(
        warp_rad_base + slub_rad_amp * (warp_slub - 0.5) + warp_rad_j,
        0.68,
        1.02,
    ).astype(np.float32)
    weft_rad = np.clip(
        weft_rad_base + slub_rad_amp * 0.78 * (weft_slub - 0.5) + weft_rad_j,
        0.68,
        1.02,
    ).astype(np.float32)

    weave_h, warp_over, top_u, top_v = common.woven_fabric_field(
        uu,
        vv,
        tu_n,
        tv_n,
        m,
        float_arch=float_arch,
        ply_twist=ply_twist,
        warp_radius=warp_rad,
        weft_radius=weft_rad,
    )

    # Smooth physical visibility of warp vs weft (accounts for under-yarn peeking in gaps)
    warp_vis = np.where(
        warp_over,
        np.clip((top_u - 0.08) / 0.24, 0.0, 1.0),
        np.clip(1.0 - (top_v - 0.06) / 0.22, 0.0, 1.0) * 0.55,
    ).astype(np.float32)

    # Micro-bevel softens the singular sqrt(1-du^2) cliff edge into realistic spun cotton fiber
    res_scale = float(size) / 512.0
    h_soft = common.periodic_gaussian_blur(weave_h, max(0.45, 0.55 * res_scale))
    slub_bump = warp_vis * (warp_slub - 0.5) * 0.14 + (1.0 - warp_vis) * (weft_slub - 0.5) * 0.10
    weave_h = np.clip(0.55 * weave_h + 0.45 * h_soft + slub_bump, 0.0, 1.25).astype(np.float32)

    return {
        "h": weave_h,
        "warp_over": warp_over,
        "warp_vis": warp_vis,
        "top_u": top_u,
        "top_v": top_v,
        "iu": iu,
        "iv": iv,
        "tu_n": np.int32(tu_n),
        "tv_n": np.int32(tv_n),
        "warp_lot": warp_lot,
        "weft_lot": weft_lot,
    }


def _interp_stops(field: np.ndarray, stops: list[tuple[int, int, int]]) -> np.ndarray:
    """Linearly interpolate a [0, 1] scalar field across an arbitrary RGB stop list."""
    f = np.clip(field, 0.0, 1.0).astype(np.float32)
    n_stops = len(stops)
    stop_pos = np.linspace(0.0, 1.0, n_stops, dtype=np.float32)
    out = np.empty((*f.shape, 3), dtype=np.float32)
    for ch in range(3):
        vals = np.array([float(s[ch]) for s in stops], dtype=np.float32)
        out[..., ch] = np.interp(f, stop_pos, vals)
    return out


def _denim_albedo(
    warp_t: np.ndarray,
    weft_t: np.ndarray,
    warp_vis: np.ndarray,
    stops: list[tuple[int, int, int]],
    core_fade: np.ndarray | None = None,
    weft_dark_mix: float = 0.0,
) -> np.ndarray:
    """Compose physically distinct warp (stops[0..3]) and ecru weft (stops[3..4]) albedo."""
    warp_rgb = _interp_stops(warp_t, stops[:4])
    if weft_dark_mix > 1e-4:
        w_lo = tuple(
            int(round(stops[0][c] * weft_dark_mix + stops[2][c] * (1.0 - weft_dark_mix)))
            for c in range(3)
        )
        w_hi = tuple(
            int(round(stops[2][c] * weft_dark_mix + stops[4][c] * (1.0 - weft_dark_mix)))
            for c in range(3)
        )
        weft_rgb = _interp_stops(weft_t, [w_lo, w_hi])
    else:
        w_shadow = tuple(int(round(0.45 * stops[0][c] + 0.55 * stops[4][c])) for c in range(3))
        w_mid = tuple(int(round(0.18 * stops[2][c] + 0.82 * stops[4][c])) for c in range(3))
        weft_rgb = _interp_stops(weft_t, [w_shadow, w_mid, stops[4]])

    if core_fade is not None:
        cf = np.clip(core_fade, 0.0, 1.0)[..., None]
        core_col = np.array(stops[4], dtype=np.float32)[None, None, :]
        warp_rgb = warp_rgb * (1.0 - cf) + core_col * cf

    wv = np.clip(warp_vis, 0.0, 1.0)[..., None]
    return warp_rgb * wv + weft_rgb * (1.0 - wv)


def _wash_mottle(
    u: np.ndarray,
    v: np.ndarray,
    rng: np.random.Generator,
    warp_strength: float = 0.08,
) -> np.ndarray:
    """Multi-scale balanced wash clouding distributed across frequency bands to avoid repeating blobs."""
    b_main, _, _ = common.periodic_domain_warp_fbm(
        u, v, freq=4, octaves=4, persistence=0.54, rng=rng, warp_strength=warp_strength, warp_freq=3
    )
    b_hi = common.periodic_fbm(u, v, 13, 2, 0.55, rng, mode="fbm")
    return common.robust_normalize(
        0.66 * b_main + 0.34 * b_hi,
        low_pct=0.35,
        high_pct=99.65,
        soft=True,
    )


def _cotton_neps(
    u: np.ndarray,
    v: np.ndarray,
    rng: np.random.Generator,
    scale: int = 86,
    threshold: float = 0.82,
) -> np.ndarray:
    """Sparse high-frequency cotton neps / botanical flecks with zero Cartesian alignment."""
    p1 = common.periodic_rotated_perlin(
        u, v, scale, 0.42, rng, phase_u=float(rng.uniform()), phase_v=float(rng.uniform())
    )
    p2 = common.periodic_rotated_perlin(
        u, v, int(round(scale * 1.43)), 1.18, rng, phase_u=float(rng.uniform()), phase_v=float(rng.uniform())
    )
    raw = p1 * p2
    return np.clip((raw - threshold * threshold) / max(1e-4, 1.0 - threshold * threshold), 0.0, 1.0).astype(
        np.float32
    )


# ---------------------------------------------------------------------------
# 12 Physically Grounded Denim & Workwear Builders
# ---------------------------------------------------------------------------


def _build_raw_selvedge_indigo(
    u: np.ndarray, v: np.ndarray, size: int, rng: np.random.Generator, stops: list[tuple[int, int, int]]
) -> tuple[Image.Image, np.ndarray, np.ndarray]:
    """Heavy 14oz unwashed 3/1 RHT raw selvedge indigo denim."""
    uw, vw = _warp_uv(u, v, freq=3, amp=0.010, rng=rng, octaves=3)
    tu, tv = 88, 92  # 22x4, 23x4 (even*4 x odd*4 eliminates hy & diag half-tile twill locks)
    warp_slub, weft_slub = _yarn_slub(uw, vw, tu, tv, rng, warp_freqy=5, weft_freqx=6)
    fine, fuzz = _cotton_micro(u, v, rng)
    mottle = _wash_mottle(u, v, rng, warp_strength=0.06)
    neps = _cotton_neps(uw, vw, rng, scale=78, threshold=0.81)

    cloth = _woven_cloth(
        uw,
        vw,
        tu,
        tv,
        _twill_matrix(3, 1, step=1),
        warp_slub,
        weft_slub,
        rng,
        size=size,
        float_arch=0.38,
        ply_twist=0.11,
        warp_rad_base=0.92,
        weft_rad_base=0.89,
        slub_rad_amp=0.15,
    )
    wh = cloth["h"]
    wv = cloth["warp_vis"]

    h = common.robust_normalize(
        0.72 * wh + 0.12 * fine + 0.08 * fuzz + 0.06 * warp_slub + 0.05 * mottle + 0.05 * neps,
        low_pct=0.25,
        high_pct=99.75,
        soft=True,
    )

    warp_t = common.robust_normalize(
        0.42 * wh
        + 0.24 * warp_slub
        + 0.14 * cloth["warp_lot"]
        + 0.12 * fine
        + 0.12 * mottle,
        low_pct=0.4,
        high_pct=99.6,
        soft=True,
    )
    weft_t = np.clip(0.38 + 0.56 * wh + 0.10 * (weft_slub - 0.5) + 0.08 * cloth["weft_lot"], 0.0, 1.0)
    core_fade = np.clip(neps * 0.72 + np.maximum(warp_slub - 0.82, 0.0) * 0.58 * wh, 0.0, 0.65)
    albedo = _denim_albedo(warp_t, weft_t, wv, stops, core_fade=core_fade)

    rough_raw = (
        0.44 * (1.0 - wh)
        + 0.22 * (1.0 - wv)
        + 0.20 * fuzz
        + 0.14 * (1.0 - warp_slub)
    )
    rough = common.match_histogram_range(rough_raw, 0.66, 0.88, target_mean=0.78, contrast=1.08)
    base = common.shade_pbr_surface(
        albedo, h, roughness=rough, normal_strength=2.5, spec_intensity=0.05, ao_strength=0.26, cavity_strength=0.22
    )
    return base, h, rough


def _build_stone_washed_indigo(
    u: np.ndarray, v: np.ndarray, size: int, rng: np.random.Generator, stops: list[tuple[int, int, int]]
) -> tuple[Image.Image, np.ndarray, np.ndarray]:
    """Pumice stone-washed mid-blue 3/1 RHT denim with chalky crown abrasion."""
    uw, vw = _warp_uv(u, v, freq=3, amp=0.012, rng=rng, octaves=3)
    tu, tv = 88, 92  # 22x4, 23x4 (even*4 x odd*4 eliminates hy & diag half-tile twill locks)
    warp_slub, weft_slub = _yarn_slub(uw, vw, tu, tv, rng, warp_freqy=5, weft_freqx=6)
    fine, fuzz = _cotton_micro(u, v, rng)
    wash = _wash_mottle(u, v, rng, warp_strength=0.09)
    pumice_ridge = common.periodic_fbm(uw, vw, 11, 3, 0.56, rng, mode="ridged")
    soft_nap = common.periodic_gabor_noise(
        u, v, grid_cells=30, f0=2.2, omega0=0.0, bandwidth=1.70, rng=rng, isotropic=True, impulses_per_cell=2
    )

    cloth = _woven_cloth(
        uw,
        vw,
        tu,
        tv,
        _twill_matrix(3, 1, step=1),
        warp_slub,
        weft_slub,
        rng,
        size=size,
        float_arch=0.38,
        ply_twist=0.10,
        warp_rad_base=0.91,
        weft_rad_base=0.90,
        slub_rad_amp=0.16,
    )
    wh = cloth["h"]
    wv = cloth["warp_vis"]

    h = common.robust_normalize(
        0.66 * wh + 0.12 * fine + 0.09 * fuzz + 0.09 * soft_nap + 0.06 * warp_slub + 0.06 * wash,
        low_pct=0.25,
        high_pct=99.75,
        soft=True,
    )

    abrasion = common.robust_normalize(
        0.38 * wash + 0.24 * pumice_ridge + 0.26 * warp_slub + 0.14 * cloth["warp_lot"] + 0.10 * fine,
        low_pct=0.5,
        high_pct=99.5,
        soft=True,
    )
    crown_mask = np.clip((wh - 0.30) / 0.58, 0.0, 1.0)
    core_fade = np.clip(np.power(abrasion, 1.60) * crown_mask * 0.56, 0.0, 0.68)

    warp_t = common.robust_normalize(
        0.36 * wh + 0.32 * abrasion + 0.18 * warp_slub + 0.12 * cloth["warp_lot"] + 0.10 * fine,
        low_pct=0.4,
        high_pct=99.6,
        soft=True,
    )
    weft_t = np.clip(0.46 + 0.50 * wh + 0.10 * (weft_slub - 0.5), 0.0, 1.0)
    albedo = _denim_albedo(warp_t, weft_t, wv, stops, core_fade=core_fade)

    rough_raw = (
        0.38 * (1.0 - wh)
        + 0.24 * fuzz
        + 0.20 * core_fade
        + 0.18 * soft_nap
    )
    rough = common.match_histogram_range(rough_raw, 0.74, 0.94, target_mean=0.85, contrast=1.08)
    base = common.shade_pbr_surface(
        albedo, h, roughness=rough, normal_strength=2.4, spec_intensity=0.035, ao_strength=0.25, cavity_strength=0.22
    )
    return base, h, rough


def _build_faded_vintage_sky(
    u: np.ndarray, v: np.ndarray, size: int, rng: np.random.Generator, stops: list[tuple[int, int, int]]
) -> tuple[Image.Image, np.ndarray, np.ndarray]:
    """Sun-faded vintage sky-blue 3/1 LHT (left-hand twill) soft denim."""
    uw, vw = _warp_uv(u, v, freq=3, amp=0.012, rng=rng, octaves=3)
    tu, tv = 88, 92  # 22x4, 23x4 (even*4 x odd*4 eliminates hy & diag half-tile twill locks)
    warp_slub, weft_slub = _yarn_slub(uw, vw, tu, tv, rng, warp_freqy=4, weft_freqx=6)
    fine, fuzz = _cotton_micro(u, v, rng)
    wash = _wash_mottle(u, v, rng, warp_strength=0.10)
    rain_fade = common.periodic_fbm(uw, vw, 38, 3, 0.56, rng, mode="fbm", freqy=3)
    soft_nap = common.periodic_gabor_noise(
        u, v, grid_cells=30, f0=2.1, omega0=0.0, bandwidth=1.65, rng=rng, isotropic=True, impulses_per_cell=2
    )

    cloth = _woven_cloth(
        uw,
        vw,
        tu,
        tv,
        _twill_matrix(3, 1, step=-1),  # Left-hand S-twill
        warp_slub,
        weft_slub,
        rng,
        size=size,
        float_arch=0.36,
        ply_twist=0.10,
        warp_rad_base=0.90,
        weft_rad_base=0.91,
        slub_rad_amp=0.15,
    )
    wh = cloth["h"]
    wv = cloth["warp_vis"]

    h = common.robust_normalize(
        0.62 * wh + 0.13 * fine + 0.11 * fuzz + 0.10 * soft_nap + 0.06 * warp_slub + 0.06 * wash,
        low_pct=0.25,
        high_pct=99.75,
        soft=True,
    )

    fade_sig = common.robust_normalize(
        0.34 * wash + 0.30 * rain_fade + 0.24 * warp_slub + 0.14 * cloth["warp_lot"],
        low_pct=0.5,
        high_pct=99.5,
        soft=True,
    )
    crown = np.clip((wh - 0.24) / 0.64, 0.0, 1.0)
    core_fade = np.clip((0.18 + 0.58 * np.power(fade_sig, 1.35)) * crown, 0.0, 0.72)

    warp_t = common.robust_normalize(
        0.34 * wh + 0.34 * fade_sig + 0.18 * warp_slub + 0.10 * fine + 0.08 * soft_nap,
        low_pct=0.4,
        high_pct=99.6,
        soft=True,
    )
    weft_t = np.clip(0.54 + 0.44 * wh + 0.08 * (weft_slub - 0.5), 0.0, 1.0)
    albedo = _denim_albedo(warp_t, weft_t, wv, stops, core_fade=core_fade)

    rough_raw = 0.36 * (1.0 - wh) + 0.26 * fuzz + 0.22 * soft_nap + 0.16 * core_fade
    rough = common.match_histogram_range(rough_raw, 0.78, 0.96, target_mean=0.88, contrast=1.06)
    base = common.shade_pbr_surface(
        albedo, h, roughness=rough, normal_strength=2.3, spec_intensity=0.03, ao_strength=0.22, cavity_strength=0.20
    )
    return base, h, rough


def _build_washed_black_denim(
    u: np.ndarray, v: np.ndarray, size: int, rng: np.random.Generator, stops: list[tuple[int, int, int]]
) -> tuple[Image.Image, np.ndarray, np.ndarray]:
    """Sulphur-dyed washed black/charcoal 3/1 RHT denim with ash-grey slub fade."""
    uw, vw = _warp_uv(u, v, freq=3, amp=0.011, rng=rng, octaves=3)
    tu, tv = 88, 92  # 22x4, 23x4 (even*4 x odd*4 eliminates hy & diag half-tile twill locks)
    warp_slub, weft_slub = _yarn_slub(uw, vw, tu, tv, rng, warp_freqy=5, weft_freqx=6)
    fine, fuzz = _cotton_micro(u, v, rng)
    wash = _wash_mottle(u, v, rng, warp_strength=0.08)
    charcoal_ridge = common.periodic_fbm(uw, vw, 13, 3, 0.55, rng, mode="ridged")

    cloth = _woven_cloth(
        uw,
        vw,
        tu,
        tv,
        _twill_matrix(3, 1, step=1),
        warp_slub,
        weft_slub,
        rng,
        size=size,
        float_arch=0.38,
        ply_twist=0.11,
        warp_rad_base=0.92,
        weft_rad_base=0.89,
        slub_rad_amp=0.14,
    )
    wh = cloth["h"]
    wv = cloth["warp_vis"]

    h = common.robust_normalize(
        0.70 * wh + 0.13 * fine + 0.08 * fuzz + 0.06 * warp_slub + 0.05 * wash,
        low_pct=0.25,
        high_pct=99.75,
        soft=True,
    )

    ash_fade = common.robust_normalize(
        0.36 * wash + 0.28 * warp_slub + 0.20 * charcoal_ridge + 0.14 * cloth["warp_lot"] + 0.10 * fine,
        low_pct=0.5,
        high_pct=99.5,
        soft=True,
    )
    crown = np.clip((wh - 0.32) / 0.56, 0.0, 1.0)
    core_fade = np.clip(np.power(ash_fade, 1.75) * crown * 0.42, 0.0, 0.50)

    warp_t = common.robust_normalize(
        0.40 * wh + 0.30 * ash_fade + 0.16 * warp_slub + 0.10 * fine + 0.10 * cloth["warp_lot"],
        low_pct=0.35,
        high_pct=99.65,
        soft=True,
    )
    weft_t = np.clip(0.32 + 0.54 * wh + 0.12 * (weft_slub - 0.5) + 0.08 * wash, 0.0, 1.0)
    albedo = _denim_albedo(warp_t, weft_t, wv, stops, core_fade=core_fade, weft_dark_mix=0.45)

    rough_raw = 0.42 * (1.0 - wh) + 0.24 * fuzz + 0.18 * ash_fade + 0.16 * (1.0 - wv)
    rough = common.match_histogram_range(rough_raw, 0.72, 0.92, target_mean=0.83, contrast=1.08)
    base = common.shade_pbr_surface(
        albedo, h, roughness=rough, normal_strength=2.5, spec_intensity=0.04, ao_strength=0.26, cavity_strength=0.24
    )
    return base, h, rough


def _build_ecru_bull_denim(
    u: np.ndarray, v: np.ndarray, size: int, rng: np.random.Generator, stops: list[tuple[int, int, int]]
) -> tuple[Image.Image, np.ndarray, np.ndarray]:
    """Heavy unbleached natural ecru 3/1 bull denim with cotton seed-coat flecks."""
    uw, vw = _warp_uv(u, v, freq=3, amp=0.011, rng=rng, octaves=3)
    tu, tv = 80, 84  # heavier 14oz bull denim yarn gauge (20x4, 21x4 breaks hy & diag locks)
    warp_slub, weft_slub = _yarn_slub(uw, vw, tu, tv, rng, warp_freqy=5, weft_freqx=6)
    fine, fuzz = _cotton_micro(u, v, rng)
    mottle = _wash_mottle(u, v, rng, warp_strength=0.07)
    seed_flecks = _cotton_neps(uw, vw, rng, scale=68, threshold=0.80)
    fine_flecks = _cotton_neps(uw, vw, rng, scale=124, threshold=0.82)
    flecks = np.clip(seed_flecks * 0.85 + fine_flecks * 0.65, 0.0, 1.0)

    cloth = _woven_cloth(
        uw,
        vw,
        tu,
        tv,
        _twill_matrix(3, 1, step=1),
        warp_slub,
        weft_slub,
        rng,
        size=size,
        float_arch=0.40,
        ply_twist=0.12,
        warp_rad_base=0.92,
        weft_rad_base=0.90,
        slub_rad_amp=0.14,
    )
    wh = cloth["h"]
    wv = cloth["warp_vis"]

    h = common.robust_normalize(
        0.72 * wh + 0.12 * fine + 0.08 * fuzz + 0.06 * warp_slub + 0.05 * mottle + 0.04 * flecks,
        low_pct=0.25,
        high_pct=99.75,
        soft=True,
    )

    tone = common.robust_normalize(
        0.46 * wh
        + 0.14 * wv
        + 0.14 * warp_slub
        + 0.12 * cloth["warp_lot"] * wv
        + 0.08 * cloth["weft_lot"] * (1.0 - wv)
        + 0.12 * fine
        + 0.10 * mottle,
        low_pct=0.35,
        high_pct=99.65,
        soft=True,
    )
    albedo = _interp_stops(tone, stops)
    hull_rgb = np.array(
        [max(34, stops[0][0] * 0.58), max(26, stops[0][1] * 0.50), max(18, stops[0][2] * 0.42)],
        dtype=np.float32,
    )[None, None, :]
    fm = np.clip(flecks * 0.82, 0.0, 0.85)[..., None]
    albedo = albedo * (1.0 - fm) + hull_rgb * fm

    rough_raw = 0.44 * (1.0 - wh) + 0.26 * fuzz + 0.18 * (1.0 - fine) + 0.14 * flecks
    rough = common.match_histogram_range(rough_raw, 0.76, 0.94, target_mean=0.86, contrast=1.08)
    base = common.shade_pbr_surface(
        albedo, h, roughness=rough, normal_strength=2.6, spec_intensity=0.03, ao_strength=0.24, cavity_strength=0.22
    )
    return base, h, rough


def _build_indigo_chambray(
    u: np.ndarray, v: np.ndarray, size: int, rng: np.random.Generator, stops: list[tuple[int, int, int]]
) -> tuple[Image.Image, np.ndarray, np.ndarray]:
    """Lightweight 1/1 plain-weave indigo-warp / white-weft workshirt chambray."""
    uw, vw = _warp_uv(u, v, freq=4, amp=0.011, rng=rng, octaves=3)
    tu, tv = 114, 110  # finer shirting weight (57x2, 55x2 -> ~18px/yarn at 2048px)
    warp_slub, weft_slub = _yarn_slub(uw, vw, tu, tv, rng, warp_freqy=7, weft_freqx=7)
    fine, fuzz = _cotton_micro(u, v, rng)
    mottle = _wash_mottle(u, v, rng, warp_strength=0.07)
    plain_1x1 = np.array([[1, 0], [0, 1]], dtype=np.int32)

    cloth = _woven_cloth(
        uw,
        vw,
        tu,
        tv,
        plain_1x1,
        warp_slub,
        weft_slub,
        rng,
        size=size,
        float_arch=0.42,
        ply_twist=0.08,
        warp_rad_base=0.89,
        weft_rad_base=0.89,
        slub_rad_amp=0.14,
    )
    wh = cloth["h"]
    wv = cloth["warp_vis"]

    h = common.robust_normalize(
        0.68 * wh + 0.13 * fine + 0.09 * fuzz + 0.07 * (0.5 * warp_slub + 0.5 * weft_slub) + 0.07 * mottle,
        low_pct=0.25,
        high_pct=99.75,
        soft=True,
    )

    warp_t = common.robust_normalize(
        0.40 * wh + 0.24 * warp_slub + 0.16 * cloth["warp_lot"] + 0.16 * mottle + 0.10 * fine,
        low_pct=0.4,
        high_pct=99.6,
        soft=True,
    )
    weft_t = np.clip(0.50 + 0.46 * wh + 0.10 * (weft_slub - 0.5) + 0.08 * cloth["weft_lot"], 0.0, 1.0)
    albedo = _denim_albedo(warp_t, weft_t, wv, stops, core_fade=None)

    rough_raw = 0.42 * (1.0 - wh) + 0.26 * fuzz + 0.18 * (1.0 - fine) + 0.14 * mottle
    rough = common.match_histogram_range(rough_raw, 0.74, 0.92, target_mean=0.84, contrast=1.06)
    base = common.shade_pbr_surface(
        albedo, h, roughness=rough, normal_strength=2.1, spec_intensity=0.035, ao_strength=0.22, cavity_strength=0.20
    )
    return base, h, rough


def _build_canvas_duck_brown(
    u: np.ndarray, v: np.ndarray, size: int, rng: np.random.Generator, stops: list[tuple[int, int, int]]
) -> tuple[Image.Image, np.ndarray, np.ndarray]:
    """Heavy 12oz tobacco-brown firm cotton duck canvas (paired twin-warp plain duck)."""
    uw, vw = _warp_uv(u, v, freq=3, amp=0.011, rng=rng, octaves=3)
    duck_matrix = np.array([[1, 1, 0, 0], [0, 0, 1, 1]], dtype=np.int32)
    tu, tv = 100, 66  # 25x4 warp ends (50 twin pairs), 33x2 thick weft picks
    warp_slub, weft_slub = _yarn_slub(uw, vw, tu, tv, rng, warp_freqy=6, weft_freqx=5)
    fine, fuzz = _cotton_micro(u, v, rng)
    wear = _wash_mottle(u, v, rng, warp_strength=0.08)

    cloth = _woven_cloth(
        uw,
        vw,
        tu,
        tv,
        duck_matrix,
        warp_slub,
        weft_slub,
        rng,
        size=size,
        float_arch=0.44,
        ply_twist=0.12,
        warp_rad_base=0.93,
        weft_rad_base=0.92,
        slub_rad_amp=0.13,
    )
    wh = cloth["h"]
    wv = cloth["warp_vis"]

    h = common.robust_normalize(
        0.72 * wh + 0.13 * fine + 0.08 * fuzz + 0.06 * (0.55 * warp_slub + 0.45 * weft_slub) + 0.06 * wear,
        low_pct=0.25,
        high_pct=99.75,
        soft=True,
    )

    knuckle_wear = np.clip((wh - 0.42) / 0.52, 0.0, 1.0) * (0.45 + 0.55 * wear)
    tone = common.robust_normalize(
        0.42 * wh
        + 0.20 * knuckle_wear
        + 0.16 * wear
        + 0.12 * fine
        + 0.10 * (wv * warp_slub + (1.0 - wv) * weft_slub)
        + 0.10 * (wv * cloth["warp_lot"] + (1.0 - wv) * cloth["weft_lot"])
        + 0.06 * wv,
        low_pct=0.35,
        high_pct=99.65,
        soft=True,
    )
    albedo = _interp_stops(tone, stops)

    rough_raw = 0.44 * (1.0 - wh) + 0.24 * fuzz + 0.18 * (1.0 - fine) + 0.14 * knuckle_wear
    rough = common.match_histogram_range(rough_raw, 0.74, 0.94, target_mean=0.85, contrast=1.10)
    base = common.shade_pbr_surface(
        albedo, h, roughness=rough, normal_strength=2.7, spec_intensity=0.035, ao_strength=0.28, cavity_strength=0.24
    )
    return base, h, rough


def _build_waxed_canvas_olive(
    u: np.ndarray, v: np.ndarray, size: int, rng: np.random.Generator, stops: list[tuple[int, int, int]]
) -> tuple[Image.Image, np.ndarray, np.ndarray]:
    """Paraffin-waxed olive 2/2 basketweave field canvas with chalky pull-up creases."""
    uw, vw = _warp_uv(u, v, freq=3, amp=0.011, rng=rng, octaves=3)
    basket_2x2 = np.array(
        [
            [1, 1, 0, 0],
            [1, 1, 0, 0],
            [0, 0, 1, 1],
            [0, 0, 1, 1],
        ],
        dtype=np.int32,
    )
    tu, tv = 88, 92  # 22x4, 23x4
    warp_slub, weft_slub = _yarn_slub(uw, vw, tu, tv, rng, warp_freqy=6, weft_freqx=6)
    fine, fuzz = _cotton_micro(u, v, rng)
    wax_patina = _wash_mottle(u, v, rng, warp_strength=0.09)

    c_u, c_v = common.periodic_curl_noise(u, v, 3, 3, 0.52, rng)
    uc = _wrap01(u + c_u * 0.045)
    vc = _wrap01(v + c_v * 0.045)
    ridge_broad = common.periodic_fbm(uc, vc, 4, 3, 0.52, rng, mode="ridged")
    ridge_fine = common.periodic_fbm(uc, vc, 10, 3, 0.55, rng, mode="ridged")
    crease_lines = np.clip(
        0.55 * np.power(ridge_broad, 3.4) + 0.45 * np.power(ridge_fine, 3.8),
        0.0,
        1.0,
    ).astype(np.float32)

    cloth = _woven_cloth(
        uw,
        vw,
        tu,
        tv,
        basket_2x2,
        warp_slub,
        weft_slub,
        rng,
        size=size,
        float_arch=0.38,
        ply_twist=0.09,
        warp_rad_base=0.93,
        weft_rad_base=0.93,
        slub_rad_amp=0.11,
    )
    wh = cloth["h"]
    wv = cloth["warp_vis"]

    wax_fill = common.periodic_gaussian_blur(wh, max(0.9, 1.5 * (size / 512.0)))
    wh_waxed = 0.68 * wh + 0.32 * wax_fill

    h = common.robust_normalize(
        0.70 * wh_waxed + 0.12 * fine + 0.06 * fuzz + 0.08 * wax_patina - 0.09 * crease_lines,
        low_pct=0.25,
        high_pct=99.75,
        soft=True,
    )

    tone = common.robust_normalize(
        0.34 * wh_waxed
        + 0.36 * crease_lines
        + 0.18 * wax_patina
        + 0.10 * fine
        + 0.07 * (wv * cloth["warp_lot"] + (1.0 - wv) * cloth["weft_lot"]),
        low_pct=0.35,
        high_pct=99.65,
        soft=True,
    )
    albedo = _interp_stops(tone, stops)

    rough_raw = (
        0.36 * crease_lines
        + 0.32 * (1.0 - wh_waxed)
        + 0.18 * fuzz
        + 0.14 * (1.0 - wax_patina)
    )
    rough = common.match_histogram_range(rough_raw, 0.44, 0.78, target_mean=0.60, contrast=1.12)
    base = common.shade_pbr_surface(
        albedo, h, roughness=rough, normal_strength=2.4, spec_intensity=0.13, ao_strength=0.26, cavity_strength=0.22
    )
    return base, h, rough


def _build_corduroy_rust(
    u: np.ndarray, v: np.ndarray, size: int, rng: np.random.Generator, stops: list[tuple[int, int, int]]
) -> tuple[Image.Image, np.ndarray, np.ndarray]:
    """Plush 8-wale workwear corduroy in warm rust with Gabor cut-pile tufts & woven backing."""
    uw, vw = _warp_uv(u, v, freq=3, amp=0.011, rng=rng, octaves=3)
    wales = 17  # odd wale count (~120px/wale at 2048px)
    fine, fuzz = _cotton_micro(u, v, rng)
    brush_mottle = _wash_mottle(u, v, rng, warp_strength=0.10)
    # Velveteen nap crush: organic brushed light/dark patches breaking uniform vertical tubes
    pile_crush, _, _ = common.periodic_domain_warp_fbm(
        u, v, freq=5, octaves=3, persistence=0.55, rng=rng, warp_strength=0.12, warp_freq=3
    )

    nap_vert = common.periodic_gabor_noise(
        uw, vw, grid_cells=36, f0=2.6, omega0=0.0, bandwidth=1.45, rng=rng, impulses_per_cell=3
    )
    nap_velvet = common.periodic_gabor_noise(
        u, v, grid_cells=42, f0=2.5, omega0=0.0, bandwidth=1.65, rng=rng, isotropic=True, impulses_per_cell=3
    )
    bristle = common.periodic_fbm(uw, vw, min(144, wales * 8), 3, 0.60, rng, mode="fbm", freqy=28)

    # Half-wale offset puts tile edge cleanly on a plush wale crown rather than trough wall
    uu = _wrap01(uw + 0.5 / float(wales))
    gx = uu * wales
    iw = np.floor(gx).astype(np.int32) % wales
    wu = gx - np.floor(gx)

    w_half = (0.375 + rng.uniform(-0.024, 0.024, size=wales)).astype(np.float32)[iw]
    w_amp = rng.uniform(0.90, 1.10, size=wales).astype(np.float32)[iw]
    w_lean = rng.uniform(-0.026, 0.026, size=wales).astype(np.float32)[iw]
    w_tone = rng.uniform(-0.075, 0.075, size=wales).astype(np.float32)[iw]

    splay = (nap_velvet - 0.5) * 0.16 + (bristle - 0.5) * 0.11 + (pile_crush - 0.5) * 0.06
    du = np.abs(wu - 0.5 + w_lean) / np.maximum(w_half, 0.18) + splay
    crown = np.power(np.clip(1.0 - np.power(np.clip(du, 0.0, 1.0), 2.35), 0.0, 1.0), 0.58)
    wale_mask = np.clip((1.05 - du) / 0.24, 0.0, 1.0)

    plain_1x1 = np.array([[1, 0], [0, 1]], dtype=np.int32)
    base_weave, _, _, _ = common.woven_fabric_field(
        uu, vw, wales * 6, wales * 6, plain_1x1, float_arch=0.38, ply_twist=0.08
    )

    pile_tex = 0.38 * nap_vert + 0.36 * nap_velvet + 0.26 * bristle
    pile_h = crown * w_amp * (0.62 + 0.38 * pile_tex)
    trough_h = base_weave * 0.30 * (1.0 - 0.65 * wale_mask)

    raw_h = np.maximum(pile_h, trough_h)
    res_scale = float(size) / 512.0
    raw_h = 0.84 * raw_h + 0.16 * common.periodic_gaussian_blur(raw_h, max(0.4, 0.45 * res_scale))

    h = common.robust_normalize(
        raw_h + 0.16 * fine + 0.12 * fuzz + 0.07 * brush_mottle,
        low_pct=0.25,
        high_pct=99.75,
        soft=True,
    )

    c = common.robust_normalize(
        0.44 * pile_h
        + 0.22 * pile_crush * wale_mask
        + 0.16 * brush_mottle * wale_mask
        + 0.18 * pile_tex * wale_mask
        + 0.12 * fine
        + 0.06 * trough_h
        + w_tone * wale_mask,
        low_pct=0.35,
        high_pct=99.65,
        soft=True,
    )
    albedo = _interp_stops(c, stops)

    rough_raw = (
        0.38 * (1.0 - wale_mask)
        + 0.24 * fuzz
        + 0.20 * (1.0 - pile_h)
        + 0.18 * (1.0 - pile_crush * 0.5 - nap_velvet * 0.5)
    )
    rough = common.match_histogram_range(rough_raw, 0.80, 0.98, target_mean=0.90, contrast=1.08)
    base = common.shade_pbr_surface(
        albedo, h, roughness=rough, normal_strength=4.2, spec_intensity=0.03, ao_strength=0.34, cavity_strength=0.28
    )
    return base, h, rough


def _build_herringbone_twill_olive(
    u: np.ndarray, v: np.ndarray, size: int, rng: np.random.Generator, stops: list[tuple[int, int, int]]
) -> tuple[Image.Image, np.ndarray, np.ndarray]:
    """Vintage USMC / chore-coat Olive Drab 2/2 broken-twill Herringbone (HBT)."""
    uw, vw = _warp_uv(u, v, freq=3, amp=0.010, rng=rng, octaves=3)
    band_w = 6
    hbt_matrix = _herringbone_matrix(band_w=band_w)  # (4, 12) exact broken-twill
    tu = 9 * (2 * band_w)  # 108 threads = 9 full chevrons (odd count!)
    tv = 100  # 25x4 (odd*4 breaks vertical half-tile twill phase lock!)
    warp_slub, weft_slub = _yarn_slub(uw, vw, tu, tv, rng, warp_freqy=5, weft_freqx=6)
    fine, fuzz = _cotton_micro(u, v, rng)
    wash = _wash_mottle(u, v, rng, warp_strength=0.08)

    cloth = _woven_cloth(
        uw,
        vw,
        tu,
        tv,
        hbt_matrix,
        warp_slub,
        weft_slub,
        rng,
        size=size,
        float_arch=0.38,
        ply_twist=0.11,
        warp_rad_base=0.91,
        weft_rad_base=0.90,
        slub_rad_amp=0.14,
    )
    wh = cloth["h"]
    wv = cloth["warp_vis"]

    band_side = ((cloth["iu"] // band_w) % 2).astype(np.float32)
    chevron_sheen = (band_side - 0.5) * 0.11 * (0.65 + 0.35 * wh)

    h = common.robust_normalize(
        0.72 * wh + 0.12 * fine + 0.08 * fuzz + 0.06 * (0.6 * warp_slub + 0.4 * weft_slub) + 0.06 * wash,
        low_pct=0.25,
        high_pct=99.75,
        soft=True,
    )

    tone = common.robust_normalize(
        0.44 * wh
        + 0.18 * wash
        + 0.12 * fine
        + 0.12 * (wv * warp_slub + (1.0 - wv) * weft_slub)
        + 0.10 * (wv * cloth["warp_lot"] + (1.0 - wv) * cloth["weft_lot"])
        + 0.07 * wv
        + chevron_sheen,
        low_pct=0.35,
        high_pct=99.65,
        soft=True,
    )
    albedo = _interp_stops(tone, stops)

    rough_raw = 0.44 * (1.0 - wh) + 0.24 * fuzz + 0.18 * (1.0 - fine) + 0.14 * wash
    rough = common.match_histogram_range(rough_raw, 0.74, 0.93, target_mean=0.84, contrast=1.08)
    base = common.shade_pbr_surface(
        albedo, h, roughness=rough, normal_strength=2.5, spec_intensity=0.035, ao_strength=0.26, cavity_strength=0.22
    )
    return base, h, rough


def _build_hickory_stripe_indigo(
    u: np.ndarray, v: np.ndarray, size: int, rng: np.random.Generator, stops: list[tuple[int, int, int]]
) -> tuple[Image.Image, np.ndarray, np.ndarray]:
    """Classic railroad engineer yarn-dyed 2/1 RHT hickory stripe twill."""
    uw, vw = _warp_uv(u, v, freq=3, amp=0.011, rng=rng, octaves=3)
    stripe_period = 12  # 6 indigo + 6 natural ecru warp ends per stripe cycle
    tu = 11 * stripe_period  # 132 ends (11 odd cycles, 44x3 divisible by 3!)
    tv = 129  # 43x3 (odd multiple of 3)
    warp_slub, weft_slub = _yarn_slub(uw, vw, tu, tv, rng, warp_freqy=5, weft_freqx=6)
    fine, fuzz = _cotton_micro(u, v, rng)
    wash = _wash_mottle(u, v, rng, warp_strength=0.09)

    cloth = _woven_cloth(
        uw,
        vw,
        tu,
        tv,
        _twill_matrix(2, 1, step=1),
        warp_slub,
        weft_slub,
        rng,
        size=size,
        float_arch=0.38,
        ply_twist=0.10,
        warp_rad_base=0.91,
        weft_rad_base=0.90,
        slub_rad_amp=0.14,
    )
    wh = cloth["h"]
    wv = cloth["warp_vis"]

    # Center the stripe boundary cleanly inside the tile rather than on iu=0
    is_indigo_warp = (((cloth["iu"] + 3) % stripe_period) < 6).astype(np.float32)

    h = common.robust_normalize(
        0.72 * wh + 0.12 * fine + 0.08 * fuzz + 0.06 * warp_slub + 0.06 * wash,
        low_pct=0.25,
        high_pct=99.75,
        soft=True,
    )

    indigo_t = common.robust_normalize(
        0.38 * wh + 0.24 * warp_slub + 0.22 * wash + 0.14 * cloth["warp_lot"] + 0.10 * fine,
        low_pct=0.4,
        high_pct=99.6,
        soft=True,
    )
    ecru_t = np.clip(
        0.44 + 0.46 * wh + 0.10 * (wash - 0.5) + 0.08 * (weft_slub - 0.5) + 0.06 * cloth["weft_lot"],
        0.0,
        1.0,
    )

    indigo_rgb = _interp_stops(indigo_t, stops[:3])
    ecru_shadow = tuple(int(round(0.35 * stops[0][c] + 0.65 * stops[4][c])) for c in range(3))
    ecru_rgb = _interp_stops(ecru_t, [ecru_shadow, stops[3], stops[4]])

    blue_fade = np.clip((wh - 0.32) * 0.40 * wash, 0.0, 0.32)[..., None]
    indigo_rgb = indigo_rgb * (1.0 - blue_fade) + ecru_rgb * blue_fade

    warp_rgb = indigo_rgb * is_indigo_warp[..., None] + ecru_rgb * (1.0 - is_indigo_warp[..., None])
    albedo = warp_rgb * wv[..., None] + ecru_rgb * (1.0 - wv[..., None])

    rough_raw = 0.44 * (1.0 - wh) + 0.24 * fuzz + 0.18 * (1.0 - fine) + 0.14 * wash
    rough = common.match_histogram_range(rough_raw, 0.72, 0.92, target_mean=0.83, contrast=1.08)
    base = common.shade_pbr_surface(
        albedo, h, roughness=rough, normal_strength=2.3, spec_intensity=0.04, ao_strength=0.25, cavity_strength=0.22
    )
    return base, h, rough


def _build_wabash_stripe_indigo(
    u: np.ndarray, v: np.ndarray, size: int, rng: np.random.Generator, stops: list[tuple[int, int, int]]
) -> tuple[Image.Image, np.ndarray, np.ndarray]:
    """Deep indigo 2/1 workwear twill with bleach-discharged Stifel Wabash dot stripes."""
    uw, vw = _warp_uv(u, v, freq=3, amp=0.010, rng=rng, octaves=3)
    tu, tv = 126, 123  # 42x3, 41x3
    warp_slub, weft_slub = _yarn_slub(uw, vw, tu, tv, rng, warp_freqy=5, weft_freqx=6)
    fine, fuzz = _cotton_micro(u, v, rng)
    wash = _wash_mottle(u, v, rng, warp_strength=0.08)

    cloth = _woven_cloth(
        uw,
        vw,
        tu,
        tv,
        _twill_matrix(2, 1, step=1),
        warp_slub,
        weft_slub,
        rng,
        size=size,
        float_arch=0.38,
        ply_twist=0.10,
        warp_rad_base=0.92,
        weft_rad_base=0.90,
        slub_rad_amp=0.13,
    )
    wh = cloth["h"]
    wv = cloth["warp_vis"]

    stripe_cols = 11  # odd column count avoids half-tile phase lock
    dot_rows = 43  # odd vertical count avoids half-tile phase lock
    sx = uw * stripe_cols
    sy = vw * dot_rows
    ix = np.floor(sx).astype(np.int32) % stripe_cols
    iy = np.floor(sy).astype(np.int32) % dot_rows
    fx = sx - np.floor(sx) - 0.5
    fy = sy - np.floor(sy) - 0.5

    jx = rng.uniform(-0.005, 0.005, size=(dot_rows, stripe_cols)).astype(np.float32)[iy, ix]
    jy = rng.uniform(-0.050, 0.050, size=(dot_rows, stripe_cols)).astype(np.float32)[iy, ix]
    j_scale = rng.uniform(0.86, 1.14, size=(dot_rows, stripe_cols)).astype(np.float32)[iy, ix]
    j_alpha = rng.uniform(0.74, 1.00, size=(dot_rows, stripe_cols)).astype(np.float32)[iy, ix]

    rx = 0.035 * j_scale
    ry = 0.235 * j_scale
    dist = np.sqrt(((fx - jx) / rx) ** 2 + ((fy - jy) / ry) ** 2)
    wick = (fine - 0.5) * 0.24 + (warp_slub - 0.5) * 0.18 + (weft_slub - 0.5) * 0.12
    dot_raw = np.clip((1.08 - (dist + wick)) / 0.34, 0.0, 1.0)
    discharge = np.clip(dot_raw * j_alpha * (0.58 + 0.42 * np.clip(wh, 0.0, 1.0)), 0.0, 0.94).astype(np.float32)

    h = common.robust_normalize(
        0.72 * wh + 0.12 * fine + 0.08 * fuzz + 0.06 * warp_slub + 0.05 * wash,
        low_pct=0.25,
        high_pct=99.75,
        soft=True,
    )

    warp_t = common.robust_normalize(
        0.40 * wh + 0.24 * warp_slub + 0.18 * wash + 0.14 * cloth["warp_lot"] + 0.10 * fine,
        low_pct=0.4,
        high_pct=99.6,
        soft=True,
    )
    weft_t = np.clip(0.32 + 0.52 * wh + 0.10 * (weft_slub - 0.5), 0.0, 1.0)
    albedo = _denim_albedo(warp_t, weft_t, wv, stops, core_fade=discharge * 0.92, weft_dark_mix=0.62)
    bleach_col = np.array(stops[4], dtype=np.float32)[None, None, :]
    d_weft = (discharge * (1.0 - wv) * 0.78)[..., None]
    albedo = albedo * (1.0 - d_weft) + bleach_col * d_weft

    rough_raw = 0.42 * (1.0 - wh) + 0.24 * fuzz + 0.18 * (1.0 - fine) + 0.16 * discharge
    rough = common.match_histogram_range(rough_raw, 0.70, 0.90, target_mean=0.81, contrast=1.08)
    base = common.shade_pbr_surface(
        albedo, h, roughness=rough, normal_strength=2.3, spec_intensity=0.045, ao_strength=0.26, cavity_strength=0.22
    )
    return base, h, rough


_BUILDERS = (
    _build_raw_selvedge_indigo,
    _build_stone_washed_indigo,
    _build_faded_vintage_sky,
    _build_washed_black_denim,
    _build_ecru_bull_denim,
    _build_indigo_chambray,
    _build_canvas_duck_brown,
    _build_waxed_canvas_olive,
    _build_corduroy_rust,
    _build_herringbone_twill_olive,
    _build_hickory_stripe_indigo,
    _build_wabash_stripe_indigo,
)


def _compute_bundle(
    seed: int = 1,
    palette: PaletteArg = None,
    size: int = 512,
    variant: int = 0,
) -> tuple[Image.Image, np.ndarray, np.ndarray, DenimStyleSpec]:
    style = DENIM_STYLES[int(variant) % len(DENIM_STYLES)]
    rng = np.random.default_rng(int(seed))
    u, v = common.make_uv_grid(int(size))
    stops = common.resolve_palette(palette, style.default_palette, rng)
    base, h, rough = _BUILDERS[style.variant](u, v, int(size), rng, stops)
    return base, np.asarray(h, dtype=np.float32), np.asarray(rough, dtype=np.float32), style


def render_denim_bundle(
    style_or_variant: str | int,
    seed: int = 1,
    size: int = 2048,
    palette: PaletteArg = None,
) -> tuple[Image.Image, Image.Image, Image.Image, np.ndarray, np.ndarray, DenimStyleSpec]:
    """Single-pass PBR bundle renderer returning (base, normal, roughness, height, rough_field, style)."""
    if isinstance(style_or_variant, str):
        style = STYLE_BY_SLUG[style_or_variant]
        variant = style.variant
    else:
        variant = int(style_or_variant) % len(DENIM_STYLES)
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
    name="denim-workwear",
    kind="texture",
    tileable=True,
    origin="tools/generators/types/denim_workwear.py",
    default_palette="raw_indigo_selvedge",
    description=(
        "Apparel denim & workwear suite (12 variants: raw selvedge indigo denim, "
        "stone-washed indigo denim, faded vintage sky LHT denim, washed black charcoal denim, "
        "natural ecru bull denim, indigo workwear chambray, heavy brown duck canvas, "
        "waxed olive field canvas, 8-wale rust corduroy, olive chore herringbone twill, "
        "railroad hickory stripe, and heritage Wabash dot stripe) with calibrated PBR "
        "height, normal, and roughness."
    ),
    render_fn=generate,
    normal_source="native",
    normal_rule="Native per-style woven denim & workwear height field (h) converted with calibrated Scharr normal strength.",
    normal_strength=2.5,
    height_fn=generate_height,
    normal_fn=generate_normal,
    roughness_fn=generate_roughness,
    default_roughness=0.83,
)
