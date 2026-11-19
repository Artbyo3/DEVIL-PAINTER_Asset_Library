"""Procedural Sheer Micro Mesh generator suite (10 tileable RGBA PBR materials).

Built strictly on the committed periodic math primitives in ``tools/generators/common.py``:
  - Self-contained 5-stop surface palettes defined inside this module.
  - Exact toroidal dual-sublattice hexagonal and warp-knit tricot micro-mesh geometry
    with smooth C1 rounded-cylinder filament profiles, vertex twist junctions, and
    fully transparent open apertures (alpha = 0) between semi-transparent monofilaments.
  - Integrated raised textile motifs that elevate both opacity (alpha) and 3D relief:
    staggered woven dots, alternating satin-woven stripes with corded selvedge ribs,
    dual-scale braided honeycomb ribs over fine sheer netting, plush matte flocked-pile
    dots with fibrous edges, and helical metallic-sheen twisted filaments.
  - Halo-free straight-alpha RGBA base color output paired with resolution-independent
    OpenGL (+Y up) Scharr normal maps and calibrated PBR roughness maps.
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
class SheerStyleSpec:
    variant: int
    slug: str
    default_palette: str
    normal_strength: float
    rough_range: tuple[float, float]
    description: str


# Self-contained 5-stop RGB palettes (dark shadow -> filament midtone -> bright sheen).
SHEER_PALETTES: dict[str, list[tuple[int, int, int]]] = {
    "micro-mesh-ultra-sheer-black": [
        (10, 11, 14),
        (22, 24, 29),
        (38, 41, 48),
        (60, 65, 74),
        (92, 98, 110),
    ],
    "micro-mesh-sheer-nude": [
        (138, 104, 84),
        (170, 134, 110),
        (198, 164, 140),
        (222, 192, 170),
        (242, 218, 200),
    ],
    "micro-mesh-sheer-white": [
        (178, 180, 186),
        (206, 208, 214),
        (228, 230, 234),
        (242, 244, 247),
        (253, 254, 255),
    ],
    "micro-mesh-sheer-red": [
        (72, 10, 18),
        (118, 16, 28),
        (166, 24, 38),
        (206, 46, 60),
        (238, 96, 108),
    ],
    "fine-tulle-soft": [
        (172, 166, 158),
        (200, 194, 186),
        (224, 219, 212),
        (240, 236, 230),
        (252, 250, 246),
    ],
    "sheer-dot-mesh": [
        (12, 13, 17),
        (24, 26, 32),
        (42, 46, 54),
        (68, 74, 86),
        (106, 114, 128),
    ],
    "sheer-stripe-mesh": [
        (14, 15, 19),
        (28, 30, 37),
        (48, 52, 62),
        (78, 84, 98),
        (122, 130, 146),
    ],
    "sheer-honeycomb-mesh": [
        (22, 24, 28),
        (44, 48, 55),
        (72, 78, 88),
        (108, 116, 128),
        (152, 160, 172),
    ],
    "flocked-dot-mesh": [
        (8, 8, 10),
        (16, 17, 21),
        (28, 30, 36),
        (48, 52, 60),
        (78, 84, 94),
    ],
    "micro-mesh-metallic-sheen": [
        (88, 80, 68),
        (144, 134, 116),
        (194, 184, 164),
        (230, 222, 204),
        (252, 248, 238),
    ],
}


SHEER_STYLES: tuple[SheerStyleSpec, ...] = (
    SheerStyleSpec(
        0,
        "micro-mesh-ultra-sheer-black",
        "micro-mesh-ultra-sheer-black",
        1.15,
        (0.40, 0.62),
        "Ultra-sheer black fine warp-knit micro-mesh with slender low-alpha monofilaments over open transparent pores.",
    ),
    SheerStyleSpec(
        1,
        "micro-mesh-sheer-nude",
        "micro-mesh-sheer-nude",
        1.25,
        (0.48, 0.68),
        "Sheer nude fine stretch tricot micro-mesh with soft rounded apertures and even matte-silky filament relief.",
    ),
    SheerStyleSpec(
        2,
        "micro-mesh-sheer-white",
        "micro-mesh-sheer-white",
        1.25,
        (0.46, 0.66),
        "Sheer white fine micro-mesh with crisp symmetrical open pores and clean satin-matte monofilaments.",
    ),
    SheerStyleSpec(
        3,
        "micro-mesh-sheer-red",
        "micro-mesh-sheer-red",
        1.25,
        (0.42, 0.62),
        "Sheer red fine warp-knit micro-mesh with translucent crimson filaments and subtle lustrous edge highlights.",
    ),
    SheerStyleSpec(
        4,
        "fine-tulle-soft",
        "fine-tulle-soft",
        1.35,
        (0.56, 0.76),
        "Soft fine hexagonal bobbinet tulle with twisted vertex knots, delicate open pores, and airy matte gauze finish.",
    ),
    SheerStyleSpec(
        5,
        "sheer-dot-mesh",
        "sheer-dot-mesh",
        1.85,
        (0.34, 0.64),
        "Fine sheer micro-mesh integrated with staggered dense satin-woven dots raising opacity and domed thread relief.",
    ),
    SheerStyleSpec(
        6,
        "sheer-stripe-mesh",
        "sheer-stripe-mesh",
        1.85,
        (0.26, 0.66),
        "Alternating sheer micro-mesh channels and opaque satin-woven vertical stripes framed by fine corded selvedge edges.",
    ),
    SheerStyleSpec(
        7,
        "sheer-honeycomb-mesh",
        "sheer-honeycomb-mesh",
        1.95,
        (0.40, 0.68),
        "Dual-scale sheer honeycomb mesh with raised braided hexagonal structural ribs spanning an ultra-fine inner sheer net.",
    ),
    SheerStyleSpec(
        8,
        "flocked-dot-mesh",
        "flocked-dot-mesh",
        2.10,
        (0.44, 0.96),
        "Fine sheer micro-mesh overlaid with lofty, opaque, ultra-matte velvety flocked fiber dots with fibrous edges.",
    ),
    SheerStyleSpec(
        9,
        "micro-mesh-metallic-sheen",
        "micro-mesh-metallic-sheen",
        1.45,
        (0.16, 0.52),
        "Fine sheer micro-mesh knitted with twisted metallic-sheen filaments producing low-roughness specular micro-glints.",
    ),
)

STYLE_BY_SLUG: dict[str, SheerStyleSpec] = {s.slug: s for s in SHEER_STYLES}

_SQRT3 = math.sqrt(3.0)
_HALF_SQRT3 = 0.5 * _SQRT3
_INV_SQRT3 = 1.0 / _SQRT3
_HALF_INV_SQRT3 = 0.5 * _INV_SQRT3


def _parse_hex_color(token: str) -> tuple[int, int, int]:
    h = token.strip().lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    if len(h) != 6:
        raise ValueError(f"Invalid hex color token: {token!r}")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def _resolve_sheer_palette(
    palette: PaletteArg,
    default_name: str,
    rng: np.random.Generator,
) -> list[tuple[int, int, int]]:
    """Resolve palettes strictly from module-local SHEER_PALETTES or explicit RGB/hex stops."""
    if palette is None:
        return list(SHEER_PALETTES[default_name])
    if isinstance(palette, str):
        key = palette.strip().lower()
        if key == "auto":
            names = tuple(SHEER_PALETTES.keys())
            return list(SHEER_PALETTES[names[int(rng.integers(0, len(names)))]])
        if key in SHEER_PALETTES:
            return list(SHEER_PALETTES[key])
        if "," in palette or palette.strip().startswith("#"):
            parts = [p.strip() for p in palette.split(",") if p.strip()]
            return [_parse_hex_color(p) for p in parts]
        return list(SHEER_PALETTES[default_name])
    stops: list[tuple[int, int, int]] = []
    for item in palette:
        if isinstance(item, str):
            stops.append(_parse_hex_color(item))
        else:
            r, g, b = item
            stops.append((int(r), int(g), int(b)))
    return stops if stops else list(SHEER_PALETTES[default_name])


def _smin(a: np.ndarray, b: np.ndarray, k: float) -> np.ndarray:
    """Smooth quadratic polynomial minimum with fillet radius k."""
    kk = max(1e-5, float(k))
    h = np.clip(0.5 + 0.5 * (b - a) / kk, 0.0, 1.0)
    return (b * (1.0 - h) + a * h - kk * h * (1.0 - h)).astype(np.float32)


def _smoothstep(edge0: float, edge1: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - edge0) / max(1e-6, edge1 - edge0), 0.0, 1.0)
    return (t * t * (3.0 - 2.0 * t)).astype(np.float32)


def _gentle_drape_uv(
    u: np.ndarray,
    v: np.ndarray,
    rng: np.random.Generator,
    amp: float = 0.0007,
    freq: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Tiny periodic textile tension drift so fine filaments feel woven rather than rigid."""
    if amp <= 1e-6:
        return u, v
    du = (common.periodic_fbm(u, v, freq, 3, 0.5, rng, mode="fbm") - 0.5) * (2.0 * amp)
    dv = (common.periodic_fbm(u, v, freq, 3, 0.5, rng, mode="fbm") - 0.5) * (2.0 * amp)
    return ((u + du) % 1.0).astype(np.float32), ((v + dv) % 1.0).astype(np.float32)


def _hex_tricot_mesh_fields(
    u: np.ndarray,
    v: np.ndarray,
    nx: int,
    ny: int,
    strut_width: float,
    corner_round: float = 0.06,
    knot_boost: float = 0.22,
    wale_boost: float = 0.14,
    twist_amp: float = 0.07,
    twist_cycles: float = 3.0,
) -> dict[str, np.ndarray]:
    """Exact toroidal 2-sublattice hexagonal / warp-knit tricot micro-mesh field.

    When nx / ny ~= sqrt(3), apertures are regular hexagons (tulle / honeycomb).
    When nx / ny > sqrt(3), apertures gently elongate vertically (warp-knit tricot).

    Returns dict with keys:
      - ``profile``: filament cylindrical height in [0, 1], exact 0.0 in open pores
      - ``coverage``: smooth anti-aliased filament coverage mask in [0, 1], exact 0.0 in open pores
      - ``knot``: vertex junction weight in [0, 1]
      - ``wale``: vertical pillar strand weight in [0, 1]
      - ``twist``: micro-ply helical variation along filaments in [-1, 1]
      - ``pore_dist``: signed distance inside open aperture (0 on strut centerline, >0 in pore)
    """
    fx = float(nx)
    fy = float(ny)

    # Sublattice A centered at (i + 0.5, j + 0.5), Sublattice B at (i, j)
    ax = (u * fx) % 1.0 - 0.5
    ay = (v * fy) % 1.0 - 0.5
    bx = (u * fx + 0.5) % 1.0 - 0.5
    by = (v * fy + 0.5) % 1.0 - 0.5

    ay_s = ay * _SQRT3
    by_s = by * _SQRT3

    use_a = (ax * ax + ay_s * ay_s) <= (bx * bx + by_s * by_s)
    px = np.where(use_a, ax, bx).astype(np.float32)
    py = np.where(use_a, ay_s, by_s).astype(np.float32)

    abs_x = np.abs(px)
    abs_y = np.abs(py)

    # Exact perpendicular distances to the three positive-quadrant hexagon faces (apothem = 0.5)
    e0 = 0.5 - abs_x
    e1 = 0.5 - (0.5 * abs_x + _HALF_SQRT3 * abs_y)
    e2 = 0.5 - (_HALF_SQRT3 * abs_y - 0.5 * abs_x)

    # Smooth-min rounds inner pore corners and thickens 3-way junctions naturally
    edge_d = _smin(_smin(e0, e1, corner_round), e2, corner_round)

    # Distance to nearest hexagon vertex (where 3 filaments interlock)
    dv1 = np.hypot(abs_x - 0.5, abs_y - _HALF_INV_SQRT3)
    dv2 = np.hypot(abs_x, abs_y - _INV_SQRT3)
    v_dist = np.minimum(dv1, dv2).astype(np.float32)

    w = max(0.03, float(strut_width))
    # Subtle vertical wale thickening for warp-knit tricot structure
    wale_mask = (1.0 / (1.0 + np.exp(np.clip((e0 - e1) / 0.045, -12.0, 12.0)))).astype(np.float32)
    eff_w = w * (1.0 + 0.14 * float(wale_boost) * wale_mask)

    # Normalized cross-section coordinate t in [0, 1] from strut centerline to pore rim
    t_pos = np.clip(edge_d / eff_w, 0.0, 1.0)
    # Smooth C1 rounded-cylinder cross-section: plump crown, clean fade at aperture rim
    cyl = np.power(np.clip(1.0 - t_pos * t_pos, 0.0, 1.0), 0.72).astype(np.float32)
    # Crisp anti-aliased coverage mask (1.0 across core of filament, exact 0.0 inside pore)
    cov = np.power(np.clip(1.0 - t_pos * t_pos, 0.0, 1.0), 0.48).astype(np.float32)

    # Vertex twist knot bead
    knot_r = w * 1.18
    knot = np.exp(-((v_dist / max(1e-4, knot_r)) ** 2)).astype(np.float32) * (cov > 1e-4)

    # Fine helical ply twist along struts (periodic because derived from local px, py & wrapped uv)
    along_vert = np.sin((v * fy * twist_cycles * 2.0 + px * 3.0) * math.pi)
    along_diag = np.sin(((u * fx + v * fy) * twist_cycles + (px - py) * 2.5) * math.pi)
    twist = (wale_mask * along_vert + (1.0 - wale_mask) * along_diag).astype(np.float32)

    # Combine cylinder, subtle wale reinforcement, vertex knot, and micro-ply twist
    raw_h = (
        cyl
        * (1.0 + float(wale_boost) * 0.22 * wale_mask)
        * (1.0 + float(twist_amp) * twist)
        + float(knot_boost) * knot * (0.45 + 0.55 * cyl)
    )
    # Keep open pores strictly at 0.0 height
    raw_h = np.where(cov > 1e-4, raw_h, 0.0).astype(np.float32)
    peak = max(1e-5, float(np.percentile(raw_h[cov > 0.15], 99.5))) if np.any(cov > 0.15) else 1.0
    profile = np.clip(raw_h / peak, 0.0, 1.0).astype(np.float32)

    return {
        "profile": profile,
        "coverage": cov,
        "knot": knot,
        "wale": wale_mask,
        "twist": twist,
        "pore_dist": np.maximum(edge_d, 0.0).astype(np.float32),
    }


def _staggered_dot_lattice(
    u: np.ndarray,
    v: np.ndarray,
    cols: int,
    rows: int,
    rng: np.random.Generator,
    jitter: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Exact toroidal staggered (diamond/hex-brick) dot lattice on [0, 1)^2.

    Returns ``(dist, dx, dy, dot_seed)`` where ``dist`` is in unit-cell units and
    each unit cell hosts two staggered dots at (0.5, 0.5) and (0.0, 0.0).
    """
    fc = float(cols)
    fr = float(rows)

    # Sublattice A
    ux_a = u * fc
    vy_a = v * fr
    ix_a = np.floor(ux_a).astype(np.int32) % cols
    iy_a = np.floor(vy_a).astype(np.int32) % rows
    dx_a = (ux_a % 1.0) - 0.5
    dy_a = (vy_a % 1.0) - 0.5

    # Sublattice B shifted by (0.5, 0.5)
    ux_b = u * fc + 0.5
    vy_b = v * fr + 0.5
    ix_b = np.floor(ux_b).astype(np.int32) % cols
    iy_b = np.floor(vy_b).astype(np.int32) % rows
    dx_b = (ux_b % 1.0) - 0.5
    dy_b = (vy_b % 1.0) - 0.5

    seed_a = rng.uniform(0.0, 1.0, size=(rows, cols)).astype(np.float32)
    seed_b = rng.uniform(0.0, 1.0, size=(rows, cols)).astype(np.float32)

    if jitter > 1e-5:
        jx_a = rng.uniform(-jitter, jitter, size=(rows, cols)).astype(np.float32)
        jy_a = rng.uniform(-jitter, jitter, size=(rows, cols)).astype(np.float32)
        jx_b = rng.uniform(-jitter, jitter, size=(rows, cols)).astype(np.float32)
        jy_b = rng.uniform(-jitter, jitter, size=(rows, cols)).astype(np.float32)
        dx_a = dx_a - jx_a[iy_a, ix_a]
        dy_a = dy_a - jy_a[iy_a, ix_a]
        dx_b = dx_b - jx_b[iy_b, ix_b]
        dy_b = dy_b - jy_b[iy_b, ix_b]

    # Convert to isotropic distance if cols != rows
    aspect = fc / fr
    d2_a = dx_a * dx_a + (dy_a * aspect) ** 2
    d2_b = dx_b * dx_b + (dy_b * aspect) ** 2

    use_a = d2_a <= d2_b
    dist = np.sqrt(np.where(use_a, d2_a, d2_b)).astype(np.float32)
    dx = np.where(use_a, dx_a, dx_b).astype(np.float32)
    dy = np.where(use_a, dy_a * aspect, dy_b * aspect).astype(np.float32)
    dot_seed = np.where(use_a, seed_a[iy_a, ix_a], seed_b[iy_b, ix_b]).astype(np.float32)
    return dist, dx, dy, dot_seed


def _shade_sheer_rgba(
    albedo_rgb: np.ndarray,
    height: np.ndarray,
    alpha: np.ndarray,
    roughness: np.ndarray,
    normal_strength: float,
    spec_intensity: float = 0.14,
    ao_strength: float = 0.18,
    cavity_strength: float = 0.10,
    sheen_tint: tuple[float, float, float] | None = None,
    sheen_strength: float = 0.0,
) -> Image.Image:
    """Shade straight-alpha RGBA sheer fabric with halo-free open-pore color extension.

    Uses ``common.shade_pbr_surface`` for Scharr-normal lighting and Fresnel response,
    then extends clean filament RGB cleanly across transparent open pores so bilinear
    sampling and mipmapping produce zero dark edge fringes.
    """
    shaded_img = common.shade_pbr_surface(
        albedo_rgb,
        height,
        roughness=roughness,
        normal_strength=normal_strength,
        spec_intensity=spec_intensity,
        ao_strength=ao_strength,
        cavity_strength=cavity_strength,
    )
    rgb = np.asarray(shaded_img, dtype=np.float32)

    if sheen_strength > 1e-4:
        dh_dx, dh_dy_up = common.periodic_scharr_gradients(common.normalize_field(height), reference_size=512)
        s = float(normal_strength)
        nx = -dh_dx * s
        ny = dh_dy_up * s
        nz = 1.0 / np.sqrt(nx * nx + ny * ny + 1.0 + 1e-8)
        rim = np.power(np.clip(1.0 - nz, 0.0, 1.0), 1.55) * float(sheen_strength)
        tint = np.array(sheen_tint if sheen_tint is not None else (255.0, 255.0, 255.0), dtype=np.float32)
        rgb = rgb + rim[..., None] * tint[None, None, :]

    # Halo-free straight-alpha color bleed: transparent pores inherit clean unshadowed albedo
    a_clean = np.clip(np.nan_to_num(alpha, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0).astype(np.float32)
    vis_blend = _smoothstep(0.0, 0.07, a_clean)[..., None]
    rgb_bleed = np.nan_to_num(rgb * vis_blend + albedo_rgb * (1.0 - vis_blend), nan=0.0, posinf=255.0, neginf=0.0)

    rgba = np.empty((rgb.shape[0], rgb.shape[1], 4), dtype=np.uint8)
    rgba[..., :3] = np.rint(np.clip(rgb_bleed, 0.0, 255.0)).astype(np.uint8)
    rgba[..., 3] = np.rint(a_clean * 255.0).astype(np.uint8)
    return Image.fromarray(rgba, mode="RGBA")


def _build_plain_micro_mesh(
    u: np.ndarray,
    v: np.ndarray,
    rng: np.random.Generator,
    stops: Sequence[tuple[int, int, int]],
    *,
    nx: int,
    ny: int,
    strut_width: float,
    corner_round: float,
    knot_boost: float,
    wale_boost: float,
    alpha_filament: float,
    alpha_knot: float,
    normal_strength: float,
    rough_min: float,
    rough_max: float,
    rough_mean: float,
    spec_intensity: float = 0.14,
    sheen_strength: float = 0.05,
    drape_amp: float = 0.0006,
) -> tuple[Image.Image, np.ndarray, np.ndarray]:
    """Shared fine even sheer micro-mesh builder parameterized per material variant."""
    uu, vv = _gentle_drape_uv(u, v, rng, amp=drape_amp, freq=3)
    mesh = _hex_tricot_mesh_fields(
        uu,
        vv,
        nx=nx,
        ny=ny,
        strut_width=strut_width,
        corner_round=corner_round,
        knot_boost=knot_boost,
        wale_boost=wale_boost,
        twist_amp=0.075,
        twist_cycles=3.0,
    )
    prof = mesh["profile"]
    cov = mesh["coverage"]
    knot = mesh["knot"]
    wale = mesh["wale"]
    twist = mesh["twist"]

    # Whisper-subtle fiber dye & sheen micro-variation (keeps mesh extremely fine and even)
    micro_tex = common.periodic_fbm(u, v, 24, 4, 0.52, rng, mode="fbm")
    macro_even = common.periodic_fbm(u, v, 4, 3, 0.48, rng, mode="fbm")

    # Height: exact 0.0 in open apertures, smooth cylindrical relief on monofilaments
    h_raw = prof * (0.94 + 0.06 * micro_tex)
    h = np.where(cov > 1e-4, np.clip(h_raw, 0.0, 1.0), 0.0).astype(np.float32)

    # Alpha: low semi-transparent alpha on fine filaments, exact 0.0 in open pores
    alpha_raw = (
        cov * float(alpha_filament) * (0.86 + 0.14 * prof + 0.06 * (micro_tex - 0.5))
        + knot * float(alpha_knot) * cov
    )
    alpha = np.where(cov > 1e-4, np.clip(alpha_raw, 0.0, 0.85), 0.0).astype(np.float32)

    # Color field mapped cleanly onto the 5-stop palette; pores hold midtone filament color
    tone_filament = np.clip(
        0.34
        + 0.44 * prof
        + 0.10 * knot
        + 0.06 * wale
        + 0.05 * twist
        + 0.06 * (micro_tex - 0.5)
        + 0.04 * (macro_even - 0.5),
        0.05,
        0.96,
    )
    tone = np.where(cov > 1e-4, tone_filament, 0.48 + 0.04 * (macro_even - 0.5)).astype(np.float32)
    albedo = common.apply_palette_5stop(tone, micro_detail=None, stops=stops, contrast=1.04, gamma=1.0)

    # Roughness: smoother specular crowns along cylindrical filaments, slightly rougher knots & pores
    rough_raw = (
        0.48 * (1.0 - prof)
        + 0.24 * knot
        + 0.16 * (1.0 - micro_tex)
        + 0.12 * (0.5 - 0.5 * twist)
    )
    rough = common.match_histogram_range(
        rough_raw,
        target_min=rough_min,
        target_max=rough_max,
        target_mean=rough_mean,
        contrast=1.06,
    )

    sheen_tint = tuple(float(c) for c in stops[-1])
    base = _shade_sheer_rgba(
        albedo,
        h,
        alpha,
        rough,
        normal_strength=normal_strength,
        spec_intensity=spec_intensity,
        ao_strength=0.16,
        cavity_strength=0.10,
        sheen_tint=sheen_tint,
        sheen_strength=sheen_strength,
    )
    return base, h, rough


def _build_micro_mesh_ultra_sheer_black(
    u: np.ndarray,
    v: np.ndarray,
    size: int,
    rng: np.random.Generator,
    stops: Sequence[tuple[int, int, int]],
) -> tuple[Image.Image, np.ndarray, np.ndarray]:
    _ = size
    return _build_plain_micro_mesh(
        u,
        v,
        rng,
        stops,
        nx=156,
        ny=80,
        strut_width=0.135,
        corner_round=0.075,
        knot_boost=0.20,
        wale_boost=0.18,
        alpha_filament=0.28,
        alpha_knot=0.14,
        normal_strength=1.15,
        rough_min=0.40,
        rough_max=0.62,
        rough_mean=0.50,
        spec_intensity=0.16,
        sheen_strength=0.06,
        drape_amp=0.0005,
    )


def _build_micro_mesh_sheer_nude(
    u: np.ndarray,
    v: np.ndarray,
    size: int,
    rng: np.random.Generator,
    stops: Sequence[tuple[int, int, int]],
) -> tuple[Image.Image, np.ndarray, np.ndarray]:
    _ = size
    return _build_plain_micro_mesh(
        u,
        v,
        rng,
        stops,
        nx=144,
        ny=74,
        strut_width=0.155,
        corner_round=0.09,
        knot_boost=0.22,
        wale_boost=0.16,
        alpha_filament=0.38,
        alpha_knot=0.16,
        normal_strength=1.25,
        rough_min=0.48,
        rough_max=0.68,
        rough_mean=0.58,
        spec_intensity=0.11,
        sheen_strength=0.04,
        drape_amp=0.0006,
    )


def _build_micro_mesh_sheer_white(
    u: np.ndarray,
    v: np.ndarray,
    size: int,
    rng: np.random.Generator,
    stops: Sequence[tuple[int, int, int]],
) -> tuple[Image.Image, np.ndarray, np.ndarray]:
    _ = size
    return _build_plain_micro_mesh(
        u,
        v,
        rng,
        stops,
        nx=148,
        ny=76,
        strut_width=0.155,
        corner_round=0.08,
        knot_boost=0.22,
        wale_boost=0.14,
        alpha_filament=0.42,
        alpha_knot=0.16,
        normal_strength=1.25,
        rough_min=0.46,
        rough_max=0.66,
        rough_mean=0.56,
        spec_intensity=0.13,
        sheen_strength=0.04,
        drape_amp=0.0005,
    )


def _build_micro_mesh_sheer_red(
    u: np.ndarray,
    v: np.ndarray,
    size: int,
    rng: np.random.Generator,
    stops: Sequence[tuple[int, int, int]],
) -> tuple[Image.Image, np.ndarray, np.ndarray]:
    _ = size
    return _build_plain_micro_mesh(
        u,
        v,
        rng,
        stops,
        nx=144,
        ny=74,
        strut_width=0.155,
        corner_round=0.08,
        knot_boost=0.22,
        wale_boost=0.16,
        alpha_filament=0.40,
        alpha_knot=0.16,
        normal_strength=1.25,
        rough_min=0.42,
        rough_max=0.62,
        rough_mean=0.52,
        spec_intensity=0.15,
        sheen_strength=0.06,
        drape_amp=0.0006,
    )


def _build_fine_tulle_soft(
    u: np.ndarray,
    v: np.ndarray,
    size: int,
    rng: np.random.Generator,
    stops: Sequence[tuple[int, int, int]],
) -> tuple[Image.Image, np.ndarray, np.ndarray]:
    """Authentic soft regular-hexagonal bobbinet tulle with twisted vertex knots."""
    _ = size
    # nx=112, ny=65 -> 112/65 = 1.723 (regular isotropic hexagons with gentle airy drape)
    uu, vv = _gentle_drape_uv(u, v, rng, amp=0.0014, freq=3)
    mesh = _hex_tricot_mesh_fields(
        uu,
        vv,
        nx=112,
        ny=65,
        strut_width=0.135,
        corner_round=0.045,
        knot_boost=0.34,
        wale_boost=0.0,
        twist_amp=0.11,
        twist_cycles=4.0,
    )
    prof = mesh["profile"]
    cov = mesh["coverage"]
    knot = mesh["knot"]
    twist = mesh["twist"]

    micro_gauze = common.periodic_fbm(u, v, 28, 4, 0.54, rng, mode="fbm")
    soft_drape = common.periodic_fbm(u, v, 3, 3, 0.50, rng, mode="fbm")

    # Crisp hexagonal bobbinet relief with pronounced twisted 3-way vertex knots
    h_raw = prof * (0.90 + 0.10 * micro_gauze) + 0.08 * knot
    h = np.where(cov > 1e-4, np.clip(h_raw, 0.0, 1.0), 0.0).astype(np.float32)

    # Airy low-alpha tulle struts with slightly denser bobbinet knots; full transparency in pores
    alpha_raw = cov * (0.34 * (0.85 + 0.15 * prof) + 0.18 * knot)
    alpha = np.where(cov > 1e-4, np.clip(alpha_raw, 0.0, 0.72), 0.0).astype(np.float32)

    tone_fil = np.clip(
        0.38 + 0.40 * prof + 0.14 * knot + 0.06 * twist + 0.06 * (soft_drape - 0.5),
        0.08,
        0.96,
    )
    tone = np.where(cov > 1e-4, tone_fil, 0.52 + 0.04 * (soft_drape - 0.5)).astype(np.float32)
    albedo = common.apply_palette_5stop(tone, micro_detail=None, stops=stops, contrast=1.02, gamma=1.0)

    rough_raw = 0.44 * (1.0 - prof) + 0.28 * knot + 0.18 * micro_gauze + 0.10 * (0.5 - 0.5 * twist)
    rough = common.match_histogram_range(rough_raw, 0.56, 0.76, target_mean=0.66, contrast=1.05)

    base = _shade_sheer_rgba(
        albedo,
        h,
        alpha,
        rough,
        normal_strength=1.35,
        spec_intensity=0.09,
        ao_strength=0.14,
        cavity_strength=0.09,
        sheen_tint=(252.0, 250.0, 246.0),
        sheen_strength=0.035,
    )
    return base, h, rough


def _build_sheer_dot_mesh(
    u: np.ndarray,
    v: np.ndarray,
    size: int,
    rng: np.random.Generator,
    stops: Sequence[tuple[int, int, int]],
) -> tuple[Image.Image, np.ndarray, np.ndarray]:
    """Fine sheer micro-mesh integrated with staggered dense satin-woven dots."""
    _ = size
    uu, vv = _gentle_drape_uv(u, v, rng, amp=0.0005, freq=3)
    mesh = _hex_tricot_mesh_fields(
        uu,
        vv,
        nx=144,
        ny=74,
        strut_width=0.145,
        corner_round=0.075,
        knot_boost=0.22,
        wale_boost=0.14,
        twist_amp=0.07,
        twist_cycles=3.0,
    )
    prof = mesh["profile"]
    cov = mesh["coverage"]
    knot = mesh["knot"]

    # Staggered dot lattice: 16x16 unit cells (32 staggered rows) -> crisp rhythmic dots
    dot_dist, dx, dy, dot_seed = _staggered_dot_lattice(uu, vv, cols=16, rows=16, rng=rng, jitter=0.004)
    micro_grain = common.periodic_fbm(u, v, 48, 4, 0.54, rng, mode="fbm")

    # Dot radius with subtle natural stitched-thread perimeter irregularity
    dot_r = 0.145 + (dot_seed - 0.5) * 0.006 + (micro_grain - 0.5) * 0.006
    t_dot = np.clip(dot_dist / np.maximum(dot_r, 1e-4), 0.0, 1.5)
    dot_mask = (1.0 - _smoothstep(0.84, 1.04, t_dot)).astype(np.float32)

    # Domed woven cushion profile + fine satin weft threads locked into the dot
    dot_dome = np.power(np.clip(1.0 - np.clip(t_dot, 0.0, 1.0) ** 2, 0.0, 1.0), 0.52).astype(np.float32)
    satin_mat = np.array(
        [
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
        ],
        dtype=np.int32,
    )
    dot_weave, _, _, _ = common.woven_fabric_field(
        uu, vv, threads_u=320, threads_v=320, matrix=satin_mat, float_arch=0.38, ply_twist=0.10
    )
    dot_weave = np.nan_to_num(dot_weave, nan=0.5, posinf=1.0, neginf=0.0).astype(np.float32)
    # Subtle perimeter stitch collar where dot anchors onto underlying mesh
    rim_ring = np.exp(-(((t_dot - 0.88) / 0.09) ** 2)).astype(np.float32) * dot_mask

    # Two-tier height: fine mesh sits at [0, 0.34], raised woven dots rise cleanly to [0.38, 1.0]
    mesh_h = np.where(cov > 1e-4, prof * 0.34, 0.0).astype(np.float32)
    dot_h = dot_mask * (0.36 + 0.50 * dot_dome + 0.11 * dot_weave + 0.05 * rim_ring)
    h = np.clip(np.maximum(mesh_h, dot_h), 0.0, 1.0).astype(np.float32)

    # Alpha: open pores = 0.0, sheer filaments = 0.22-0.42, woven dots = 0.86-0.96
    mesh_alpha = np.where(cov > 1e-4, cov * (0.32 * (0.85 + 0.15 * prof) + 0.14 * knot), 0.0)
    dot_alpha = dot_mask * (0.86 + 0.10 * dot_dome * (0.75 + 0.25 * dot_weave))
    alpha = np.clip(np.maximum(mesh_alpha, dot_alpha), 0.0, 0.97).astype(np.float32)

    # Color: tonal sheer charcoal ground + richer satin-woven dark dots with thread sheen
    tone_mesh = np.where(cov > 1e-4, 0.36 + 0.36 * prof + 0.08 * knot, 0.44)
    tone_dot = 0.18 + 0.34 * dot_dome + 0.22 * dot_weave + 0.08 * rim_ring + 0.05 * (dot_seed - 0.5)
    tone = np.clip(tone_mesh * (1.0 - dot_mask) + tone_dot * dot_mask, 0.04, 0.96).astype(np.float32)
    albedo = common.apply_palette_5stop(tone, micro_detail=None, stops=stops, contrast=1.06, gamma=1.0)

    # PBR roughness: drier sheer mesh (0.50-0.64) vs smoother satin-woven dots (0.34-0.48)
    r_mesh = common.match_histogram_range(
        0.55 * (1.0 - prof) + 0.25 * knot + 0.20 * micro_grain,
        0.50,
        0.64,
        target_mean=0.57,
    )
    r_dot = common.match_histogram_range(
        0.52 * (1.0 - dot_weave) + 0.28 * (1.0 - dot_dome) + 0.20 * rim_ring,
        0.34,
        0.48,
        target_mean=0.40,
    )
    rough = np.clip(r_mesh * (1.0 - dot_mask) + r_dot * dot_mask, 0.25, 0.95).astype(np.float32)

    base = _shade_sheer_rgba(
        albedo,
        h,
        alpha,
        rough,
        normal_strength=1.85,
        spec_intensity=0.20,
        ao_strength=0.18,
        cavity_strength=0.11,
        sheen_tint=tuple(float(c) for c in stops[-1]),
        sheen_strength=0.05,
    )
    return base, h, rough


def _build_sheer_stripe_mesh(
    u: np.ndarray,
    v: np.ndarray,
    size: int,
    rng: np.random.Generator,
    stops: Sequence[tuple[int, int, int]],
) -> tuple[Image.Image, np.ndarray, np.ndarray]:
    """Alternating sheer micro-mesh channels and opaque satin-woven vertical stripes."""
    _ = size
    uu, vv = _gentle_drape_uv(u, v, rng, amp=0.0004, freq=3)
    mesh = _hex_tricot_mesh_fields(
        uu,
        vv,
        nx=148,
        ny=76,
        strut_width=0.145,
        corner_round=0.075,
        knot_boost=0.20,
        wale_boost=0.16,
        twist_amp=0.07,
        twist_cycles=3.0,
    )
    prof = mesh["profile"]
    cov = mesh["coverage"]
    knot = mesh["knot"]

    # 16 periodic vertical stripe cycles across UV (41% opaque satin stripe, 59% sheer mesh)
    n_stripes = 16
    s_phase = ((uu * float(n_stripes)) % 1.0) - 0.5
    abs_s = np.abs(s_phase)

    half_w = 0.205
    stripe_mask = (1.0 - _smoothstep(half_w - 0.018, half_w + 0.018, abs_s)).astype(np.float32)
    # Fine raised corded selvedge piping along both edges of each opaque stripe
    selvedge = np.exp(-(((abs_s - (half_w - 0.012)) / 0.014) ** 2)).astype(np.float32)
    # Subtle domed micro-bevel across each opaque satin stripe width
    stripe_crown = np.power(np.clip(1.0 - np.clip(abs_s / max(half_w, 1e-4), 0.0, 1.0) ** 4, 0.0, 1.0), 0.45)

    # Dense 5-end satin weave inside the opaque stripes (long warp floats = smooth luster)
    satin5 = np.array(
        [
            [1, 1, 1, 1, 0],
            [1, 1, 0, 1, 1],
            [0, 1, 1, 1, 1],
            [1, 1, 1, 0, 1],
            [1, 0, 1, 1, 1],
        ],
        dtype=np.int32,
    )
    satin_h, _, _, _ = common.woven_fabric_field(
        uu, vv, threads_u=380, threads_v=360, matrix=satin5, float_arch=0.32, ply_twist=0.08
    )
    satin_h = np.nan_to_num(satin_h, nan=0.5, posinf=1.0, neginf=0.0).astype(np.float32)
    fine_grain = common.periodic_fbm(u, v, 32, 4, 0.50, rng, mode="fbm")

    # Height: sheer micro-mesh at [0, 0.34]; satin stripes + selvedge piping at [0.42, 1.0]
    mesh_h = np.where(cov > 1e-4, prof * 0.34, 0.0).astype(np.float32)
    stripe_h = stripe_mask * (0.44 + 0.24 * stripe_crown + 0.22 * satin_h) + 0.14 * selvedge
    h = np.clip(np.maximum(mesh_h, stripe_h), 0.0, 1.0).astype(np.float32)

    # Alpha: open pores in sheer bands = 0.0, filaments = 0.22-0.42, satin stripes = 0.86-0.96
    mesh_alpha = np.where(cov > 1e-4, cov * (0.34 * (0.85 + 0.15 * prof) + 0.14 * knot), 0.0)
    stripe_alpha = stripe_mask * (0.86 + 0.10 * (0.65 * stripe_crown + 0.35 * satin_h)) + 0.12 * selvedge
    alpha = np.clip(np.maximum(mesh_alpha, stripe_alpha), 0.0, 0.96).astype(np.float32)

    # Color: sheer slate-onyx mesh vs luminous satin-ribbed stripe bands
    tone_mesh = np.where(cov > 1e-4, 0.34 + 0.34 * prof + 0.08 * knot, 0.42)
    tone_stripe = 0.26 + 0.32 * satin_h + 0.18 * stripe_crown + 0.14 * selvedge + 0.05 * (fine_grain - 0.5)
    tone = np.clip(tone_mesh * (1.0 - stripe_mask) + tone_stripe * stripe_mask, 0.05, 0.96).astype(np.float32)
    albedo = common.apply_palette_5stop(tone, micro_detail=None, stops=stops, contrast=1.08, gamma=1.0)

    # PBR roughness: crisp satin luster (0.26-0.40) on stripes vs drier sheer mesh (0.52-0.66)
    r_mesh = common.match_histogram_range(
        0.54 * (1.0 - prof) + 0.26 * knot + 0.20 * fine_grain,
        0.52,
        0.66,
        target_mean=0.59,
    )
    r_stripe = common.match_histogram_range(
        0.58 * (1.0 - satin_h) + 0.24 * selvedge + 0.18 * (1.0 - fine_grain),
        0.26,
        0.40,
        target_mean=0.32,
    )
    rough = np.clip(r_mesh * (1.0 - stripe_mask) + r_stripe * stripe_mask, 0.20, 0.95).astype(np.float32)

    base = _shade_sheer_rgba(
        albedo,
        h,
        alpha,
        rough,
        normal_strength=1.85,
        spec_intensity=0.24,
        ao_strength=0.18,
        cavity_strength=0.12,
        sheen_tint=tuple(float(c) for c in stops[-1]),
        sheen_strength=0.06,
    )
    return base, h, rough


def _build_sheer_honeycomb_mesh(
    u: np.ndarray,
    v: np.ndarray,
    size: int,
    rng: np.random.Generator,
    stops: Sequence[tuple[int, int, int]],
) -> tuple[Image.Image, np.ndarray, np.ndarray]:
    """Dual-scale sheer honeycomb mesh: raised braided hex ribs over fine inner sheer net."""
    _ = size
    uu, vv = _gentle_drape_uv(u, v, rng, amp=0.0006, freq=3)

    # Macro structural honeycomb frame: 48 x 28 regular hexagons (48/28 = 1.714 ~= sqrt(3))
    macro = _hex_tricot_mesh_fields(
        uu,
        vv,
        nx=48,
        ny=28,
        strut_width=0.17,
        corner_round=0.055,
        knot_boost=0.34,
        wale_boost=0.06,
        twist_amp=0.16,
        twist_cycles=6.0,
    )
    # Exact 3x harmonic micro sheer netting spanning the open honeycomb cells (144 x 84)
    micro = _hex_tricot_mesh_fields(
        uu,
        vv,
        nx=144,
        ny=84,
        strut_width=0.125,
        corner_round=0.065,
        knot_boost=0.20,
        wale_boost=0.0,
        twist_amp=0.06,
        twist_cycles=3.0,
    )
    fine_tex = common.periodic_fbm(u, v, 36, 4, 0.52, rng, mode="fbm")

    m_prof = macro["profile"]
    m_cov = macro["coverage"]
    m_knot = macro["knot"]
    m_twist = macro["twist"]

    s_prof = micro["profile"]
    s_cov = micro["coverage"]
    s_knot = micro["knot"]

    # Height: inner fine net at [0, 0.28], raised braided honeycomb ribs & knots at [0.35, 1.0]
    inner_h = np.where(s_cov > 1e-4, s_prof * 0.28, 0.0).astype(np.float32)
    rib_h = np.where(
        m_cov > 1e-4,
        (0.28 + 0.62 * m_prof) * (0.94 + 0.08 * m_twist) + 0.12 * m_knot,
        0.0,
    ).astype(np.float32)
    h = np.clip(np.maximum(inner_h, rib_h), 0.0, 1.0).astype(np.float32)

    # Alpha: tiny inner pores = 0.0, inner micro-filaments = 0.16-0.28, honeycomb ribs = 0.72-0.92
    inner_alpha = np.where(s_cov > 1e-4, s_cov * (0.20 * (0.85 + 0.15 * s_prof) + 0.08 * s_knot), 0.0)
    rib_alpha = np.where(m_cov > 1e-4, m_cov * (0.76 * (0.78 + 0.22 * m_prof) + 0.18 * m_knot), 0.0)
    alpha = np.clip(np.maximum(inner_alpha, rib_alpha), 0.0, 0.94).astype(np.float32)

    # Color: lighter smoky fine net + deeper sculptural braided honeycomb ribs with crisp highlights
    rib_w = _smoothstep(0.05, 0.45, m_cov)
    tone_inner = np.where(s_cov > 1e-4, 0.42 + 0.28 * s_prof, 0.46)
    tone_rib = 0.26 + 0.48 * m_prof + 0.14 * m_knot + 0.08 * m_twist + 0.06 * (fine_tex - 0.5)
    tone = np.clip(tone_inner * (1.0 - rib_w) + tone_rib * rib_w, 0.05, 0.96).astype(np.float32)
    albedo = common.apply_palette_5stop(tone, micro_detail=None, stops=stops, contrast=1.06, gamma=1.0)

    # PBR roughness: crisp braided ribs (0.40-0.56) vs softer fine inner net (0.54-0.68)
    r_inner = common.match_histogram_range(
        0.55 * (1.0 - s_prof) + 0.25 * s_knot + 0.20 * fine_tex,
        0.54,
        0.68,
        target_mean=0.61,
    )
    r_rib = common.match_histogram_range(
        0.48 * (1.0 - m_prof) + 0.28 * m_knot + 0.24 * (0.5 - 0.5 * m_twist),
        0.40,
        0.56,
        target_mean=0.47,
    )
    rough = np.clip(r_inner * (1.0 - rib_w) + r_rib * rib_w, 0.30, 0.95).astype(np.float32)

    base = _shade_sheer_rgba(
        albedo,
        h,
        alpha,
        rough,
        normal_strength=1.95,
        spec_intensity=0.18,
        ao_strength=0.20,
        cavity_strength=0.13,
        sheen_tint=tuple(float(c) for c in stops[-1]),
        sheen_strength=0.05,
    )
    return base, h, rough


def _build_flocked_dot_mesh(
    u: np.ndarray,
    v: np.ndarray,
    size: int,
    rng: np.random.Generator,
    stops: Sequence[tuple[int, int, int]],
) -> tuple[Image.Image, np.ndarray, np.ndarray]:
    """Fine sheer micro-mesh overlaid with lofty, opaque, matte velvety flocked dots."""
    _ = size
    uu, vv = _gentle_drape_uv(u, v, rng, amp=0.0005, freq=3)
    mesh = _hex_tricot_mesh_fields(
        uu,
        vv,
        nx=144,
        ny=74,
        strut_width=0.14,
        corner_round=0.08,
        knot_boost=0.20,
        wale_boost=0.14,
        twist_amp=0.07,
        twist_cycles=3.0,
    )
    prof = mesh["profile"]
    cov = mesh["coverage"]
    knot = mesh["knot"]

    # Staggered dot lattice: 12x12 unit cells -> slightly larger, loftier plush flocked dots
    dot_dist, _, _, dot_seed = _staggered_dot_lattice(uu, vv, cols=12, rows=12, rng=rng, jitter=0.006)

    # High-frequency upright cut-pile velvet fiber texture & fibrous perimeter fuzz
    pile_fine = common.periodic_fbm(u, v, 96, 5, 0.62, rng, mode="fbm")
    pile_tuft = common.periodic_fbm(u, v, 48, 4, 0.56, rng, mode="billow")
    edge_fuzz = common.periodic_fbm(u, v, 72, 4, 0.60, rng, mode="turbulence")

    # Organic feathery flocked dot silhouette
    dot_r = 0.168 + (dot_seed - 0.5) * 0.008 + (edge_fuzz - 0.5) * 0.026
    t_dot = np.clip(dot_dist / np.maximum(dot_r, 1e-4), 0.0, 1.5)
    flock_core = (1.0 - _smoothstep(0.76, 1.02, t_dot)).astype(np.float32)
    # Feathery stray flock fibers slightly beyond the dense dot core
    flock_fringe = (1.0 - _smoothstep(0.92, 1.14, t_dot)).astype(np.float32) * (0.45 + 0.55 * pile_fine)
    flock_mask = np.clip(np.maximum(flock_core, flock_fringe * 0.62), 0.0, 1.0).astype(np.float32)

    # Lofty plush velvet dome with dense granular micro-fiber pile relief
    dome = np.power(np.clip(1.0 - np.clip(t_dot / 1.02, 0.0, 1.0) ** 2, 0.0, 1.0), 0.42).astype(np.float32)
    pile_micro = 0.60 * pile_fine + 0.40 * pile_tuft

    # Height: delicate sheer mesh at [0, 0.28]; lofty fuzzy velvet dots rise to [0.36, 1.0]
    mesh_h = np.where(cov > 1e-4, prof * 0.28, 0.0).astype(np.float32)
    flock_h = flock_mask * (0.34 + 0.46 * dome + 0.20 * pile_micro * (0.45 + 0.55 * dome))
    h = np.clip(np.maximum(mesh_h, flock_h), 0.0, 1.0).astype(np.float32)

    # Alpha: open pores = 0.0, sheer filaments = 0.20-0.38, opaque velvet flocked dots = 0.90-0.99
    mesh_alpha = np.where(cov > 1e-4, cov * (0.30 * (0.85 + 0.15 * prof) + 0.12 * knot), 0.0)
    flock_alpha = flock_core * (0.91 + 0.08 * dome) + (1.0 - flock_core) * flock_fringe * 0.55
    alpha = np.clip(np.maximum(mesh_alpha, flock_alpha), 0.0, 0.99).astype(np.float32)

    # Color: sheer dark mesh + deep light-absorbing velvet pile core with soft grazing fiber rim
    velvet_rim = np.power(np.clip(t_dot, 0.0, 1.0), 2.2) * flock_core
    tone_mesh = np.where(cov > 1e-4, 0.40 + 0.34 * prof + 0.08 * knot, 0.46)
    tone_flock = 0.06 + 0.18 * pile_micro + 0.22 * velvet_rim + 0.06 * dome
    tone = np.clip(tone_mesh * (1.0 - flock_mask) + tone_flock * flock_mask, 0.02, 0.94).astype(np.float32)
    albedo = common.apply_palette_5stop(tone, micro_detail=None, stops=stops, contrast=1.08, gamma=1.0)

    # PBR roughness: maximum contrast — smooth sheer nylon mesh (0.44-0.58) vs ultra-matte flock (0.86-0.97)
    r_mesh = common.match_histogram_range(
        0.54 * (1.0 - prof) + 0.26 * knot + 0.20 * pile_tuft,
        0.44,
        0.58,
        target_mean=0.51,
    )
    r_flock = common.match_histogram_range(
        0.55 * pile_fine + 0.30 * (1.0 - dome) + 0.15 * pile_tuft,
        0.86,
        0.97,
        target_mean=0.92,
    )
    rough = np.clip(r_mesh * (1.0 - flock_mask) + r_flock * flock_mask, 0.35, 0.98).astype(np.float32)

    base = _shade_sheer_rgba(
        albedo,
        h,
        alpha,
        rough,
        normal_strength=2.10,
        spec_intensity=0.10,
        ao_strength=0.20,
        cavity_strength=0.14,
        sheen_tint=(136.0, 142.0, 154.0),
        sheen_strength=0.045,
    )
    return base, h, rough


def _build_micro_mesh_metallic_sheen(
    u: np.ndarray,
    v: np.ndarray,
    size: int,
    rng: np.random.Generator,
    stops: Sequence[tuple[int, int, int]],
) -> tuple[Image.Image, np.ndarray, np.ndarray]:
    """Fine sheer micro-mesh knitted with twisted metallic-sheen ribbon filaments."""
    _ = size
    uu, vv = _gentle_drape_uv(u, v, rng, amp=0.0005, freq=3)
    mesh = _hex_tricot_mesh_fields(
        uu,
        vv,
        nx=144,
        ny=74,
        strut_width=0.155,
        corner_round=0.075,
        knot_boost=0.26,
        wale_boost=0.16,
        twist_amp=0.14,
        twist_cycles=5.0,
    )
    prof = mesh["profile"]
    cov = mesh["coverage"]
    knot = mesh["knot"]
    twist = mesh["twist"]

    # Twisted metallic ribbon micro-facets creating realistic shimmering glints along filaments
    phase_jit = common.periodic_fbm(u, v, 18, 3, 0.50, rng, mode="fbm")
    glint_wave_a = 0.5 + 0.5 * np.cos((u * 288.0 + v * 148.0 + phase_jit * 3.0) * math.pi)
    glint_wave_b = 0.5 + 0.5 * np.cos((u * 144.0 - v * 296.0 - phase_jit * 2.5) * math.pi)
    glint = np.power(np.maximum(glint_wave_a, glint_wave_b), 3.2).astype(np.float32) * cov
    micro_shimmer = common.periodic_fbm(u, v, 42, 4, 0.55, rng, mode="fbm")

    # Height: crisp faceted metallic filament relief; exact 0.0 in open pores
    h_raw = prof * (0.86 + 0.14 * (0.5 + 0.5 * twist)) + 0.10 * glint
    h = np.where(cov > 1e-4, np.clip(h_raw, 0.0, 1.0), 0.0).astype(np.float32)

    # Alpha: fine metallic-sheen sheer mesh (open pores = 0.0, filaments = 0.30-0.58, knots/glints up to 0.68)
    alpha_raw = cov * (0.44 * (0.82 + 0.18 * prof) + 0.18 * knot + 0.08 * glint)
    alpha = np.where(cov > 1e-4, np.clip(alpha_raw, 0.0, 0.78), 0.0).astype(np.float32)

    # Color: lustrous champagne-platinum metallic tones with crisp twisted glint highlights
    tone_fil = np.clip(
        0.24
        + 0.42 * prof
        + 0.24 * glint
        + 0.10 * knot
        + 0.08 * twist
        + 0.06 * (micro_shimmer - 0.5),
        0.04,
        0.99,
    )
    tone = np.where(cov > 1e-4, tone_fil, 0.52 + 0.04 * (micro_shimmer - 0.5)).astype(np.float32)
    albedo = common.apply_palette_5stop(tone, micro_detail=None, stops=stops, contrast=1.10, gamma=0.96)

    # PBR roughness: distinctly low metallic-sheen roughness (0.16-0.38 on filaments, 0.40-0.52 in crevices)
    rough_raw = (
        0.46 * (1.0 - prof)
        + 0.30 * (1.0 - glint)
        + 0.14 * knot
        + 0.10 * (1.0 - micro_shimmer)
    )
    rough = common.match_histogram_range(rough_raw, 0.16, 0.52, target_mean=0.31, contrast=1.12)

    base = _shade_sheer_rgba(
        albedo,
        h,
        alpha,
        rough,
        normal_strength=1.45,
        spec_intensity=0.38,
        ao_strength=0.16,
        cavity_strength=0.10,
        sheen_tint=(255.0, 248.0, 232.0),
        sheen_strength=0.12,
    )
    return base, h, rough


_BUILDERS = (
    _build_micro_mesh_ultra_sheer_black,
    _build_micro_mesh_sheer_nude,
    _build_micro_mesh_sheer_white,
    _build_micro_mesh_sheer_red,
    _build_fine_tulle_soft,
    _build_sheer_dot_mesh,
    _build_sheer_stripe_mesh,
    _build_sheer_honeycomb_mesh,
    _build_flocked_dot_mesh,
    _build_micro_mesh_metallic_sheen,
)


def _compute_bundle(
    seed: int = 1,
    palette: PaletteArg = None,
    size: int = 512,
    variant: int = 0,
) -> tuple[Image.Image, np.ndarray, np.ndarray, SheerStyleSpec]:
    style = SHEER_STYLES[int(variant) % len(SHEER_STYLES)]
    rng = np.random.default_rng(int(seed))
    u, v = common.make_uv_grid(int(size))
    stops = _resolve_sheer_palette(palette, style.default_palette, rng)
    base, h, rough = _BUILDERS[style.variant](u, v, int(size), rng, stops)
    return base, np.asarray(h, dtype=np.float32), np.asarray(rough, dtype=np.float32), style


def render_sheer_bundle(
    style_or_variant: str | int,
    seed: int = 1,
    size: int = 2048,
    palette: PaletteArg = None,
) -> tuple[Image.Image, Image.Image, Image.Image, np.ndarray, np.ndarray, SheerStyleSpec]:
    """Single-pass PBR bundle renderer returning (base_rgba, normal, roughness, height, rough_field, style)."""
    if isinstance(style_or_variant, str):
        style = STYLE_BY_SLUG[style_or_variant]
        variant = style.variant
    else:
        variant = int(style_or_variant) % len(SHEER_STYLES)
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
    name="sheer-micro-mesh",
    kind="texture",
    tileable=True,
    origin="tools/generators/types/sheer_pack.py",
    default_palette="micro-mesh-ultra-sheer-black",
    description=(
        "Sheer Micro Mesh suite (10 tileable RGBA variants: micro-mesh-ultra-sheer-black, "
        "micro-mesh-sheer-nude, micro-mesh-sheer-white, micro-mesh-sheer-red, fine-tulle-soft, "
        "sheer-dot-mesh, sheer-stripe-mesh, sheer-honeycomb-mesh, flocked-dot-mesh, and "
        "micro-mesh-metallic-sheen) with calibrated PBR height, normal, and roughness."
    ),
    render_fn=generate,
    normal_source="native",
    normal_rule="Native per-style sheer filament and raised motif height field converted with calibrated Scharr normal strength.",
    normal_strength=1.45,
    height_fn=generate_height,
    normal_fn=generate_normal,
    roughness_fn=generate_roughness,
    default_roughness=0.55,
)
