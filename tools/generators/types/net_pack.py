"""Procedural open net & mesh generator suite (10 tileable RGBA materials).

Built strictly on the committed periodic math primitives in ``tools/generators/common.py``
with self-contained palettes defined inside this module:
  - Diamond nets across fine, medium, and wide gauges with exact Euclidean toroidal
    distance, helical 2-ply twisted strands, subtle over/under weave, and tied knot
    junctions (black, wide-gauge charcoal, fine white, fine nude, crimson red).
  - Exact periodic hexagonal honeycomb nets with uniform perpendicular strut width,
    warp-knit vertical pillar emphasis, filleted inner apertures, and 3-way junctions
    (black and white).
  - Athletic open square grid net with rounded-box superellipse apertures, twin-wale
    vertical knit ribs, laid-in horizontal bars, and reinforced grid intersections.
  - Seamless toroidal cobweb net built on periodic Lloyd Voronoi anchor frames with
    radial spokes, inward-sagging catenary concentric rings, and silk tie junctions.
  - Silver metallic woven diamond wire net with alternating over/under wire arch
    crimp, longitudinal drawn-wire micro-grain, and low metallic strand roughness.

All 10 materials are open RGBA base-color maps with true transparency in the apertures,
color-matched RGB outside the mask to prevent mipmap haloing, subtle tangent-space
OpenGL (+Y up) strand normal relief, and calibrated PBR roughness maps.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from PIL import Image

from .. import common
from ..interface import GeneratorSpec, PaletteArg


# ---------------------------------------------------------------------------
# Self-contained 5-stop RGB palettes (shadow -> midtone -> highlight)
# ---------------------------------------------------------------------------

NET_PALETTES: dict[str, list[tuple[int, int, int]]] = {
    "diamond_net_black": [
        (10, 11, 14),
        (20, 22, 27),
        (34, 37, 44),
        (56, 61, 71),
        (92, 99, 112),
    ],
    "diamond_net_wide_gauge": [
        (12, 13, 16),
        (24, 26, 31),
        (42, 46, 54),
        (70, 75, 86),
        (110, 117, 130),
    ],
    "diamond_net_fine_white": [
        (178, 180, 184),
        (208, 210, 214),
        (230, 232, 236),
        (245, 246, 249),
        (255, 255, 255),
    ],
    "diamond_net_fine_nude": [
        (124, 96, 78),
        (164, 134, 110),
        (196, 166, 140),
        (220, 194, 170),
        (238, 218, 198),
    ],
    "diamond_net_red": [
        (54, 8, 14),
        (112, 16, 26),
        (172, 24, 38),
        (214, 46, 58),
        (244, 114, 120),
    ],
    "hexagonal_net_black": [
        (11, 12, 15),
        (23, 25, 30),
        (39, 43, 51),
        (64, 69, 80),
        (102, 108, 122),
    ],
    "hexagonal_net_white": [
        (174, 176, 180),
        (206, 208, 212),
        (230, 232, 236),
        (245, 246, 249),
        (255, 255, 255),
    ],
    "grid_net_sport": [
        (18, 22, 28),
        (34, 41, 52),
        (56, 66, 80),
        (88, 101, 118),
        (134, 149, 168),
    ],
    "cobweb_net": [
        (16, 17, 21),
        (32, 34, 40),
        (54, 58, 67),
        (86, 91, 104),
        (132, 138, 152),
    ],
    "metallic_net_silver": [
        (78, 84, 94),
        (136, 144, 156),
        (186, 194, 206),
        (226, 232, 240),
        (249, 252, 255),
    ],
}


@dataclass(frozen=True)
class NetStyleSpec:
    variant: int
    slug: str
    label: str
    default_palette: str
    normal_strength: float
    rough_range: tuple[float, float]
    rough_mean: float
    metallic: bool
    description: str


NET_STYLES: tuple[NetStyleSpec, ...] = (
    NetStyleSpec(
        0,
        "diamond-net-black",
        "Diamond Net Black",
        "diamond_net_black",
        2.1,
        (0.56, 0.78),
        0.67,
        False,
        "Medium-gauge open elongated diamond net in deep black with 2-ply twisted strands, tied junctions, and matte-satin fiber roughness.",
    ),
    NetStyleSpec(
        1,
        "diamond-net-wide-gauge",
        "Diamond Net Wide Gauge",
        "diamond_net_wide_gauge",
        2.5,
        (0.58, 0.80),
        0.69,
        False,
        "Wide-gauge open diamond net with heavy braided dark charcoal cords, pronounced knotted intersections, and large open apertures.",
    ),
    NetStyleSpec(
        2,
        "diamond-net-fine-white",
        "Diamond Net Fine White",
        "diamond_net_fine_white",
        1.7,
        (0.60, 0.82),
        0.71,
        False,
        "Fine-gauge open diamond net in clean bright white with slender twisted filaments, delicate junctions, and soft fiber roughness.",
    ),
    NetStyleSpec(
        3,
        "diamond-net-fine-nude",
        "Diamond Net Fine Nude",
        "diamond_net_fine_nude",
        1.6,
        (0.54, 0.76),
        0.65,
        False,
        "Fine-gauge open diamond net in warm neutral beige-tan with slender stretch-filament strands and smooth matte-sheen finish.",
    ),
    NetStyleSpec(
        4,
        "diamond-net-red",
        "Diamond Net Red",
        "diamond_net_red",
        2.1,
        (0.54, 0.76),
        0.65,
        False,
        "Medium-gauge open diamond net in crimson red with 2-ply helical twist, snug knotted crossovers, and subtle fiber sheen.",
    ),
    NetStyleSpec(
        5,
        "hexagonal-net-black",
        "Hexagonal Net Black",
        "hexagonal_net_black",
        2.0,
        (0.58, 0.80),
        0.69,
        False,
        "Medium-gauge open hexagonal honeycomb net in deep black with reinforced 3-way junctions, filleted apertures, and twisted vertical pillars.",
    ),
    NetStyleSpec(
        6,
        "hexagonal-net-white",
        "Hexagonal Net White",
        "hexagonal_net_white",
        1.7,
        (0.62, 0.84),
        0.73,
        False,
        "Fine-gauge open hexagonal honeycomb net in bright white with airy slender struts, tidy 3-way junctions, and matte synthetic fiber texture.",
    ),
    NetStyleSpec(
        7,
        "grid-net-sport",
        "Grid Net Sport",
        "grid_net_sport",
        2.3,
        (0.48, 0.72),
        0.60,
        False,
        "Open orthogonal athletic grid mesh in dark slate-charcoal with filleted square apertures, twin-wale knit vertical ribs, and technical polyester sheen.",
    ),
    NetStyleSpec(
        8,
        "cobweb-net",
        "Cobweb Net",
        "cobweb_net",
        1.9,
        (0.62, 0.84),
        0.73,
        False,
        "Seamless organic open cobweb net with Voronoi anchor frames, radial spokes, inward-sagging concentric catenary rings, and matte fibrous strands.",
    ),
    NetStyleSpec(
        9,
        "metallic-net-silver",
        "Metallic Net Silver",
        "metallic_net_silver",
        2.5,
        (0.16, 0.34),
        0.24,
        True,
        "Open 45-degree silver metallic diamond wire net with alternating over/under wire crimp weave, drawn-wire micro-grain, and low glossy strand roughness.",
    ),
)

STYLE_BY_SLUG: dict[str, NetStyleSpec] = {s.slug: s for s in NET_STYLES}


# ---------------------------------------------------------------------------
# Shared toroidal utilities (strictly built on common.py)
# ---------------------------------------------------------------------------


def _resolve_local_palette(
    palette: PaletteArg, default_name: str, rng: np.random.Generator
) -> list[tuple[int, int, int]]:
    if palette is None:
        return list(NET_PALETTES[default_name])
    if isinstance(palette, str):
        key = palette.strip().lower().replace("-", "_")
        if key in NET_PALETTES:
            return list(NET_PALETTES[key])
    return common.resolve_palette(palette, "charcoal_flannel", rng)


def _wrap01(arr: np.ndarray) -> np.ndarray:
    """Safe float32 modulo [0, 1) avoiding float32 (-eps % 1.0 == 1.0)."""
    out = np.mod(arr.astype(np.float64), 1.0).astype(np.float32)
    return np.where(out >= 1.0, np.float32(0.0), out)


def _wrap_signed(arr: np.ndarray) -> np.ndarray:
    """Wrap scalar phase cleanly into [-0.5, 0.5) using float64 intermediate."""
    w = np.mod(arr.astype(np.float64) + 0.5, 1.0) - 0.5
    return w.astype(np.float32)


def _smoothstep(edge0: float | np.ndarray, edge1: float | np.ndarray, x: np.ndarray) -> np.ndarray:
    denom = np.where(np.abs(edge1 - edge0) > 1e-12, edge1 - edge0, 1e-12)
    t = np.clip((x - edge0) / denom, 0.0, 1.0).astype(np.float32)
    return t * t * (3.0 - 2.0 * t)


def _interp_stops(field: np.ndarray, stops: list[tuple[int, int, int]]) -> np.ndarray:
    f = np.clip(field, 0.0, 1.0).astype(np.float32)
    pos = np.linspace(0.0, 1.0, len(stops), dtype=np.float32)
    out = np.empty((*f.shape, 3), dtype=np.float32)
    for ch in range(3):
        vals = np.array([float(s[ch]) for s in stops], dtype=np.float32)
        out[..., ch] = np.interp(f, pos, vals)
    return out


def _warp_uv(
    u: np.ndarray,
    v: np.ndarray,
    freq: int,
    amp: float,
    rng: np.random.Generator,
    octaves: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Subtle periodic tension warp preserving exact toroidal tiling."""
    if amp <= 1e-7:
        return u, v
    du = common.periodic_fbm(u, v, freq, octaves, 0.52, rng, mode="fbm") - 0.5
    dv = common.periodic_fbm(u, v, freq, octaves, 0.52, rng, mode="fbm") - 0.5
    return _wrap01(u + du * float(amp)), _wrap01(v + dv * float(amp))


def _fiber_micro(u: np.ndarray, v: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """High-frequency periodic fiber micro-grain & fuzz for 2048px strand realism."""
    fine = common.periodic_fbm(u, v, 64, 3, 0.58, rng, mode="fbm")
    fuzz = common.periodic_fbm(u, v, 140, 2, 0.62, rng, mode="turbulence")
    return fine.astype(np.float32), fuzz.astype(np.float32)


def _strand_profile(dist_uv: np.ndarray, radius_uv: np.ndarray | float, aa_uv: float) -> tuple[np.ndarray, np.ndarray]:
    """Return smooth `(alpha_mask, cylinder_dome_height)` for a strand of given radius."""
    r = np.maximum(radius_uv, aa_uv * 0.65)
    alpha = _smoothstep(r + aa_uv, r - aa_uv, dist_uv)
    u_rad = np.clip(dist_uv / np.maximum(r + 0.55 * aa_uv, 1e-7), 0.0, 1.0)
    # Soft-fillet cylindrical dome avoids extreme normal spikes at cutout boundary
    dome = np.power(np.clip(1.0 - u_rad * u_rad, 0.0, 1.0), 0.72).astype(np.float32)
    return alpha.astype(np.float32), dome


def _calibrate_strand_roughness(
    raw_strand_field: np.ndarray,
    alpha: np.ndarray,
    r_min: float,
    r_max: float,
    r_mean: float,
    hole_neutral: float | None = None,
) -> np.ndarray:
    """Calibrate PBR roughness strictly over the strand pixels, bleeding cleanly across edges."""
    strand_mask = alpha > 0.25
    if not np.any(strand_mask):
        return np.full_like(alpha, r_mean, dtype=np.float32)

    vals = raw_strand_field[strand_mask]
    lo, hi = np.percentile(vals, [0.5, 99.5])
    span = float(hi - lo)
    if span < 1e-6:
        norm = np.full_like(raw_strand_field, 0.5, dtype=np.float32)
    else:
        norm = np.clip((raw_strand_field - float(lo)) / span, 0.0, 1.0).astype(np.float32)

    desired = float(np.clip((r_mean - r_min) / max(r_max - r_min, 1e-6), 0.08, 0.92))
    g_lo, g_hi = 0.2, 5.0
    sample_vals = np.clip(norm[strand_mask], 1e-5, 1.0)
    for _ in range(12):
        g_mid = 0.5 * (g_lo + g_hi)
        if float(np.mean(np.power(sample_vals, g_mid))) > desired:
            g_lo = g_mid
        else:
            g_hi = g_mid
    gamma = 0.5 * (g_lo + g_hi)
    shaped = np.power(np.clip(norm, 1e-5, 1.0), gamma).astype(np.float32)
    strand_r = np.clip(r_min + shaped * (r_max - r_min), 0.04, 0.98).astype(np.float32)

    # Dilate alpha mask slightly so strand roughness bleeds past the anti-aliased edge
    # (prevents wrong roughness bleeding during 3D engine mipmapping), while deep hole
    # interiors hold a clean uniform reference value.
    bg_val = float(r_mean if hole_neutral is None else hole_neutral)
    res_scale = alpha.shape[0] / 512.0
    halo = np.clip(common.periodic_gaussian_blur(alpha, max(1.2, 1.8 * res_scale)) * 3.2, 0.0, 1.0)
    bleed_mask = np.maximum(alpha, halo)
    out = strand_r * bleed_mask + bg_val * (1.0 - bleed_mask)
    return np.clip(out, 0.04, 0.98).astype(np.float32)


def _shade_open_net_rgba(
    albedo_rgb: np.ndarray,
    height: np.ndarray,
    alpha: np.ndarray,
    roughness: np.ndarray,
    normal_strength: float,
    metallic: bool = False,
    sheen_amt: float = 12.0,
    spec_intensity: float = 0.14,
    reference_size: int = 512,
) -> Image.Image:
    """Shade open net strands into an RGBA PIL image with clean RGB bleed in transparent holes."""
    h = np.clip(height, 0.0, 1.0).astype(np.float32)
    r = np.clip(roughness, 0.04, 0.98).astype(np.float32)
    a = np.clip(alpha, 0.0, 1.0).astype(np.float32)

    dh_dx, dh_dy_up = common.periodic_scharr_gradients(h, reference_size=reference_size)
    nx = -dh_dx * float(normal_strength)
    ny = dh_dy_up * float(normal_strength)
    nz = np.ones_like(h, dtype=np.float32)
    inv_len = 1.0 / np.sqrt(nx * nx + ny * ny + nz * nz + 1e-8)
    nx, ny, nz = nx * inv_len, ny * inv_len, nz * inv_len

    lx, ly, lz = 0.44, -0.54, 0.72
    ln = math.sqrt(lx * lx + ly * ly + lz * lz)
    lx, ly, lz = lx / ln, ly / ln, lz / ln

    ndotl = np.clip(nx * lx + ny * ly + nz * lz, 0.0, 1.0)
    hx, hy, hz = lx, ly, lz + 1.0
    hn = math.sqrt(hx * hx + hy * hy + hz * hz)
    hx, hy, hz = hx / hn, hy / hn, hz / hn
    ndoth = np.clip(nx * hx + ny * hy + nz * hz, 0.0, 1.0)

    # Secondary fill light from opposite rim keeps cylindrical cord undersides legible
    ndotl_fill = np.clip(nx * (-0.36) + ny * 0.44 + nz * 0.82, 0.0, 1.0)

    # Subtle cylindrical crown-to-edge shading (never blackens edges of bright nets)
    cyl_shade = 0.74 + 0.26 * np.power(np.clip(h, 0.0, 1.0), 0.55)

    if metallic:
        shininess = np.clip((1.0 - r) ** 2 * 140.0 + 10.0, 10.0, 150.0)
        spec_peak = np.power(ndoth, shininess) * spec_intensity * (1.0 - r * 0.65)
        # Metallic tinted specular reflection + crisp glancing Fresnel
        fresnel = 0.25 + 0.75 * np.power(np.clip(1.0 - nz, 0.0, 1.0), 3.0)
        diffuse = (0.52 + 0.42 * ndotl + 0.14 * ndotl_fill) * cyl_shade
        spec_tint = 0.72 * albedo_rgb + 0.28 * 255.0
        lit = albedo_rgb * diffuse[..., None] + spec_tint * (spec_peak * 1.45 + fresnel * 0.16)[..., None]
    else:
        shininess = np.clip((1.0 - r) ** 2 * 64.0 + 4.0, 4.0, 68.0)
        spec_peak = np.power(ndoth, shininess) * spec_intensity * (1.0 - r * 0.82)
        sheen = np.power(np.clip(1.0 - nz, 0.0, 1.0), 1.8) * sheen_amt
        diffuse = (0.64 + 0.30 * ndotl + 0.12 * ndotl_fill) * cyl_shade
        lit = albedo_rgb * diffuse[..., None] + (spec_peak * 255.0 + sheen)[..., None]

    # Clean RGB edge bleed into transparent holes so bilinear sampling & mipmaps stay halo-free
    res_scale = h.shape[0] / 512.0
    rgb_clipped = np.clip(lit, 0.0, 255.0).astype(np.float32)
    mask_blur = common.periodic_gaussian_blur(a, max(2.0, 3.0 * res_scale))
    bleed_rgb = np.empty_like(rgb_clipped)
    for ch in range(3):
        ch_blur = common.periodic_gaussian_blur(rgb_clipped[..., ch] * a, max(2.0, 3.0 * res_scale))
        mean_strand = float(np.sum(rgb_clipped[..., ch] * a) / max(float(np.sum(a)), 1.0))
        extrap = np.where(mask_blur > 1e-4, ch_blur / np.maximum(mask_blur, 1e-4), mean_strand)
        bleed_rgb[..., ch] = np.where(a > 0.02, rgb_clipped[..., ch], extrap)

    rgba = np.empty((*h.shape, 4), dtype=np.uint8)
    rgba[..., :3] = np.rint(np.clip(bleed_rgb, 0.0, 255.0)).astype(np.uint8)
    rgba[..., 3] = np.rint(np.clip(a * 255.0, 0.0, 255.0)).astype(np.uint8)
    return Image.fromarray(rgba, mode="RGBA")


# ---------------------------------------------------------------------------
# Core geometric net builders
# ---------------------------------------------------------------------------


def _build_diamond_net(
    u: np.ndarray,
    v: np.ndarray,
    size: int,
    rng: np.random.Generator,
    stops: list[tuple[int, int, int]],
    style: NetStyleSpec,
    *,
    a: int,
    b: int,
    radius_uv: float,
    warp_amp: float,
    twists_per_span: int,
    twist_depth: float,
    knot_scale: float,
    knot_boost: float,
    wire_weave: bool = False,
    sheen_amt: float = 11.0,
    spec_intensity: float = 0.14,
) -> tuple[Image.Image, np.ndarray, np.ndarray]:
    """Build a seamless toroidal diamond net (textile knotted or woven metallic wire)."""
    uw, vw = _warp_uv(u, v, freq=3, amp=warp_amp, rng=rng, octaves=3)
    fine, fuzz = _fiber_micro(u, v, rng)
    slub = common.periodic_fbm(uw, vw, max(12, a), 3, 0.55, rng, mode="fbm", freqy=max(10, b))

    aa_uv = 1.30 / float(size)
    inv_norm = 1.0 / math.hypot(a, b)

    p1 = a * uw.astype(np.float64) + b * vw.astype(np.float64)
    p2 = a * uw.astype(np.float64) - b * vw.astype(np.float64)

    w1 = _wrap_signed(p1)
    w2 = _wrap_signed(p2)

    # Periodic integer indices: (idx1 + idx2) & 1 is exact toroidal invariant for all (a, b)
    idx1 = np.floor(p1 + 0.5).astype(np.int32)
    idx2 = np.floor(p2 + 0.5).astype(np.int32)
    over1 = ((idx1 + idx2) & 1) == 0

    d1 = np.abs(w1) * np.float32(inv_norm)
    d2 = np.abs(w2) * np.float32(inv_norm)

    # Subtle per-strand diameter and tension modulation
    rad_mod = (1.0 + (slub - 0.5) * (0.08 if wire_weave else 0.18) + (fine - 0.5) * 0.06).astype(np.float32)
    r1 = np.float32(radius_uv) * rad_mod
    r2 = np.float32(radius_uv) * (1.0 + (0.5 - slub) * (0.08 if wire_weave else 0.18) + (fine - 0.5) * 0.06).astype(
        np.float32
    )

    a1, dome1 = _strand_profile(d1, r1, aa_uv)
    a2, dome2 = _strand_profile(d2, r2, aa_uv)

    # Exact Euclidean distance to nearest diamond intersection
    du_int = (w1 + w2) / np.float32(2.0 * a)
    dv_int = (w1 - w2) / np.float32(2.0 * b)
    r_int = np.sqrt(du_int * du_int + dv_int * dv_int).astype(np.float32)

    # Periodic helical 2-ply twist (or drawn-wire micro-grooves for metallic wire)
    r_phase = float(radius_uv) / inv_norm
    u1_cross = np.clip(w1 / max(r_phase, 1e-6), -1.0, 1.0)
    u2_cross = np.clip(w2 / max(r_phase, 1e-6), -1.0, 1.0)

    if wire_weave:
        # Subtle longitudinal drawn-wire striations & slight metallic crimp arch over/under
        draw1 = 0.5 + 0.5 * np.sin((u1_cross * 3.5 + (fine - 0.5) * 0.6) * math.pi)
        draw2 = 0.5 + 0.5 * np.sin((u2_cross * 3.5 + (fine - 0.5) * 0.6) * math.pi)
        twist1 = 1.0 + twist_depth * (draw1 - 0.5)
        twist2 = 1.0 + twist_depth * (draw2 - 0.5)

        # Smooth over/under wire crimp around intersections
        int_env = np.exp(-(r_int * r_int) / max((radius_uv * 2.6) ** 2, 1e-8)).astype(np.float32)
        crimp1 = np.where(over1, 1.0 + 0.24 * int_env, 1.0 - 0.34 * int_env)
        crimp2 = np.where(over1, 1.0 - 0.34 * int_env, 1.0 + 0.24 * int_env)

        h1 = dome1 * twist1 * crimp1 * 0.78
        h2 = dome2 * twist2 * crimp2 * 0.78
        h_raw = np.maximum(h1, h2)
        alpha = np.maximum(a1, a2)
        knot_mask = int_env * alpha
        top_is_1 = h1 >= h2
        under_ao = np.clip(1.0 - 0.30 * int_env * np.where(top_is_1, ~over1, over1).astype(np.float32), 0.68, 1.0)
    else:
        # Authentic helical 2-ply yarn twist along each diagonal strand
        t1 = 0.5 + 0.5 * np.sin(2.0 * math.pi * (twists_per_span * p2.astype(np.float32) + 1.15 * u1_cross))
        t2 = 0.5 + 0.5 * np.sin(2.0 * math.pi * (twists_per_span * p1.astype(np.float32) - 1.15 * u2_cross))
        twist1 = (1.0 - twist_depth * 0.45) + twist_depth * t1
        twist2 = (1.0 - twist_depth * 0.45) + twist_depth * t2

        # Slight over/under lap into the knotted junction
        int_env = np.exp(-(r_int * r_int) / max((radius_uv * 2.1) ** 2, 1e-8)).astype(np.float32)
        lap1 = np.where(over1, 1.0 + 0.10 * int_env, 1.0 - 0.16 * int_env)
        lap2 = np.where(over1, 1.0 - 0.16 * int_env, 1.0 + 0.10 * int_env)

        h1 = dome1 * twist1 * lap1 * 0.74
        h2 = dome2 * twist2 * lap2 * 0.74

        # Tied/knotted junction bead slightly elongated along vertical net tension
        knot_dist = np.sqrt((du_int * 1.10) ** 2 + (dv_int * 0.90) ** 2).astype(np.float32)
        knot_r = np.float32(radius_uv * knot_scale)
        a_knot, dome_knot = _strand_profile(knot_dist, knot_r, aa_uv)
        knot_twist = 0.90 + 0.10 * np.sin((du_int * a * 6.0 + dv_int * b * 6.0) * math.pi)
        h_knot = dome_knot * knot_twist * np.float32(0.76 + knot_boost)

        h_raw = np.maximum(np.maximum(h1, h2), h_knot)
        alpha = np.maximum(np.maximum(a1, a2), a_knot)
        knot_mask = a_knot
        under_ao = np.clip(1.0 - 0.20 * int_env * (1.0 - dome_knot), 0.76, 1.0)

    # Add subtle micro-fiber / micro-wire relief strictly on the strands (0 in holes!)
    micro_amp = 0.045 if wire_weave else 0.085
    h_strand = np.clip(h_raw + alpha * micro_amp * ((fine - 0.5) * 0.65 + (fuzz - 0.5) * 0.35), 0.0, 1.0)
    h = (h_strand * alpha).astype(np.float32)

    # Tone mapping along strand crown, twist, knot, and subtle dye/wire variation
    tone_raw = (
        0.58 * h_strand
        + 0.14 * knot_mask
        + 0.14 * fine
        + 0.08 * slub
        + 0.06 * (1.0 - fuzz)
    ) * under_ao
    tone = np.clip(tone_raw, 0.0, 1.0)
    albedo = _interp_stops(tone, stops)

    # Calibrated strand roughness
    if wire_weave:
        rough_raw = 0.36 * (1.0 - h_strand) + 0.28 * fuzz + 0.20 * (1.0 - fine) + 0.16 * (1.0 - knot_mask)
        hole_neutral = 0.52
    else:
        rough_raw = 0.40 * (1.0 - h_strand) + 0.26 * fuzz + 0.18 * (1.0 - fine) + 0.16 * knot_mask
        hole_neutral = style.rough_mean

    rough = _calibrate_strand_roughness(
        rough_raw,
        alpha,
        style.rough_range[0],
        style.rough_range[1],
        style.rough_mean,
        hole_neutral=hole_neutral,
    )
    base = _shade_open_net_rgba(
        albedo,
        h,
        alpha,
        rough,
        normal_strength=style.normal_strength,
        metallic=style.metallic,
        sheen_amt=sheen_amt,
        spec_intensity=spec_intensity,
    )
    return base, h, rough


def _build_hexagonal_net(
    u: np.ndarray,
    v: np.ndarray,
    size: int,
    rng: np.random.Generator,
    stops: list[tuple[int, int, int]],
    style: NetStyleSpec,
    *,
    nx: int,
    my: int,
    radius_uv: float,
    warp_amp: float,
    pillar_boost: float,
    vertex_scale: float,
    sheen_amt: float = 11.0,
    spec_intensity: float = 0.13,
) -> tuple[Image.Image, np.ndarray, np.ndarray]:
    """Build an exact periodic pointy-top hexagonal honeycomb net with 3-way junctions."""
    uw, vw = _warp_uv(u, v, freq=3, amp=warp_amp, rng=rng, octaves=3)
    fine, fuzz = _fiber_micro(u, v, rng)
    slub = common.periodic_fbm(uw, vw, max(14, nx), 3, 0.56, rng, mode="fbm", freqy=max(14, my * 2))

    aa_uv = 1.30 / float(size)
    wx = 1.0 / float(nx)
    hy = 1.0 / float(2 * my)
    h2 = 2.0 * hy

    # Quarter-cell horizontal offset keeps tile edge off vertical hex wall centerlines
    uu = _wrap01(uw + np.float32(0.25 * wx))

    # Staggered sub-lattices A (even rows) and B (odd rows)
    dax = _wrap_signed(uu / wx) * np.float32(wx)
    day = _wrap_signed(vw / h2) * np.float32(h2)
    dbx = _wrap_signed((uu - 0.5 * wx) / wx) * np.float32(wx)
    dby = _wrap_signed((vw - hy) / h2) * np.float32(h2)

    use_a = (dax * dax + day * day) <= (dbx * dbx + dby * dby)
    dx = np.where(use_a, dax, dbx).astype(np.float32)
    dy = np.where(use_a, day, dby).astype(np.float32)

    # Exact perpendicular distance to all 6 Voronoi bisector walls of the hexagon
    neighbors = (
        (wx, 0.0, True),
        (-wx, 0.0, True),
        (0.5 * wx, hy, False),
        (-0.5 * wx, hy, False),
        (0.5 * wx, -hy, False),
        (-0.5 * wx, -hy, False),
    )

    d1 = np.full_like(dx, 1e9, dtype=np.float32)
    d2 = np.full_like(dx, 1e9, dtype=np.float32)
    t_along = np.zeros_like(dx, dtype=np.float32)
    is_vert = np.zeros_like(dx, dtype=np.bool_)

    for px, py, vert_flag in neighbors:
        length = math.hypot(px, py)
        inv_l = np.float32(1.0 / length)
        half_l = np.float32(0.5 * length)
        dist = half_l - (dx * np.float32(px) + dy * np.float32(py)) * inv_l
        tang = (-dx * np.float32(py) + dy * np.float32(px)) * inv_l

        closer1 = dist < d1
        between = (~closer1) & (dist < d2)
        d2 = np.where(closer1, d1, np.where(between, dist, d2))
        d1 = np.where(closer1, dist, d1)
        t_along = np.where(closer1, tang, t_along)
        is_vert = np.where(closer1, vert_flag, is_vert)

    d1 = np.maximum(d1, 0.0)
    d2 = np.maximum(d2, 0.0)

    # Warp-knitted vertical hex pillars are slightly fuller than angled traverse bars
    strut_rad = np.where(is_vert, radius_uv * pillar_boost, radius_uv * 0.94).astype(np.float32)
    strut_rad = strut_rad * (1.0 + (slub - 0.5) * 0.14 + (fine - 0.5) * 0.06).astype(np.float32)

    a_strut, dome_strut = _strand_profile(d1, strut_rad, aa_uv)

    # Helical yarn twist along each hexagonal strut (symmetric in t_along so vertices match)
    edge_len = max(wx * 0.58, 1e-5)
    twist_wave = 0.5 + 0.5 * np.sin((np.abs(t_along) / edge_len * 4.5 + d1 / np.maximum(strut_rad, 1e-6) * 0.8) * 2.0 * math.pi)
    strut_h = dome_strut * (0.86 + 0.14 * twist_wave) * np.where(is_vert, 0.80, 0.74).astype(np.float32)

    # 3-way vertex distance (exact Euclidean distance to hex corner via two adjacent walls at 120 deg)
    # For angle 120 deg between walls, distance to vertex from wall distances (d1, d2) is:
    # r_v = sqrt(d1^2 + d2^2 + d1*d2) * (2 / sqrt(3))
    r_vert = (np.sqrt(d1 * d1 + d2 * d2 + d1 * d2) * np.float32(2.0 / math.sqrt(3.0))).astype(np.float32)
    vert_r = np.float32(radius_uv * vertex_scale) * (1.0 + (slub - 0.5) * 0.10).astype(np.float32)
    a_vert, dome_vert = _strand_profile(r_vert, vert_r, aa_uv)
    h_vert = dome_vert * np.float32(0.92)

    # Gentle inner corner filleting right at the 3-way junction
    fillet_r = np.float32(radius_uv * 1.85)
    fillet_dist = np.maximum(0.0, d1 + d2 - fillet_r)
    fillet_alpha = _smoothstep(fillet_r * 0.45 + aa_uv, fillet_r * 0.45 - aa_uv, fillet_dist) * (
        _smoothstep(fillet_r * 1.35, 0.0, r_vert)
    )

    alpha = np.maximum(np.maximum(a_strut, a_vert), fillet_alpha * 0.95)
    h_raw = np.maximum(np.maximum(strut_h, h_vert), fillet_alpha * 0.56)

    h_strand = np.clip(h_raw + alpha * 0.08 * ((fine - 0.5) * 0.65 + (fuzz - 0.5) * 0.35), 0.0, 1.0)
    h = (h_strand * alpha).astype(np.float32)

    tone = np.clip(0.56 * h_strand + 0.16 * a_vert + 0.14 * fine + 0.08 * slub + 0.06 * (1.0 - fuzz), 0.0, 1.0)
    albedo = _interp_stops(tone, stops)

    rough_raw = 0.42 * (1.0 - h_strand) + 0.24 * fuzz + 0.18 * (1.0 - fine) + 0.16 * a_vert
    rough = _calibrate_strand_roughness(
        rough_raw,
        alpha,
        style.rough_range[0],
        style.rough_range[1],
        style.rough_mean,
        hole_neutral=style.rough_mean,
    )
    base = _shade_open_net_rgba(
        albedo,
        h,
        alpha,
        rough,
        normal_strength=style.normal_strength,
        metallic=False,
        sheen_amt=sheen_amt,
        spec_intensity=spec_intensity,
    )
    return base, h, rough


def _build_grid_net_sport(
    u: np.ndarray,
    v: np.ndarray,
    size: int,
    rng: np.random.Generator,
    stops: list[tuple[int, int, int]],
    style: NetStyleSpec,
) -> tuple[Image.Image, np.ndarray, np.ndarray]:
    """Build an open athletic square/rectangular grid mesh with filleted holes and knit ribs."""
    uw, vw = _warp_uv(u, v, freq=3, amp=0.0028, rng=rng, octaves=3)
    fine, fuzz = _fiber_micro(u, v, rng)
    cols = 22
    rows = 22
    slub = common.periodic_fbm(uw, vw, cols, 3, 0.55, rng, mode="fbm", freqy=rows)

    aa_uv = 1.30 / float(size)
    cell_w = 1.0 / float(cols)
    cell_h = 1.0 / float(rows)

    # Half-cell shift places tile boundaries across open apertures instead of rib centerlines
    uu = _wrap01(uw + np.float32(0.5 * cell_w))
    vv = _wrap01(vw + np.float32(0.5 * cell_h))

    # Hole-centered local cell coordinates in [-0.5, 0.5)
    hx = _wrap_signed(uu * cols - 0.5)
    hy = _wrap_signed(vv * rows - 0.5)

    # Rounded-square aperture SDF in UV units (~52% open athletic mesh area)
    bx = 0.365 * cell_w
    by = 0.380 * cell_h
    cr = 0.110 * min(cell_w, cell_h)

    ax = np.abs(hx) * np.float32(cell_w)
    ay = np.abs(hy) * np.float32(cell_h)
    qx = np.maximum(ax - (bx - cr), 0.0)
    qy = np.maximum(ay - (by - cr), 0.0)
    inner_max = np.minimum(np.maximum(ax - (bx - cr), ay - (by - cr)), 0.0)
    # Positive inside the solid mesh web, negative inside the open aperture
    web_sdf = (np.sqrt(qx * qx + qy * qy) + inner_max - cr).astype(np.float32)

    # Slight organic aperture micro-variation
    web_sdf = web_sdf + (slub - 0.5) * np.float32(0.00040)
    alpha = _smoothstep(-aa_uv, aa_uv, web_sdf)

    # Rib-centered coordinates in UV units (0 on vertical/horizontal bar centerlines)
    lx = _wrap_signed(uu * cols) * np.float32(cell_w)
    ly = _wrap_signed(vv * rows) * np.float32(cell_h)
    dx = np.abs(lx)
    dy = np.abs(ly)

    # Vertical twin-wale knit pillars (subtle double-cord rib + chevron knit loops)
    rv = np.float32(0.148 * cell_w)
    rh = np.float32(0.132 * cell_h)
    _, dome_v = _strand_profile(dx, rv, aa_uv)
    _, dome_h = _strand_profile(dy, rh, aa_uv)

    wale_split = 0.88 + 0.12 * np.cos((lx / max(float(rv), 1e-6)) * math.pi * 2.0)
    v_loop_phase = (vv * rows * 4.0 + np.abs(lx) / max(float(rv), 1e-6) * 0.55) * 2.0 * math.pi
    v_knit = 0.86 + 0.14 * (0.5 + 0.5 * np.sin(v_loop_phase))
    h_braid = 0.88 + 0.12 * (0.5 + 0.5 * np.sin((uu * cols * 4.0 - ly / max(float(rh), 1e-6) * 0.9) * 2.0 * math.pi))

    hv = dome_v * wale_split * v_knit * np.float32(0.84)
    hh = dome_h * h_braid * np.float32(0.72)

    # Reinforced rounded intersection bridge pad
    r_cross = np.sqrt((dx / np.float32(1.05)) ** 2 + (dy / np.float32(0.95)) ** 2)
    _, dome_cross = _strand_profile(r_cross, np.float32(0.175 * cell_w), aa_uv)
    h_cross = dome_cross * np.float32(0.92)

    # Smooth fillet height from aperture edge inward so rounded corners have clean normal bevel
    fillet_bevel = np.power(_smoothstep(0.0, float(0.095 * cell_w), web_sdf), 0.60) * np.float32(0.68)
    h_raw = np.maximum(np.maximum(np.maximum(hv, hh), h_cross), fillet_bevel)
    h_strand = np.clip(h_raw + alpha * 0.065 * ((fine - 0.5) * 0.70 + (fuzz - 0.5) * 0.30), 0.0, 1.0)
    h = (h_strand * alpha).astype(np.float32)

    tone = np.clip(
        0.54 * h_strand + 0.14 * dome_cross + 0.12 * dome_v + 0.12 * fine + 0.08 * slub,
        0.0,
        1.0,
    )
    albedo = _interp_stops(tone, stops)

    rough_raw = 0.40 * (1.0 - h_strand) + 0.24 * fuzz + 0.20 * (1.0 - fine) + 0.16 * (1.0 - dome_v)
    rough = _calibrate_strand_roughness(
        rough_raw,
        alpha,
        style.rough_range[0],
        style.rough_range[1],
        style.rough_mean,
        hole_neutral=style.rough_mean,
    )
    base = _shade_open_net_rgba(
        albedo,
        h,
        alpha,
        rough,
        normal_strength=style.normal_strength,
        metallic=False,
        sheen_amt=14.0,
        spec_intensity=0.18,
    )
    return base, h, rough


def _build_cobweb_net(
    u: np.ndarray,
    v: np.ndarray,
    size: int,
    rng: np.random.Generator,
    stops: list[tuple[int, int, int]],
    style: NetStyleSpec,
) -> tuple[Image.Image, np.ndarray, np.ndarray]:
    """Build a seamless toroidal multi-hub cobweb net using periodic Lloyd Voronoi anchors."""
    # Gentle organic domain warp keeps every web spoke & frame thread naturally tensioned
    uw, vw = _warp_uv(u, v, freq=3, amp=0.014, rng=rng, octaves=3)
    fine, fuzz = _fiber_micro(u, v, rng)
    var_noise = common.periodic_fbm(uw, vw, 10, 3, 0.55, rng, mode="fbm")
    ring_wobble = common.periodic_fbm(uw, vw, 14, 3, 0.52, rng, mode="fbm")

    grid_cells = 3
    aa_uv = 1.30 / float(size)

    _, edge_norm, cell_id, best_dx, best_dy = common.periodic_lloyd_voronoi(
        uw, vw, grid_cells=grid_cells, jitter=0.72, rng=rng, lloyd_iters=2, metric="euclidean"
    )

    # Convert best_dx, best_dy from cell-units to UV-units
    dx_uv = best_dx / np.float32(grid_cells)
    dy_uv = best_dy / np.float32(grid_cells)
    r_uv = np.sqrt(dx_uv * dx_uv + dy_uv * dy_uv).astype(np.float32)

    # Estimate true UV perpendicular distance to Voronoi anchor frame borders
    # (edge_norm is exact IQ perpendicular distance normalized to [0, 1])
    max_inradius_uv = float(np.percentile(r_uv, 92)) * 0.72
    d_anchor = (edge_norm * np.float32(max_inradius_uv)).astype(np.float32)

    # Anchor frame strands along Voronoi borders (thicker structural frame tying all hubs)
    r_anchor = np.float32(0.00245) * (0.86 + 0.28 * var_noise).astype(np.float32)
    a_anchor, dome_anchor = _strand_profile(d_anchor, r_anchor, aa_uv)

    # Per-cell radial spoke count (9 to 13 spokes) and organic rotation phase
    k_spokes = (9.0 + np.floor(cell_id * 5.0)).astype(np.float32)
    phi0 = (cell_id * np.float32(2.0 * math.pi * 1.6180339)).astype(np.float32)
    theta = (np.arctan2(dy_uv, dx_uv) + phi0 + (var_noise - 0.5) * 0.14).astype(np.float32)

    sector = theta * k_spokes / np.float32(2.0 * math.pi)
    ang_err = _wrap_signed(sector) * (np.float32(2.0 * math.pi) / k_spokes)
    d_spoke = (r_uv * np.abs(ang_err)).astype(np.float32)

    r_spoke = np.float32(0.00185) * (0.92 + 0.18 * (1.0 - np.clip(r_uv * 4.5, 0.0, 1.0)) + (var_noise - 0.5) * 0.22)
    a_spoke, dome_spoke = _strand_profile(d_spoke, r_spoke, aa_uv)

    # Concentric sagging catenary spiral rings between adjacent spokes
    s_frac = (sector - np.floor(sector)).astype(np.float32)
    sag_curve = (4.0 * s_frac * (1.0 - s_frac)).astype(np.float32)

    # Slightly expanding spacing from hub outward (authentic orb-web radial progression)
    ring_freq = np.float32(24.0)
    ring_exp = np.float32(0.84)
    r_safe = np.maximum(r_uv, np.float32(0.004))
    r_scaled = np.power(r_safe, ring_exp) * ring_freq
    # Positive sag term shifts integer contours to smaller r_uv between spokes (inward sag!)
    ring_phase = r_scaled + np.float32(0.36) * sag_curve + (ring_wobble - 0.5) * np.float32(0.18)
    d_ring_phase = np.abs(_wrap_signed(ring_phase))
    dr_scale = ring_freq * ring_exp * np.power(r_safe, ring_exp - np.float32(1.0))
    d_ring = (d_ring_phase / np.maximum(dr_scale, np.float32(1.0))).astype(np.float32)

    # Mask concentric rings cleanly outside the central hub eye
    ring_region = _smoothstep(0.010, 0.016, r_uv)
    r_ring = np.float32(0.00145) * (0.85 + 0.30 * ring_wobble).astype(np.float32)
    a_ring_raw, dome_ring_raw = _strand_profile(d_ring, r_ring, aa_uv)
    a_ring = a_ring_raw * ring_region
    dome_ring = dome_ring_raw * ring_region

    # Central woven hub ring/knot at each web center
    hub_ring_d = np.abs(r_uv - np.float32(0.0095))
    a_hub, dome_hub = _strand_profile(hub_ring_d, np.float32(0.0019), aa_uv)
    a_core, dome_core = _strand_profile(r_uv, np.float32(0.0042), aa_uv)

    # Small silk tie-beads at spoke-ring intersections
    d_joint = np.sqrt(d_spoke * d_spoke + d_ring * d_ring).astype(np.float32)
    a_joint_raw, dome_joint_raw = _strand_profile(d_joint, np.float32(0.0023), aa_uv)
    a_joint = a_joint_raw * ring_region
    dome_joint = dome_joint_raw * ring_region

    # Combine strand hierarchy: anchor frame > spokes/hub > concentric rings
    h_anchor = dome_anchor * np.float32(0.92)
    h_spoke = dome_spoke * np.float32(0.80)
    h_ring = dome_ring * np.float32(0.68)
    h_hub = np.maximum(dome_hub * np.float32(0.84), dome_core * np.float32(0.88))
    h_joint = dome_joint * np.float32(0.88)

    # Subtle fibrous twist/texture along threads
    strand_tex = 0.90 + 0.10 * fine
    h_raw = np.maximum.reduce([h_anchor, h_spoke, h_ring, h_hub, h_joint]) * strand_tex
    alpha = np.maximum.reduce([a_anchor, a_spoke, a_ring, a_hub, a_core, a_joint])

    h_strand = np.clip(h_raw + alpha * 0.08 * ((fine - 0.5) * 0.60 + (fuzz - 0.5) * 0.40), 0.0, 1.0)
    h = (h_strand * alpha).astype(np.float32)

    tone = np.clip(
        0.52 * h_strand
        + 0.16 * a_anchor
        + 0.12 * a_joint
        + 0.12 * fine
        + 0.08 * (1.0 - fuzz),
        0.0,
        1.0,
    )
    albedo = _interp_stops(tone, stops)

    rough_raw = 0.42 * (1.0 - h_strand) + 0.26 * fuzz + 0.18 * (1.0 - fine) + 0.14 * a_anchor
    rough = _calibrate_strand_roughness(
        rough_raw,
        alpha,
        style.rough_range[0],
        style.rough_range[1],
        style.rough_mean,
        hole_neutral=style.rough_mean,
    )
    base = _shade_open_net_rgba(
        albedo,
        h,
        alpha,
        rough,
        normal_strength=style.normal_strength,
        metallic=False,
        sheen_amt=12.0,
        spec_intensity=0.11,
    )
    return base, h, rough


# ---------------------------------------------------------------------------
# Per-style dispatch table (all 10 approved styles)
# ---------------------------------------------------------------------------


def _build_style(
    style: NetStyleSpec,
    u: np.ndarray,
    v: np.ndarray,
    size: int,
    rng: np.random.Generator,
    stops: list[tuple[int, int, int]],
) -> tuple[Image.Image, np.ndarray, np.ndarray]:
    slug = style.slug

    if slug == "diamond-net-black":
        return _build_diamond_net(
            u,
            v,
            size,
            rng,
            stops,
            style,
            a=18,
            b=11,
            radius_uv=0.0038,
            warp_amp=0.0040,
            twists_per_span=5,
            twist_depth=0.18,
            knot_scale=1.42,
            knot_boost=0.18,
            sheen_amt=12.0,
            spec_intensity=0.14,
        )

    if slug == "diamond-net-wide-gauge":
        return _build_diamond_net(
            u,
            v,
            size,
            rng,
            stops,
            style,
            a=8,
            b=5,
            radius_uv=0.0082,
            warp_amp=0.0052,
            twists_per_span=7,
            twist_depth=0.24,
            knot_scale=1.46,
            knot_boost=0.20,
            sheen_amt=13.0,
            spec_intensity=0.15,
        )

    if slug == "diamond-net-fine-white":
        return _build_diamond_net(
            u,
            v,
            size,
            rng,
            stops,
            style,
            a=32,
            b=20,
            radius_uv=0.0022,
            warp_amp=0.0030,
            twists_per_span=4,
            twist_depth=0.14,
            knot_scale=1.34,
            knot_boost=0.14,
            sheen_amt=10.0,
            spec_intensity=0.11,
        )

    if slug == "diamond-net-fine-nude":
        return _build_diamond_net(
            u,
            v,
            size,
            rng,
            stops,
            style,
            a=30,
            b=19,
            radius_uv=0.0021,
            warp_amp=0.0032,
            twists_per_span=4,
            twist_depth=0.13,
            knot_scale=1.32,
            knot_boost=0.13,
            sheen_amt=11.0,
            spec_intensity=0.13,
        )

    if slug == "diamond-net-red":
        return _build_diamond_net(
            u,
            v,
            size,
            rng,
            stops,
            style,
            a=16,
            b=10,
            radius_uv=0.0042,
            warp_amp=0.0042,
            twists_per_span=5,
            twist_depth=0.19,
            knot_scale=1.44,
            knot_boost=0.18,
            sheen_amt=12.0,
            spec_intensity=0.15,
        )

    if slug == "hexagonal-net-black":
        return _build_hexagonal_net(
            u,
            v,
            size,
            rng,
            stops,
            style,
            nx=24,
            my=14,
            radius_uv=0.0034,
            warp_amp=0.0036,
            pillar_boost=1.14,
            vertex_scale=1.42,
            sheen_amt=12.0,
            spec_intensity=0.13,
        )

    if slug == "hexagonal-net-white":
        return _build_hexagonal_net(
            u,
            v,
            size,
            rng,
            stops,
            style,
            nx=29,
            my=17,
            radius_uv=0.0026,
            warp_amp=0.0032,
            pillar_boost=1.12,
            vertex_scale=1.38,
            sheen_amt=9.0,
            spec_intensity=0.10,
        )

    if slug == "grid-net-sport":
        return _build_grid_net_sport(u, v, size, rng, stops, style)

    if slug == "cobweb-net":
        return _build_cobweb_net(u, v, size, rng, stops, style)

    if slug == "metallic-net-silver":
        return _build_diamond_net(
            u,
            v,
            size,
            rng,
            stops,
            style,
            a=14,
            b=14,
            radius_uv=0.0044,
            warp_amp=0.0012,
            twists_per_span=4,
            twist_depth=0.12,
            knot_scale=1.10,
            knot_boost=0.0,
            wire_weave=True,
            sheen_amt=0.0,
            spec_intensity=0.52,
        )

    raise KeyError(f"Unknown net style slug: {slug!r}")


def _compute_bundle(
    seed: int = 1,
    palette: PaletteArg = None,
    size: int = 512,
    variant: int = 0,
) -> tuple[Image.Image, np.ndarray, np.ndarray, NetStyleSpec]:
    style = NET_STYLES[int(variant) % len(NET_STYLES)]
    rng = np.random.default_rng(int(seed))
    u, v = common.make_uv_grid(int(size))
    stops = _resolve_local_palette(palette, style.default_palette, rng)
    base, h, rough = _build_style(style, u, v, int(size), rng, stops)
    return base, np.asarray(h, dtype=np.float32), np.asarray(rough, dtype=np.float32), style


def render_net_bundle(
    style_or_variant: str | int,
    seed: int = 1,
    size: int = 2048,
    palette: PaletteArg = None,
) -> tuple[Image.Image, Image.Image, Image.Image, np.ndarray, np.ndarray, NetStyleSpec]:
    """Single-pass PBR bundle renderer returning (base_rgba, normal, roughness, height, rough_field, style)."""
    if isinstance(style_or_variant, str):
        style = STYLE_BY_SLUG[style_or_variant]
        variant = style.variant
    else:
        variant = int(style_or_variant) % len(NET_STYLES)
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
    name="net-pack",
    kind="texture",
    tileable=True,
    origin="tools/generators/types/net_pack.py",
    default_palette="diamond_net_black",
    description=(
        "Open net & mesh suite (10 tileable RGBA variants: diamond-net-black, "
        "diamond-net-wide-gauge, diamond-net-fine-white, diamond-net-fine-nude, "
        "diamond-net-red, hexagonal-net-black, hexagonal-net-white, grid-net-sport, "
        "cobweb-net, and metallic-net-silver) with true hole transparency, subtle "
        "strand normal relief, and calibrated PBR roughness."
    ),
    render_fn=generate,
    normal_source="native",
    normal_rule="Native per-style open net strand height field (h, 0 in apertures) converted with calibrated Scharr normal strength.",
    normal_strength=2.1,
    height_fn=generate_height,
    normal_fn=generate_normal,
    roughness_fn=generate_roughness,
    default_roughness=0.67,
)
