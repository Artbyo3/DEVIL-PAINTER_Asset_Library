"""German Bundeswehr Flecktarn camouflage with layered clustered spots and flecks."""

from __future__ import annotations

import math
import numpy as np
from PIL import Image

from .. import common
from ..interface import GeneratorSpec, PaletteArg


def _smoothstep(x: np.ndarray, edge0: float, edge1: float) -> np.ndarray:
    t = np.clip((x - edge0) / max(edge1 - edge0, 1e-6), 0.0, 1.0)
    return (t * t * (3.0 - 2.0 * t)).astype(np.float32)


def _spot_cluster_score(
    u: np.ndarray,
    v: np.ndarray,
    shift_u: float,
    shift_v: float,
    spot_cells: int,
    cluster_freq: int,
    rng: np.random.Generator,
) -> np.ndarray:
    uu = (u + shift_u) % 1.0
    vv = (v + shift_v) % 1.0
    f1, _, _, _, _ = common.periodic_lloyd_voronoi(uu, vv, spot_cells, 0.90, rng, lloyd_iters=1)
    wobble = common.periodic_fbm(uu, vv, spot_cells, 2, 0.5, rng)
    cluster = common.periodic_fbm((u + shift_v) % 1.0, (v + shift_u) % 1.0, cluster_freq, 2, 0.5, rng)
    return 0.65 * (1.0 - f1) + 0.35 * wobble + 0.32 * (cluster - 0.5)


def _compute(
    seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0
) -> tuple[Image.Image, np.ndarray]:
    rng = np.random.default_rng(seed)
    u, v = common.make_uv_grid(size)

    base_cells = [18, 22, 16, 24][variant % 4]
    s_sage = _spot_cluster_score(u, v, 0.13, 0.27, base_cells, 3, rng)
    s_moss = _spot_cluster_score(u, v, 0.41, 0.63, base_cells - 2, 3, rng)
    s_rust = _spot_cluster_score(u, v, 0.73, 0.19, base_cells + 2, 3, rng)
    s_black = _spot_cluster_score(u, v, 0.29, 0.83, base_cells + 4, 4, rng)

    # Bundeswehr TL 8305-0290 Flecktarn area ratios: 10% Black, 20% Red-Brown, 35% Dark Green, 15% Sage/Light Green, 20% Light Olive ground
    qk = float(np.quantile(s_black, 1.0 - 0.10))
    m_black = _smoothstep(s_black, qk - 0.02, qk + 0.02)
    free1 = s_black < qk
    qr = float(np.quantile(s_rust[free1], 1.0 - (0.20 / 0.90)))
    m_rust = _smoothstep(s_rust, qr - 0.02, qr + 0.02)
    free2 = free1 & (s_rust < qr)
    qm = float(np.quantile(s_moss[free2], 1.0 - (0.35 / 0.70)))
    m_moss = _smoothstep(s_moss, qm - 0.02, qm + 0.02)
    free3 = free2 & (s_moss < qm)
    qs = float(np.quantile(s_sage[free3], 1.0 - (0.15 / 0.35)))
    m_sage = _smoothstep(s_sage, qs - 0.02, qs + 0.02)

    stops = np.array(common.resolve_palette(palette, SPEC.default_palette, rng), dtype=np.float32)
    slub = common.periodic_fbm(u, v, 10, 2, 0.5, rng)

    albedo = np.broadcast_to(stops[3], (size, size, 3)).copy()
    albedo = albedo * (1.0 - m_sage[..., None]) + stops[4] * m_sage[..., None]
    albedo = albedo * (1.0 - m_moss[..., None]) + stops[2] * m_moss[..., None]
    albedo = albedo * (1.0 - m_rust[..., None]) + stops[1] * m_rust[..., None]
    albedo = albedo * (1.0 - m_black[..., None]) + stops[0] * m_black[..., None]
    albedo = albedo * (0.96 + 0.08 * slub[..., None])

    threads = max(24, ((size // 6) // 4) * 4)
    twill_22 = np.array([[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1], [1, 0, 0, 1]], dtype=np.float32)
    weave, _, _, _ = common.woven_fabric_field(u, v, threads, threads, twill_22, float_arch=0.38, ply_twist=0.10)
    h = common.normalize_field(0.75 * weave + 0.25 * slub)
    return common.shade_cloth_surface(albedo, h, normal_strength=2.1, sheen_amt=5.5), np.asarray(h, dtype=np.float32)


def generate(seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0) -> Image.Image:
    return _compute(seed=seed, palette=palette, size=size, variant=variant)[0]


def generate_height(seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0) -> np.ndarray:
    return _compute(seed=seed, palette=palette, size=size, variant=variant)[1]

SPEC = GeneratorSpec(
    name="flecktarn-camo",
    kind="texture",
    tileable=True,
    origin="tools/generators/types/flecktarn_camo.py (Bundeswehr Flecktarn)",
    default_palette="bundeswehr_flecktarn",
    description="German Flecktarn camouflage with dense layered clusters of small organic dots and spots over a muted green base.",
    render_fn=generate,
    normal_source="native",
    normal_rule="Native physical surface height field (h) used by shade_cloth_surface.",
    normal_strength=2.1,
    height_fn=generate_height,
)
