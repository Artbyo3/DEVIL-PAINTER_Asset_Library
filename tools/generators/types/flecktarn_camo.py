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


def _spot_cluster_layer(
    u: np.ndarray,
    v: np.ndarray,
    shift_u: float,
    shift_v: float,
    spot_cells: int,
    cluster_freq: int,
    threshold: float,
    rng: np.random.Generator,
) -> np.ndarray:
    uu = (u + shift_u) % 1.0
    vv = (v + shift_v) % 1.0
    f1, _, _, _, _ = common.periodic_voronoi(uu, vv, spot_cells, 0.90, rng)
    wobble = common.periodic_fbm(uu, vv, spot_cells, 2, 0.5, rng)
    cluster = common.periodic_fbm((u + shift_v) % 1.0, (v + shift_u) % 1.0, cluster_freq, 2, 0.5, rng)
    spot_score = 0.65 * (1.0 - f1) + 0.35 * wobble + 0.32 * (cluster - 0.5)
    return _smoothstep(spot_score, threshold, threshold + 0.045)


def generate(seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0) -> Image.Image:
    rng = np.random.default_rng(seed)
    u, v = common.make_uv_grid(size)

    base_cells = [18, 22, 16, 24][variant % 4]
    m_sage = _spot_cluster_layer(u, v, 0.13, 0.27, base_cells, 3, 0.54, rng)
    m_moss = _spot_cluster_layer(u, v, 0.41, 0.63, base_cells - 2, 3, 0.52, rng)
    m_rust = _spot_cluster_layer(u, v, 0.73, 0.19, base_cells + 2, 3, 0.58, rng)
    m_black = _spot_cluster_layer(u, v, 0.29, 0.83, base_cells + 4, 4, 0.68, rng)

    stops = np.array(common.resolve_palette(palette, SPEC.default_palette, rng), dtype=np.float32)
    slub = common.periodic_fbm(u, v, 10, 2, 0.5, rng)

    albedo = np.broadcast_to(stops[3], (size, size, 3)).copy()
    albedo = albedo * (1.0 - m_sage[..., None]) + stops[4] * m_sage[..., None]
    albedo = albedo * (1.0 - m_moss[..., None]) + stops[2] * m_moss[..., None]
    albedo = albedo * (1.0 - m_rust[..., None]) + stops[1] * m_rust[..., None]
    albedo = albedo * (1.0 - m_black[..., None]) + stops[0] * m_black[..., None]
    albedo = albedo * (0.96 + 0.08 * slub[..., None])

    threads = max(24, min(96, size // 6))
    twill = 0.5 + 0.5 * np.sin(((u + v) * threads) * 2.0 * math.pi) * np.sin((u * threads * 2.0) * math.pi)
    h = common.normalize_field(0.75 * twill + 0.25 * slub)
    return common.shade_cloth_surface(albedo, h, normal_strength=2.1, sheen_amt=5.5)


SPEC = GeneratorSpec(
    name="flecktarn-camo",
    kind="texture",
    tileable=True,
    origin="tools/generators/types/flecktarn_camo.py (Bundeswehr Flecktarn)",
    default_palette="bundeswehr_flecktarn",
    description="German Flecktarn camouflage with dense layered clusters of small organic dots and spots over a muted green base.",
    render_fn=generate,
)
