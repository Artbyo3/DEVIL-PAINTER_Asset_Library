"""Classic M81 woodland camouflage with large organic green, brown, and black blobs."""

from __future__ import annotations

import math
import numpy as np
from PIL import Image

from .. import common
from ..interface import GeneratorSpec, PaletteArg


def _smoothstep(x: np.ndarray, edge0: float, edge1: float) -> np.ndarray:
    t = np.clip((x - edge0) / max(edge1 - edge0, 1e-6), 0.0, 1.0)
    return (t * t * (3.0 - 2.0 * t)).astype(np.float32)


def _compute(
    seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0
) -> tuple[Image.Image, np.ndarray]:
    rng = np.random.default_rng(seed)
    u, v = common.make_uv_grid(size)

    fx_base = 2 + (variant % 2)
    fy_base = 3 + ((variant // 2) % 2)

    wx = common.periodic_fbm(u, v, 2, 3, 0.55, rng, freqy=3)
    wy = common.periodic_fbm((u + 0.37) % 1.0, (v + 0.61) % 1.0, 2, 3, 0.55, rng, freqy=3)
    uu = (u + 0.24 * (wx - 0.5)) % 1.0
    vv = (v + 0.20 * (wy - 0.5)) % 1.0

    f_green = common.periodic_fbm(uu, vv, fx_base, 3, 0.48, rng, freqy=fy_base)
    f_brown = common.periodic_fbm((uu + 0.53) % 1.0, (vv + 0.19) % 1.0, fx_base, 3, 0.48, rng, freqy=fy_base + 1)
    f_black = common.periodic_fbm((uu + 0.29) % 1.0, (vv + 0.71) % 1.0, fx_base + 1, 3, 0.46, rng, freqy=fy_base + 2)

    # MIL-DTL-44436D M81 Woodland exact area fractions: 16% Black 357, 34% Brown 356, 30% Dark Green 355, 20% Light Green 354
    q_k = float(np.quantile(f_black, 1.0 - 0.16))
    m_black = _smoothstep(f_black, q_k - 0.012, q_k + 0.012)
    free_b = f_black < q_k
    q_b = float(np.quantile(f_brown[free_b], 1.0 - (0.34 / 0.84)))
    m_brown = _smoothstep(f_brown, q_b - 0.012, q_b + 0.012)
    free_g = free_b & (f_brown < q_b)
    q_g = float(np.quantile(f_green[free_g], 1.0 - (0.30 / 0.50)))
    m_green = _smoothstep(f_green, q_g - 0.012, q_g + 0.012)

    stops = np.array(common.resolve_palette(palette, SPEC.default_palette, rng), dtype=np.float32)
    slub = common.periodic_fbm(u, v, 10, 2, 0.5, rng)

    albedo = np.broadcast_to(stops[3], (size, size, 3)).copy()
    albedo = albedo * (1.0 - m_green[..., None]) + stops[2] * m_green[..., None]
    albedo = albedo * (1.0 - m_brown[..., None]) + stops[1] * m_brown[..., None]
    albedo = albedo * (1.0 - m_black[..., None]) + stops[0] * m_black[..., None]
    albedo = albedo * (0.95 + 0.09 * slub[..., None]) + (stops[4] - stops[3]) * (0.06 * slub[..., None])

    threads = max(24, ((size // 6) // 4) * 4)
    twill_22 = np.array([[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1], [1, 0, 0, 1]], dtype=np.float32)
    weave, _, _, _ = common.woven_fabric_field(u, v, threads, threads, twill_22, float_arch=0.38, ply_twist=0.10)
    h = common.normalize_field(0.76 * weave + 0.24 * slub)
    return common.shade_cloth_surface(albedo, h, normal_strength=2.2, sheen_amt=6.0), np.asarray(h, dtype=np.float32)


def generate(seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0) -> Image.Image:
    return _compute(seed=seed, palette=palette, size=size, variant=variant)[0]


def generate_height(seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0) -> np.ndarray:
    return _compute(seed=seed, palette=palette, size=size, variant=variant)[1]

SPEC = GeneratorSpec(
    name="woodland-camo",
    kind="texture",
    tileable=True,
    origin="tools/generators/types/woodland_camo.py (M81 Woodland)",
    default_palette="m81_woodland_camo",
    description="Classic M81 woodland camouflage with large interlocking green, brown, and black organic blobs over an olive-khaki base.",
    render_fn=generate,
    normal_source="native",
    normal_rule="Native physical surface height field (h) used by shade_cloth_surface.",
    normal_strength=2.2,
    height_fn=generate_height,
)
