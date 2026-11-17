"""Urban metro camouflage with woodland-style organic blobs in cool tactical greys and black."""

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
    wy = common.periodic_fbm((u + 0.41) % 1.0, (v + 0.59) % 1.0, 2, 3, 0.55, rng, freqy=3)
    uu = (u + 0.24 * (wx - 0.5)) % 1.0
    vv = (v + 0.20 * (wy - 0.5)) % 1.0

    f_mid = common.periodic_fbm(uu, vv, fx_base, 3, 0.48, rng, freqy=fy_base)
    f_slate = common.periodic_fbm((uu + 0.51) % 1.0, (vv + 0.17) % 1.0, fx_base, 3, 0.48, rng, freqy=fy_base + 1)
    f_black = common.periodic_fbm((uu + 0.27) % 1.0, (vv + 0.73) % 1.0, fx_base + 1, 3, 0.46, rng, freqy=fy_base + 2)

    # Balanced M81-style Urban Metro coverage fractions: 16% black, 34% slate, 30% mid-grey, 20% light ash
    qk = float(np.quantile(f_black, 1.0 - 0.16))
    m_black = _smoothstep(f_black, qk - 0.012, qk + 0.012)
    free_s = f_black < qk
    qs = float(np.quantile(f_slate[free_s], 1.0 - (0.34 / 0.84)))
    m_slate = _smoothstep(f_slate, qs - 0.012, qs + 0.012)
    free_m = free_s & (f_slate < qs)
    qm = float(np.quantile(f_mid[free_m], 1.0 - (0.30 / 0.50)))
    m_mid = _smoothstep(f_mid, qm - 0.012, qm + 0.012)

    stops = np.array(common.resolve_palette(palette, SPEC.default_palette, rng), dtype=np.float32)
    slub = common.periodic_fbm(u, v, 10, 2, 0.5, rng)

    albedo = np.broadcast_to(stops[3], (size, size, 3)).copy()
    albedo = albedo * (1.0 - m_mid[..., None]) + stops[2] * m_mid[..., None]
    albedo = albedo * (1.0 - m_slate[..., None]) + stops[1] * m_slate[..., None]
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
    name="urban-camo",
    kind="texture",
    tileable=True,
    origin="tools/generators/types/urban_camo.py (Urban Metro)",
    default_palette="urban_metro_camo",
    description="Urban camouflage featuring M81-style organic blob shapes in cool tactical greys and black over light ash.",
    render_fn=generate,
    normal_source="native",
    normal_rule="Native physical surface height field (h) used by shade_cloth_surface.",
    normal_strength=2.2,
    height_fn=generate_height,
)
