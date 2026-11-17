"""Vietnam-era tiger-stripe camouflage with long horizontal frayed brush strokes."""

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

    fy_stripes = 7 + (variant % 4) * 2
    wy = common.periodic_fbm(u, v, 2, 3, 0.5, rng, freqy=4)
    vv = (v + 0.065 * (wy - 0.5)) % 1.0

    # Horizontal brush-bristle striations
    bristle = common.periodic_fbm(u, vv, 3, 2, 0.5, rng, freqy=44)
    bristle_fine = common.periodic_fbm((u + 0.31) % 1.0, vv, 5, 2, 0.5, rng, freqy=64)

    f_tan = common.periodic_fbm((u + 0.17) % 1.0, (vv + 0.29) % 1.0, 1, 3, 0.48, rng, freqy=fy_stripes - 2)
    f_olive_dk = common.periodic_fbm((u + 0.53) % 1.0, (vv + 0.11) % 1.0, 1, 3, 0.48, rng, freqy=fy_stripes)
    f_black = common.periodic_fbm(u, vv, 2, 3, 0.46, rng, freqy=fy_stripes + 2)

    m_tan = _smoothstep(f_tan + 0.16 * (bristle - 0.5), 0.56, 0.60)
    m_olive_dk = _smoothstep(f_olive_dk + 0.20 * (bristle - 0.5), 0.48, 0.52)
    m_black = _smoothstep(f_black + 0.24 * (bristle_fine - 0.5), 0.55, 0.585)

    stops = np.array(common.resolve_palette(palette, SPEC.default_palette, rng), dtype=np.float32)
    slub = common.periodic_fbm(u, v, 10, 2, 0.5, rng)

    albedo = np.broadcast_to(stops[2], (size, size, 3)).copy()
    albedo = albedo * (1.0 - m_tan[..., None]) + stops[3] * m_tan[..., None]
    albedo = albedo * (1.0 - m_olive_dk[..., None]) + stops[1] * m_olive_dk[..., None]
    albedo = albedo * (1.0 - m_black[..., None]) + stops[0] * m_black[..., None]
    albedo = albedo * (0.95 + 0.09 * slub[..., None]) + (stops[4] - stops[3]) * (0.05 * slub[..., None])

    threads = max(24, min(96, size // 6))
    twill = 0.5 + 0.5 * np.sin(((u + v) * threads) * 2.0 * math.pi) * np.sin((u * threads * 2.0) * math.pi)
    h = common.normalize_field(0.74 * twill + 0.26 * slub)
    return common.shade_cloth_surface(albedo, h, normal_strength=2.2, sheen_amt=5.5), np.asarray(h, dtype=np.float32)


def generate(seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0) -> Image.Image:
    return _compute(seed=seed, palette=palette, size=size, variant=variant)[0]


def generate_height(seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0) -> np.ndarray:
    return _compute(seed=seed, palette=palette, size=size, variant=variant)[1]

SPEC = GeneratorSpec(
    name="tiger-stripe-camo",
    kind="texture",
    tileable=True,
    origin="tools/generators/types/tiger_stripe_camo.py (Vietnam Tiger Stripe)",
    default_palette="vietnam_tiger_stripe",
    description="Vietnam-era tiger-stripe camouflage with long horizontal frayed brush-stroke bands in black and dark olive over olive-brown.",
    render_fn=generate,
    normal_source="native",
    normal_rule="Native physical surface height field (h) used by shade_cloth_surface.",
    normal_strength=2.2,
    height_fn=generate_height,
)
