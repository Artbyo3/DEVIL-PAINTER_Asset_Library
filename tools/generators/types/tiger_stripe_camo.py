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

    # Horizontal brush-bristle striations via anisotropic Gabor noise (omega0=0)
    bristle = common.periodic_gabor_noise(u, vv, 14, 5.0, 0.0, 2.0, rng, impulses_per_cell=3)
    bristle_fine = common.periodic_gabor_noise((u + 0.31) % 1.0, vv, 18, 6.5, 0.0, 2.2, rng, impulses_per_cell=3)

    f_tan = common.periodic_fbm((u + 0.17) % 1.0, (vv + 0.29) % 1.0, 1, 3, 0.48, rng, freqy=fy_stripes - 2) + 0.16 * (bristle - 0.5)
    f_olive_dk = common.periodic_fbm((u + 0.53) % 1.0, (vv + 0.11) % 1.0, 1, 3, 0.48, rng, freqy=fy_stripes) + 0.20 * (bristle - 0.5)
    f_black = common.periodic_fbm(u, vv, 2, 3, 0.46, rng, freqy=fy_stripes + 2) + 0.24 * (bristle_fine - 0.5)

    # Vietnam Tiger Stripe interlocking stripe proportions: ~30% black, ~22% deep olive, ~10% tan trace, ~38% olive ground
    qk = float(np.quantile(f_black, 1.0 - 0.30))
    m_black = _smoothstep(f_black, qk - 0.018, qk + 0.018)
    free_o = f_black < qk
    qo = float(np.quantile(f_olive_dk[free_o], 1.0 - (0.22 / 0.70)))
    m_olive_dk = _smoothstep(f_olive_dk, qo - 0.018, qo + 0.018)
    free_t = free_o & (f_olive_dk < qo)
    qt = float(np.quantile(f_tan[free_t], 1.0 - (0.10 / 0.48)))
    m_tan = _smoothstep(f_tan, qt - 0.018, qt + 0.018)

    stops = np.array(common.resolve_palette(palette, SPEC.default_palette, rng), dtype=np.float32)
    slub = common.periodic_fbm(u, v, 10, 2, 0.5, rng)

    albedo = np.broadcast_to(stops[2], (size, size, 3)).copy()
    albedo = albedo * (1.0 - m_tan[..., None]) + stops[3] * m_tan[..., None]
    albedo = albedo * (1.0 - m_olive_dk[..., None]) + stops[1] * m_olive_dk[..., None]
    albedo = albedo * (1.0 - m_black[..., None]) + stops[0] * m_black[..., None]
    albedo = albedo * (0.95 + 0.09 * slub[..., None]) + (stops[4] - stops[3]) * (0.05 * slub[..., None])

    threads = max(24, ((size // 6) // 4) * 4)
    twill_22 = np.array([[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1], [1, 0, 0, 1]], dtype=np.float32)
    weave, _, _, _ = common.woven_fabric_field(u, v, threads, threads, twill_22, float_arch=0.38, ply_twist=0.10)
    h = common.normalize_field(0.75 * weave + 0.25 * slub)
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
