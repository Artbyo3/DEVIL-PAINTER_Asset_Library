"""US 3-color desert (DCU) camouflage with soft tan and brown coffee-stain blobs over sand."""

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

    fx_base = 1 + (variant % 2)
    fy_base = 3 + ((variant // 2) % 2)

    wx = common.periodic_fbm(u, v, 2, 3, 0.52, rng, freqy=3)
    wy = common.periodic_fbm((u + 0.43) % 1.0, (v + 0.29) % 1.0, 2, 3, 0.52, rng, freqy=3)
    uu = (u + 0.22 * (wx - 0.5)) % 1.0
    vv = (v + 0.16 * (wy - 0.5)) % 1.0

    f_pale = common.periodic_fbm(uu, vv, fx_base, 3, 0.45, rng, freqy=fy_base)
    f_tan = common.periodic_fbm((uu + 0.41) % 1.0, (vv + 0.67) % 1.0, fx_base, 3, 0.45, rng, freqy=fy_base)
    f_brown = common.periodic_fbm((uu + 0.77) % 1.0, (vv + 0.13) % 1.0, fx_base + 1, 3, 0.45, rng, freqy=fy_base + 2)

    # US Drawing 2-1-2240 (DCU 3-Color Desert): ~16% Light Brown streaks, ~34% Khaki bands, ~50% Light Tan ground
    q_br = float(np.quantile(f_brown, 1.0 - 0.16))
    m_brown = _smoothstep(f_brown, q_br - 0.032, q_br + 0.032)
    free_t = f_brown < q_br
    q_tn = float(np.quantile(f_tan[free_t], 1.0 - (0.34 / 0.84)))
    m_tan = _smoothstep(f_tan, q_tn - 0.038, q_tn + 0.038)
    m_pale = _smoothstep(f_pale, 0.48, 0.58)

    stops = np.array(common.resolve_palette(palette, SPEC.default_palette, rng), dtype=np.float32)
    slub = common.periodic_fbm(u, v, 10, 2, 0.5, rng)

    albedo = np.broadcast_to(stops[2], (size, size, 3)).copy()
    albedo = albedo * (1.0 - m_pale[..., None]) + stops[3] * m_pale[..., None]
    albedo = albedo * (1.0 - m_tan[..., None]) + stops[1] * m_tan[..., None]
    albedo = albedo * (1.0 - m_brown[..., None]) + stops[0] * m_brown[..., None]
    albedo = albedo * (0.96 + 0.08 * slub[..., None]) + (stops[4] - stops[3]) * (0.08 * slub[..., None])

    threads = max(24, ((size // 6) // 4) * 4)
    twill_22 = np.array([[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1], [1, 0, 0, 1]], dtype=np.float32)
    weave, _, _, _ = common.woven_fabric_field(u, v, threads, threads, twill_22, float_arch=0.38, ply_twist=0.10)
    h = common.normalize_field(0.75 * weave + 0.25 * slub)
    return common.shade_cloth_surface(albedo, h, normal_strength=2.0, sheen_amt=5.0), np.asarray(h, dtype=np.float32)


def generate(seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0) -> Image.Image:
    return _compute(seed=seed, palette=palette, size=size, variant=variant)[0]


def generate_height(seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0) -> np.ndarray:
    return _compute(seed=seed, palette=palette, size=size, variant=variant)[1]

SPEC = GeneratorSpec(
    name="desert-camo",
    kind="texture",
    tileable=True,
    origin="tools/generators/types/desert_camo.py (DCU 3-Color Desert)",
    default_palette="dcu_3color_desert",
    description="3-color desert camouflage (DCU) with soft horizontally flowing tan and coffee-brown blobs over warm sand.",
    render_fn=generate,
    normal_source="native",
    normal_rule="Native physical surface height field (h) used by shade_cloth_surface.",
    normal_strength=2.0,
    height_fn=generate_height,
)
