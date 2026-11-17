"""6-color desert (DBDU 'chocolate chip') camouflage with tan strata and dark chips with light outlines."""

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

    wx = common.periodic_fbm(u, v, 2, 3, 0.5, rng, freqy=4)
    wy = common.periodic_fbm((u + 0.31) % 1.0, (v + 0.73) % 1.0, 2, 3, 0.5, rng, freqy=4)
    uu = (u + 0.20 * (wx - 0.5)) % 1.0
    vv = (v + 0.14 * (wy - 0.5)) % 1.0

    # 4 desert strata tones via conditional quantiles (~14% earth brown, ~26% khaki, ~30% light tan, ~30% sand)
    f_pale = common.periodic_fbm(uu, vv, 2, 3, 0.48, rng, freqy=4)
    f_med = common.periodic_fbm((uu + 0.47) % 1.0, (vv + 0.23) % 1.0, 2, 3, 0.48, rng, freqy=5)
    f_brown = common.periodic_fbm((uu + 0.19) % 1.0, (vv + 0.61) % 1.0, 3, 3, 0.48, rng, freqy=5)

    q_br = float(np.quantile(f_brown, 1.0 - 0.14))
    m_brown = _smoothstep(f_brown, q_br - 0.025, q_br + 0.025)
    free_m = f_brown < q_br
    q_m = float(np.quantile(f_med[free_m], 1.0 - 0.30))
    m_med = _smoothstep(f_med, q_m - 0.025, q_m + 0.025)
    q_p = float(np.quantile(f_pale[free_m & (f_med < q_m)], 1.0 - 0.50))
    m_pale = _smoothstep(f_pale, q_p - 0.025, q_p + 0.025)

    # Dark irregular chocolate chips using Lloyd-relaxed Voronoi pebbles + paired shadow offset
    cells = [14, 16, 18, 12][variant % 4]
    f1, _, cid, _, _ = common.periodic_lloyd_voronoi(u, v, cells, 0.90, rng, lloyd_iters=1)
    chip_warp = common.periodic_fbm(u, v, cells, 2, 0.5, rng)
    chip_active = (cid > 0.42).astype(np.float32)
    chip_shape = (1.0 - f1) * chip_active + 0.22 * (chip_warp - 0.5)
    chip_mask = _smoothstep(chip_shape, 0.72, 0.76)

    # Periodic offset crescent + soft rim for light outline around each dark chip
    off_u = max(1, round(size * 0.007))
    off_v = max(1, round(size * 0.007))
    shifted = np.roll(chip_mask, shift=(-off_v, -off_u), axis=(0, 1))
    rim = np.maximum.reduce(
        [
            shifted,
            np.roll(chip_mask, shift=(-off_v, 0), axis=(0, 1)),
            np.roll(chip_mask, shift=(0, -off_u), axis=(0, 1)),
            np.roll(chip_mask, shift=(1, 1), axis=(0, 1)),
        ]
    )
    outline_mask = np.clip(rim - chip_mask * 0.85, 0.0, 1.0)

    stops = np.array(common.resolve_palette(palette, SPEC.default_palette, rng), dtype=np.float32)
    pale_tan = 0.52 * stops[3] + 0.48 * stops[2]
    slub = common.periodic_fbm(u, v, 10, 2, 0.5, rng)

    albedo = np.broadcast_to(stops[3], (size, size, 3)).copy()
    albedo = albedo * (1.0 - m_pale[..., None]) + pale_tan * m_pale[..., None]
    albedo = albedo * (1.0 - m_med[..., None]) + stops[2] * m_med[..., None]
    albedo = albedo * (1.0 - m_brown[..., None]) + stops[1] * m_brown[..., None]
    albedo = albedo * (1.0 - outline_mask[..., None]) + stops[4] * outline_mask[..., None]
    albedo = albedo * (1.0 - chip_mask[..., None]) + stops[0] * chip_mask[..., None]
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
    name="chocolate-chip-camo",
    kind="texture",
    tileable=True,
    origin="tools/generators/types/chocolate_chip_camo.py (DBDU 6-Color Desert)",
    default_palette="dbdu_chocolate_chip",
    description="6-color desert (DBDU chocolate-chip) camouflage with wavy tan strata and dark chips with light cream outlines.",
    render_fn=generate,
    normal_source="native",
    normal_rule="Native physical surface height field (h) used by shade_cloth_surface.",
    normal_strength=2.1,
    height_fn=generate_height,
)
