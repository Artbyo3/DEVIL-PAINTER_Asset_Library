"""MARPAT-style pixelated woodland camouflage composed of crisp small square pixels."""

from __future__ import annotations

import math
import numpy as np
from PIL import Image

from .. import common
from ..interface import GeneratorSpec, PaletteArg


def _compute(
    seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0
) -> tuple[Image.Image, np.ndarray]:
    rng = np.random.default_rng(seed)
    u, v = common.make_uv_grid(size)

    grid = [64, 80, 96, 128][variant % 4]
    iu = np.floor((u + 0.5 / float(grid)) * grid).astype(np.int32) % grid
    iv = np.floor((v + 0.5 / float(grid)) * grid).astype(np.int32) % grid
    ug = (iu.astype(np.float32) + 0.5) / float(grid)
    vg = (iv.astype(np.float32) + 0.5) / float(grid)

    wx = common.periodic_fbm(ug, vg, 2, 2, 0.5, rng, freqy=4)
    wy = common.periodic_fbm((ug + 0.37) % 1.0, (vg + 0.61) % 1.0, 2, 2, 0.5, rng, freqy=4)
    uu = (ug + 0.20 * (wx - 0.5)) % 1.0
    vv = (vg + 0.14 * (wy - 0.5)) % 1.0

    # US6805957B1 MARPAT dual-scale digital optels (2x2 block scatter + 1x1 fine fringe) at exact 47% / 30% / 18% / 5% coverage
    macro_green = common.periodic_fbm(uu, vv, 2, 3, 0.5, rng, freqy=4)
    macro_brown = common.periodic_fbm((uu + 0.43) % 1.0, (vv + 0.19) % 1.0, 3, 3, 0.5, rng, freqy=5)
    macro_black = common.periodic_fbm((uu + 0.71) % 1.0, (vv + 0.53) % 1.0, 3, 3, 0.5, rng, freqy=6)

    iu2, iv2 = (iu // 2) % (grid // 2), (iv // 2) % (grid // 2)
    b_g = rng.normal(0.0, 0.055, size=(grid // 2, grid // 2)).astype(np.float32)[iv2, iu2]
    b_b = rng.normal(0.0, 0.055, size=(grid // 2, grid // 2)).astype(np.float32)[iv2, iu2]
    b_k = rng.normal(0.0, 0.050, size=(grid // 2, grid // 2)).astype(np.float32)[iv2, iu2]
    f_g = macro_green + b_g + rng.normal(0.0, 0.045, size=(grid, grid)).astype(np.float32)[iv, iu]
    f_b = macro_brown + b_b + rng.normal(0.0, 0.045, size=(grid, grid)).astype(np.float32)[iv, iu]
    f_k = macro_black + b_k + rng.normal(0.0, 0.040, size=(grid, grid)).astype(np.float32)[iv, iu]

    # Hierarchical conditional quantile thresholds: 5% Black, 18% Coyote Brown, 47% Woodland Green, 30% Light Tan
    qk = float(np.quantile(f_k, 1.0 - 0.05))
    is_black = f_k >= qk
    free_b = ~is_black
    qb = float(np.quantile(f_b[free_b], 1.0 - (0.18 / 0.95)))
    is_brown = free_b & (f_b >= qb)
    free_g = free_b & (~is_brown)
    qg = float(np.quantile(f_g[free_g], 1.0 - (0.47 / 0.77)))
    is_green = free_g & (f_g >= qg)

    stops = np.array(common.resolve_palette(palette, SPEC.default_palette, rng), dtype=np.float32)
    slub = common.periodic_fbm(u, v, 12, 2, 0.5, rng)

    albedo = np.broadcast_to(stops[3], (size, size, 3)).copy()
    albedo[is_green] = stops[2]
    albedo[is_brown] = stops[1]
    albedo[is_black] = stops[0]
    albedo = albedo * (0.96 + 0.08 * slub[..., None]) + (stops[4] - stops[3]) * (0.05 * slub[..., None])

    # Authentic MARPAT 1/1 plain weave NYCO with 8-thread ripstop reinforcement ribs
    threads = max(24, ((size // 6) // 8) * 8)
    plain_11 = np.array([[1, 0], [0, 1]], dtype=np.float32)
    weave, _, _, _ = common.woven_fabric_field(u, v, threads, threads, plain_11, float_arch=0.35, ply_twist=0.08)
    rib_u = np.exp(-((((u * threads / 8.0) % 1.0) - 0.5) ** 2) / 0.008)
    rib_v = np.exp(-((((v * threads / 8.0) % 1.0) - 0.5) ** 2) / 0.008)
    rip_grid = np.maximum(rib_u, rib_v)
    h = common.normalize_field(0.68 * weave + 0.16 * rip_grid + 0.16 * slub)
    return common.shade_cloth_surface(albedo, h, normal_strength=1.9, sheen_amt=4.5), np.asarray(h, dtype=np.float32)


def generate(seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0) -> Image.Image:
    return _compute(seed=seed, palette=palette, size=size, variant=variant)[0]


def generate_height(seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0) -> np.ndarray:
    return _compute(seed=seed, palette=palette, size=size, variant=variant)[1]

SPEC = GeneratorSpec(
    name="digital-woodland-camo",
    kind="texture",
    tileable=True,
    origin="tools/generators/types/digital_woodland_camo.py (MARPAT Woodland)",
    default_palette="marpat_woodland_camo",
    description="MARPAT-style digital woodland camouflage built from quantized periodic square pixel clusters.",
    render_fn=generate,
    normal_source="native",
    normal_rule="Native physical surface height field (h) used by shade_cloth_surface.",
    normal_strength=1.9,
    height_fn=generate_height,
)
