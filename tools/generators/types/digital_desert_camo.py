"""MARPAT-style pixelated desert camouflage composed of crisp small square pixels."""

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
    wy = common.periodic_fbm((ug + 0.31) % 1.0, (vg + 0.59) % 1.0, 2, 2, 0.5, rng, freqy=4)
    uu = (ug + 0.20 * (wx - 0.5)) % 1.0
    vv = (vg + 0.14 * (wy - 0.5)) % 1.0

    macro_pale = common.periodic_fbm((uu + 0.11) % 1.0, (vv + 0.23) % 1.0, 2, 3, 0.5, rng, freqy=4)
    macro_khaki = common.periodic_fbm(uu, vv, 2, 3, 0.5, rng, freqy=4)
    macro_coyote = common.periodic_fbm((uu + 0.47) % 1.0, (vv + 0.37) % 1.0, 3, 3, 0.5, rng, freqy=5)
    macro_brown = common.periodic_fbm((uu + 0.79) % 1.0, (vv + 0.67) % 1.0, 3, 3, 0.5, rng, freqy=6)

    jit_p = rng.normal(0.0, 0.080, size=(grid, grid)).astype(np.float32)[iv, iu]
    jit_k = rng.normal(0.0, 0.085, size=(grid, grid)).astype(np.float32)[iv, iu]
    jit_c = rng.normal(0.0, 0.085, size=(grid, grid)).astype(np.float32)[iv, iu]
    jit_b = rng.normal(0.0, 0.080, size=(grid, grid)).astype(np.float32)[iv, iu]

    is_pale = (macro_pale + jit_p) > 0.62
    is_khaki = (macro_khaki + jit_k) > 0.47
    is_coyote = (macro_coyote + jit_c) > 0.57
    is_brown = (macro_brown + jit_b) > 0.67

    stops = np.array(common.resolve_palette(palette, SPEC.default_palette, rng), dtype=np.float32)
    slub = common.periodic_fbm(u, v, 12, 2, 0.5, rng)

    albedo = np.broadcast_to(stops[3], (size, size, 3)).copy()
    albedo[is_pale] = stops[4]
    albedo[is_khaki] = stops[2]
    albedo[is_coyote] = stops[1]
    albedo[is_brown] = stops[0]
    albedo = albedo * (0.96 + 0.08 * slub[..., None])

    threads = max(24, min(96, size // 6))
    twill = 0.5 + 0.5 * np.sin(((u + v) * threads) * 2.0 * math.pi) * np.sin((u * threads * 2.0) * math.pi)
    h = common.normalize_field(0.76 * twill + 0.24 * slub)
    return common.shade_cloth_surface(albedo, h, normal_strength=1.9, sheen_amt=4.5), np.asarray(h, dtype=np.float32)


def generate(seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0) -> Image.Image:
    return _compute(seed=seed, palette=palette, size=size, variant=variant)[0]


def generate_height(seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0) -> np.ndarray:
    return _compute(seed=seed, palette=palette, size=size, variant=variant)[1]

SPEC = GeneratorSpec(
    name="digital-desert-camo",
    kind="texture",
    tileable=True,
    origin="tools/generators/types/digital_desert_camo.py (MARPAT Desert)",
    default_palette="marpat_desert_camo",
    description="MARPAT-style digital desert camouflage using quantized periodic square pixels in sand, khaki, and coyote brown.",
    render_fn=generate,
    normal_source="native",
    normal_rule="Native physical surface height field (h) used by shade_cloth_surface.",
    normal_strength=1.9,
    height_fn=generate_height,
)
