"""Herringbone chevron twill suiting weave texture generator."""

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

    cols = int(rng.choice([4, 6, 8, 10]))
    band_w = int(rng.choice([4, 8]))
    threads = cols * band_w

    # Construct authentic (4, 2*band_w) broken 2/2 twill matrix:
    # phase shifts by +2 picks at every reversal boundary so floats stay uniformly <= 2
    p = 2 * band_w
    matrix = np.zeros((4, p), dtype=np.int32)
    for col in range(p):
        band_idx = col // band_w
        direction = 1 if (band_idx % 2 == 0) else -1
        break_shift = 2 if (band_idx % 2 == 1) else 0
        for row in range(4):
            matrix[row, col] = 1 if ((col + direction * row + break_shift) % 4) < 2 else 0

    thread_h, warp_over, _, _ = common.woven_fabric_field(
        u, v, threads, threads, matrix, float_arch=0.38, ply_twist=0.14
    )
    slub = common.periodic_fbm(u, v, 8, 3, 0.5, rng)

    h = common.normalize_field(thread_h * 0.84 + slub * 0.16)
    col_f = common.normalize_field(np.where(warp_over, 0.72 + 0.25 * thread_h, 0.18 + 0.25 * thread_h) + 0.10 * (slub - 0.5))
    stops = common.resolve_palette(palette, SPEC.default_palette, rng)
    albedo = common.apply_palette_5stop(col_f, slub, stops, contrast=1.15)
    return common.shade_cloth_surface(albedo, h, normal_strength=5.6, sheen_amt=14.0), np.asarray(h, dtype=np.float32)


def generate(seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0) -> Image.Image:
    return _compute(seed=seed, palette=palette, size=size, variant=variant)[0]


def generate_height(seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0) -> np.ndarray:
    return _compute(seed=seed, palette=palette, size=size, variant=variant)[1]

SPEC = GeneratorSpec(
    name="herringbone-weave",
    kind="texture",
    tileable=True,
    origin="p1_work/scripts/custom_generators.py (gen_cloth_herringbone_chevron) + v2 tweed",
    default_palette="savile_row_tweed",
    description="Alternating herringbone chevron twill wool suiting texture.",
    render_fn=generate,
    normal_source="native",
    normal_rule="Native physical surface height field (h) used by shade_cloth_surface.",
    normal_strength=5.6,
    height_fn=generate_height,
)
