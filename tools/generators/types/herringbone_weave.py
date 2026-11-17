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

    cols = int(rng.choice([4, 6, 8, 10, 12]))
    threads = int(cols * rng.choice([4, 8]))
    band = np.floor(u * cols).astype(np.int32)
    direction = np.where(band % 2 == 0, 1.0, -1.0)
    tu, tv = u * threads, v * threads
    iu, iv = np.floor(tu).astype(np.int32), np.floor(tv).astype(np.int32)
    fu, fv = tu - iu, tv - iv
    twill_phase = (iu + direction * iv).astype(np.int32) % 4
    warp_over = twill_phase < 2
    thread_h = np.where(warp_over, np.sin(fu * math.pi), np.sin(fv * math.pi) * 0.9)
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
