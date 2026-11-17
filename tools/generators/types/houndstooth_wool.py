"""Classic houndstooth (pied-de-poule) wool suiting texture generator."""

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
    fuzz = common.periodic_fbm(u, v, 8, 3, 0.5, rng)

    cells = 4 + (variant % 4) * 2
    gx = (u * cells * 2.0) % 2.0
    gy = (v * cells * 2.0) % 2.0
    in_x = gx < 1.0
    in_y = gy < 1.0
    diag = ((gx - np.floor(gx)) + (gy - np.floor(gy))) < 1.0
    dark_tooth = np.where(in_x & in_y, True, np.where((~in_x) & (~in_y), False, np.where(in_x & (~in_y), diag, ~diag)))

    threads = cells * 8
    tu, tv = u * threads, v * threads
    iu, iv = np.floor(tu).astype(np.int32), np.floor(tv).astype(np.int32)
    fu, fv = tu - iu, tv - iv
    warp_over = ((iu + iv) % 2) == 0
    weave = np.where(warp_over, np.sin(fu * math.pi), np.sin(fv * math.pi) * 0.88)

    h = common.normalize_field(0.82 * weave + 0.18 * fuzz)
    c = common.normalize_field(np.where(dark_tooth, 0.15 + 0.20 * h, 0.75 + 0.22 * h))
    stops = common.resolve_palette(palette, SPEC.default_palette, rng)
    albedo = common.apply_palette_5stop(c, fuzz, stops, contrast=1.16)
    return common.shade_cloth_surface(albedo, h, normal_strength=5.2, sheen_amt=15.0), np.asarray(h, dtype=np.float32)


def generate(seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0) -> Image.Image:
    return _compute(seed=seed, palette=palette, size=size, variant=variant)[0]


def generate_height(seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0) -> np.ndarray:
    return _compute(seed=seed, palette=palette, size=size, variant=variant)[1]

SPEC = GeneratorSpec(
    name="houndstooth-wool",
    kind="texture",
    tileable=True,
    origin="cloth_apparel_master_600_pack/procedural_texture_generator.py (arch=4)",
    default_palette="savile_row_houndstooth",
    description="Classic pied-de-poule houndstooth check with fine wool twill weave.",
    render_fn=generate,
    normal_source="native",
    normal_rule="Native physical surface height field (h) used by shade_cloth_surface.",
    normal_strength=5.2,
    height_fn=generate_height,
)
