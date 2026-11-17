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

    # Classic houndstooth: balanced 2/2 twill + alternating K-dark / K-light color blocks (K=4 classic)
    repeats = 4 + (variant % 4) * 2
    threads = repeats * 8
    idx = np.arange(4)
    ii, jj = np.meshgrid(idx, idx)
    twill_2x2 = (((ii + jj) % 4) < 2).astype(np.int32)

    weave_h, warp_over, _, _ = common.woven_fabric_field(
        u, v, threads, threads, twill_2x2, float_arch=0.36, ply_twist=0.14
    )

    iu = np.floor(u * threads).astype(np.int32)
    iv = np.floor(v * threads).astype(np.int32)
    warp_dark = (iu % 8) < 4
    weft_dark = (iv % 8) < 4
    dark_yarn = np.where(warp_over, warp_dark, weft_dark)

    h = common.normalize_field(0.82 * weave_h + 0.18 * fuzz)
    c = common.normalize_field(np.where(dark_yarn, 0.12 + 0.22 * h, 0.78 + 0.20 * h))
    stops = common.resolve_palette(palette, SPEC.default_palette, rng)
    albedo = common.apply_palette_5stop(c, fuzz, stops, contrast=1.18)
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
