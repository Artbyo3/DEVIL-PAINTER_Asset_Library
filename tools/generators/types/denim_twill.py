"""3x1 selvedge denim twill weave texture generator."""

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
    # Ring-spun warp yarn slub varies along warp axis (u periodic freq 24, v freq 4)
    slub = common.periodic_fbm(u, v, 24, 3, 0.55, rng, freqy=4)

    threads = 24 + (variant % 5) * 4
    # 3/1 twill weave matrix: Z-twill (RHT, +1 shift) on even variants, S-twill (LHT, -1) on odd
    step = 1 if variant % 2 == 0 else -1
    idx = np.arange(4)
    ii, jj = np.meshgrid(idx, idx)
    twill_3x1 = (((ii + step * jj) % 4) < 3).astype(np.int32)

    warp_rad = np.clip(0.88 + 0.14 * (slub - 0.5), 0.78, 1.0)
    weave_h, warp_over, _, _ = common.woven_fabric_field(
        u, v, threads, threads, twill_3x1, float_arch=0.38, ply_twist=0.14, warp_radius=warp_rad, weft_radius=0.90
    )

    h = common.normalize_field(weave_h + 0.08 * fuzz)
    # Indigo-dyed warp floats vs undyed/ecru weft dots showing through on 1/4 picks
    c = common.normalize_field(np.where(warp_over, 0.18 + 0.36 * h + 0.16 * slub, 0.78 + 0.20 * h))
    stops = common.resolve_palette(palette, SPEC.default_palette, rng)
    micro = common.periodic_fbm(u, v, 12, 3, 0.5, rng)
    albedo = common.apply_palette_5stop(c, micro, stops, contrast=1.15)
    return common.shade_cloth_surface(albedo, h, normal_strength=5.6, sheen_amt=18.0), np.asarray(h, dtype=np.float32)


def generate(seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0) -> Image.Image:
    return _compute(seed=seed, palette=palette, size=size, variant=variant)[0]


def generate_height(seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0) -> np.ndarray:
    return _compute(seed=seed, palette=palette, size=size, variant=variant)[1]

SPEC = GeneratorSpec(
    name="denim-twill",
    kind="texture",
    tileable=True,
    origin="cloth_apparel_master_600_pack/procedural_texture_generator.py (arch=0)",
    default_palette="selvedge_indigo_denim",
    description="3x1 selvedge denim twill with diagonal wale, 2-ply yarn twist, and weft show-through.",
    render_fn=generate,
    normal_source="native",
    normal_rule="Native physical surface height field (h) used by shade_cloth_surface.",
    normal_strength=5.6,
    height_fn=generate_height,
)
