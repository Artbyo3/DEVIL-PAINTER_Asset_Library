"""Scottish tartan plaid woven sett texture generator."""

from __future__ import annotations

import math
import numpy as np
from PIL import Image

from .. import common
from ..interface import GeneratorSpec, PaletteArg


_SETTS: tuple[list[tuple[int, float]], ...] = (
    # Classic symmetric half-sett ratios (reflected around pivot): (thread_count, palette_pos)
    [(12, 0.34), (4, 0.12), (2, 0.94), (4, 0.12), (8, 0.56), (2, 0.78), (6, 0.34)],
    [(10, 0.16), (3, 0.42), (6, 0.16), (3, 0.76), (10, 0.28), (2, 0.92), (4, 0.42)],
    [(14, 0.22), (2, 0.90), (4, 0.10), (6, 0.58), (2, 0.82), (4, 0.22), (8, 0.46)],
)


def _sett_lookup(sett: list[tuple[int, float]]) -> np.ndarray:
    half = []
    for cnt, val in sett:
        half.extend([val] * cnt)
    full = half + half[-2:0:-1]  # palindromic reflection around pivot threads
    return np.array(full, dtype=np.float32)


def _compute(
    seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0
) -> tuple[Image.Image, np.ndarray]:
    rng = np.random.default_rng(seed)
    u, v = common.make_uv_grid(size)
    fuzz = common.periodic_fbm(u, v, 8, 3, 0.5, rng)

    sett_arr = _sett_lookup(_SETTS[variant % len(_SETTS)])
    # Ensure total thread count across tile is an integer multiple of 4 (2/2 twill repeat)
    repeats = 1 + (variant % 2)
    n_sett = len(sett_arr)
    threads = ((n_sett * repeats + 3) // 4) * 4

    # Balanced 2/2 Z-twill weave matrix
    idx = np.arange(4)
    ii, jj = np.meshgrid(idx, idx)
    twill_2x2 = (((ii + jj) % 4) < 2).astype(np.int32)

    weave_h, warp_over, _, _ = common.woven_fabric_field(
        u, v, threads, threads, twill_2x2, float_arch=0.35, ply_twist=0.12
    )

    iu = (np.floor(u * threads).astype(np.int32)) % n_sett
    iv = (np.floor(v * threads).astype(np.int32)) % n_sett
    warp_col = sett_arr[iu]
    weft_col = sett_arr[iv]
    chosen_col = np.where(warp_over, warp_col, weft_col)

    h = common.normalize_field(0.85 * weave_h + 0.15 * fuzz)
    c = common.normalize_field(0.76 * chosen_col + 0.24 * h)
    stops = common.resolve_palette(palette, SPEC.default_palette, rng)
    albedo = common.apply_palette_5stop(c, fuzz, stops, contrast=1.18)
    return common.shade_cloth_surface(albedo, h, normal_strength=5.4, sheen_amt=18.0), np.asarray(h, dtype=np.float32)


def generate(seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0) -> Image.Image:
    return _compute(seed=seed, palette=palette, size=size, variant=variant)[0]


def generate_height(seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0) -> np.ndarray:
    return _compute(seed=seed, palette=palette, size=size, variant=variant)[1]

SPEC = GeneratorSpec(
    name="tartan-plaid",
    kind="texture",
    tileable=True,
    origin="cloth_apparel_master_600_pack/procedural_texture_generator.py (arch=3)",
    default_palette="royal_stewart_tartan",
    description="Scottish tartan plaid woven sett with intersecting warp/weft bands and twill weave.",
    render_fn=generate,
    normal_source="native",
    normal_rule="Native physical surface height field (h) used by shade_cloth_surface.",
    normal_strength=5.4,
    height_fn=generate_height,
)
