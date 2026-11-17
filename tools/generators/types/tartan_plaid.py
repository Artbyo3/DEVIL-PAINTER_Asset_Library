"""Scottish tartan plaid woven sett texture generator."""

from __future__ import annotations

import math
import numpy as np
from PIL import Image

from .. import common
from ..interface import GeneratorSpec, PaletteArg


def _tartan_band(coord: np.ndarray) -> np.ndarray:
    d = np.abs(coord - 0.5) * 2.0
    return np.where(
        d < 0.06,
        0.92,
        np.where(d < 0.32, 0.18, np.where(d < 0.58, 0.52, np.where(d < 0.64, 0.88, 0.35))),
    )


def generate(seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0) -> Image.Image:
    rng = np.random.default_rng(seed)
    u, v = common.make_uv_grid(size)
    fuzz = common.periodic_fbm(u, v, 8, 3, 0.5, rng)

    threads = 32
    tu, tv = u * threads, v * threads
    iu, iv = np.floor(tu).astype(np.int32), np.floor(tv).astype(np.int32)
    fu, fv = tu - iu, tv - iv
    warp_over = ((iu + iv) // 2) % 2 == 0
    weave = np.where(warp_over, np.sin(fu * math.pi), np.sin(fv * math.pi) * 0.88)

    repeats = 2 + (variant % 3)
    warp_col = _tartan_band((u * repeats) % 1.0)
    weft_col = _tartan_band((v * repeats) % 1.0)
    chosen_col = np.where(warp_over, warp_col, weft_col)

    h = common.normalize_field(0.85 * weave + 0.15 * fuzz)
    c = common.normalize_field(0.78 * chosen_col + 0.22 * h)
    stops = common.resolve_palette(palette, SPEC.default_palette, rng)
    albedo = common.apply_palette_5stop(c, fuzz, stops, contrast=1.18)
    return common.shade_cloth_surface(albedo, h, normal_strength=5.4, sheen_amt=18.0)


SPEC = GeneratorSpec(
    name="tartan-plaid",
    kind="texture",
    tileable=True,
    origin="cloth_apparel_master_600_pack/procedural_texture_generator.py (arch=3)",
    default_palette="royal_stewart_tartan",
    description="Scottish tartan plaid woven sett with intersecting warp/weft bands and twill weave.",
    render_fn=generate,
)
