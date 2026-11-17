"""Interlocking metallic zipper teeth and tape trim alpha generator."""

from __future__ import annotations

import numpy as np
from PIL import Image

from .. import common
from ..interface import GeneratorSpec, PaletteArg


def generate(seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0) -> Image.Image:
    _ = (np.random.default_rng(seed), palette)
    u, v = common.make_uv_grid(size)
    x = (u - 0.5) * 2.0
    stitches = 12 + (variant % 6) * 4

    tape = (np.abs(x) < 0.55).astype(np.float32) * 0.22
    cord = np.exp(-((np.abs(x) - 0.16) ** 2) / 0.003) * 0.35
    tooth_v = (v * (stitches * 3 // 2)) % 1.0
    left_tooth = ((x > -0.15) & (x < 0.03)) * np.exp(-((tooth_v - 0.25) ** 2) / 0.018)
    right_tooth = ((x < 0.15) & (x > -0.03)) * np.exp(-((tooth_v - 0.75) ** 2) / 0.018)
    return common.alpha_image(tape + cord + 0.78 * np.maximum(left_tooth, right_tooth))


SPEC = GeneratorSpec(
    name="zipper-teeth-trim",
    kind="alpha",
    tileable=False,
    tile_axes="v",
    origin="cloth_apparel_master_600_pack/procedural_texture_generator.py (Suite 4 style=1)",
    default_palette="charcoal_flannel",
    description="Interlocking metallic zipper teeth with cord bead and fabric tape trim (tiles vertically).",
    render_fn=generate,
)
