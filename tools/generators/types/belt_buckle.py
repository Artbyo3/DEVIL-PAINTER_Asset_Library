"""Rectangular pin belt buckle hardware decal alpha stamp."""

from __future__ import annotations

import numpy as np
from PIL import Image

from .. import common
from ..interface import GeneratorSpec, PaletteArg


def generate(seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0) -> Image.Image:
    rng = np.random.default_rng(seed)
    _ = (palette, variant)
    u, v = common.make_uv_grid(size)
    x = (u - 0.5) * 2.0
    y = (v - 0.5) * 2.0
    w_half = float(rng.uniform(0.46, 0.72))
    h_half = float(rng.uniform(0.36, 0.60))
    thick = float(rng.uniform(0.09, 0.15))
    box_dist = np.maximum(np.abs(x) - w_half, np.abs(y) - h_half)
    frame = np.exp(-((np.abs(box_dist) / thick) ** 2))
    pin = ((np.abs(y) < 0.05) & (x > -w_half) & (x < 0.15)).astype(np.float32) * 0.9
    return common.alpha_image(np.maximum(frame, pin))


SPEC = GeneratorSpec(
    name="belt-buckle",
    kind="alpha",
    tileable=False,
    tile_axes="none",
    origin="p1_work/scripts/custom_generators.py (gen_cloth_alpha_custom kind=4)",
    default_palette="charcoal_flannel",
    description="Rectangular pin belt buckle hardware decal alpha stamp.",
    render_fn=generate,
)
