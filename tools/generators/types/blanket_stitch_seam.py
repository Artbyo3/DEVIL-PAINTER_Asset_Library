"""Blanket whip-stitch edge seam alpha trim."""

from __future__ import annotations

import numpy as np
from PIL import Image

from .. import common
from ..interface import GeneratorSpec, PaletteArg


def generate(seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0) -> Image.Image:
    rng = np.random.default_rng(seed)
    _ = palette
    u, v = common.make_uv_grid(size)
    x = (u - 0.5) * 2.0
    stitches = [8, 12, 16, 22][(variant + int(rng.integers(0, 4))) % 4]
    sv = (v * stitches) % 1.0
    spine = np.exp(-((x + 0.15) ** 2) / 0.0025)
    cross_bar = ((x >= -0.15) & (x <= 0.25)).astype(np.float32) * np.exp(-((sv - 0.5) ** 2) / 0.015)
    seam_ridge = np.exp(-((x + 0.15) ** 2) / 0.025) * 0.35
    return common.alpha_image(np.maximum(spine, cross_bar) + seam_ridge)


SPEC = GeneratorSpec(
    name="blanket-stitch-seam",
    kind="alpha",
    tileable=False,
    tile_axes="v",
    origin="p1_work/scripts/custom_generators.py (gen_cloth_alpha_custom kind=5)",
    default_palette="charcoal_flannel",
    description="Blanket / whip-stitch edge seam trim alpha with vertical spine and perpendicular loops (tiles vertically).",
    render_fn=generate,
)
