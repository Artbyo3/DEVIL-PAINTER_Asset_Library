"""Overlock serger edge stitch and zigzag reinforcement seam alpha."""

from __future__ import annotations

import math
import numpy as np
from PIL import Image

from .. import common
from ..interface import GeneratorSpec, PaletteArg


def generate(seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0) -> Image.Image:
    _ = (np.random.default_rng(seed), palette)
    u, v = common.make_uv_grid(size)
    x = (u - 0.5) * 2.0
    stitches = 12 + (variant % 6) * 4
    sv = (v * stitches) % 1.0

    zig_x = (np.abs(sv - 0.5) * 2.0 - 0.5) * 0.38
    diag_dist = np.abs(x - zig_x)
    zig_thread = np.exp(-(diag_dist * diag_dist) / 0.0018)
    border_rails = np.exp(-((np.abs(x) - 0.20) ** 2) / 0.0012) * np.sin(sv * math.pi)
    return common.alpha_image(np.maximum(zig_thread, border_rails * 0.85))


SPEC = GeneratorSpec(
    name="overlock-zigzag-seam",
    kind="alpha",
    tileable=False,
    tile_axes="v",
    origin="cloth_apparel_master_600_pack/procedural_texture_generator.py (Suite 4 style=2)",
    default_palette="charcoal_flannel",
    description="Overlock serger zigzag edge stitch and dual rail threads (tiles vertically).",
    render_fn=generate,
)
