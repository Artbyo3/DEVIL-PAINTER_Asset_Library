"""Double-needle denim topstitch seam alpha trim."""

from __future__ import annotations

import math
import numpy as np
from PIL import Image

from .. import common
from ..interface import GeneratorSpec, PaletteArg


def _compute(
    seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0
) -> tuple[Image.Image, np.ndarray]:
    _ = (np.random.default_rng(seed), palette)
    u, v = common.make_uv_grid(size)
    x = (u - 0.5) * 2.0
    stitches = 12 + (variant % 6) * 4
    sv = (v * stitches) % 1.0

    seam_step = 1.0 / (1.0 + np.exp(-x * 18.0)) * 0.35
    puckering = (0.5 + 0.5 * np.sin(v * stitches * 2.0 * math.pi)) * np.exp(-(x * x) / 0.04) * 0.15
    s_left = np.exp(-((x + 0.22) ** 2) / 0.0025) * np.clip(np.sin(sv * math.pi) * 1.3, 0.0, 1.0)
    s_right = np.exp(-((x - 0.22) ** 2) / 0.0025) * np.clip(np.sin(sv * math.pi) * 1.3, 0.0, 1.0)
    _h = (seam_step + puckering + 0.65 * np.maximum(s_left, s_right)).astype(np.float32)
    return common.alpha_image(_h), _h


def generate(seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0) -> Image.Image:
    return _compute(seed=seed, palette=palette, size=size, variant=variant)[0]


def generate_height(seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0) -> np.ndarray:
    return _compute(seed=seed, palette=palette, size=size, variant=variant)[1]

SPEC = GeneratorSpec(
    name="double-topstitch-seam",
    kind="alpha",
    tileable=False,
    tile_axes="v",
    origin="cloth_apparel_master_600_pack/procedural_texture_generator.py (Suite 4 style=0)",
    default_palette="charcoal_flannel",
    description="Double-needle denim topstitch seam with rolled fabric step and stitch puckering (tiles vertically).",
    render_fn=generate,
    normal_source="native",
    normal_rule="Native folded seam step, fabric puckering, and twin stitch thread relief.",
    normal_strength=6.0,
    height_fn=generate_height,
)
