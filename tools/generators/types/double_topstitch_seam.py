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

    seam_step = 1.0 / (1.0 + np.exp(-x * 18.0)) * 0.34
    puckering = (0.5 + 0.5 * np.sin(v * stitches * 2.0 * math.pi)) * np.exp(-(x * x) / 0.04) * 0.14
    r_thread = 0.048
    dl = np.abs(x + 0.22) / r_thread
    dr = np.abs(x - 0.22) / r_thread
    # Elliptical cylindrical stitch dash arching above seam and diving at needle holes (sv=0, 1)
    dash_arch = np.power(np.clip(np.sin(sv * math.pi), 0.0, 1.0), 0.55)
    twist = 0.90 + 0.10 * np.sin((sv * 3.0 + x * 12.0) * 2.0 * math.pi)
    s_left = np.sqrt(np.clip(1.0 - dl * dl, 0.0, 1.0)) * dash_arch * twist
    s_right = np.sqrt(np.clip(1.0 - dr * dr, 0.0, 1.0)) * dash_arch * twist
    # Needle puncture dimples at stitch endpoints
    dv_hole = np.minimum(sv, 1.0 - sv) * stitches
    hole_l = np.exp(-(((x + 0.22) ** 2) / 0.0009 + ((np.minimum(sv, 1.0 - sv)) ** 2) / 0.012)) * 0.12
    hole_r = np.exp(-(((x - 0.22) ** 2) / 0.0009 + ((np.minimum(sv, 1.0 - sv)) ** 2) / 0.012)) * 0.12
    _ = dv_hole
    _h = np.clip(seam_step + puckering + 0.65 * np.maximum(s_left, s_right) - hole_l - hole_r, 0.0, 1.0).astype(np.float32)
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
