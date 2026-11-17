"""Overlock serger edge stitch and zigzag reinforcement seam alpha."""

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

    # ISO 4915 Class 504 (3-thread, 1 needle rail) vs Class 514 (4-thread, 2 needle rails)
    amp = 0.22
    r_yarn = 0.038
    phase = sv * 2.0 * math.pi
    x_upper = amp * np.sin(phase)
    x_lower = -amp * np.sin(phase)

    du_up = np.abs(x - x_upper) / r_yarn
    du_lo = np.abs(x - x_lower) / r_yarn
    # Depth modulation so upper looper passes smoothly over lower looper at center crossings
    z_up = 0.80 + 0.20 * np.cos(phase)
    z_lo = 0.80 - 0.20 * np.cos(phase)
    upper_looper = np.sqrt(np.clip(1.0 - du_up * du_up, 0.0, 1.0)) * z_up
    lower_looper = np.sqrt(np.clip(1.0 - du_lo * du_lo, 0.0, 1.0)) * z_lo

    # Twisted edge purl along +amp border and needle-thread lock rails along left bite edge
    purl = np.sqrt(np.clip(1.0 - (np.abs(x - amp) / 0.044) ** 2, 0.0, 1.0)) * (0.72 + 0.28 * np.abs(np.cos(phase)))
    rail1 = np.sqrt(np.clip(1.0 - (np.abs(x + amp) / 0.034) ** 2, 0.0, 1.0)) * (0.75 + 0.25 * np.sin(sv * math.pi))
    if variant % 2 == 1:
        rail2 = np.sqrt(np.clip(1.0 - (np.abs(x + amp * 0.45) / 0.030) ** 2, 0.0, 1.0)) * (0.75 + 0.25 * np.sin(sv * math.pi))
        rails = np.maximum(rail1, rail2)
    else:
        rails = rail1

    _h = np.maximum.reduce([upper_looper, lower_looper, purl * 0.92, rails * 0.88]).astype(np.float32)
    return common.alpha_image(_h), _h


def generate(seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0) -> Image.Image:
    return _compute(seed=seed, palette=palette, size=size, variant=variant)[0]


def generate_height(seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0) -> np.ndarray:
    return _compute(seed=seed, palette=palette, size=size, variant=variant)[1]

SPEC = GeneratorSpec(
    name="overlock-zigzag-seam",
    kind="alpha",
    tileable=False,
    tile_axes="v",
    origin="cloth_apparel_master_600_pack/procedural_texture_generator.py (Suite 4 style=2)",
    default_palette="charcoal_flannel",
    description="Overlock serger zigzag edge stitch and dual rail threads (tiles vertically).",
    render_fn=generate,
    normal_source="native",
    normal_rule="Native zigzag overlock serger thread and dual border rail relief.",
    normal_strength=6.0,
    height_fn=generate_height,
)
