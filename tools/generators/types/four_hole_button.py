"""Four-hole shirt button and sewn thread cross decal alpha stamp."""

from __future__ import annotations

import numpy as np
from PIL import Image

from .. import common
from ..interface import GeneratorSpec, PaletteArg


def _compute(
    seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0
) -> tuple[Image.Image, np.ndarray]:
    rng = np.random.default_rng(seed)
    _ = palette
    u, v = common.make_uv_grid(size)
    x = (u - 0.5) * 2.0
    y = (v - 0.5) * 2.0
    r = np.sqrt(x * x + y * y)
    btn_r = float(rng.uniform(0.58, 0.72)) if variant > 0 else 0.68

    disc = np.clip((btn_r - r) * 28.0, 0.0, 1.0)
    rim = np.exp(-((r - (btn_r - 0.10)) ** 2) / 0.004) * 0.45
    recess = np.clip((btn_r - 0.20 - r) * 16.0, 0.0, 1.0) * 0.25
    h_dist = np.sqrt((np.abs(x) - 0.18) ** 2 + (np.abs(y) - 0.18) ** 2)
    holes = np.clip((0.085 - h_dist) * 40.0, 0.0, 1.0)
    threads_cross = ((np.abs(x) < 0.05) & (np.abs(y) < 0.22)) | ((np.abs(y) < 0.05) & (np.abs(x) < 0.22))
    btn = disc * (0.65 + rim - recess - 0.60 * holes) + 0.35 * threads_cross.astype(np.float32)
    _h = (np.clip(btn, 0.0, 1.0)).astype(np.float32)
    return common.alpha_image(_h), _h


def generate(seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0) -> Image.Image:
    return _compute(seed=seed, palette=palette, size=size, variant=variant)[0]


def generate_height(seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0) -> np.ndarray:
    return _compute(seed=seed, palette=palette, size=size, variant=variant)[1]

SPEC = GeneratorSpec(
    name="four-hole-button",
    kind="alpha",
    tileable=False,
    tile_axes="none",
    origin="cloth_apparel_master_600_pack/procedural_texture_generator.py (Suite 4 style=3)",
    default_palette="charcoal_flannel",
    description="Radial 4-hole shirt button decal alpha stamp with beveled rim and crossed sewing threads.",
    render_fn=generate,
    normal_source="native",
    normal_rule="Native beveled button dish, raised rim, thread holes, and center X-stitch relief.",
    normal_strength=6.0,
    height_fn=generate_height,
)
