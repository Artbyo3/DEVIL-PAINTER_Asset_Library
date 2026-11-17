"""Cross-stitch embroidery seam band alpha trim."""

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
    y = (v - 0.5) * 2.0
    cols = [8, 12, 16, 20][(variant + int(rng.integers(0, 4))) % 4]
    band_half = float(rng.uniform(0.32, 0.56))
    cu = (u * cols) % 1.0 - 0.5
    cv = (v * cols) % 1.0 - 0.5
    band = (np.abs(y) < band_half).astype(np.float32)
    d1 = np.abs(cu - cv)
    d2 = np.abs(cu + cv)
    box = (np.abs(cu) < 0.36) & (np.abs(cv) < 0.36)
    x_stitch = np.maximum(np.exp(-(d1 * d1) / 0.012), np.exp(-(d2 * d2) / 0.012)) * box
    _h = (x_stitch * band).astype(np.float32)
    return common.alpha_image(_h), _h


def generate(seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0) -> Image.Image:
    return _compute(seed=seed, palette=palette, size=size, variant=variant)[0]


def generate_height(seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0) -> np.ndarray:
    return _compute(seed=seed, palette=palette, size=size, variant=variant)[1]

SPEC = GeneratorSpec(
    name="cross-stitch-seam",
    kind="alpha",
    tileable=False,
    tile_axes="u",
    origin="p1_work/scripts/custom_generators.py (gen_cloth_alpha_custom kind=0)",
    default_palette="charcoal_flannel",
    description="Cross-stitch X-embroidery decorative seam band alpha (tiles horizontally).",
    render_fn=generate,
    normal_source="native",
    normal_rule="Native intersecting diagonal X-stitch thread relief along seam band.",
    normal_strength=6.0,
    height_fn=generate_height,
)
