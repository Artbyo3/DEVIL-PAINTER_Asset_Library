"""Rectangular pin belt buckle hardware decal alpha stamp."""

from __future__ import annotations

import numpy as np
from PIL import Image

from .. import common
from ..interface import GeneratorSpec, PaletteArg


def _compute(
    seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0
) -> tuple[Image.Image, np.ndarray]:
    rng = np.random.default_rng(seed)
    _ = (palette, variant)
    u, v = common.make_uv_grid(size)
    x = (u - 0.5) * 2.0
    y = (v - 0.5) * 2.0
    w_half = float(rng.uniform(0.46, 0.72))
    h_half = float(rng.uniform(0.36, 0.60))
    thick = float(rng.uniform(0.085, 0.135))
    r_corner = 0.09
    qx = np.maximum(np.abs(x) - (w_half - r_corner), 0.0)
    qy = np.maximum(np.abs(y) - (h_half - r_corner), 0.0)
    box_sdf = np.sqrt(qx * qx + qy * qy) + np.minimum(np.maximum(np.abs(x) - (w_half - r_corner), np.abs(y) - (h_half - r_corner)), 0.0) - r_corner
    frame = np.sqrt(np.clip(1.0 - (np.abs(box_sdf) / thick) ** 2, 0.0, 1.0))

    # Tapered capsule SDF for buckle tongue/pin from left frame bar (-w_half) to right keeper (+0.18)
    ax, bx = -w_half, 0.18
    t_pin = np.clip((x - ax) / max(bx - ax, 1e-6), 0.0, 1.0)
    dx_pin = x - (ax + t_pin * (bx - ax))
    d_pin = np.sqrt(dx_pin * dx_pin + y * y)
    r_pin = 0.055 * (1.0 - 0.30 * t_pin)
    pin = np.sqrt(np.clip(1.0 - (d_pin / r_pin) ** 2, 0.0, 1.0)) * 0.92
    _h = np.maximum(frame, pin).astype(np.float32)
    return common.alpha_image(_h), _h


def generate(seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0) -> Image.Image:
    return _compute(seed=seed, palette=palette, size=size, variant=variant)[0]


def generate_height(seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0) -> np.ndarray:
    return _compute(seed=seed, palette=palette, size=size, variant=variant)[1]

SPEC = GeneratorSpec(
    name="belt-buckle",
    kind="alpha",
    tileable=False,
    tile_axes="none",
    origin="p1_work/scripts/custom_generators.py (gen_cloth_alpha_custom kind=4)",
    default_palette="charcoal_flannel",
    description="Rectangular pin belt buckle hardware decal alpha stamp.",
    render_fn=generate,
    normal_source="native",
    normal_rule="Native beveled metallic buckle frame and pin relief field.",
    normal_strength=6.0,
    height_fn=generate_height,
)
