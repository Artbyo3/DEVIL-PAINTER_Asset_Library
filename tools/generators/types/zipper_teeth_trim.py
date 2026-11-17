"""Interlocking metallic zipper teeth and tape trim alpha generator."""

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

    # Woven herringbone/twill cloth tape with smooth lateral taper
    tape_mask = np.clip((0.56 - np.abs(x)) * 28.0, 0.0, 1.0)
    tape_twill = 0.85 + 0.15 * np.sin(((u * 64.0 + v * 64.0) % 2.0) * math.pi)
    tape = tape_mask * tape_twill * 0.22

    # Cylindrical braided cord beads framing the tooth track
    dcord = np.abs(np.abs(x) - 0.165) / 0.055
    cord = np.sqrt(np.clip(1.0 - dcord * dcord, 0.0, 1.0)) * (0.90 + 0.10 * np.sin(v * stitches * 4.0 * 3.14159265)) * 0.36

    # Exact rounded-box SDF for interlocking staggered metal teeth (no boolean cliff edges)
    tooth_freq = (stitches * 3) // 2
    tv_l = ((v * tooth_freq) % 1.0) - 0.25
    tv_r = (((v * tooth_freq) + 0.5) % 1.0) - 0.25
    # Wrap signed periodic vertical tooth offsets in [-0.5, 0.5]
    tv_l = ((v * tooth_freq) % 1.0) - 0.5
    tv_r = (((v * tooth_freq) + 0.5) % 1.0) - 0.5

    def _tooth_sdf(cx: float, dy: np.ndarray) -> np.ndarray:
        dx = np.abs(x - cx) - 0.075
        d_y = np.abs(dy) - 0.14
        qx = np.maximum(dx, 0.0)
        qy = np.maximum(d_y, 0.0)
        sdf = np.sqrt(qx * qx + qy * qy) + np.minimum(np.maximum(dx, d_y), 0.0) - 0.045
        # Chamfered top crown plateau
        return np.clip(-sdf * 18.0, 0.0, 1.0) * np.clip(0.55 + 0.45 * np.cos(np.clip(dy / 0.22, -1.0, 1.0) * 1.4), 0.0, 1.0)

    left_tooth = _tooth_sdf(-0.065, tv_l)
    right_tooth = _tooth_sdf(0.065, tv_r)
    _h = np.clip(tape + cord + 0.78 * np.maximum(left_tooth, right_tooth), 0.0, 1.0).astype(np.float32)
    return common.alpha_image(_h), _h


def generate(seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0) -> Image.Image:
    return _compute(seed=seed, palette=palette, size=size, variant=variant)[0]


def generate_height(seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0) -> np.ndarray:
    return _compute(seed=seed, palette=palette, size=size, variant=variant)[1]

SPEC = GeneratorSpec(
    name="zipper-teeth-trim",
    kind="alpha",
    tileable=False,
    tile_axes="v",
    origin="cloth_apparel_master_600_pack/procedural_texture_generator.py (Suite 4 style=1)",
    default_palette="charcoal_flannel",
    description="Interlocking metallic zipper teeth with cord bead and fabric tape trim (tiles vertically).",
    render_fn=generate,
    normal_source="native",
    normal_rule="Native interlocking metallic zipper teeth, cord bead, and tape relief.",
    normal_strength=6.0,
    height_fn=generate_height,
)
