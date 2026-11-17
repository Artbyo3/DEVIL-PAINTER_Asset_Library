"""Sculpting height alphas & hard-surface radial stamps (bolt, crater, rock chisel, weld, sigil)."""

from __future__ import annotations

import math
import numpy as np
from PIL import Image

from .. import common
from ..interface import GeneratorSpec, PaletteArg


def generate(seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0) -> Image.Image:
    rng = np.random.default_rng(seed)
    _ = palette
    u, v = common.make_uv_grid(size)
    x = (u - 0.5) * 2.0
    y = (v - 0.5) * 2.0
    r = np.sqrt(x * x + y * y)
    theta = np.arctan2(y, x)
    falloff = np.clip(1.0 - np.power(r / 0.92, 3.2), 0.0, 1.0)

    style = variant % 5
    micro = common.periodic_fbm(u, v, 6, 4, 0.5, rng)

    if style == 0:
        sides = [6, 8, 4, 12][(variant + int(rng.integers(0, 4))) % 4]
        poly_r = r * np.cos((theta + math.pi / sides) % (2.0 * math.pi / sides) - math.pi / sides)
        outer = np.clip((0.68 - poly_r) * 24.0, 0.0, 1.0)
        inner_ring = np.clip(1.0 - np.abs(r - 0.42) * 28.0, 0.0, 1.0)
        ang = float(rng.uniform(0.0, math.pi))
        slot_mask = (np.abs(x * math.cos(ang) + y * math.sin(ang)) < 0.07) & (r < 0.32)
        stamp = outer * (0.75 + 0.25 * inner_ring - 0.45 * slot_mask.astype(np.float32))
        return common.alpha_image(stamp * falloff)
    if style == 1:
        rim = np.exp(-((r - 0.48) ** 2) / 0.018) * (0.7 + 0.5 * micro)
        bowl = np.clip(1.0 - (r / 0.48) ** 2, 0.0, 1.0) * 0.55
        ejecta = np.power(0.5 + 0.5 * np.sin(theta * (7 + variant % 6) + micro * 3.0), 2.0) * np.exp(
            -((r - 0.62) ** 2) / 0.04
        )
        stamp = np.clip(rim - bowl + 0.35 * ejecta + 0.25 * micro * falloff, 0.0, 1.0)
        return common.alpha_image(stamp * falloff)
    if style == 2:
        _, edge, _, _, _ = common.periodic_voronoi(u, v, 4 + (variant % 4), 0.8, rng)
        swiss = common.periodic_fbm(u, v, 3, 5, 0.55, rng, mode="swiss")
        return common.alpha_image((0.60 * swiss + 0.40 * edge) * falloff)
    if style == 3:
        freq = 10 + (variant % 6) * 2
        bead_y = (v * freq) % 1.0 - 0.5
        bead_x = x + (micro - 0.5) * 0.08
        dist = np.sqrt((bead_x * 3.2) ** 2 + (bead_y - 0.35 * bead_x * bead_x * 8.0) ** 2)
        weld = np.clip(1.0 - dist * 1.4, 0.0, 1.0) * np.exp(-(x * x) / 0.08)
        return common.alpha_image(weld)

    rings = np.clip(1.0 - np.abs(r - 0.62) * 32.0, 0.0, 1.0) + np.clip(1.0 - np.abs(r - 0.48) * 36.0, 0.0, 1.0)
    spokes = np.clip(np.cos(theta * (4 + (variant % 5) * 2)) - 0.75, 0.0, 1.0) * 4.0 * (r < 0.62)
    core = np.clip(1.0 - np.abs(r - 0.22) * 28.0, 0.0, 1.0)
    return common.alpha_image(np.clip(rings + spokes + core, 0.0, 1.0) * falloff)


SPEC = GeneratorSpec(
    name="sculpt-stamp-alpha",
    kind="alpha",
    tileable=False,
    tile_axes="none",
    origin="substance_painter_master_600_pack/procedural_texture_generator.py (Suite 4)",
    default_palette="charcoal_flannel",
    description="Sculpting height alphas: hex bolt (v0), crater (v1), rock chisel (v2), weld bead (v3), arcane sigil (v4).",
    render_fn=generate,
)
