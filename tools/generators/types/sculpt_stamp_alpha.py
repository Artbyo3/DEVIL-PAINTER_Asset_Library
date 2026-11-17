"""Sculpting height alphas & hard-surface radial stamps (bolt, crater, rock chisel, weld, sigil)."""

from __future__ import annotations

import math
import numpy as np
from PIL import Image

from .. import common
from ..interface import GeneratorSpec, PaletteArg


def generate_height(seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0) -> np.ndarray:
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
        outer = np.clip((0.68 - poly_r) * 20.0, 0.0, 1.0)
        bevel = np.clip(0.82 + 0.18 * np.sin(np.clip((0.68 - poly_r) / 0.14, 0.0, 1.0) * math.pi * 0.5), 0.0, 1.0)
        inner_ring = np.clip(1.0 - np.abs(r - 0.42) * 26.0, 0.0, 1.0)
        ang = float(rng.uniform(0.0, math.pi))
        # Capsule SDF for screwdriver drive slot
        u_rot = x * math.cos(ang) + y * math.sin(ang)
        v_rot = -x * math.sin(ang) + y * math.cos(ang)
        d_slot = np.hypot(u_rot, np.maximum(0.0, np.abs(v_rot) - 0.26))
        t_s = np.clip((0.085 - d_slot) / 0.035, 0.0, 1.0)
        slot_sdf = t_s * t_s * (3.0 - 2.0 * t_s)
        stamp = outer * bevel * (0.76 + 0.24 * inner_ring - 0.46 * slot_sdf)
        return (stamp * falloff).astype(np.float32)
    if style == 1:
        rim = np.exp(-((r - 0.48) ** 2) / 0.018) * (0.7 + 0.5 * micro)
        bowl = np.clip(1.0 - (r / 0.48) ** 2, 0.0, 1.0) * 0.42
        ejecta = np.power(0.5 + 0.5 * np.sin(theta * (7 + variant % 6) + micro * 3.0), 2.0) * np.exp(
            -((r - 0.62) ** 2) / 0.04
        )
        stamp = np.clip(0.32 + rim - bowl + 0.35 * ejecta + 0.22 * micro, 0.0, 1.0)
        return (stamp * falloff).astype(np.float32)
    if style == 2:
        _, edge, _, _, _ = common.periodic_lloyd_voronoi(u, v, 4 + (variant % 4), 0.85, rng, lloyd_iters=1)
        swiss = common.periodic_fbm(u, v, 3, 5, 0.55, rng, mode="swiss")
        return ((0.60 * swiss + 0.40 * edge) * falloff).astype(np.float32)
    if style == 3:
        # Overlapping crescent TIG weld dime stack along central axis with smooth falloff
        freq = 8 + (variant % 5) * 2
        row0 = np.floor(v * freq).astype(np.int32)
        weld = np.zeros_like(u, dtype=np.float32)
        for dr in (-1, 0, 1):
            by = (v * freq - (row0 + dr)) - 0.5
            bx = x + (micro - 0.5) * 0.06
            dist = np.hypot(bx * 3.1, by - 0.32 * bx * bx * 7.5)
            dime = np.clip(1.0 - dist * 1.15, 0.0, 1.0)
            weld = np.maximum(weld, dime)
        envelope = np.exp(-(x * x) / 0.075) * np.clip(1.0 - np.abs(y) ** 3.2 / (0.92 ** 3.2), 0.0, 1.0)
        return (weld * envelope).astype(np.float32)

    rings = np.clip(1.0 - np.abs(r - 0.62) * 32.0, 0.0, 1.0) + np.clip(1.0 - np.abs(r - 0.48) * 36.0, 0.0, 1.0)
    spoke_r_mask = np.clip((0.63 - r) / 0.04, 0.0, 1.0) * np.clip((r - 0.20) / 0.04, 0.0, 1.0)
    spokes = np.clip((np.cos(theta * (4 + (variant % 5) * 2)) - 0.72) * 3.8, 0.0, 1.0) * spoke_r_mask
    core = np.clip(1.0 - np.abs(r - 0.22) * 28.0, 0.0, 1.0)
    return (np.clip(rings + spokes + core, 0.0, 1.0) * falloff).astype(np.float32)


def generate(seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0) -> Image.Image:
    return common.alpha_image(generate_height(seed=seed, palette=palette, size=size, variant=variant))


SPEC = GeneratorSpec(
    name="sculpt-stamp-alpha",
    kind="alpha",
    tileable=False,
    tile_axes="none",
    origin="substance_painter_master_600_pack/procedural_texture_generator.py (Suite 4)",
    default_palette="charcoal_flannel",
    description="Sculpting height alphas: hex bolt (v0), crater (v1), rock chisel (v2), weld bead (v3), arcane sigil (v4).",
    render_fn=generate,
    normal_source="native",
    normal_rule="Native sculpting brush height stamp relief (hex bolt, crater, rock chisel, weld bead, arcane sigil).",
    normal_strength=6.5,
    height_fn=generate_height,
)
