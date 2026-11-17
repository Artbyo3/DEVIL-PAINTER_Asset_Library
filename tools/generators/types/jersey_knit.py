"""Interlocking V-loop jersey knitwear texture generator."""

from __future__ import annotations

import math
import numpy as np
from PIL import Image

from .. import common
from ..interface import GeneratorSpec, PaletteArg


def _compute(
    seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0
) -> tuple[Image.Image, np.ndarray]:
    rng = np.random.default_rng(seed)
    u, v = common.make_uv_grid(size)
    fuzz = common.periodic_fbm(u, v, 8, 3, 0.5, rng)
    slub = common.periodic_fbm(u, v, 4, 3, 0.5, rng, freqy=32)

    cols = 14 + (variant % 5) * 4
    # Authentic single-jersey course:wale density ratio ~ 1.35:1 (rounded even for clean stripe period)
    rows = max(12, ((int(round(cols * 1.35)) + 1) // 2) * 2)
    gx, gy = u * cols, v * rows
    fx = gx - np.floor(gx)
    iy_base = np.floor(gy).astype(np.int32)

    # Evaluate Chamberlain/Leaf-Glaskin V-loop legs across neighboring courses dy in (-1, 0, 1)
    best_h = np.zeros_like(u, dtype=np.float32)
    r_yarn = 0.205
    for dy in (-1, 0, 1):
        fy = gy - (iy_base + dy)
        # Active loop body spans fy in [-0.28, 1.18]
        t = np.clip((fy + 0.20) / 1.32, 0.0, 1.0)
        # Sinuous V-leg centerline converges inward toward top loop head, flaring out at sinker bottom
        x_leg_l = 0.18 + 0.145 * np.sin(t * math.pi * 0.85) + 0.04 * (1.0 - np.cos(t * math.pi))
        x_leg_r = 1.0 - x_leg_l
        # Depth z(t): legs emerge forward over lower course (z peaks around t=0.55) and dive under upper sinker
        z_leg = 0.52 + 0.48 * np.sin(t * math.pi)
        in_span = ((fy >= -0.20) & (fy <= 1.12)).astype(np.float32)

        dl = np.abs(fx - x_leg_l) / r_yarn
        dr = np.abs(fx - x_leg_r) / r_yarn
        pl = np.sqrt(np.clip(1.0 - dl * dl, 0.0, 1.0)) * in_span
        pr = np.sqrt(np.clip(1.0 - dr * dr, 0.0, 1.0)) * in_span

        # 2-ply twist along loop tangent
        twist_l = 0.86 + 0.14 * np.sin((t * 5.0 + (fx - x_leg_l) * 4.0) * 2.0 * math.pi)
        twist_r = 0.86 + 0.14 * np.sin((t * 5.0 - (fx - x_leg_r) * 4.0) * 2.0 * math.pi)

        # Horizontal needle crown arch connecting top of legs at fy ~ 1.05 (recessed behind next legs)
        crown_y = 1.02 - 0.08 * np.cos((fx - 0.5) * math.pi)
        dc = np.abs(fy - crown_y) / (r_yarn * 1.15)
        in_crown = ((fx >= 0.24) & (fx <= 0.76)).astype(np.float32)
        pc = np.sqrt(np.clip(1.0 - dc * dc, 0.0, 1.0)) * in_crown * 0.45

        cand = np.maximum(np.maximum(pl * twist_l, pr * twist_r) * z_leg, pc)
        best_h = np.maximum(best_h, cand)

    h = common.normalize_field(best_h + 0.12 * fuzz)
    row_stripe = ((iy_base // 2) % 2) * 0.22
    c = common.normalize_field(0.68 * h + row_stripe + 0.10 * slub)
    stops = common.resolve_palette(palette, SPEC.default_palette, rng)
    albedo = common.apply_palette_5stop(c, fuzz, stops, contrast=1.14)
    return common.shade_cloth_surface(albedo, h, normal_strength=5.8, sheen_amt=16.0), np.asarray(h, dtype=np.float32)


def generate(seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0) -> Image.Image:
    return _compute(seed=seed, palette=palette, size=size, variant=variant)[0]


def generate_height(seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0) -> np.ndarray:
    return _compute(seed=seed, palette=palette, size=size, variant=variant)[1]

SPEC = GeneratorSpec(
    name="jersey-knit",
    kind="texture",
    tileable=True,
    origin="cloth_apparel_master_600_pack/procedural_texture_generator.py (arch=1)",
    default_palette="heather_grey_jersey",
    description="Interlocking V-loop jersey knitwear for t-shirts, hoodies, and sweaters.",
    render_fn=generate,
    normal_source="native",
    normal_rule="Native physical surface height field (h) used by shade_cloth_surface.",
    normal_strength=5.8,
    height_fn=generate_height,
)
