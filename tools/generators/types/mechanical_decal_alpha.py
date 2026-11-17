"""Mechanical decals, claw scratches, rosettes, dents, and reticle sculpting alpha stamps."""

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
    kind = variant % 8
    x = (u - 0.5) * 2.0
    y = (v - 0.5) * 2.0

    rot = float(rng.uniform(-math.pi, math.pi))
    sx = float(rng.uniform(0.72, 1.15))
    sy = float(rng.uniform(0.72, 1.15))
    xr = (x * math.cos(rot) - y * math.sin(rot)) / sx
    yr = (x * math.sin(rot) + y * math.cos(rot)) / sy
    r = np.sqrt(xr * xr + yr * yr)
    theta = np.arctan2(yr, xr)
    falloff = np.clip(1.0 - np.power(np.sqrt(x * x + y * y) / 0.92, 3.2), 0.0, 1.0)
    micro = common.periodic_fbm(u, v, [4, 6, 9][int(rng.integers(0, 3))], 4, 0.5, rng)

    if kind == 0:
        teeth = [6, 8, 10, 12, 16, 22, 28][int(rng.integers(0, 7))]
        tooth_profile = np.clip(np.cos(theta * teeth) * 3.0, -0.5, 0.5) * float(rng.uniform(0.06, 0.13))
        gear_r = float(rng.uniform(0.45, 0.72)) + tooth_profile
        gear = np.clip((gear_r - r) * 30.0, 0.0, 1.0)
        hole_r = float(rng.uniform(0.10, 0.28))
        hub = np.clip((r - hole_r) * 30.0, 0.0, 1.0)
        spokes = [3, 4, 5, 6][int(rng.integers(0, 4))]
        ring_cut = (r > hole_r + 0.08) & (r < 0.42) & (np.abs(np.sin(theta * spokes)) > 0.35)
        return (gear * hub * np.where(ring_cut, 0.15, 1.0) * falloff).astype(np.float32)
    if kind == 1:
        slots = [3, 4, 5, 7, 9][int(rng.integers(0, 5))]
        w_box = float(rng.uniform(0.45, 0.78))
        h_box = float(rng.uniform(0.40, 0.74))
        box = (np.abs(xr) < w_box) & (np.abs(yr) < h_box)
        slot_phase = ((yr / (h_box * 2.0) + 0.5) * slots) % 1.0
        louver = np.where(slot_phase < 0.65, slot_phase / 0.65, 0.0)
        frame = np.clip(1.0 - np.maximum(np.abs(xr) / w_box, np.abs(yr) / h_box), 0.0, 0.15) * 6.0
        return (np.where(box, np.maximum(louver * 0.85, frame * 0.4), 0.0) * falloff).astype(np.float32)
    if kind == 2:
        claws = [1, 2, 3, 4, 5][int(rng.integers(0, 5))]
        acc = np.zeros_like(u, dtype=np.float32)
        spacing = float(rng.uniform(0.12, 0.28))
        curve_k = float(rng.uniform(-0.25, 0.25))
        width = float(rng.uniform(0.0008, 0.0025))
        for k in range(claws):
            off = (k - (claws - 1) / 2.0) * spacing
            curve = curve_k * xr * xr + (micro - 0.5) * 0.07
            taper = np.clip(1.0 - (xr / 0.82) ** 2, 0.0, 1.0)
            dist = np.abs(yr - off - curve)
            acc = np.maximum(acc, np.exp(-(dist * dist) / width) * taper)
        return acc.astype(np.float32)
    if kind == 3:
        rays = [4, 5, 7, 9, 13][int(rng.integers(0, 5))]
        ang_jitter = theta + (micro - 0.5) * 0.7
        spoke = np.power(np.abs(np.cos(ang_jitter * rays * 0.5)), float(rng.uniform(10.0, 22.0)))
        rings = np.power(np.abs(np.cos(r * float(rng.uniform(6.0, 15.0)) + micro * 2.5)), 14.0)
        return (np.maximum(spoke, rings * 0.7) * np.clip(1.0 - r / 0.88, 0.0, 1.0)).astype(np.float32)
    if kind == 4:
        petals = [4, 5, 6, 8, 10, 12][int(rng.integers(0, 6))]
        wave_r = float(rng.uniform(0.38, 0.55)) + float(rng.uniform(0.10, 0.22)) * np.cos(theta * petals)
        petal_rim = np.exp(-((r - wave_r) ** 2) / 0.003)
        inner_star = np.clip((0.35 + 0.12 * np.cos(theta * petals + math.pi / petals) - r) * 14.0, 0.0, 1.0)
        return (np.clip(petal_rim + 0.65 * inner_star, 0.0, 1.0) * falloff).astype(np.float32)
    if kind == 5:
        cells = [5, 7, 10, 14][int(rng.integers(0, 4))]
        f1, _, cid, _, _ = common.periodic_voronoi(u, v, cells, 0.88, rng)
        bumps = np.clip(1.0 - f1 * 1.4, 0.0, 1.0) * (0.55 + 0.45 * cid)
        return (bumps * falloff).astype(np.float32)
    if kind == 6:
        n_dents = int(rng.integers(4, 15))
        acc = np.zeros_like(u, dtype=np.float32)
        for _ in range(n_dents):
            cx = float(rng.uniform(-0.52, 0.52))
            cy = float(rng.uniform(-0.52, 0.52))
            rad = float(rng.uniform(0.07, 0.26))
            d = np.sqrt((x - cx) ** 2 + (y - cy) ** 2) / rad
            acc = np.maximum(acc, np.clip(1.0 - d * d, 0.0, 1.0))
        return (acc * falloff).astype(np.float32)

    sub = variant % 3
    if sub == 0:
        r1 = np.exp(-((r - float(rng.uniform(0.50, 0.72))) ** 2) / 0.0018)
        r2 = np.exp(-((r - float(rng.uniform(0.22, 0.42))) ** 2) / 0.0018)
        ticks = ((np.abs(xr) < 0.03) | (np.abs(yr) < 0.03)) & (r > 0.18) & (r < 0.75)
        return (np.clip(r1 + r2 + ticks.astype(np.float32) * 0.9, 0.0, 1.0) * falloff).astype(np.float32)
    if sub == 1:
        chev = ((np.abs(xr) + yr * 0.8) * [2.5, 3.5, 5.0][int(rng.integers(0, 3))]) % 1.0
        bars = (chev < 0.5).astype(np.float32)
        box = (np.abs(xr) < 0.68) & (np.abs(yr) < 0.55)
        return (bars * box.astype(np.float32) * falloff).astype(np.float32)

    sides = [3, 4, 5, 6, 8][int(rng.integers(0, 5))]
    pr = r * np.cos((theta + math.pi / sides) % (2.0 * math.pi / sides) - math.pi / sides)
    ring1 = np.exp(-((pr - float(rng.uniform(0.45, 0.65))) ** 2) / 0.0025)
    ring2 = np.exp(-((pr - float(rng.uniform(0.20, 0.38))) ** 2) / 0.0025)
    return (np.clip(ring1 + ring2, 0.0, 1.0) * falloff).astype(np.float32)


def generate(seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0) -> Image.Image:
    return common.alpha_image(generate_height(seed=seed, palette=palette, size=size, variant=variant))


SPEC = GeneratorSpec(
    name="mechanical-decal-alpha",
    kind="alpha",
    tileable=False,
    tile_axes="none",
    origin="p1_work/scripts/custom_generators.py (gen_sp_alpha_custom kinds 0-7)",
    default_palette="charcoal_flannel",
    description="Hard-surface & organic radial alphas: gear (v0), louver (v1), claw scratch (v2), shatter (v3), rosette (v4), scales (v5), dents (v6), reticle (v7).",
    render_fn=generate,
    normal_source="native",
    normal_rule="Native hard-surface and sculpted decal relief field (gear, louver, scratch, shatter, rosette, scales, dents, reticle).",
    normal_strength=6.5,
    height_fn=generate_height,
)
