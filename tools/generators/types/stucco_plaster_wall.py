"""Smooth warm architectural stucco plaster wall with subtle arcing trowel marks and sand float grain."""

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

    trowel_cells = [3, 4, 5, 6][variant % 4]
    wx = common.periodic_fbm(u, v, 2, 3, 0.5, rng)
    wy = common.periodic_fbm((u + 0.37) % 1.0, (v + 0.61) % 1.0, 2, 3, 0.5, rng)
    uu = (u + 0.08 * (wx - 0.5)) % 1.0
    vv = (v + 0.08 * (wy - 0.5)) % 1.0

    # Curved steel trowel-blade arc sweeps: overlapping periodic circular arc ridges exp(-((r - R)^2) / w)
    arc_field = np.zeros_like(u, dtype=np.float32)
    skim_field = np.zeros_like(u, dtype=np.float32)
    n_arcs = trowel_cells * 5
    for _ in range(n_arcs):
        cx = float(rng.uniform(0.0, 1.0))
        cy = float(rng.uniform(0.0, 1.0))
        rad = float(rng.uniform(0.14, 0.28))
        ang0 = float(rng.uniform(0.0, 2.0 * math.pi))
        dx = ((uu - cx + 0.5) % 1.0) - 0.5
        dy = ((vv - cy + 0.5) % 1.0) - 0.5
        r = np.sqrt(dx * dx + dy * dy + 1e-8)
        theta = np.arctan2(dy, dx)
        # Directional sweep envelope over ~120 deg arc segment
        arc_ang = 0.5 + 0.5 * np.cos(theta - ang0)
        arc_env = np.power(np.clip(arc_ang, 0.0, 1.0), 2.2)
        ridge = np.exp(-((r - rad) ** 2) / 0.00055) * arc_env
        arc_field += ridge
        skim_field += np.exp(-(r * r) / (rad * rad * 1.4)) * arc_env

    trowel_ridge = common.normalize_field(arc_field)
    skim_swell = common.normalize_field(skim_field)
    lime_wash = common.periodic_fbm(u, v, 3, 3, 0.48, rng)
    sand_float = common.periodic_fbm(u, v, 42, 3, 0.52, rng)

    h = common.normalize_field(0.42 * lime_wash + 0.34 * trowel_ridge + 0.16 * skim_swell + 0.08 * sand_float)
    c = common.normalize_field(0.54 * lime_wash + 0.24 * skim_swell + 0.12 * trowel_ridge + 0.10 * sand_float)
    rough = np.clip(0.68 + 0.14 * sand_float, 0.55, 0.88).astype(np.float32)

    stops = common.resolve_palette(palette, SPEC.default_palette, rng)
    albedo = common.apply_palette_5stop(c, sand_float, stops, contrast=1.08)
    return common.shade_pbr_surface(albedo, h, roughness=rough, normal_strength=2.4, spec_intensity=0.09), np.asarray(h, dtype=np.float32)


def generate(seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0) -> Image.Image:
    return _compute(seed=seed, palette=palette, size=size, variant=variant)[0]


def generate_height(seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0) -> np.ndarray:
    return _compute(seed=seed, palette=palette, size=size, variant=variant)[1]

SPEC = GeneratorSpec(
    name="stucco-plaster-wall",
    kind="texture",
    tileable=True,
    origin="tools/generators/types/stucco_plaster_wall.py",
    default_palette="warm_stucco_plaster",
    description="Smooth warm architectural stucco plaster wall with subtle sweeping trowel marks and fine sand float.",
    render_fn=generate,
    normal_source="native",
    normal_rule="Native physical surface height field (h) used by shade_pbr_surface.",
    normal_strength=2.4,
    height_fn=generate_height,
)
