"""Smooth warm architectural stucco plaster wall with subtle arcing trowel marks and sand float grain."""

from __future__ import annotations

import numpy as np
from PIL import Image

from .. import common
from ..interface import GeneratorSpec, PaletteArg


def generate(seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0) -> Image.Image:
    rng = np.random.default_rng(seed)
    u, v = common.make_uv_grid(size)

    trowel_cells = [3, 4, 5, 6][variant % 4]
    wx = common.periodic_fbm(u, v, 2, 3, 0.5, rng)
    wy = common.periodic_fbm((u + 0.37) % 1.0, (v + 0.61) % 1.0, 2, 3, 0.5, rng)
    uu = (u + 0.10 * (wx - 0.5)) % 1.0
    vv = (v + 0.10 * (wy - 0.5)) % 1.0

    # Broad shallow overlapping trowel sweeps & feather ridges
    f1_a, edge_a, cid_a, _, _ = common.periodic_voronoi(uu, vv, trowel_cells, 0.85, rng)
    f1_b, edge_b, _, _, _ = common.periodic_voronoi((uu + 0.43) % 1.0, (vv + 0.29) % 1.0, trowel_cells + 2, 0.85, rng)

    trowel_ridge = 0.55 * np.exp(-((edge_a - 0.06) ** 2) / 0.003) + 0.45 * np.exp(-((edge_b - 0.06) ** 2) / 0.003)
    lime_wash = common.periodic_fbm(u, v, 3, 3, 0.48, rng)
    sand_float = common.periodic_fbm(u, v, 42, 3, 0.52, rng)

    h = common.normalize_field(0.45 * lime_wash + 0.35 * trowel_ridge + 0.12 * (1.0 - f1_a) + 0.08 * sand_float)
    c = common.normalize_field(0.55 * lime_wash + 0.22 * (1.0 - f1_b) + 0.13 * cid_a + 0.10 * sand_float)
    rough = np.clip(0.68 + 0.14 * sand_float, 0.55, 0.88).astype(np.float32)

    stops = common.resolve_palette(palette, SPEC.default_palette, rng)
    albedo = common.apply_palette_5stop(c, sand_float, stops, contrast=1.08)
    return common.shade_pbr_surface(albedo, h, roughness=rough, normal_strength=2.4, spec_intensity=0.09)


SPEC = GeneratorSpec(
    name="stucco-plaster-wall",
    kind="texture",
    tileable=True,
    origin="tools/generators/types/stucco_plaster_wall.py",
    default_palette="warm_stucco_plaster",
    description="Smooth warm architectural stucco plaster wall with subtle sweeping trowel marks and fine sand float.",
    render_fn=generate,
)
