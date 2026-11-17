"""Dense crushed angular gravel aggregate with sharp fractured stone facets."""

from __future__ import annotations

import numpy as np
from PIL import Image

from .. import common
from ..interface import GeneratorSpec, PaletteArg


def generate(seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0) -> Image.Image:
    rng = np.random.default_rng(seed)
    u, v = common.make_uv_grid(size)

    cells_a = [18, 22, 26, 30][variant % 4]
    cells_b = cells_a + 14

    f1_a, edge_a, cid_a, _, _ = common.periodic_voronoi(u, v, cells_a, 0.94, rng, metric="chebyshev")
    f1_b, edge_b, cid_b, _, _ = common.periodic_voronoi(
        (u + 0.37) % 1.0, (v + 0.59) % 1.0, cells_b, 0.94, rng, metric="manhattan"
    )
    grit = common.periodic_fbm(u, v, 40, 3, 0.52, rng)

    # Sharp faceted beveled chips
    chip_a = np.clip(edge_a * 6.5, 0.0, 1.0) * (0.65 + 0.35 * (1.0 - f1_a))
    chip_b = np.clip(edge_b * 7.5, 0.0, 1.0) * (0.55 + 0.35 * (1.0 - f1_b))
    use_a = chip_a >= chip_b * 0.85

    stone_h = np.where(use_a, chip_a, chip_b * 0.85)
    stone_id = np.where(use_a, cid_a, cid_b)

    h = common.normalize_field(0.84 * stone_h + 0.16 * grit)
    c = common.normalize_field(0.62 * stone_id + 0.24 * stone_h + 0.14 * grit)
    rough = np.clip(0.72 + 0.20 * (1.0 - stone_h) + 0.08 * grit, 0.62, 0.95).astype(np.float32)

    stops = common.resolve_palette(palette, SPEC.default_palette, rng)
    albedo = common.apply_palette_5stop(c, grit, stops, contrast=1.22)
    return common.shade_pbr_surface(albedo, h, roughness=rough, normal_strength=6.5, spec_intensity=0.14)


SPEC = GeneratorSpec(
    name="gravel",
    kind="texture",
    tileable=True,
    origin="tools/generators/types/gravel.py",
    default_palette="crushed_angular_gravel",
    description="Dense angular crushed gravel aggregate with sharp fractured stone chips and fine dusty grit.",
    render_fn=generate,
)
