"""Full-grain pebbled leather with Worley grain cells and natural creases."""

from __future__ import annotations

import numpy as np
from PIL import Image

from .. import common
from ..interface import GeneratorSpec, PaletteArg


def generate(seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0) -> Image.Image:
    rng = np.random.default_rng(seed)
    u, v = common.make_uv_grid(size)
    micro = common.periodic_fbm(u, v, 10, 4, 0.5, rng)
    f1, edge, cid, _, _ = common.periodic_voronoi(u, v, 14 + (variant % 5) * 2, 0.88, rng)
    creases = common.periodic_fbm(u, v, 4, 4, 0.55, rng, mode="ridged")
    pebble = np.sqrt(np.clip(1.0 - f1 * 0.85, 0.0, 1.0)) * np.clip(edge * 5.0, 0.2, 1.0)

    h = common.normalize_field(0.68 * pebble + 0.20 * creases + 0.12 * micro)
    c = common.normalize_field(0.55 * h + 0.25 * cid + 0.20 * micro)
    rough = np.clip(0.32 + 0.38 * (1.0 - pebble), 0.2, 0.85).astype(np.float32)

    stops = common.resolve_palette(palette, SPEC.default_palette, rng)
    albedo = common.apply_palette_5stop(c, micro, stops, contrast=1.15)
    return common.shade_pbr_surface(albedo, h, roughness=rough, normal_strength=6.0, spec_intensity=0.26)


SPEC = GeneratorSpec(
    name="pebbled-leather",
    kind="texture",
    tileable=True,
    origin="procedural_textures_v2_300_pack/procedural_texture_generator.py (leather)",
    default_palette="saddle_cognac_leather",
    description="Full-grain pebbled hide leather with organic Worley grain cells and natural wrinkle creases.",
    render_fn=generate,
)
