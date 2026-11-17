"""Full-grain pebbled leather with Worley grain cells and natural creases."""

from __future__ import annotations

import numpy as np
from PIL import Image

from .. import common
from ..interface import GeneratorSpec, PaletteArg


def _compute(
    seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0
) -> tuple[Image.Image, np.ndarray]:
    rng = np.random.default_rng(seed)
    u, v = common.make_uv_grid(size)
    micro = common.periodic_fbm(u, v, 10, 4, 0.5, rng)
    cells = 14 + (variant % 5) * 2
    # Lloyd-relaxed periodic Voronoi gives authentic compact rounded leather grain pebbles
    f1, edge, cid, _, _ = common.periodic_lloyd_voronoi(u, v, cells, 0.86, rng, lloyd_iters=2)
    creases = common.periodic_fbm(u, v, 4, 4, 0.55, rng, mode="ridged")

    # Convex pebble crown + recessed epidermal valleys + fine hair-follicle pores in crease junctions
    pebble = np.sqrt(np.clip(1.0 - f1 * 0.78, 0.0, 1.0)) * np.clip(edge * 4.5, 0.16, 1.0)
    pore_noise = common.periodic_gradient_perlin(u, v, min(128, cells * 3), min(128, cells * 3), rng)
    follicle_pores = np.exp(-((edge) ** 2) / 0.008) * np.clip((pore_noise - 0.62) * 3.2, 0.0, 1.0)

    h = common.normalize_field(0.68 * pebble + 0.18 * creases + 0.14 * micro - 0.16 * follicle_pores)
    c = common.normalize_field(0.55 * h + 0.25 * cid + 0.20 * micro)
    rough = np.clip(0.30 + 0.42 * (1.0 - pebble) + 0.14 * follicle_pores, 0.2, 0.88).astype(np.float32)

    stops = common.resolve_palette(palette, SPEC.default_palette, rng)
    albedo = common.apply_palette_5stop(c, micro, stops, contrast=1.15)
    return common.shade_pbr_surface(albedo, h, roughness=rough, normal_strength=6.0, spec_intensity=0.26), np.asarray(h, dtype=np.float32)


def generate(seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0) -> Image.Image:
    return _compute(seed=seed, palette=palette, size=size, variant=variant)[0]


def generate_height(seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0) -> np.ndarray:
    return _compute(seed=seed, palette=palette, size=size, variant=variant)[1]

SPEC = GeneratorSpec(
    name="pebbled-leather",
    kind="texture",
    tileable=True,
    origin="procedural_textures_v2_300_pack/procedural_texture_generator.py (leather)",
    default_palette="saddle_cognac_leather",
    description="Full-grain pebbled hide leather with organic Worley grain cells and natural wrinkle creases.",
    render_fn=generate,
    normal_source="native",
    normal_rule="Native physical surface height field (h) used by shade_pbr_surface.",
    normal_strength=6.0,
    height_fn=generate_height,
)
