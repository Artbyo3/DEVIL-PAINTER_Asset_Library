"""Dual-scale Voronoi fracture network with per-cell facet tilt and deep fissures."""

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
    cells_major = 4 + (variant % 5) * 2
    # Goehring et al. (2009): columnar basalt stress relaxation converges toward Centroidal Voronoi Tessellations (~120 deg Y-junctions)
    _, edge_a, cid_a, rx_a, ry_a = common.periodic_lloyd_voronoi(u, v, cells_major, 0.88, rng, lloyd_iters=2)
    _, edge_b, _, _, _ = common.periodic_voronoi(u, v, cells_major * 2, 0.90, rng)
    micro = common.periodic_fbm(u, v, cells_major * 3, 4, 0.5, rng)

    def _smoothstep(edge0: float, edge1: float, x: np.ndarray) -> np.ndarray:
        t = np.clip((x - edge0) / (edge1 - edge0 + 1e-8), 0.0, 1.0)
        return t * t * (3.0 - 2.0 * t)

    joint_w = 0.045 - 0.008 * (variant % 3)
    crack_major = _smoothstep(0.01, joint_w + 0.055, edge_a)
    crack_minor = 0.35 + 0.65 * _smoothstep(0.008, 0.065, edge_b)
    # Planar cleavage facet tilt per column + subtle chisel relief
    ang = cid_a * 6.2831853
    tilt = np.clip(0.62 + 0.30 * (rx_a * np.cos(ang) + ry_a * np.sin(ang)), 0.25, 1.0)

    height = common.normalize_field(crack_major * crack_minor * (0.72 * tilt + 0.28 * micro))
    color_field = common.normalize_field(crack_major * (0.52 * cid_a + 0.26 * micro + 0.22 * crack_minor))
    roughness = np.clip(0.32 + 0.52 * (1.0 - crack_major) + 0.14 * micro, 0.15, 0.95).astype(np.float32)

    stops = common.resolve_palette(palette, SPEC.default_palette, rng)
    albedo = common.apply_palette_5stop(color_field, micro, stops, contrast=1.20)
    return common.shade_pbr_surface(albedo, height, roughness=roughness, normal_strength=6.8, spec_intensity=0.26), np.asarray(height, dtype=np.float32)


def generate(seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0) -> Image.Image:
    return _compute(seed=seed, palette=palette, size=size, variant=variant)[0]


def generate_height(seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0) -> np.ndarray:
    return _compute(seed=seed, palette=palette, size=size, variant=variant)[1]

SPEC = GeneratorSpec(
    name="voronoi-basalt-crackle",
    kind="texture",
    tileable=True,
    origin="procedural_textures_v2_300_pack (gen_voronoi_crackle_and_basalt; merges V1 voronoi & SP arch=1)",
    default_palette="basalt_magma_fissure",
    description="Dual-scale Voronoi crackle basalt columns, fractured stone, and faceted ceramic glaze.",
    render_fn=generate,
    normal_source="native",
    normal_rule="Native physical surface height field (height) used by shade_pbr_surface.",
    normal_strength=6.8,
    height_fn=generate_height,
)
