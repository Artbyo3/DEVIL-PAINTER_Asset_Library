"""Top-down patchy grass thinning organically into bare compacted brown earth soil."""

from __future__ import annotations

import math
import numpy as np
from PIL import Image

from .. import common
from ..interface import GeneratorSpec, PaletteArg


def _smoothstep(x: np.ndarray, edge0: float, edge1: float) -> np.ndarray:
    t = np.clip((x - edge0) / max(edge1 - edge0, 1e-6), 0.0, 1.0)
    return (t * t * (3.0 - 2.0 * t)).astype(np.float32)


def _compute(
    seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0
) -> tuple[Image.Image, np.ndarray]:
    rng = np.random.default_rng(seed)
    u, v = common.make_uv_grid(size)

    patch_freq = 2 + (variant % 2)
    patch_macro = common.periodic_fbm(u, v, patch_freq, 4, 0.54, rng)
    patch_edge = common.periodic_fbm((u + 0.37) % 1.0, (v + 0.59) % 1.0, 10, 3, 0.5, rng)

    # Directional blades
    warp = common.periodic_fbm(u, v, 14, 2, 0.5, rng)
    b1 = np.power(0.5 + 0.5 * np.sin((u * 26 + v * 12 + 0.35 * (warp - 0.5)) * 2.0 * math.pi), 2.4)
    b2 = np.power(0.5 + 0.5 * np.sin((u * (-14) + v * 28 + 0.35 * (warp - 0.5)) * 2.0 * math.pi), 2.4)
    blades = np.maximum(b1, b2 * 0.90)

    # Grass coverage thins organically at edges into isolated blades
    grass_zone = patch_macro + 0.24 * (patch_edge - 0.5) + 0.16 * (blades - 0.4)
    grass_mask = _smoothstep(grass_zone, 0.44, 0.58)

    # Dirt surface grain + tiny stones
    dirt_grain = common.periodic_fbm(u, v, 24, 4, 0.55, rng)
    f1_s, _, cid_s, _, _ = common.periodic_voronoi(u, v, 18, 0.92, rng)
    small_stones = np.clip(((1.0 - f1_s) * (cid_s > 0.72).astype(np.float32) - 0.70) * 14.0, 0.0, 1.0)

    stops = np.array(common.resolve_palette(palette, SPEC.default_palette, rng), dtype=np.float32)
    dirt_rgb = stops[0] * (1.0 - dirt_grain[..., None]) + stops[1] * dirt_grain[..., None]
    dirt_rgb = dirt_rgb * (1.0 - 0.35 * small_stones[..., None]) + (stops[1] * 1.15) * (0.35 * small_stones[..., None])

    blade_tone = np.clip(0.45 * blades + 0.35 * patch_macro + 0.20 * patch_edge, 0.0, 1.0)
    grass_rgb = np.where(
        blade_tone[..., None] < 0.5,
        stops[2] * (1.0 - blade_tone[..., None] * 2.0) + stops[3] * (blade_tone[..., None] * 2.0),
        stops[3] * (1.0 - (blade_tone[..., None] - 0.5) * 2.0) + stops[4] * ((blade_tone[..., None] - 0.5) * 2.0),
    )

    albedo = dirt_rgb * (1.0 - grass_mask[..., None]) + grass_rgb * grass_mask[..., None]
    h_dirt = 0.25 + 0.20 * dirt_grain + 0.22 * small_stones
    h_grass = 0.52 + 0.42 * blades
    h = common.normalize_field(h_dirt * (1.0 - grass_mask) + h_grass * grass_mask)
    rough = np.where(grass_mask > 0.4, 0.74, 0.88).astype(np.float32)

    return common.shade_pbr_surface(albedo, h, roughness=rough, normal_strength=5.4, spec_intensity=0.09), np.asarray(h, dtype=np.float32)


def generate(seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0) -> Image.Image:
    return _compute(seed=seed, palette=palette, size=size, variant=variant)[0]


def generate_height(seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0) -> np.ndarray:
    return _compute(seed=seed, palette=palette, size=size, variant=variant)[1]

SPEC = GeneratorSpec(
    name="grass-dirt-patch",
    kind="texture",
    tileable=True,
    origin="tools/generators/types/grass_dirt_patch.py",
    default_palette="patchy_grass_earth",
    description="Top-down patchy meadow grass thinning organically with stray blades into bare compacted earth.",
    render_fn=generate,
    normal_source="native",
    normal_rule="Native physical surface height field (h) used by shade_pbr_surface.",
    normal_strength=5.4,
    height_fn=generate_height,
)
