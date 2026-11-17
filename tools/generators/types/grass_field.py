"""Top-down meadow grass field with multi-directional blades, clump density, and green variation."""

from __future__ import annotations

import math
import numpy as np
from PIL import Image

from .. import common
from ..interface import GeneratorSpec, PaletteArg


def _blade_layer(
    u: np.ndarray,
    v: np.ndarray,
    kx: int,
    ky: int,
    warp: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    # Integer linear combination guarantees exact periodicity across [0, 1)^2
    p = (u * kx + v * ky + 0.35 * (warp - 0.5)) % 1.0
    q = (u * (-ky) + v * kx + 0.35 * (warp - 0.5)) % 1.0
    # Dense short blade clusters via modulated high-frequency wave ridges
    env = common.periodic_fbm((u + 0.17) % 1.0, (v + 0.31) % 1.0, 12, 2, 0.5, rng)
    ridge = np.power(0.5 + 0.5 * np.sin(p * 2.0 * math.pi), 2.5)
    taper = np.power(0.5 + 0.5 * np.cos(q * 0.5 * 2.0 * math.pi), 1.2)
    return np.clip(ridge * (0.45 + 0.55 * taper) * (0.55 + 0.65 * env), 0.0, 1.0).astype(np.float32)


def _compute(
    seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0
) -> tuple[Image.Image, np.ndarray]:
    rng = np.random.default_rng(seed)
    u, v = common.make_uv_grid(size)

    density_scale = [3, 4, 5, 6][variant % 4]
    density = common.periodic_fbm(u, v, density_scale, 4, 0.55, rng)
    clump = common.periodic_fbm((u + 0.43) % 1.0, (v + 0.71) % 1.0, density_scale * 2, 3, 0.5, rng)
    warp_a = common.periodic_fbm(u, v, 14, 2, 0.5, rng)
    warp_b = common.periodic_fbm((u + 0.29) % 1.0, (v + 0.61) % 1.0, 18, 2, 0.5, rng)

    b1 = _blade_layer(u, v, 28, 12, warp_a, rng)
    b2 = _blade_layer(u, v, -16, 26, warp_b, rng)
    b3 = _blade_layer(u, v, 20, -24, warp_a, rng)
    blades = np.maximum(b1, np.maximum(b2 * 0.92, b3 * 0.88))

    h = common.normalize_field(0.58 * blades + 0.26 * clump + 0.16 * density)
    c = common.normalize_field(0.44 * blades + 0.34 * density + 0.22 * clump)
    rough = np.clip(0.68 + 0.22 * (1.0 - blades), 0.55, 0.94).astype(np.float32)

    stops = common.resolve_palette(palette, SPEC.default_palette, rng)
    albedo = common.apply_palette_5stop(c, blades, stops, contrast=1.18)
    return common.shade_pbr_surface(albedo, h, roughness=rough, normal_strength=5.6, spec_intensity=0.10), np.asarray(h, dtype=np.float32)


def generate(seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0) -> Image.Image:
    return _compute(seed=seed, palette=palette, size=size, variant=variant)[0]


def generate_height(seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0) -> np.ndarray:
    return _compute(seed=seed, palette=palette, size=size, variant=variant)[1]

SPEC = GeneratorSpec(
    name="grass-field",
    kind="texture",
    tileable=True,
    origin="tools/generators/types/grass_field.py",
    default_palette="meadow_grass_field",
    description="Top-down meadow grass field with dense multi-directional grass blades and natural green density variation.",
    render_fn=generate,
    normal_source="native",
    normal_rule="Native physical surface height field (h) used by shade_pbr_surface.",
    normal_strength=5.6,
    height_fn=generate_height,
)
