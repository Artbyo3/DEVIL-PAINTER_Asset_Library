"""Top-down meadow grass field with multi-directional blades, clump density, and green variation."""

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

    density_scale = [3, 4, 5, 6][variant % 4]
    density = common.periodic_fbm(u, v, density_scale, 4, 0.55, rng)
    clump = common.periodic_fbm((u + 0.43) % 1.0, (v + 0.71) % 1.0, density_scale * 2, 3, 0.5, rng)
    warp = common.periodic_fbm(u, v, 8, 2, 0.5, rng)
    wu = (u + 0.03 * (warp - 0.5)) % 1.0
    wv = (v + 0.03 * (warp - 0.5)) % 1.0

    # Lagae-Lefebvre Gabor bladelets oriented along 4 skewed azimuthal wind-lay angles (finite envelope, zero grid stripes)
    b1 = np.power(np.clip((common.periodic_gabor_noise(wu, wv, 18, 4.5, 0.35, 2.0, rng, impulses_per_cell=3) - 0.38) * 1.65, 0.0, 1.0), 1.4)
    b2 = np.power(np.clip((common.periodic_gabor_noise(wu, wv, 20, 4.8, 1.45, 2.1, rng, impulses_per_cell=3) - 0.38) * 1.65, 0.0, 1.0), 1.4)
    b3 = np.power(np.clip((common.periodic_gabor_noise(wu, wv, 16, 4.2, 2.25, 1.9, rng, impulses_per_cell=3) - 0.38) * 1.65, 0.0, 1.0), 1.4)
    blades = np.maximum(b1, np.maximum(b2 * 0.94, b3 * 0.88))

    h = common.normalize_field(0.58 * blades + 0.26 * clump + 0.16 * density)
    c = common.normalize_field(0.46 * blades + 0.32 * density + 0.22 * clump)
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
