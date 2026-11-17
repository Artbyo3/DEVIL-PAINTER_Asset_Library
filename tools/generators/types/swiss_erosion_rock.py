"""Slope-eroded Swiss ridged multifractal alpine rock and sedimentary strata."""

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
    freq = 2 + (variant % 5)
    swiss = common.periodic_fbm(u, v, freq, 6, 0.54, rng, mode="swiss")
    strata = 0.5 + 0.5 * np.sin((swiss * (5 + variant % 5) + u) * 2.0 * math.pi)
    micro = common.periodic_fbm(u, v, freq * 4, 4, 0.5, rng, mode="turbulence")

    height = common.normalize_field(0.72 * swiss + 0.16 * strata + 0.12 * micro)
    color_field = common.normalize_field(0.60 * swiss + 0.25 * strata + 0.15 * micro)
    roughness = np.clip(0.55 + 0.35 * (1.0 - swiss), 0.2, 0.95).astype(np.float32)

    stops = common.resolve_palette(palette, SPEC.default_palette, rng)
    albedo = common.apply_palette_5stop(color_field, micro, stops, contrast=1.18)
    return common.shade_pbr_surface(albedo, height, roughness=roughness, normal_strength=6.2, spec_intensity=0.22), np.asarray(height, dtype=np.float32)


def generate(seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0) -> Image.Image:
    return _compute(seed=seed, palette=palette, size=size, variant=variant)[0]


def generate_height(seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0) -> np.ndarray:
    return _compute(seed=seed, palette=palette, size=size, variant=variant)[1]

SPEC = GeneratorSpec(
    name="swiss-erosion-rock",
    kind="texture",
    tileable=True,
    origin="procedural_textures_v2_300_pack (gen_true_perlin_swiss_erosion; merges V1 organic_rock & SP arch=0)",
    default_palette="alpine_lichen_granite",
    description="Slope-eroded Swiss ridged multifractal rock with sedimentary strata and mineral grit.",
    render_fn=generate,
    normal_source="native",
    normal_rule="Native physical surface height field (height) used by shade_pbr_surface.",
    normal_strength=6.2,
    height_fn=generate_height,
)
