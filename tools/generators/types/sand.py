"""Flat beach and desert floor sand with fine low-relief wave ripples and crisp mineral grain."""

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

    ripples = [14, 18, 22, 26][variant % 4]
    skew = [-1, 0, 1, 2][variant % 4]
    ru = (u * ripples + v * skew) % float(ripples)
    warp = common.periodic_fbm(u, v, 3, 3, 0.5, rng)

    # Gentle low-relief flat-floor ripples (distinct from steep wind-dunes ridges)
    fine_ripple = 0.5 + 0.5 * np.sin((ru + 1.15 * (warp - 0.5)) * 2.0 * math.pi)
    broad_undulation = common.periodic_fbm(u, v, 2, 2, 0.5, rng)

    # Crisp granular sand / quartz & shell mineral speckle
    grain_a = common.periodic_fbm(u, v, 44, 3, 0.52, rng)
    grain_b = common.periodic_fbm((u + 0.37) % 1.0, (v + 0.61) % 1.0, 64, 2, 0.5, rng)
    speck = np.clip((grain_b - 0.62) * 3.0, 0.0, 1.0)

    h = common.normalize_field(0.46 * fine_ripple + 0.16 * broad_undulation + 0.26 * grain_a + 0.12 * grain_b)
    c = common.normalize_field(0.36 * fine_ripple + 0.24 * broad_undulation + 0.26 * grain_a + 0.14 * speck)
    rough = np.clip(0.76 + 0.14 * grain_a - 0.10 * speck, 0.55, 0.94).astype(np.float32)

    stops = common.resolve_palette(palette, SPEC.default_palette, rng)
    albedo = common.apply_palette_5stop(c, grain_b, stops, contrast=1.12)
    return common.shade_pbr_surface(albedo, h, roughness=rough, normal_strength=2.8, spec_intensity=0.14), np.asarray(h, dtype=np.float32)


def generate(seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0) -> Image.Image:
    return _compute(seed=seed, palette=palette, size=size, variant=variant)[0]


def generate_height(seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0) -> np.ndarray:
    return _compute(seed=seed, palette=palette, size=size, variant=variant)[1]

SPEC = GeneratorSpec(
    name="sand",
    kind="texture",
    tileable=True,
    origin="tools/generators/types/sand.py",
    default_palette="rippled_beach_sand",
    description="Flat beach and desert floor sand with fine low-relief wave ripples and crisp granular mineral sparkle.",
    render_fn=generate,
    normal_source="native",
    normal_rule="Native physical surface height field (h) used by shade_pbr_surface.",
    normal_strength=2.8,
    height_fn=generate_height,
)
