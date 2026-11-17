"""Layered sedimentary sandstone strata with undulating horizontal mineral bands."""

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

    bands = [8, 12, 16, 20][variant % 4]
    warp = common.periodic_fbm(u, v, 2, 4, 0.55, rng, freqy=3)
    cross = common.periodic_fbm((u + 0.41) % 1.0, (v + 0.19) % 1.0, 4, 3, 0.50, rng, freqy=6)
    grit = common.periodic_fbm(u, v, 32, 3, 0.52, rng)

    phase = (v * bands + 1.65 * (warp - 0.5) + 0.45 * (cross - 0.5))
    strata_main = 0.5 + 0.5 * np.sin(phase * 2.0 * math.pi)
    strata_sub = 0.5 + 0.5 * np.sin((phase * 3.0 + 0.4 * cross) * 2.0 * math.pi)
    ledge = np.power(np.clip(strata_main, 0.0, 1.0), 0.72)

    h = common.normalize_field(0.62 * ledge + 0.22 * strata_sub + 0.16 * grit)
    c = common.normalize_field(0.55 * strata_main + 0.28 * strata_sub + 0.17 * warp)
    rough = np.clip(0.76 + 0.16 * grit, 0.65, 0.95).astype(np.float32)

    stops = common.resolve_palette(palette, SPEC.default_palette, rng)
    albedo = common.apply_palette_5stop(c, grit, stops, contrast=1.18)
    return common.shade_pbr_surface(albedo, h, roughness=rough, normal_strength=5.8, spec_intensity=0.10), np.asarray(h, dtype=np.float32)


def generate(seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0) -> Image.Image:
    return _compute(seed=seed, palette=palette, size=size, variant=variant)[0]


def generate_height(seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0) -> np.ndarray:
    return _compute(seed=seed, palette=palette, size=size, variant=variant)[1]

SPEC = GeneratorSpec(
    name="sandstone-strata",
    kind="texture",
    tileable=True,
    origin="tools/generators/types/sandstone_strata.py",
    default_palette="layered_sandstone_strata",
    description="Layered sedimentary sandstone rock with undulating mineral strata bands and fine quartz grit.",
    render_fn=generate,
    normal_source="native",
    normal_rule="Native physical surface height field (h) used by shade_pbr_surface.",
    normal_strength=5.8,
    height_fn=generate_height,
)
