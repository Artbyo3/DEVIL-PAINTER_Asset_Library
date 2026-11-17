"""Architectural fluted travertine limestone columns with horizontal bedding and pores."""

from __future__ import annotations

import math
import numpy as np
from PIL import Image

from .. import common
from ..interface import GeneratorSpec, PaletteArg


def generate(seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0) -> Image.Image:
    rng = np.random.default_rng(seed)
    u, v = common.make_uv_grid(size)
    flutes = [6, 10, 14, 20][(variant + int(rng.integers(0, 4))) % 4]
    flute_wave = np.abs(np.sin(u * flutes * math.pi))
    strata = common.periodic_fbm(u, v, 4, 5, 0.58, rng, freqy=16)
    pores = np.clip((common.periodic_fbm(u, v, 14, 4, 0.6, rng, freqy=28) - 0.68) * 5.0, 0.0, 1.0)

    h = common.normalize_field(0.60 * flute_wave + 0.30 * strata - 0.35 * pores)
    c = common.normalize_field(0.55 * strata + 0.30 * flute_wave - 0.25 * pores)
    rough = np.clip(0.45 + 0.40 * pores, 0.30, 0.90).astype(np.float32)

    stops = common.resolve_palette(palette, SPEC.default_palette, rng)
    albedo = common.apply_palette_5stop(c, strata, stops, contrast=1.15)
    return common.shade_pbr_surface(albedo, h, roughness=rough, normal_strength=6.8, spec_intensity=0.16)


SPEC = GeneratorSpec(
    name="fluted-travertine",
    kind="texture",
    tileable=True,
    origin="p1_work/scripts/custom_generators.py (gen_v2_fluted_travertine)",
    default_palette="desert_travertine",
    description="Fluted architectural travertine stone with sedimentary bedding strata and pitted cavities.",
    render_fn=generate,
)
