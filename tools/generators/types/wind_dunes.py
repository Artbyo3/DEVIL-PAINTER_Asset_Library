"""Wind-sculpted asymmetric sand dunes with micro wind ripples."""

from __future__ import annotations

import math
import numpy as np
from PIL import Image

from .. import common
from ..interface import GeneratorSpec, PaletteArg


def generate(seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0) -> Image.Image:
    rng = np.random.default_rng(seed)
    u, v = common.make_uv_grid(size)
    freq = [4, 6, 8, 12][(variant + int(rng.integers(0, 4))) % 4]
    skew = [-2, -1, 0, 1, 2][int(rng.integers(0, 5))]
    # Integer directional linear combination guarantees clean toroidal edge wrapping.
    ru = (u * freq + v * skew) % float(freq)
    warp = common.periodic_fbm(u, v, 3, 4, 0.55, rng)

    phase = (ru + warp * 1.4) % 1.0
    dune = np.where(phase < 0.72, phase / 0.72, 1.0 - (phase - 0.72) / 0.28)
    micro_ripple = 0.5 + 0.5 * np.sin((ru * 5.0 + warp * 3.0) * 2.0 * math.pi)
    h = common.normalize_field(0.82 * dune + 0.18 * micro_ripple)
    c = common.normalize_field(0.70 * h + 0.30 * warp)

    stops = common.resolve_palette(palette, SPEC.default_palette, rng)
    albedo = common.apply_palette_5stop(c, micro_ripple, stops, contrast=1.10)
    return common.shade_pbr_surface(albedo, h, roughness=0.78, normal_strength=5.0, spec_intensity=0.08)


SPEC = GeneratorSpec(
    name="wind-dunes",
    kind="texture",
    tileable=True,
    origin="p1_work/scripts/custom_generators.py (gen_v1_wind_dunes; integer-shear seamless fix)",
    default_palette="desert_sandstone",
    description="Asymmetric wind-blown desert sand dunes with secondary slip-face micro-ripples.",
    render_fn=generate,
)
