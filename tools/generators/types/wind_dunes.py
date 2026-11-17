"""Wind-sculpted asymmetric sand dunes with micro wind ripples."""

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
    freq = [4, 6, 8, 12][(variant + int(rng.integers(0, 4))) % 4]
    skew = [-2, -1, 0, 1, 2][int(rng.integers(0, 5))]
    # Integer directional linear combination guarantees clean toroidal edge wrapping.
    ru = (u * freq + v * skew) % float(freq)
    warp = common.periodic_fbm(u, v, 3, 4, 0.55, rng)

    phase = (ru + warp * 1.4) % 1.0
    # Aeolian dune cross-section: gentle windward stoss (<15 deg) -> rounded brink -> steep ~34 deg slipface
    brink = 0.74
    brink_w = 0.08
    stoss = np.power(np.clip(phase / brink, 0.0, 1.0), 0.90)
    lee = 1.0 - np.clip((phase - brink) / (1.0 - brink), 0.0, 1.0)
    # C1 crest brink blend + trough smoothing
    t_b = np.clip((phase - (brink - brink_w)) / (2.0 * brink_w), 0.0, 1.0)
    w_b = t_b * t_b * (3.0 - 2.0 * t_b)
    dune = stoss * (1.0 - w_b) + lee * w_b
    # Bagnold (1941) / Anderson (1987) saltation ripples on windward stoss, quenched on avalanching slipface
    ripple_phase = (ru * 6.0 + warp * 2.8) % 1.0
    ripple_asym = np.where(ripple_phase < 0.68, ripple_phase / 0.68, 1.0 - (ripple_phase - 0.68) / 0.32)
    ripple_asym = ripple_asym * ripple_asym * (3.0 - 2.0 * ripple_asym)
    stoss_weight = 1.0 - w_b * 0.85
    h = common.normalize_field(0.84 * dune + 0.16 * ripple_asym * stoss_weight)
    c = common.normalize_field(0.72 * h + 0.28 * warp)

    stops = common.resolve_palette(palette, SPEC.default_palette, rng)
    albedo = common.apply_palette_5stop(c, ripple_asym, stops, contrast=1.10)
    return common.shade_pbr_surface(albedo, h, roughness=0.78, normal_strength=5.0, spec_intensity=0.08), np.asarray(h, dtype=np.float32)


def generate(seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0) -> Image.Image:
    return _compute(seed=seed, palette=palette, size=size, variant=variant)[0]


def generate_height(seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0) -> np.ndarray:
    return _compute(seed=seed, palette=palette, size=size, variant=variant)[1]

SPEC = GeneratorSpec(
    name="wind-dunes",
    kind="texture",
    tileable=True,
    origin="p1_work/scripts/custom_generators.py (gen_v1_wind_dunes; integer-shear seamless fix)",
    default_palette="desert_sandstone",
    description="Asymmetric wind-blown desert sand dunes with secondary slip-face micro-ripples.",
    render_fn=generate,
    normal_source="native",
    normal_rule="Native physical surface height field (h) used by shade_pbr_surface.",
    normal_strength=5.0,
    height_fn=generate_height,
)
