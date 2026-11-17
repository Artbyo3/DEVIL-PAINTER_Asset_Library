"""Architectural fluted travertine limestone columns with horizontal bedding and pores."""

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
    flutes = [6, 10, 14, 20][(variant + int(rng.integers(0, 4))) % 4]
    # Classical Doric/Ionic architectural flute profile: concave circular channel separated by narrow flat arrises
    arr_frac = 0.16
    fu = np.abs(((u * flutes) % 1.0) - 0.5) * 2.0  # 0 at trough center, 1 at arris
    channel_span = 1.0 - arr_frac
    in_flute = np.clip(fu / channel_span, 0.0, 1.0)
    flute_wave = 1.0 - 0.78 * np.sqrt(np.clip(1.0 - in_flute * in_flute, 0.0, 1.0))

    strata = common.periodic_fbm(u, v, 3, 5, 0.58, rng, freqy=18)
    # Travertine carbonate dissolution vugs are horizontally elongated along bedding planes
    vug_raw = common.periodic_fbm(u, v, 6, 4, 0.58, rng, freqy=34)
    t_p = np.clip((vug_raw - 0.64) / 0.14, 0.0, 1.0)
    pores = t_p * t_p * (3.0 - 2.0 * t_p)

    h = common.normalize_field(0.62 * flute_wave + 0.26 * strata - 0.34 * pores)
    c = common.normalize_field(0.55 * strata + 0.30 * flute_wave - 0.24 * pores)
    rough = np.clip(0.45 + 0.40 * pores, 0.30, 0.90).astype(np.float32)

    stops = common.resolve_palette(palette, SPEC.default_palette, rng)
    albedo = common.apply_palette_5stop(c, strata, stops, contrast=1.15)
    return common.shade_pbr_surface(albedo, h, roughness=rough, normal_strength=6.8, spec_intensity=0.16), np.asarray(h, dtype=np.float32)


def generate(seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0) -> Image.Image:
    return _compute(seed=seed, palette=palette, size=size, variant=variant)[0]


def generate_height(seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0) -> np.ndarray:
    return _compute(seed=seed, palette=palette, size=size, variant=variant)[1]

SPEC = GeneratorSpec(
    name="fluted-travertine",
    kind="texture",
    tileable=True,
    origin="p1_work/scripts/custom_generators.py (gen_v2_fluted_travertine)",
    default_palette="desert_travertine",
    description="Fluted architectural travertine stone with sedimentary bedding strata and pitted cavities.",
    render_fn=generate,
    normal_source="native",
    normal_rule="Native physical surface height field (h) used by shade_pbr_surface.",
    normal_strength=6.8,
    height_fn=generate_height,
)
