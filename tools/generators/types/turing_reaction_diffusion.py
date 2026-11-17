"""Multi-scale Difference-of-Gaussians Turing reaction-diffusion coral and labyrinth patterns."""

from __future__ import annotations

import numpy as np
from PIL import Image

from .. import common
from ..interface import GeneratorSpec, PaletteArg


def generate(seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0) -> Image.Image:
    rng = np.random.default_rng(seed)
    u, v = common.make_uv_grid(size)
    freq = 6 + (variant % 5) * 2
    state = common.periodic_fbm(u, v, freq, 4, 0.55, rng)
    micro = common.periodic_fbm(u, v, freq * 2, 3, 0.5, rng)

    r_act = 1 + (variant % 2)
    r_inh = 3 + (variant % 3)
    for _ in range(4):
        act = common.periodic_box_blur(state, r_act)
        inh = common.periodic_box_blur(state, r_inh)
        diff = act - inh
        state = common.normalize_field(state + np.tanh(diff * 8.0) * 0.35)

    ridges = np.clip((state - 0.35) / 0.30, 0.0, 1.0)
    height = common.normalize_field(0.82 * ridges + 0.18 * micro)
    color_field = common.normalize_field(0.75 * state + 0.25 * micro)
    roughness = np.clip(0.22 + 0.50 * (1.0 - ridges), 0.15, 0.85).astype(np.float32)

    stops = common.resolve_palette(palette, SPEC.default_palette, rng)
    albedo = common.apply_palette_5stop(color_field, micro, stops, contrast=1.20)
    return common.shade_pbr_surface(albedo, height, roughness=roughness, normal_strength=6.0, spec_intensity=0.28)


SPEC = GeneratorSpec(
    name="turing-reaction-diffusion",
    kind="texture",
    tileable=True,
    origin="procedural_textures_v2_300_pack (gen_reaction_diffusion_and_turing; merges SP arch=8)",
    default_palette="bioluminescent_reef",
    description="Iterative periodic Difference-of-Gaussians Turing spots, coral ridges, and labyrinthine patterns.",
    render_fn=generate,
)
