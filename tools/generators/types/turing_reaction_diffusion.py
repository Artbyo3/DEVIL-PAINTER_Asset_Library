"""Multi-scale Difference-of-Gaussians Turing reaction-diffusion coral and labyrinth patterns."""

from __future__ import annotations

import numpy as np
from PIL import Image

from .. import common
from ..interface import GeneratorSpec, PaletteArg


def _compute(
    seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0
) -> tuple[Image.Image, np.ndarray]:
    rng = np.random.default_rng(seed)
    u, v = common.make_uv_grid(size)
    # Pearson (1993, Science 261) canonical Gray-Scott (F, k) morphology regimes:
    # 0: Mitosis/Spots (0.0367, 0.0649), 1: Coral/Brain (0.0545, 0.0620),
    # 2: Labyrinth/Maze (0.0290, 0.0570), 3: Soliton/Ribbons (0.0420, 0.0600)
    regimes = [
        (0.0367, 0.0649, 460),
        (0.0545, 0.0620, 440),
        (0.0290, 0.0570, 420),
        (0.0420, 0.0600, 440),
    ]
    feed, kill, steps = regimes[variant % 4]
    sim_res = 96 if size <= 256 else 128
    rd_u, rd_v = common.gray_scott_rd(size, feed=feed, kill=kill, steps=steps, rng=rng, sim_res=sim_res)
    micro = common.periodic_fbm(u, v, 10 + (variant % 4) * 2, 3, 0.5, rng)

    t = np.clip((rd_v - 0.18) / 0.58, 0.0, 1.0)
    ridges = t * t * (3.0 - 2.0 * t)
    height = common.normalize_field(0.84 * ridges + 0.16 * micro)
    color_field = common.normalize_field(0.72 * rd_v + 0.14 * (1.0 - rd_u) + 0.14 * micro)
    roughness = np.clip(0.20 + 0.52 * (1.0 - ridges), 0.14, 0.85).astype(np.float32)

    stops = common.resolve_palette(palette, SPEC.default_palette, rng)
    albedo = common.apply_palette_5stop(color_field, micro, stops, contrast=1.20)
    return common.shade_pbr_surface(albedo, height, roughness=roughness, normal_strength=6.0, spec_intensity=0.28), np.asarray(height, dtype=np.float32)


def generate(seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0) -> Image.Image:
    return _compute(seed=seed, palette=palette, size=size, variant=variant)[0]


def generate_height(seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0) -> np.ndarray:
    return _compute(seed=seed, palette=palette, size=size, variant=variant)[1]

SPEC = GeneratorSpec(
    name="turing-reaction-diffusion",
    kind="texture",
    tileable=True,
    origin="procedural_textures_v2_300_pack (gen_reaction_diffusion_and_turing; merges SP arch=8)",
    default_palette="bioluminescent_reef",
    description="Iterative periodic Difference-of-Gaussians Turing spots, coral ridges, and labyrinthine patterns.",
    render_fn=generate,
    normal_source="native",
    normal_rule="Native physical surface height field (height) used by shade_pbr_surface.",
    normal_strength=6.0,
    height_fn=generate_height,
)
