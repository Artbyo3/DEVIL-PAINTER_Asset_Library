"""Deeply fissured tree bark, wind-rippled sand, and mossy forest floor."""

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
    style = variant % 3
    micro = common.periodic_fbm(u, v, 8, 4, 0.5, rng)

    if style == 0:
        # Anisotropic longitudinal conifer/redwood cork-cambium bark fissures (1:3 elongation) + lenticel streaks
        warp = common.periodic_fbm(u, v, 3, 4, 0.5, rng)
        bark_u = (u * (5 + variant % 4) + 0.14 * (warp - 0.5)) % 1.0
        bark_v = (v * 2.0 + 0.05 * (warp - 0.5)) % 1.0
        _, edge, cid, _, _ = common.periodic_lloyd_voronoi(bark_u, bark_v, 6, 0.88, rng, lloyd_iters=1)
        fissures = np.power(np.clip(edge * 3.4, 0.0, 1.0), 0.72)
        fiber_nap = common.periodic_gabor_noise(u, v, 14, 4.5, math.pi * 0.5, 2.2, rng, impulses_per_cell=3)
        height = common.normalize_field(0.66 * fissures + 0.18 * fiber_nap + 0.16 * micro)
        color_field = common.normalize_field(0.58 * height + 0.26 * cid + 0.16 * warp)
    elif style == 1:
        # Aeolian saltation ground sand ripples (gentle windward stoss 68%, steeper leeward 32%)
        warp = common.periodic_fbm(u, v, 3, 4, 0.5, rng)
        freq = 8 + (variant % 5)
        phase = (v * freq + 0.35 * warp) % 1.0
        asym = np.where(phase < 0.68, phase / 0.68, 1.0 - (phase - 0.68) / 0.32)
        ripples = asym * asym * (3.0 - 2.0 * asym)
        grains = common.periodic_fbm(u, v, freq * 3, 3, 0.55, rng)
        height = common.normalize_field(0.76 * ripples + 0.14 * grains + 0.10 * micro)
        color_field = common.normalize_field(0.70 * height + 0.30 * warp)
    else:
        # Bryophyte moss cushions: Lloyd-relaxed pillow domes + Swiss leafy tuft micro-relief
        f1, edge, cid, _, _ = common.periodic_lloyd_voronoi(u, v, 7 + (variant % 4), 0.90, rng, lloyd_iters=1)
        pillow = np.sqrt(np.clip(1.0 - f1 * 1.45, 0.0, 1.0))
        swiss = common.periodic_fbm(u, v, 6, 5, 0.56, rng, mode="swiss")
        height = common.normalize_field(0.46 * pillow + 0.38 * swiss + 0.16 * micro)
        color_field = common.normalize_field(0.48 * height + 0.34 * edge + 0.18 * cid)

    roughness = np.clip(0.60 + 0.30 * (1.0 - height), 0.4, 0.95).astype(np.float32)
    stops = common.resolve_palette(palette, SPEC.default_palette, rng)
    albedo = common.apply_palette_5stop(color_field, micro, stops, contrast=1.15)
    return common.shade_pbr_surface(albedo, height, roughness=roughness, normal_strength=6.2, spec_intensity=0.14), np.asarray(height, dtype=np.float32)


def generate(seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0) -> Image.Image:
    return _compute(seed=seed, palette=palette, size=size, variant=variant)[0]


def generate_height(seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0) -> np.ndarray:
    return _compute(seed=seed, palette=palette, size=size, variant=variant)[1]

SPEC = GeneratorSpec(
    name="biome-bark-foliage",
    kind="texture",
    tileable=True,
    origin="procedural_textures_v2_300_pack (gen_biome_ground_foliage_and_bark; merges SP arch=6)",
    default_palette="ancient_redwood_bark",
    description="Biome surfaces: anisotropic tree bark (v0), rippled ground sand (v1), and mossy forest floor (v2).",
    render_fn=generate,
    normal_source="native",
    normal_rule="Native physical surface height field (height) used by shade_pbr_surface.",
    normal_strength=6.2,
    height_fn=generate_height,
)
