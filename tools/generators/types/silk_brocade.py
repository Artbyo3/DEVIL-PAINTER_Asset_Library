"""Mulberry silk satin damask and jacquard floral brocade texture generator."""

from __future__ import annotations

import math
import numpy as np
from PIL import Image

from .. import common
from ..interface import GeneratorSpec, PaletteArg


def generate(seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0) -> Image.Image:
    rng = np.random.default_rng(seed)
    u, v = common.make_uv_grid(size)

    freq = 2 + (variant % 3) * 2
    warp = common.periodic_fbm(u, v, freq * 2, 4, 0.5, rng)
    mu = np.abs(((u * freq) % 1.0) - 0.5) * 2.0
    mv = np.abs(((v * freq) % 1.0) - 0.5) * 2.0
    petal = np.cos((mu * 2.0 + 0.35 * warp) * math.pi) * np.cos((mv * 2.0 + 0.35 * warp) * math.pi)
    motif = np.clip((petal + 0.25) * 4.0, 0.0, 1.0)

    satin_u = np.sin((u * 48.0) * math.pi) * 0.5 + 0.5
    satin_v = np.sin((v * 48.0) * math.pi) * 0.5 + 0.5
    h = common.normalize_field(np.where(motif > 0.5, 0.65 + 0.35 * satin_u, 0.25 + 0.25 * satin_v))
    c = common.normalize_field(0.75 * motif + 0.25 * h)
    rough = np.where(motif > 0.5, 0.28, 0.14)
    stops = common.resolve_palette(palette, SPEC.default_palette, rng)
    albedo = common.apply_palette_5stop(c, warp, stops, contrast=1.20)
    return common.shade_pbr_surface(albedo, h, roughness=rough, normal_strength=5.2, spec_intensity=0.35)


SPEC = GeneratorSpec(
    name="silk-brocade",
    kind="texture",
    tileable=True,
    origin="cloth_apparel_master_600_pack/procedural_texture_generator.py (arch=7)",
    default_palette="crimson_mulberry_silk",
    description="Mulberry silk satin damask with raised floral brocade motif.",
    render_fn=generate,
)
