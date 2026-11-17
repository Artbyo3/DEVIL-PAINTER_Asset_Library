"""Tactical crosshatch ripstop nylon texture generator."""

from __future__ import annotations

import math
import numpy as np
from PIL import Image

from .. import common
from ..interface import GeneratorSpec, PaletteArg


def generate(seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0) -> Image.Image:
    rng = np.random.default_rng(seed)
    u, v = common.make_uv_grid(size)
    slub = common.periodic_fbm(u, v, 4, 3, 0.5, rng, freqy=32)

    cells = 8 + (variant % 5) * 4
    gu = (u * cells) % 1.0
    gv = (v * cells) % 1.0
    rib_u = np.exp(-((gu - 0.5) ** 2) / 0.008)
    rib_v = np.exp(-((gv - 0.5) ** 2) / 0.008)
    ripstop_grid = np.maximum(rib_u, rib_v)

    plain_weave = 0.5 + 0.5 * np.sin(u * cells * 8.0 * math.pi) * np.sin(v * cells * 8.0 * math.pi)
    h = common.normalize_field(0.65 * ripstop_grid + 0.35 * plain_weave)
    c = common.normalize_field(0.60 * h + 0.40 * slub)
    rough = np.clip(0.28 + 0.30 * (1.0 - ripstop_grid), 0.2, 0.65)
    stops = common.resolve_palette(palette, SPEC.default_palette, rng)
    albedo = common.apply_palette_5stop(c, slub, stops, contrast=1.18)
    return common.shade_pbr_surface(albedo, h, roughness=rough, normal_strength=5.4, spec_intensity=0.25)


SPEC = GeneratorSpec(
    name="ripstop-nylon",
    kind="texture",
    tileable=True,
    origin="cloth_apparel_master_600_pack/procedural_texture_generator.py (arch=9) + v2 ripstop",
    default_palette="tactical_coyote_ripstop",
    description="Tactical ripstop nylon with raised reinforcement grid over plain weave.",
    render_fn=generate,
)
