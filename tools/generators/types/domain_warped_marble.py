"""Nested domain-warped marble, mineral veins, and psychedelic plasma texture."""

from __future__ import annotations

import math
import numpy as np
from PIL import Image

from .. import common
from ..interface import GeneratorSpec, PaletteArg


def generate(seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0) -> Image.Image:
    rng = np.random.default_rng(seed)
    u, v = common.make_uv_grid(size)
    freq = 2 + (variant % 4)
    q_u = common.periodic_fbm(u, v, freq, 4, 0.55, rng, mode="fbm")
    q_v = common.periodic_fbm(u, v, freq, 4, 0.55, rng, mode="ridged")

    u1 = (u + 0.14 * (q_u - 0.5)) % 1.0
    v1 = (v + 0.14 * (q_v - 0.5)) % 1.0
    r_u = common.periodic_fbm(u1, v1, freq * 2, 4, 0.5, rng, mode="turbulence")
    r_v = common.periodic_fbm(u1, v1, freq * 2, 4, 0.5, rng, mode="fbm")

    u2 = (u + 0.22 * (r_u - 0.5)) % 1.0
    v2 = (v + 0.22 * (r_v - 0.5)) % 1.0
    final = common.periodic_fbm(u2, v2, freq, 5, 0.55, rng, mode="fbm")
    veins = 0.5 + 0.5 * np.sin((final * (2.0 + variant % 3) + q_u) * 2.0 * math.pi)

    color_field = common.normalize_field(0.65 * final + 0.35 * veins)
    height = common.normalize_field(0.75 * color_field + 0.25 * r_u)
    roughness = np.clip(0.25 + 0.45 * (1.0 - veins), 0.1, 0.9).astype(np.float32)

    stops = common.resolve_palette(palette, SPEC.default_palette, rng)
    albedo = common.apply_palette_5stop(color_field, r_v, stops, contrast=1.18)
    return common.shade_pbr_surface(albedo, height, roughness=roughness, normal_strength=5.5, spec_intensity=0.28)


SPEC = GeneratorSpec(
    name="domain-warped-marble",
    kind="texture",
    tileable=True,
    origin="procedural_textures_v2_300_pack (gen_iq_nested_domain_warp; merges V1 marble/clouds & SP arch=7)",
    default_palette="carrara_gold_marble",
    description="Dual-stage IQ nested domain warping for Carrara/Portoro marble veins and mineral swirls.",
    render_fn=generate,
)
