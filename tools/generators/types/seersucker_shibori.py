"""Puckered seersucker stripe and shibori tie-dye linen texture generator."""

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
    plain_m = np.array([[1, 0], [0, 1]], dtype=np.int32)

    if variant % 2 == 0:
        stripes = [6, 8, 10, 14][int(rng.integers(0, 4))]
        su = (u * stripes) % 1.0
        # Smoothly tapered tension transition across slack/tight stripe borders
        pucker_band = np.clip(0.5 + 0.5 * np.tanh(np.sin(su * 2.0 * math.pi) * 6.0), 0.0, 1.0)
        crinkle = common.periodic_fbm(u, v, stripes * 2, 4, 0.55, rng, freqy=stripes * 5)
        # Differential warp slack causes transverse compressive buckling along v in pucker bands
        buckle_freq = stripes * 3
        buckle = 0.5 + 0.5 * np.cos((v * buckle_freq + 0.45 * (crinkle - 0.5)) * 2.0 * math.pi)
        weave, _, _, _ = common.woven_fabric_field(u, v, stripes * 8, stripes * 8, plain_m, float_arch=0.36, ply_twist=0.10)
        pucker_rel = 0.60 * buckle + 0.40 * crinkle
        h = common.normalize_field(0.42 * weave + 0.58 * pucker_rel * (0.18 + 0.82 * pucker_band))
        c = common.normalize_field(0.55 * pucker_band + 0.28 * buckle + 0.17 * weave)
    else:
        warp = common.periodic_fbm(u, v, 3, 4, 0.55, rng)
        rings = [2, 3, 4][int(rng.integers(0, 3))]
        du = ((u + 0.10 * (warp - 0.5)) * rings) % 1.0 - 0.5
        dv = ((v + 0.10 * (warp - 0.5)) * rings) % 1.0 - 0.5
        r = np.sqrt(du * du + dv * dv) * 2.0
        ang = np.arctan2(dv, du)
        # Radial binding compression pleats + Fickian resist-dye diffusion profile
        pleats = 0.5 + 0.5 * np.cos(ang * 14.0 + 3.0 * warp)
        dye = np.clip(0.5 + 0.5 * np.cos(r * math.pi * float(rng.uniform(3.0, 6.0)) + warp * 4.0), 0.0, 1.0)
        linen, _, _, _ = common.woven_fabric_field(u, v, 64, 64, plain_m, float_arch=0.36, ply_twist=0.10)
        h = common.normalize_field(0.56 * linen + 0.24 * warp + 0.20 * pleats * np.exp(-r * 0.9))
        c = common.normalize_field(0.80 * dye + 0.20 * linen)

    stops = common.resolve_palette(palette, SPEC.default_palette, rng)
    albedo = common.apply_palette_5stop(c, None, stops, contrast=1.10)
    return common.shade_cloth_surface(albedo, h, normal_strength=4.8, sheen_amt=12.0), np.asarray(h, dtype=np.float32)


def generate(seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0) -> Image.Image:
    return _compute(seed=seed, palette=palette, size=size, variant=variant)[0]


def generate_height(seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0) -> np.ndarray:
    return _compute(seed=seed, palette=palette, size=size, variant=variant)[1]

SPEC = GeneratorSpec(
    name="seersucker-shibori",
    kind="texture",
    tileable=True,
    origin="p1_work/scripts/custom_generators.py (gen_cloth_seersucker_shibori)",
    default_palette="indigo_chambray",
    description="Puckered seersucker striped cotton (even variants) and indigo shibori tie-dye linen (odd variants).",
    render_fn=generate,
    normal_source="native",
    normal_rule="Native physical surface height field (h) used by shade_cloth_surface.",
    normal_strength=4.8,
    height_fn=generate_height,
)
