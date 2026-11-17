"""Tileable Chantilly floral lace, fishnet stocking, athletic eyelet mesh, and ripped denim opacity masks."""

from __future__ import annotations

import math
import numpy as np
from PIL import Image

from .. import common
from ..interface import GeneratorSpec, PaletteArg


def generate_height(seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    _ = palette
    u, v = common.make_uv_grid(size)
    style = variant % 4
    scale = 4 + (variant % 5) * 2

    if style == 0:
        lu = (u * scale) % 1.0 - 0.5
        lv = (v * scale) % 1.0 - 0.5
        r = np.sqrt(lu * lu + lv * lv) * 2.0
        theta = np.arctan2(lv, lu)
        petals = 6 + (variant % 3) * 2
        rosette = np.abs(r - (0.45 + 0.18 * np.cos(theta * petals)))
        flower = np.exp(-(rosette * rosette) / 0.006) + np.exp(-((r - 0.18) ** 2) / 0.005)
        d1 = np.abs(((u + v) * scale * 4.0) % 1.0 - 0.5)
        d2 = np.abs(((u - v) * scale * 4.0) % 1.0 - 0.5)
        tulle = np.maximum(np.exp(-(d1 * d1) / 0.004), np.exp(-(d2 * d2) / 0.004)) * 0.45
        return np.clip(flower + tulle, 0.0, 1.0).astype(np.float32)

    if style == 1:
        freq = 8 + (variant % 6) * 4
        d1 = np.abs(((u + v) * freq) % 1.0 - 0.5) * 2.0
        d2 = np.abs(((u - v) * freq) % 1.0 - 0.5) * 2.0
        thickness = 0.12 + (variant % 3) * 0.04
        c1 = np.sqrt(np.clip(1.0 - (d1 / thickness) ** 2, 0.0, 1.0))
        c2 = np.sqrt(np.clip(1.0 - (d2 / thickness) ** 2, 0.0, 1.0))
        knots = np.sqrt(np.clip(1.0 - (d1 * d1 + d2 * d2) / (thickness * thickness * 2.2), 0.0, 1.0)) * 0.35
        return np.clip(np.maximum(c1, c2) * 0.85 + knots, 0.0, 1.0).astype(np.float32)

    if style == 2:
        cols = 10 + (variant % 5) * 4
        rows = cols
        r_idx = np.floor(v * rows)
        stag = (r_idx % 2) * 0.5
        lu = ((u * cols) + stag) % 1.0 - 0.5
        lv = (v * rows) % 1.0 - 0.5
        hole_r = np.sqrt((lu * 1.2) ** 2 + (lv * 0.85) ** 2)
        web = np.clip((hole_r - 0.28) / 0.06, 0.0, 1.0)
        knit = 0.85 + 0.15 * np.sin(u * cols * 4.0 * math.pi)
        return (web * knit).astype(np.float32)

    tear_noise = common.periodic_fbm(u, v, 3, 5, 0.55, rng)
    thresh = 0.36 + (variant % 3) * 0.05
    intact_cloth = np.clip((tear_noise - thresh) * 24.0, 0.0, 1.0)
    hole_mask = 1.0 - intact_cloth
    # Cylindrical spanning white weft yarns across the tear (zero binary thresholding)
    wf = ((v * 32.0 + (tear_noise - 0.5) * 2.4) % 1.0) - 0.5
    weft_strands = np.sqrt(np.clip(1.0 - (wf / 0.26) ** 2, 0.0, 1.0))
    fray_fuzz = np.exp(-((tear_noise - thresh) ** 2) / 0.0025) * common.periodic_gradient_perlin(u, v, 32, 32, rng)
    denim_twill = 0.88 + 0.12 * np.sin((u * 48.0 + v * 48.0) * math.pi)
    return np.clip(intact_cloth * denim_twill + hole_mask * weft_strands * 0.88 - 0.22 * fray_fuzz, 0.0, 1.0).astype(
        np.float32
    )


def generate(seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0) -> Image.Image:
    return common.alpha_image(generate_height(seed=seed, palette=palette, size=size, variant=variant))


SPEC = GeneratorSpec(
    name="lace-mesh-mask",
    kind="alpha",
    tileable=True,
    origin="cloth_apparel_master_600_pack/procedural_texture_generator.py (Suite 5)",
    default_palette="bridal_ivory_lace",
    description="Tileable opacity masks: Chantilly floral lace, fishnet, jersey eyelet mesh, and ripped denim tear.",
    render_fn=generate,
    normal_source="derived",
    normal_rule="Derived thread relief from opacity/tear cutout alpha mask field.",
    normal_strength=4.5,
    height_fn=generate_height,
)
