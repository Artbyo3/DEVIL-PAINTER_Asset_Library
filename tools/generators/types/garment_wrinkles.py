"""Garment tension folds, gathered pleats, linen crumples, and bias drape folds."""

from __future__ import annotations

import math
import numpy as np
from PIL import Image

from .. import common
from ..interface import GeneratorSpec, PaletteArg


def generate(seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0) -> Image.Image:
    rng = np.random.default_rng(seed)
    u, v = common.make_uv_grid(size)
    style = variant % 4
    w1 = common.periodic_fbm(u, v, 3, 4, 0.55, rng)
    w2 = common.periodic_fbm(u, v, 6, 4, 0.50, rng, mode="billow")

    if style == 0:
        folds = 4 + (variant % 5)
        phase = (v * folds + 0.28 * (w1 - 0.5) + 0.12 * np.sin(u * 2.0 * math.pi)) * 2.0 * math.pi
        pipe = np.power(0.5 + 0.5 * np.sin(phase), 0.65)
        h = common.normalize_field(0.75 * pipe + 0.25 * w2)
    elif style == 1:
        pleats = 6 + (variant % 6) * 2
        gather = np.power(
            0.5 + 0.5 * np.sin((u * pleats + 0.18 * w1 * np.sin(v * 2.0 * math.pi)) * 2.0 * math.pi),
            0.7,
        )
        h = common.normalize_field(0.80 * gather + 0.20 * w2)
    elif style == 2:
        swiss = common.periodic_fbm(u, v, 4 + (variant % 4), 5, 0.55, rng, mode="swiss")
        billow = common.periodic_fbm(u, v, 3 + (variant % 3), 5, 0.55, rng, mode="billow")
        h = common.normalize_field(0.55 * billow + 0.45 * swiss)
    else:
        freq = 3 + (variant % 4)
        diag = ((u + v) * freq + 0.22 * (w1 - 0.5)) * 2.0 * math.pi
        drape = 0.5 + 0.5 * np.sin(diag)
        h = common.normalize_field(0.72 * drape + 0.28 * w2)

    if palette is None and variant % 2 == 1:
        normal_rgb, _, _, _, _ = common.compute_sobel_normals_and_baking(h, normal_strength=7.5)
        return common.rgb_image(normal_rgb)

    stops = common.resolve_palette(palette, SPEC.default_palette, rng)
    albedo = common.apply_palette_5stop(h, w2, stops, contrast=1.08)
    return common.shade_cloth_surface(albedo, h, normal_strength=6.5, sheen_amt=14.0)


SPEC = GeneratorSpec(
    name="garment-wrinkles",
    kind="texture",
    tileable=True,
    origin="cloth_apparel_master_600_pack/procedural_texture_generator.py (Suite 3)",
    default_palette="heather_grey_jersey",
    description="Tension pipe folds, gathered waistband pleats, crumpled linen creases, and bias drape folds.",
    render_fn=generate,
)
