"""Diamond quilted puffer jacket / matelasse padded nylon texture generator."""

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
    slub = common.periodic_fbm(u, v, 4, 3, 0.5, rng, freqy=32)

    scale = 3 + (variant % 4) * 2
    d1 = ((u + v) * scale) % 1.0
    d2 = ((u - v) * scale) % 1.0
    ed = np.minimum(np.minimum(d1, 1.0 - d1), np.minimum(d2, 1.0 - d2))

    # C1-smooth doubly-periodic Poisson/Helmholtz inflated pillow dome inside each diamond
    pillow_base = np.sin(d1 * math.pi) * np.sin(d2 * math.pi)
    puffer_pillow = np.power(np.clip(pillow_base, 0.0, 1.0), 0.52)

    # Seam trench and radial tension gather folds radiating inward perpendicular to quilt seams
    seam_groove = np.clip(ed * 12.0, 0.0, 1.0)
    warp_w = common.periodic_fbm(u, v, scale * 2, 3, 0.5, rng)
    fold_wave1 = np.cos((d1 * 18.0 + 1.8 * warp_w) * math.pi)
    fold_wave2 = np.cos((d2 * 18.0 + 1.8 * warp_w) * math.pi)
    gather_folds = (
        np.exp(-np.minimum(d1, 1.0 - d1) * 7.0) * fold_wave2
        + np.exp(-np.minimum(d2, 1.0 - d2) * 7.0) * fold_wave1
    ) * 0.075

    # Lockstitch thread dashes centered inside seam channels
    s1 = np.exp(-((np.minimum(d1, 1.0 - d1)) ** 2) / 0.00045) * np.clip(np.sin(d2 * 16.0 * math.pi), 0.0, 1.0)
    s2 = np.exp(-((np.minimum(d2, 1.0 - d2)) ** 2) / 0.00045) * np.clip(np.sin(d1 * 16.0 * math.pi), 0.0, 1.0)
    stitches = np.maximum(s1, s2)

    wrinkle = common.periodic_fbm(u, v, scale * 2, 4, 0.55, rng, freqy=scale * 6)
    puff_h = puffer_pillow * (0.86 + 0.14 * wrinkle) + gather_folds * seam_groove + 0.16 * stitches

    h = common.normalize_field(puff_h)
    c = common.normalize_field(0.84 * h + 0.16 * slub)
    stops = common.resolve_palette(palette, SPEC.default_palette, rng)
    albedo = common.apply_palette_5stop(c, wrinkle, stops, contrast=1.15)
    return common.shade_pbr_surface(albedo, h, roughness=np.clip(0.22 + 0.35 * (1.0 - seam_groove), 0.15, 0.75), normal_strength=6.2, spec_intensity=0.30), np.asarray(h, dtype=np.float32)


def generate(seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0) -> Image.Image:
    return _compute(seed=seed, palette=palette, size=size, variant=variant)[0]


def generate_height(seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0) -> np.ndarray:
    return _compute(seed=seed, palette=palette, size=size, variant=variant)[1]


SPEC = GeneratorSpec(
    name="quilted-puffer",
    kind="texture",
    tileable=True,
    origin="cloth_apparel_master_600_pack/procedural_texture_generator.py (arch=5)",
    default_palette="charcoal_flannel",
    description="Diamond quilted puffer matelasse nylon with stitched seams and internal gather wrinkles.",
    render_fn=generate,
    normal_source="native",
    normal_rule="Native physical surface height field (h) used by shade_pbr_surface.",
    normal_strength=6.2,
    height_fn=generate_height,
)
