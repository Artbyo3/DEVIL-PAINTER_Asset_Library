"""Diamond quilted puffer jacket / matelasse padded nylon texture generator."""

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

    scale = 3 + (variant % 4) * 2
    d1 = ((u + v) * scale) % 1.0
    d2 = ((u - v) * scale) % 1.0
    ed = np.minimum(np.minimum(d1, 1.0 - d1), np.minimum(d2, 1.0 - d2))

    stitch_along = np.sin(((u - v) * scale * 12.0) * math.pi) > 0.2
    seam_groove = np.clip(ed * 10.0, 0.0, 1.0)
    puffer_pillow = np.power(seam_groove, 0.42)

    wrinkle = common.periodic_fbm(u, v, scale * 2, 4, 0.55, rng, freqy=scale * 6)
    puff_h = puffer_pillow * (0.82 + 0.18 * wrinkle) + (1.0 - seam_groove) * 0.22 * stitch_along

    h = common.normalize_field(puff_h)
    c = common.normalize_field(0.85 * h + 0.15 * slub)
    stops = common.resolve_palette(palette, SPEC.default_palette, rng)
    albedo = common.apply_palette_5stop(c, wrinkle, stops, contrast=1.15)
    return common.shade_pbr_surface(albedo, h, roughness=np.clip(0.22 + 0.35 * (1.0 - seam_groove), 0.15, 0.75), normal_strength=6.2, spec_intensity=0.30)


SPEC = GeneratorSpec(
    name="quilted-puffer",
    kind="texture",
    tileable=True,
    origin="cloth_apparel_master_600_pack/procedural_texture_generator.py (arch=5)",
    default_palette="charcoal_flannel",
    description="Diamond quilted puffer matelasse nylon with stitched seams and internal gather wrinkles.",
    render_fn=generate,
)
