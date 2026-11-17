"""Tileable grunge, surface scratches, drip streaks, paint peel, and crack dirt masks."""

from __future__ import annotations

import numpy as np
from PIL import Image

from .. import common
from ..interface import GeneratorSpec, PaletteArg


def _compute_mask(seed: int = 1, size: int = 512, variant: int = 0) -> tuple[np.random.Generator, np.ndarray]:
    rng = np.random.default_rng(seed)
    u, v = common.make_uv_grid(size)
    style = variant % 5
    base = common.periodic_fbm(u, v, 3 + (variant % 4), 5, 0.58, rng)
    ridged = common.periodic_fbm(u, v, 4 + (variant % 5), 5, 0.55, rng, mode="ridged")

    if style == 0:
        # Integer-lattice shear directions ensure crisp scratches wrap cleanly across tile borders.
        accum = np.zeros_like(u, dtype=np.float32)
        dirs = ((1, 0), (0, 1), (1, 1), (1, -1))
        for k in range(4):
            sx, sy = dirs[k]
            ru = (u * sx + v * sy) % 1.0
            rv = (u * (-sy) + v * sx) % 1.0
            streak = common.periodic_gradient_perlin(ru, rv, 4, 48, rng)
            slash = np.clip((streak - 0.74) * 5.0, 0.0, 1.0)
            accum = np.maximum(accum, slash)
        mask = common.normalize_field(0.65 * accum + 0.35 * base * ridged)
    elif style == 1:
        drip_warp = common.periodic_fbm(u, v, 8, 4, 0.5, rng, freqy=2)
        drip = common.periodic_fbm((u + 0.04 * (drip_warp - 0.5)) % 1.0, v, 16, 4, 0.55, rng, freqy=2)
        mask = common.normalize_field(np.power(drip, 1.4) * (0.7 + 0.3 * base))
    elif style == 2:
        f1, edge, _, _, _ = common.periodic_voronoi(u, v, 6 + (variant % 6), 0.9, rng)
        peel = np.clip((base + (1.0 - f1) * 0.6 - 0.55) * 4.5, 0.0, 1.0)
        mask = common.normalize_field(0.75 * peel + 0.25 * edge)
    elif style == 3:
        _, edge1, _, _, _ = common.periodic_voronoi(u, v, 5 + (variant % 5), 0.85, rng)
        _, edge2, _, _, _ = common.periodic_voronoi(u, v, 12, 0.9, rng)
        cracks = np.power(1.0 - edge1, 3.5) + 0.5 * np.power(1.0 - edge2, 4.0)
        mask = common.normalize_field(0.70 * cracks + 0.30 * ridged)
    else:
        swiss = common.periodic_fbm(u, v, 4 + (variant % 4), 6, 0.55, rng, mode="swiss")
        thresh = 0.35 + (variant % 4) * 0.08
        raw = np.clip((swiss - thresh) / 0.28, 0.0, 1.0)
        mask = common.normalize_field(0.80 * raw + 0.20 * base)
    return rng, mask


def generate(seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0) -> Image.Image:
    rng, mask = _compute_mask(seed=seed, size=size, variant=variant)
    if palette is not None:
        stops = common.resolve_palette(palette, SPEC.default_palette, rng)
        return common.rgb_image(common.apply_palette_5stop(mask, None, stops))
    return common.alpha_image(mask)


def generate_height(seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0) -> np.ndarray:
    _ = palette
    return _compute_mask(seed=seed, size=size, variant=variant)[1]


SPEC = GeneratorSpec(
    name="grunge-wear-mask",
    kind="texture",
    tileable=True,
    origin="substance_painter_master_600_pack/procedural_texture_generator.py (Suite 3; seamless scratch shear)",
    default_palette="charcoal_flannel",
    description="Tileable surface grunge masks: scratches (v0), drips (v1), corrosion peel (v2), cracks (v3), smudge breakup (v4).",
    render_fn=generate,
    normal_source="derived",
    normal_rule="Derived relief height directly from the 2D grunge/scratch/drip/peel/crack wear mask.",
    normal_strength=5.5,
    height_fn=generate_height,
)
