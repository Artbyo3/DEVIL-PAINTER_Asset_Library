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
        # Lagae-Lefebvre (2009) periodic Gabor sparse-convolution filaments at skewed scratch directions
        accum = np.zeros_like(u, dtype=np.float32)
        for ang in (0.18, 1.32, 0.78, -0.68):
            g = common.periodic_gabor_noise(u, v, 14, 5.5, ang, 1.8, rng, impulses_per_cell=3)
            slash = np.clip((g - 0.68) * 4.5, 0.0, 1.0)
            accum = np.maximum(accum, slash)
        mask = common.normalize_field(0.68 * accum + 0.32 * base * ridged)
    elif style == 1:
        # Bridson (2007) curl-advected vertical drip streaks with gravity attenuation
        vx, _ = common.periodic_curl_noise(u, v, 4, 3, 0.5, rng)
        drip = common.periodic_fbm((u + 0.045 * vx) % 1.0, v, 14, 4, 0.55, rng, freqy=2)
        mask = common.normalize_field(np.power(drip, 1.45) * (0.68 + 0.32 * base))
    elif style == 2:
        f1, edge, _, _, _ = common.periodic_lloyd_voronoi(u, v, 6 + (variant % 6), 0.90, rng, lloyd_iters=1)
        score = base + (1.0 - f1) * 0.58 - 0.55
        t = np.clip(score / 0.22, 0.0, 1.0)
        peel = t * t * (3.0 - 2.0 * t)
        mask = common.normalize_field(0.75 * peel + 0.25 * edge)
    elif style == 3:
        _, edge1, _, _, _ = common.periodic_lloyd_voronoi(u, v, 5 + (variant % 5), 0.88, rng, lloyd_iters=1)
        _, edge2, _, _, _ = common.periodic_voronoi(u, v, 12, 0.90, rng)
        cracks = np.exp(-edge1 * 9.5) + 0.55 * np.exp(-edge2 * 11.0)
        mask = common.normalize_field(0.72 * cracks + 0.28 * ridged)
    else:
        swiss = common.periodic_fbm(u, v, 4 + (variant % 4), 6, 0.55, rng, mode="swiss")
        thresh = 0.35 + (variant % 4) * 0.08
        t = np.clip((swiss - thresh) / 0.28, 0.0, 1.0)
        raw = t * t * (3.0 - 2.0 * t)
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
