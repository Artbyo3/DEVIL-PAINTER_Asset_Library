"""Twisted Damascus steel, 2x2 twill carbon fiber, diamond-knurled, and brushed cast metal."""

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
    micro = common.periodic_fbm(u, v, 8, 4, 0.5, rng)

    if style == 0:
        twist = (
            0.5
            + 0.5
            * np.sin(u * (3 + variant % 3) * 2.0 * math.pi)
            * np.sin(v * (3 + variant % 3) * 2.0 * math.pi)
        )
        w = common.periodic_fbm(u, v, 4, 5, 0.55, rng, mode="turbulence")
        bands = 0.5 + 0.5 * np.sin((u * 4.0 + 2.8 * w + 1.4 * twist) * 2.0 * math.pi)
        height = common.normalize_field(0.78 * bands + 0.22 * micro)
        color_field = height
        roughness = np.clip(0.15 + 0.35 * (1.0 - bands), 0.1, 0.6).astype(np.float32)
    elif style == 1:
        n = 16 + (variant % 4) * 4
        tu, tv = u * n, v * n
        iu, iv = np.floor(tu).astype(np.int32), np.floor(tv).astype(np.int32)
        fu, fv = tu - iu, tv - iv
        warp_over = (((iu + iv) // 2) % 2) == 0
        strand_u = np.sin(fu * math.pi) * (0.85 + 0.15 * np.sin(fv * 6.0 * math.pi))
        strand_v = np.sin(fv * math.pi) * (0.85 + 0.15 * np.sin(fu * 6.0 * math.pi))
        weave = np.where(warp_over, strand_u, strand_v * 0.85)
        height = common.normalize_field(0.88 * weave + 0.12 * micro)
        color_field = common.normalize_field(0.65 * weave + 0.25 * warp_over.astype(np.float32) + 0.10 * micro)
        roughness = (np.where(warp_over, 0.18, 0.34) + 0.08 * micro).astype(np.float32)
    elif style == 2:
        freq = 10 + (variant % 5) * 2
        d1 = np.abs(((u + v) * freq) % 1.0 - 0.5) * 2.0
        d2 = np.abs(((u - v) * freq) % 1.0 - 0.5) * 2.0
        pyramid = 1.0 - np.maximum(d1, d2)
        height = common.normalize_field(0.82 * pyramid + 0.18 * micro)
        color_field = height
        roughness = np.clip(0.20 + 0.30 * (1.0 - pyramid), 0.12, 0.65).astype(np.float32)
    else:
        s1 = common.periodic_fbm(u, v, 4, 4, 0.5, rng, freqy=32)
        s2 = common.periodic_fbm(u, v, 32, 4, 0.5, rng, freqy=4)
        pits = common.periodic_voronoi(u, v, 12, 0.9, rng)[0]
        pit_mask = np.clip((pits - 0.22) / 0.15, 0.0, 1.0)
        height = common.normalize_field((0.65 * s1 + 0.20 * s2 + 0.15 * micro) * pit_mask)
        color_field = height
        roughness = np.clip(0.22 + 0.45 * (1.0 - height), 0.12, 0.80).astype(np.float32)

    stops = common.resolve_palette(palette, SPEC.default_palette, rng)
    albedo = common.apply_palette_5stop(color_field, micro, stops, contrast=1.18)
    return common.shade_pbr_surface(albedo, height, roughness=roughness, normal_strength=6.0, spec_intensity=0.38)


SPEC = GeneratorSpec(
    name="damascus-carbon-metal",
    kind="texture",
    tileable=True,
    origin="procedural_textures_v2_300_pack + substance_painter_master_600_pack (merges Damascus/carbon/knurl/brushed cast)",
    default_palette="folded_damascus_steel",
    description="Damascus steel (v0), 2x2 carbon weave (v1), diamond knurled metal (v2), and brushed cast metal (v3).",
    render_fn=generate,
)
