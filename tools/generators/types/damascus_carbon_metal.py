"""Twisted Damascus steel, 2x2 twill carbon fiber, diamond-knurled, and brushed cast metal."""

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
    style = variant % 4
    micro = common.periodic_fbm(u, v, 8, 4, 0.5, rng)

    if style == 0:
        # Pattern-welded 1095/15N20 bilayer Damascus steel: forged raindrop/ladder/twist billet depth folded & differentially etched
        fold_k = 3 + (variant % 3)
        raindrop = 0.45 * np.sin(u * fold_k * 2.0 * math.pi) * np.sin(v * fold_k * 2.0 * math.pi)
        w = common.periodic_fbm(u, v, 3, 5, 0.55, rng, mode="turbulence")
        billet_z = (u * (4 + (variant % 2) * 2) + 2.4 * w + 1.15 * raindrop) * 2.0 * math.pi
        # Acid-etch step between darker martensitic steel and brighter nickel alloy
        layer_wave = np.sin(billet_z) + 0.28 * np.sin(3.0 * billet_z)
        bands = 0.5 + 0.5 * np.tanh(layer_wave * 3.2)
        fine_polish = common.periodic_gabor_noise(u, v, 14, 4.0, 0.15, 3.0, rng, impulses_per_cell=3)
        height = common.normalize_field(0.76 * bands + 0.12 * fine_polish + 0.12 * micro)
        color_field = common.normalize_field(0.80 * bands + 0.20 * micro)
        roughness = np.clip(0.14 + 0.34 * (1.0 - bands), 0.10, 0.55).astype(np.float32)
    elif style == 1:
        # Authentic 2x2 Z-twill carbon fiber tow weave (continuous multi-float arch) + 3K filament striations
        n = 16 + (variant % 4) * 4
        twill_22 = np.array(
            [[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1], [1, 0, 0, 1]],
            dtype=np.float32,
        )
        weave, warp_over, _, _ = common.woven_fabric_field(
            u, v, n, n, twill_22, float_arch=0.45, ply_twist=0.0, warp_radius=0.96, weft_radius=0.96
        )
        is_warp = warp_over.astype(np.float32)
        # Micro carbon-filament striations aligned along each tow's axis
        gu = (u * n) % 1.0
        gv = (v * n) % 1.0
        filaments_u = 0.5 + 0.5 * np.cos(gv * 14.0 * 2.0 * math.pi)
        filaments_v = 0.5 + 0.5 * np.cos(gu * 14.0 * 2.0 * math.pi)
        tow_filaments = np.where(warp_over, filaments_u, filaments_v)
        height = common.normalize_field(0.84 * weave + 0.09 * tow_filaments + 0.07 * micro)
        color_field = common.normalize_field(0.60 * weave + 0.26 * is_warp + 0.08 * tow_filaments + 0.06 * micro)
        roughness = (np.where(warp_over, 0.16, 0.34) + 0.06 * micro).astype(np.float32)
    elif style == 2:
        # ASME B94.6M 45-degree diamond knurling: truncated flat-topped frustum pyramids
        freq = 10 + (variant % 5) * 2
        d1 = np.abs(((u + v) * freq) % 1.0 - 0.5) * 2.0
        d2 = np.abs(((u - v) * freq) % 1.0 - 0.5) * 2.0
        pyramid = np.clip((1.0 - np.maximum(d1, d2)) * 1.18, 0.0, 1.0)
        height = common.normalize_field(0.84 * pyramid + 0.16 * micro)
        color_field = height
        roughness = np.clip(0.18 + 0.34 * (1.0 - pyramid), 0.10, 0.65).astype(np.float32)
    else:
        # Anisotropic brushed metal via Lagae-Lefebvre periodic Gabor noise (directional scratches) + casting porosity
        s1 = common.periodic_gabor_noise(u, v, 18, 6.0, 0.0, 2.2, rng, impulses_per_cell=4)
        s2 = common.periodic_gabor_noise(u, v, 10, 3.5, 0.04, 1.6, rng, impulses_per_cell=3)
        pits = common.periodic_voronoi(u, v, 12, 0.9, rng)[0]
        pit_mask = np.clip((pits - 0.20) / 0.14, 0.0, 1.0)
        height = common.normalize_field((0.60 * s1 + 0.25 * s2 + 0.15 * micro) * pit_mask)
        color_field = height
        roughness = np.clip(0.20 + 0.44 * (1.0 - height), 0.12, 0.78).astype(np.float32)

    stops = common.resolve_palette(palette, SPEC.default_palette, rng)
    albedo = common.apply_palette_5stop(color_field, micro, stops, contrast=1.18)
    return common.shade_pbr_surface(albedo, height, roughness=roughness, normal_strength=6.0, spec_intensity=0.38), np.asarray(height, dtype=np.float32)


def generate(seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0) -> Image.Image:
    return _compute(seed=seed, palette=palette, size=size, variant=variant)[0]


def generate_height(seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0) -> np.ndarray:
    return _compute(seed=seed, palette=palette, size=size, variant=variant)[1]

SPEC = GeneratorSpec(
    name="damascus-carbon-metal",
    kind="texture",
    tileable=True,
    origin="procedural_textures_v2_300_pack + substance_painter_master_600_pack (merges Damascus/carbon/knurl/brushed cast)",
    default_palette="folded_damascus_steel",
    description="Damascus steel (v0), 2x2 carbon weave (v1), diamond knurled metal (v2), and brushed cast metal (v3).",
    render_fn=generate,
    normal_source="native",
    normal_rule="Native physical surface height field (height) used by shade_pbr_surface.",
    normal_strength=6.0,
    height_fn=generate_height,
)
