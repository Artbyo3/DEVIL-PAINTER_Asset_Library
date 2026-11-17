"""Architectural herringbone brickwork, interlocking pavers, and glazed Moroccan zellige tiles."""

from __future__ import annotations

import numpy as np
from PIL import Image

from .. import common
from ..interface import GeneratorSpec, PaletteArg


def _compute(
    seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0
) -> tuple[Image.Image, np.ndarray]:
    rng = np.random.default_rng(seed)
    u, v = common.make_uv_grid(size)
    scale = 4 + (variant % 5) * 2
    sub = variant % 3
    micro = common.periodic_fbm(u, v, scale * 2, 4, 0.5, rng)

    if sub == 0:
        s = u * scale
        t = v * scale
        is_ = np.floor(s).astype(np.int32)
        it_ = np.floor(t).astype(np.int32)
        zig = (is_ + it_) % 2
        local_u = np.where(zig == 0, (s * 0.5) % 1.0, (t * 0.5) % 1.0)
        local_v = np.where(zig == 0, t % 1.0, s % 1.0)
        tile_id = (((is_ % scale) * 3 + (it_ % scale) * 5) % 7) / 7.0
    elif sub == 1:
        rows = scale
        cols = max(2, scale // 2)
        r_idx = np.floor(v * rows).astype(np.int32)
        stag = (r_idx % 2) * 0.5
        local_u = (u * cols + stag) % 1.0
        local_v = (v * rows) % 1.0
        c_idx = np.floor(u * cols + stag).astype(np.int32) % cols
        tile_id = (((r_idx % rows) * 13 + c_idx * 7) % 11) / 11.0
    else:
        local_u = (u * scale) % 1.0
        local_v = (v * scale) % 1.0
        cx = np.abs(local_u - 0.5)
        cy = np.abs(local_v - 0.5)
        oct_edge = 0.5 - np.maximum(np.maximum(cx, cy), (cx + cy) * 0.7071)
        local_u = np.clip(oct_edge * 2.0, 0.0, 1.0)
        local_v = local_u
        iu = np.floor(u * scale).astype(np.int32) % scale
        iv = np.floor(v * scale).astype(np.int32) % scale
        tile_id = ((iu * 5 + iv * 9) % 8) / 8.0

    edge_d = np.minimum(np.minimum(local_u, 1.0 - local_u), np.minimum(local_v, 1.0 - local_v))
    warp = (micro - 0.5) * 0.025
    mortar = np.clip((edge_d + warp - 0.045) / 0.025, 0.0, 1.0)
    pillow_dome = np.power(mortar, 0.45)

    height = common.normalize_field(pillow_dome * (0.82 + 0.18 * micro))
    color_field = common.normalize_field(mortar * (0.55 * tile_id.astype(np.float32) + 0.45 * micro))
    roughness = np.where(mortar > 0.2, 0.18 + 0.22 * micro, 0.88).astype(np.float32)

    stops = common.resolve_palette(palette, SPEC.default_palette, rng)
    albedo = common.apply_palette_5stop(color_field, micro, stops, contrast=1.18)
    return common.shade_pbr_surface(albedo, height, roughness=roughness, normal_strength=6.0, spec_intensity=0.28), np.asarray(height, dtype=np.float32)


def generate(seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0) -> Image.Image:
    return _compute(seed=seed, palette=palette, size=size, variant=variant)[0]


def generate_height(seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0) -> np.ndarray:
    return _compute(seed=seed, palette=palette, size=size, variant=variant)[1]

SPEC = GeneratorSpec(
    name="masonry-tiles",
    kind="texture",
    tileable=True,
    origin="procedural_textures_v2_300_pack (gen_herringbone_and_moroccan_tiles; merges V1 masonry & SP arch=3)",
    default_palette="moroccan_zellige_cobalt",
    description="Herringbone pavers, running-bond brick masonry, and octagonal Moroccan zellige tiles.",
    render_fn=generate,
    normal_source="native",
    normal_rule="Native physical surface height field (height) used by shade_pbr_surface.",
    normal_strength=6.0,
    height_fn=generate_height,
)
