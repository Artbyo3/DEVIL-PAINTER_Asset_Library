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
        # Exact periodic 2:1 herringbone lattice on (grid x grid) with grid % 4 == 0
        grid = max(4, (scale // 2) * 4)
        s = u * grid
        t = v * grid
        is_ = np.floor(s).astype(np.int32)
        it_ = np.floor(t).astype(np.int32)
        fx = s - is_
        fy = t - it_
        res = (is_ - it_) % 4
        # res==0: left half of horiz 2x1; res==1: right half; res==3: bottom half of vert 1x2; res==2: top half
        h_mask = (res == 0) | (res == 1)
        ax = np.where(res == 1, is_ - 1, is_) % grid
        ay = np.where(res == 2, it_ - 1, it_) % grid
        local_long = np.where(
            res == 0,
            fx * 0.5,
            np.where(res == 1, 0.5 + fx * 0.5, np.where(res == 3, fy * 0.5, 0.5 + fy * 0.5)),
        )
        local_short = np.where(h_mask, fy, fx)
        # Scale long-axis edge distance by 2.0 so mortar thickness is physically isotropic
        edge_d = np.minimum(np.minimum(local_long, 1.0 - local_long) * 2.0, np.minimum(local_short, 1.0 - local_short))
        tile_id = (
            ((ax.astype(np.int64) * 73856093) ^ (ay.astype(np.int64) * 19349663) ^ np.int64((seed + variant) * 83492791))
            % 997
        ) / 997.0
    elif sub == 1:
        # BIA Technical Notes 10 US Modular running bond (~1:3 brick face aspect, uniform mortar width)
        rows = scale * 2
        cols = max(2, int(round(rows / 2.8)))
        r_idx = np.floor(v * rows).astype(np.int32)
        stag = (r_idx % 2) * 0.5
        lu = (u * cols + stag) % 1.0
        lv = (v * rows) % 1.0
        # Normalize both axes to physical row-height units so horizontal & vertical joints match width
        aspect = float(rows) / float(cols)
        edge_d = np.minimum(np.minimum(lu, 1.0 - lu) * aspect, np.minimum(lv, 1.0 - lv))
        c_idx = np.floor(u * cols + stag).astype(np.int32) % cols
        tile_id = (
            (((r_idx % rows).astype(np.int64) * 73856093) ^ (c_idx.astype(np.int64) * 19349663) ^ np.int64((seed + variant) * 83492791))
            % 997
        ) / 997.0
    else:
        # Exact 4.8.8 Archimedean tiling: regular octagons (edge s = sqrt(2)-1) + inset tilted squares at cell vertices
        p_u = (u * scale) % 1.0
        p_v = (v * scale) % 1.0
        cx = np.abs(p_u - 0.5)
        cy = np.abs(p_v - 0.5)
        d_oct = 0.5 - np.maximum(np.maximum(cx, cy), (cx + cy) * 0.70710678)
        # Corner diamond tile in vertex coordinates [-0.5, 0.5]^2
        vx = ((u * scale + 0.5) % 1.0) - 0.5
        vy = ((v * scale + 0.5) % 1.0) - 0.5
        d_sq = (1.0 - 0.70710678) - (np.abs(vx) + np.abs(vy)) * 0.70710678
        in_oct = d_oct >= d_sq
        edge_d = np.maximum(d_oct, d_sq) * 2.0
        iu = np.floor(u * scale).astype(np.int64) % scale
        iv = np.floor(v * scale).astype(np.int64) % scale
        vu = np.floor(u * scale + 0.5).astype(np.int64) % scale
        vv = np.floor(v * scale + 0.5).astype(np.int64) % scale
        oct_id = (((iu * 73856093) ^ (iv * 19349663)) % 997) / 997.0
        sq_id = ((((vu * 19349663) ^ (vv * 83492791)) + 313) % 997) / 997.0
        tile_id = np.where(in_oct, oct_id * 0.8 + 0.2, sq_id * 0.45)

    warp = (micro - 0.5) * 0.025
    t = np.clip((edge_d + warp - 0.045) / 0.028, 0.0, 1.0)
    mortar = t * t * (3.0 - 2.0 * t)
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
