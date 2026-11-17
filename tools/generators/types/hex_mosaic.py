"""Glazed hexagonal ceramic mosaic tile with recessed mortar grout lines."""

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
    cols = [6, 8, 10, 14][(variant + int(rng.integers(0, 4))) % 4]
    # Even row count guarantees periodic half-cell stagger wraps seamlessly; aspect matched to sqrt(3)/2
    rows = max(2, (int(round(cols / 0.8660254)) // 2) * 2)
    scale_y = float(cols) / float(rows)  # ~0.8660254

    px = u * cols
    py = v * cols  # isotropic metric vertically scaled by rows*scale_y == cols
    r_base = np.floor(v * rows).astype(np.int32)
    c_base = np.floor(px - 0.5 * (r_base % 2)).astype(np.int32)

    d1 = np.full_like(px, 1e9, dtype=np.float32)
    d2 = np.full_like(px, 1e9, dtype=np.float32)
    best_c = np.zeros_like(c_base, dtype=np.int32)
    best_r = np.zeros_like(r_base, dtype=np.int32)

    for dr in (-1, 0, 1):
        rr = r_base + dr
        stagger = 0.5 * (rr % 2)
        cy = rr * scale_y + 0.5 * scale_y
        for dc in (-1, 0, 1, 2):
            cc = c_base + dc
            cx = cc + stagger + 0.5
            dist = np.hypot(px - cx, py - cy)
            nearer1 = dist < d1
            d2 = np.where(nearer1, d1, np.minimum(d2, dist))
            d1 = np.where(nearer1, dist, d1)
            best_c = np.where(nearer1, cc % cols, best_c)
            best_r = np.where(nearer1, rr % rows, best_r)

    # Exact perpendicular distance to regular hexagonal Voronoi edge = 0.5 * (d2 - d1) / (cx-c' vector approx) -> true metric boundary
    edge_d = d2 - d1
    t_g = np.clip((edge_d - 0.055) / 0.035, 0.0, 1.0)
    grout = t_g * t_g * (3.0 - 2.0 * t_g)
    tile_rand = (
        (
            (best_c.astype(np.int64) * 73856093)
            ^ (best_r.astype(np.int64) * 19349663)
            ^ np.int64((seed + variant) * 83492791)
        )
        % 997
    ) / 997.0

    glaze = common.periodic_fbm(u, v, cols, 3, 0.5, rng)
    pillow = np.sqrt(np.clip(edge_d * 1.1, 0.0, 0.5))
    h = common.normalize_field(grout * (0.66 + 0.24 * pillow) + 0.10 * glaze)
    c = common.normalize_field((1.0 - grout) * 0.12 + grout * (0.25 + 0.55 * tile_rand.astype(np.float32) + 0.20 * glaze))

    stops = common.resolve_palette(palette, SPEC.default_palette, rng)
    albedo = common.apply_palette_5stop(c, glaze, stops, contrast=1.15)
    rough = np.where(grout > 0.2, 0.16, 0.85).astype(np.float32)
    return common.shade_pbr_surface(albedo, h, roughness=rough, normal_strength=5.5, spec_intensity=0.25), np.asarray(h, dtype=np.float32)


def generate(seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0) -> Image.Image:
    return _compute(seed=seed, palette=palette, size=size, variant=variant)[0]


def generate_height(seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0) -> np.ndarray:
    return _compute(seed=seed, palette=palette, size=size, variant=variant)[1]

SPEC = GeneratorSpec(
    name="hex-mosaic",
    kind="texture",
    tileable=True,
    origin="p1_work/scripts/custom_generators.py (gen_v1_hex_mosaic)",
    default_palette="moroccan_zellige",
    description="Beveled hexagonal ceramic mosaic tiles with per-tile glaze color variation and recessed grout.",
    render_fn=generate,
    normal_source="native",
    normal_rule="Native physical surface height field (h) used by shade_pbr_surface.",
    normal_strength=5.5,
    height_fn=generate_height,
)
