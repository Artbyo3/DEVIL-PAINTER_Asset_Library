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
    rows = max(2, int(round(cols * 1.1547)) // 2 * 2)

    px = u * cols
    py = v * rows
    row_i = np.floor(py).astype(np.int32)
    px_shift = px + (row_i % 2) * 0.5
    col_i = np.floor(px_shift).astype(np.int32)
    fx = (px_shift - col_i) - 0.5
    fy = (py - row_i) - 0.5

    dist = np.maximum(np.abs(fx), np.abs(fx) * 0.5 + np.abs(fy) * 0.866)
    grout = np.clip((0.46 - dist) * 24.0, 0.0, 1.0)
    tile_rand = (
        (
            ((col_i % cols).astype(np.int64) * 73856093)
            ^ ((row_i % rows).astype(np.int64) * 19349663)
            ^ np.int64((seed + variant) * 83492791)
        )
        % 997
    ) / 997.0

    glaze = common.periodic_fbm(u, v, cols, 3, 0.5, rng)
    h = common.normalize_field(grout * (0.65 + 0.25 * np.sqrt(np.clip(0.5 - dist, 0.0, 0.5))) + 0.10 * glaze)
    c = common.normalize_field(np.where(grout > 0.2, 0.25 + 0.55 * tile_rand.astype(np.float32) + 0.20 * glaze, 0.12))

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
