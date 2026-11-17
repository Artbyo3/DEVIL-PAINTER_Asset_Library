"""Channel-packed PBR baking utility maps: ORM (AO/Rough/Curvature), Color ID, and Anisotropic Flow."""

from __future__ import annotations

import math
import numpy as np
from PIL import Image

from .. import common
from ..interface import GeneratorSpec, PaletteArg

COLOR_ID_PALETTE = np.array(
    [
        [235, 52, 52],
        [46, 204, 113],
        [52, 152, 219],
        [241, 196, 15],
        [155, 89, 182],
        [26, 188, 156],
        [230, 126, 34],
        [232, 67, 147],
    ],
    dtype=np.uint8,
)


def _base_surface_fields(
    u: np.ndarray, v: np.ndarray, rng: np.random.Generator, variant: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mode = variant % 3
    micro = common.periodic_fbm(u, v, 8, 4, 0.5, rng)
    if mode == 0:
        f1, edge, cid, _, _ = common.periodic_voronoi(u, v, 6 + (variant % 4) * 2, 0.85, rng)
        crack = np.clip(edge * 6.0, 0.0, 1.0)
        h = common.normalize_field(crack * (0.75 * (1.0 - f1) + 0.25 * micro))
        r = np.clip(0.30 + 0.55 * (1.0 - crack), 0.15, 0.95).astype(np.float32)
        return h, r, cid
    if mode == 1:
        swiss = common.periodic_fbm(u, v, 3 + (variant % 4), 5, 0.55, rng, mode="swiss")
        h = common.normalize_field(0.8 * swiss + 0.2 * micro)
        r = np.clip(0.45 + 0.40 * (1.0 - swiss), 0.2, 0.92).astype(np.float32)
        return h, r, np.floor(h * 5.99).astype(np.float32) / 5.0
    grid = 4 + (variant % 3) * 2
    cx = np.floor(u * grid).astype(np.int32) % grid
    cy = np.floor(v * grid).astype(np.int32) % grid
    lu, lv = (u * grid) % 1.0, (v * grid) % 1.0
    pmap = rng.random((grid, grid), dtype=np.float32)
    seam = np.clip((np.minimum(np.minimum(lu, 1.0 - lu), np.minimum(lv, 1.0 - lv)) - 0.04) * 30.0, 0.0, 1.0)
    h = common.normalize_field(seam * (0.75 * pmap[cy, cx] + 0.25 * micro))
    r = np.clip(0.25 + 0.45 * micro, 0.15, 0.85).astype(np.float32)
    return h, r, pmap[cy, cx]


def generate(seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0) -> Image.Image:
    rng = np.random.default_rng(seed)
    _ = palette
    u, v = common.make_uv_grid(size)
    style = variant % 3
    h, r, id_field = _base_surface_fields(u, v, rng, variant)
    _, ao, curv, nx, ny = common.compute_sobel_normals_and_baking(h, normal_strength=6.5)

    if style == 0:
        orm = np.stack([ao * 255.0, r * 255.0, curv * 255.0], axis=-1)
        return common.rgb_image(orm)
    if style == 1:
        idx = np.clip(np.floor(id_field * len(COLOR_ID_PALETTE)).astype(np.int32), 0, len(COLOR_ID_PALETTE) - 1)
        return Image.fromarray(COLOR_ID_PALETTE[idx], mode="RGB")

    pot = common.periodic_fbm(u, v, 3 + (variant % 4), 4, 0.5, rng)
    tx = -ny * 0.7 + np.cos(pot * 2.0 * math.pi) * 0.3
    ty = nx * 0.7 + np.sin(pot * 2.0 * math.pi) * 0.3
    tnorm = np.sqrt(tx * tx + ty * ty) + 1e-7
    tx, ty = tx / tnorm, ty / tnorm
    flow_rgb = np.stack([(tx * 0.5 + 0.5) * 255.0, (ty * 0.5 + 0.5) * 255.0, np.full_like(tx, 128.0)], axis=-1)
    return common.rgb_image(flow_rgb)


SPEC = GeneratorSpec(
    name="baking-utility-map",
    kind="texture",
    tileable=True,
    origin="substance_painter_master_600_pack/procedural_texture_generator.py (Suite 6 styles 0-2)",
    default_palette="cyber_bulkhead",
    description="Tileable baking utility maps: Channel-packed ORM (v0), Color Selection ID (v1), Anisotropic Flow Tangent (v2).",
    render_fn=generate,
)
