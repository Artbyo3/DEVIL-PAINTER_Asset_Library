"""Classic running-bond red clay brick masonry wall with weathered mortar joints."""

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

    rows = [8, 10, 12, 14][variant % 4]
    cols = max(2, rows // 2)

    r_idx = np.floor(v * rows).astype(np.int32) % rows
    stag = (r_idx % 2) * 0.5
    lu = (u * cols + stag) % 1.0
    lv = (v * rows) % 1.0
    c_idx = np.floor(u * cols + stag).astype(np.int32) % cols

    brick_rand = rng.uniform(0.0, 1.0, size=(rows, cols)).astype(np.float32)
    brick_id = brick_rand[r_idx, c_idx]

    clay = common.periodic_fbm(u, v, rows * 2, 4, 0.55, rng)
    pits = common.periodic_fbm((u + 0.43) % 1.0, (v + 0.19) % 1.0, rows * 3, 3, 0.5, rng)

    # Organic brick-edge chip/wear (scaled proportionally by aspect ratio cols/rows)
    edge_u = np.minimum(lu, 1.0 - lu) * (float(rows) / float(cols))
    edge_v = np.minimum(lv, 1.0 - lv)
    edge_d = np.minimum(edge_u, edge_v) + (clay - 0.5) * 0.032

    brick_mask = np.clip((edge_d - 0.065) / 0.030, 0.0, 1.0)
    brick_face = np.power(brick_mask, 0.35) * (0.76 + 0.16 * clay - 0.10 * np.clip((0.36 - pits) * 3.0, 0.0, 1.0))
    mortar_h = 0.16 + 0.10 * pits

    h = common.normalize_field(brick_mask * brick_face + (1.0 - brick_mask) * mortar_h)

    stops = np.array(common.resolve_palette(palette, SPEC.default_palette, rng), dtype=np.float32)
    # Brick albedo varies across stops[0..3]; mortar uses stops[4] softened into recessed grey-buff lime
    t_brick = np.clip(0.22 + 0.56 * brick_id + 0.24 * (clay - 0.5), 0.0, 1.0)
    b_rgb = np.where(
        t_brick[..., None] < 0.33,
        stops[0] * (1.0 - t_brick[..., None] / 0.33) + stops[1] * (t_brick[..., None] / 0.33),
        np.where(
            t_brick[..., None] < 0.68,
            stops[1] * (1.0 - (t_brick[..., None] - 0.33) / 0.35) + stops[2] * ((t_brick[..., None] - 0.33) / 0.35),
            stops[2] * (1.0 - (t_brick[..., None] - 0.68) / 0.32) + stops[3] * ((t_brick[..., None] - 0.68) / 0.32),
        ),
    )
    mortar_rgb = stops[4] * (0.86 + 0.16 * pits[..., None])
    albedo = mortar_rgb * (1.0 - brick_mask[..., None]) + b_rgb * brick_mask[..., None]
    rough = np.where(brick_mask > 0.35, 0.72 + 0.16 * clay, 0.90).astype(np.float32)

    return common.shade_pbr_surface(albedo, h, roughness=rough, normal_strength=6.2, spec_intensity=0.10), np.asarray(h, dtype=np.float32)


def generate(seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0) -> Image.Image:
    return _compute(seed=seed, palette=palette, size=size, variant=variant)[0]


def generate_height(seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0) -> np.ndarray:
    return _compute(seed=seed, palette=palette, size=size, variant=variant)[1]

SPEC = GeneratorSpec(
    name="red-brick-wall",
    kind="texture",
    tileable=True,
    origin="tools/generators/types/red_brick_wall.py",
    default_palette="classic_red_brick",
    description="Classic running-bond red clay brick masonry wall with weathered faces and recessed mortar joints.",
    render_fn=generate,
    normal_source="native",
    normal_rule="Native physical surface height field (h) used by shade_pbr_surface.",
    normal_strength=6.2,
    height_fn=generate_height,
)
