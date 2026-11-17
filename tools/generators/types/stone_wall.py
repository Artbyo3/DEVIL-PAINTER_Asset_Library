"""Stacked rough-hewn coursed stone masonry wall with recessed mortar gaps."""

from __future__ import annotations

import numpy as np
from PIL import Image

from .. import common
from ..interface import GeneratorSpec, PaletteArg


def _periodic_rect_voronoi(
    u: np.ndarray,
    v: np.ndarray,
    nx: int,
    ny: int,
    jitter_x: float,
    jitter_y: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    px = rng.uniform(0.5 - 0.5 * jitter_x, 0.5 + 0.5 * jitter_x, size=(ny, nx)).astype(np.float32)
    py = rng.uniform(0.5 - 0.5 * jitter_y, 0.5 + 0.5 * jitter_y, size=(ny, nx)).astype(np.float32)
    cid_map = rng.uniform(0.0, 1.0, size=(ny, nx)).astype(np.float32)
    tx_map = rng.uniform(-0.22, 0.22, size=(ny, nx)).astype(np.float32)
    ty_map = rng.uniform(-0.22, 0.22, size=(ny, nx)).astype(np.float32)

    gx = u * nx
    gy = v * ny
    ix = np.floor(gx).astype(np.int32)
    iy = np.floor(gy).astype(np.int32)

    f1 = np.full_like(u, 1e9, dtype=np.float32)
    f2 = np.full_like(u, 1e9, dtype=np.float32)
    cid = np.zeros_like(u, dtype=np.float32)
    cleavage = np.zeros_like(u, dtype=np.float32)

    aspect = float(nx) / float(ny)
    for oy in (-1, 0, 1):
        for ox in (-1, 0, 1):
            cx = (ix + ox) % nx
            cy = (iy + oy) % ny
            fx = ((ix + ox) + px[cy, cx] - gx) / aspect
            fy = (iy + oy) + py[cy, cx] - gy
            dist = np.sqrt(fx * fx + fy * fy)
            closer1 = dist < f1
            between = (~closer1) & (dist < f2)
            f2 = np.where(closer1, f1, np.where(between, dist, f2))
            f1 = np.where(closer1, dist, f1)
            cid = np.where(closer1, cid_map[cy, cx], cid)
            cleavage = np.where(closer1, tx_map[cy, cx] * fx + ty_map[cy, cx] * fy, cleavage)

    edge = np.clip(f2 - f1, 0.0, 1.0)
    return common.normalize_field(f1), common.normalize_field(edge), cid, cleavage


def _compute(
    seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0
) -> tuple[Image.Image, np.ndarray]:
    rng = np.random.default_rng(seed)
    u, v = common.make_uv_grid(size)

    nx, ny = [(4, 8), (5, 10), (6, 11), (4, 9)][variant % 4]
    warp = common.periodic_fbm(u, v, 8, 3, 0.5, rng)
    uu = (u + 0.03 * (warp - 0.5)) % 1.0
    vv = (v + 0.02 * (warp - 0.5)) % 1.0

    _, edge, cid, cleavage = _periodic_rect_voronoi(uu, vv, nx, ny, 0.65, 0.40, rng)
    rock = common.periodic_fbm(u, v, ny * 2, 4, 0.55, rng, mode="swiss")
    grit = common.periodic_fbm(u, v, 32, 3, 0.5, rng)

    tm = np.clip((edge - 0.042) / 0.038, 0.0, 1.0)
    stone_mask = tm * tm * (3.0 - 2.0 * tm)
    td = np.clip(edge / 0.25, 0.0, 1.0)
    stone_dome = np.power(td * td * (3.0 - 2.0 * td), 0.45) * (0.68 + 0.24 * rock + 0.38 * cleavage)

    h = common.normalize_field(stone_mask * stone_dome + (1.0 - stone_mask) * 0.12 * grit)
    c = common.normalize_field(
        stone_mask * (0.24 + 0.52 * cid + 0.24 * rock) + (1.0 - stone_mask) * (0.10 + 0.18 * grit)
    )
    rough = np.where(stone_mask > 0.30, 0.72 + 0.18 * grit, 0.92).astype(np.float32)

    stops = common.resolve_palette(palette, SPEC.default_palette, rng)
    albedo = common.apply_palette_5stop(c, rock, stops, contrast=1.18)
    return common.shade_pbr_surface(albedo, h, roughness=rough, normal_strength=6.6, spec_intensity=0.14), np.asarray(h, dtype=np.float32)


def generate(seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0) -> Image.Image:
    return _compute(seed=seed, palette=palette, size=size, variant=variant)[0]


def generate_height(seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0) -> np.ndarray:
    return _compute(seed=seed, palette=palette, size=size, variant=variant)[1]

SPEC = GeneratorSpec(
    name="stone-wall",
    kind="texture",
    tileable=True,
    origin="tools/generators/types/stone_wall.py",
    default_palette="rough_fieldstone_wall",
    description="Stacked rough-hewn coursed stone masonry wall with chiseled rock faces and recessed mortar gaps.",
    render_fn=generate,
    normal_source="native",
    normal_rule="Native physical surface height field (h) used by shade_pbr_surface.",
    normal_strength=6.6,
    height_fn=generate_height,
)
