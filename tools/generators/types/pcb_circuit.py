"""Printed circuit board copper traces, solder mask weave, and annular vias."""

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
    grid = [8, 12, 14, 18][(variant + int(rng.integers(0, 4))) % 4]
    gu = (u * grid) % 1.0
    gv = (v * grid) % 1.0
    iu = np.floor(u * grid).astype(np.int32) % grid
    iv = np.floor(v * grid).astype(np.int32) % grid

    # Build a periodic connected octilinear trace graph (horizontal, vertical, +45/-45 edges + pad nodes)
    e_r = np.zeros((grid, grid), dtype=bool)
    e_d = np.zeros((grid, grid), dtype=bool)
    e_dr = np.zeros((grid, grid), dtype=bool)
    e_dl = np.zeros((grid, grid), dtype=bool)
    pads = np.zeros((grid, grid), dtype=bool)

    n_walks = grid * 2
    dirs = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1)]
    for _ in range(n_walks):
        cx = int(rng.integers(0, grid))
        cy = int(rng.integers(0, grid))
        pads[cy, cx] = True
        dx, dy = dirs[int(rng.integers(0, len(dirs)))]
        steps = int(rng.integers(2, max(4, grid // 2 + 2)))
        for s_i in range(steps):
            nx = (cx + dx) % grid
            ny = (cy + dy) % grid
            if dx == 1 and dy == 0:
                e_r[cy, cx] = True
            elif dx == -1 and dy == 0:
                e_r[ny, nx] = True
            elif dx == 0 and dy == 1:
                e_d[cy, cx] = True
            elif dx == 0 and dy == -1:
                e_d[ny, nx] = True
            elif dx == 1 and dy == 1:
                e_dr[cy, cx] = True
            elif dx == -1 and dy == 1:
                e_dl[cy, cx] = True
            cx, cy = nx, ny
            if s_i == steps - 1 or rng.random() < 0.25:
                pads[cy, cx] = True
            if rng.random() < 0.35:
                dx, dy = dirs[int(rng.integers(0, len(dirs)))]

    def _seg_dist(px: np.ndarray, py: np.ndarray, ax: float, ay: float, bx: float, by: float) -> np.ndarray:
        vx, vy = bx - ax, by - ay
        wx, wy = px - ax, py - ay
        proj = np.clip((wx * vx + wy * vy) / (vx * vx + vy * vy + 1e-12), 0.0, 1.0)
        return np.hypot(wx - proj * vx, wy - proj * vy)

    # Evaluate outgoing & incoming half-segments inside each cell [0,1]^2 for exact continuity across borders
    d_min = np.full_like(gu, 10.0, dtype=np.float32)
    iv_m1 = (iv - 1) % grid
    iu_m1 = (iu - 1) % grid
    iu_p1 = (iu + 1) % grid

    d_min = np.where(e_r[iv, iu], np.minimum(d_min, _seg_dist(gu, gv, 0.5, 0.5, 1.0, 0.5)), d_min)
    d_min = np.where(e_r[iv, iu_m1], np.minimum(d_min, _seg_dist(gu, gv, 0.0, 0.5, 0.5, 0.5)), d_min)
    d_min = np.where(e_d[iv, iu], np.minimum(d_min, _seg_dist(gu, gv, 0.5, 0.5, 0.5, 1.0)), d_min)
    d_min = np.where(e_d[iv_m1, iu], np.minimum(d_min, _seg_dist(gu, gv, 0.5, 0.0, 0.5, 0.5)), d_min)
    d_min = np.where(e_dr[iv, iu], np.minimum(d_min, _seg_dist(gu, gv, 0.5, 0.5, 1.0, 1.0)), d_min)
    d_min = np.where(e_dr[iv_m1, iu_m1], np.minimum(d_min, _seg_dist(gu, gv, 0.0, 0.0, 0.5, 0.5)), d_min)
    d_min = np.where(e_dl[iv, iu], np.minimum(d_min, _seg_dist(gu, gv, 0.5, 0.5, 0.0, 1.0)), d_min)
    d_min = np.where(e_dl[iv_m1, iu_p1], np.minimum(d_min, _seg_dist(gu, gv, 1.0, 0.0, 0.5, 0.5)), d_min)

    def _smooth(edge0: float, edge1: float, x: np.ndarray) -> np.ndarray:
        t = np.clip((x - edge0) / (edge1 - edge0 + 1e-8), 0.0, 1.0)
        return t * t * (3.0 - 2.0 * t)

    trace_w = 0.095
    traces = 1.0 - _smooth(trace_w - 0.025, trace_w + 0.025, d_min)

    rc = np.hypot(gu - 0.5, gv - 0.5)
    has_pad = pads[iv, iu]
    annular = (1.0 - _smooth(0.20, 0.24, rc)) * _smooth(0.08, 0.11, rc)
    via = np.where(has_pad, annular, 0.0).astype(np.float32)
    drill = np.where(has_pad, 1.0 - _smooth(0.08, 0.105, rc), 0.0).astype(np.float32)

    plain_11 = np.array([[1, 0], [0, 1]], dtype=np.float32)
    fg_n = grid * 4
    fiberglass, _, _, _ = common.woven_fabric_field(u, v, fg_n, fg_n, plain_11, float_arch=0.35, ply_twist=0.0)
    mask = np.clip(np.maximum(traces * 0.78, via), 0.0, 1.0) * (1.0 - 0.85 * drill)

    h = common.normalize_field(0.76 * mask + 0.16 * fiberglass * (1.0 - 0.6 * mask) - 0.28 * drill)
    c = common.normalize_field(0.16 + 0.78 * mask + 0.08 * fiberglass - 0.14 * drill)
    rough = np.where(via > 0.35, 0.18, np.where(traces > 0.35, 0.26, 0.54)).astype(np.float32)

    stops = common.resolve_palette(palette, SPEC.default_palette, rng)
    albedo = common.apply_palette_5stop(c, None, stops, contrast=1.22)
    return common.shade_pbr_surface(albedo, h, roughness=rough, normal_strength=6.0, spec_intensity=0.30), np.asarray(h, dtype=np.float32)


def generate(seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0) -> Image.Image:
    return _compute(seed=seed, palette=palette, size=size, variant=variant)[0]


def generate_height(seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0) -> np.ndarray:
    return _compute(seed=seed, palette=palette, size=size, variant=variant)[1]

SPEC = GeneratorSpec(
    name="pcb-circuit",
    kind="texture",
    tileable=True,
    origin="p1_work/scripts/custom_generators.py (gen_v1_pcb_circuit)",
    default_palette="solder_pcb_green",
    description="Printed circuit board copper routing traces, annular pad vias, and fiberglass weave substrate.",
    render_fn=generate,
    normal_source="native",
    normal_rule="Native physical surface height field (h) used by shade_pbr_surface.",
    normal_strength=6.0,
    height_fn=generate_height,
)
