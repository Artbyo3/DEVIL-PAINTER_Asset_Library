"""Four-hole shirt button and sewn thread cross decal alpha stamp."""

from __future__ import annotations

import numpy as np
from PIL import Image

from .. import common
from ..interface import GeneratorSpec, PaletteArg


def _capsule_dist(x: np.ndarray, y: np.ndarray, ax: float, ay: float, bx: float, by: float) -> np.ndarray:
    vx, vy = bx - ax, by - ay
    denom = vx * vx + vy * vy + 1e-12
    t = np.clip(((x - ax) * vx + (y - ay) * vy) / denom, 0.0, 1.0)
    dx = x - (ax + t * vx)
    dy = y - (ay + t * vy)
    return np.sqrt(dx * dx + dy * dy)


def _compute(
    seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0
) -> tuple[Image.Image, np.ndarray]:
    rng = np.random.default_rng(seed)
    _ = palette
    u, v = common.make_uv_grid(size)
    x = (u - 0.5) * 2.0
    y = (v - 0.5) * 2.0
    r = np.sqrt(x * x + y * y)
    btn_r = float(rng.uniform(0.58, 0.72)) if variant > 0 else 0.68

    disc = np.clip((btn_r - r) * 28.0, 0.0, 1.0)
    rim = np.exp(-((r - (btn_r - 0.09)) ** 2) / 0.0045) * 0.44
    recess = np.clip((btn_r - 0.19 - r) * 16.0, 0.0, 1.0) * 0.24
    # 4 thread holes at (+-0.18, +-0.18)
    h_dist = np.sqrt((np.abs(x) - 0.18) ** 2 + (np.abs(y) - 0.18) ** 2)
    holes = np.clip((0.082 - h_dist) * 42.0, 0.0, 1.0)

    # Smooth cylindrical capsule threads genuinely threading between opposing holes (no binary + mask)
    r_th = 0.048
    if variant % 2 == 0:
        d_a = _capsule_dist(x, y, -0.18, -0.18, 0.18, 0.18) / r_th
        d_b = _capsule_dist(x, y, -0.18, 0.18, 0.18, -0.18) / r_th
        t_a = np.sqrt(np.clip(1.0 - d_a * d_a, 0.0, 1.0)) * 0.85
        t_b = np.sqrt(np.clip(1.0 - d_b * d_b, 0.0, 1.0)) * 1.00
    else:
        d_a = _capsule_dist(x, y, -0.18, -0.18, 0.18, -0.18) / r_th
        d_b = _capsule_dist(x, y, -0.18, 0.18, 0.18, 0.18) / r_th
        t_a = np.sqrt(np.clip(1.0 - d_a * d_a, 0.0, 1.0))
        t_b = np.sqrt(np.clip(1.0 - d_b * d_b, 0.0, 1.0))
    threads = np.maximum(t_a, t_b) * 0.42

    btn = disc * (0.64 + rim - recess) * (1.0 - 0.72 * holes) + threads * disc
    _h = np.clip(btn, 0.0, 1.0).astype(np.float32)
    return common.alpha_image(_h), _h


def generate(seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0) -> Image.Image:
    return _compute(seed=seed, palette=palette, size=size, variant=variant)[0]


def generate_height(seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0) -> np.ndarray:
    return _compute(seed=seed, palette=palette, size=size, variant=variant)[1]

SPEC = GeneratorSpec(
    name="four-hole-button",
    kind="alpha",
    tileable=False,
    tile_axes="none",
    origin="cloth_apparel_master_600_pack/procedural_texture_generator.py (Suite 4 style=3)",
    default_palette="charcoal_flannel",
    description="Radial 4-hole shirt button decal alpha stamp with beveled rim and crossed sewing threads.",
    render_fn=generate,
    normal_source="native",
    normal_rule="Native beveled button dish, raised rim, thread holes, and center X-stitch relief.",
    normal_strength=6.0,
    height_fn=generate_height,
)
