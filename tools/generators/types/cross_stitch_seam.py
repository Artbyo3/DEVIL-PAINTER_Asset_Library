"""Cross-stitch embroidery seam band alpha trim."""

from __future__ import annotations

import math
import numpy as np
from PIL import Image

from .. import common
from ..interface import GeneratorSpec, PaletteArg


def _seg_dist(px: np.ndarray, py: np.ndarray, ax: float, ay: float, bx: float, by: float) -> tuple[np.ndarray, np.ndarray]:
    vx, vy = bx - ax, by - ay
    t = np.clip(((px - ax) * vx + (py - ay) * vy) / (vx * vx + vy * vy + 1e-12), 0.0, 1.0)
    dx = px - (ax + t * vx)
    dy = py - (ay + t * vy)
    return np.sqrt(dx * dx + dy * dy), t


def _compute(
    seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0
) -> tuple[Image.Image, np.ndarray]:
    rng = np.random.default_rng(seed)
    _ = palette
    u, v = common.make_uv_grid(size)
    y = (v - 0.5) * 2.0
    cols = [8, 12, 16, 20][(variant + int(rng.integers(0, 4))) % 4]
    band_half = float(rng.uniform(0.32, 0.56))
    cu = (u * cols) % 1.0 - 0.5
    cv = (v * cols) % 1.0 - 0.5
    band = np.clip((band_half - np.abs(y)) * 18.0, 0.0, 1.0)

    r_th = 0.115
    d1, t1 = _seg_dist(cu, cv, -0.34, -0.34, 0.34, 0.34)
    d2, t2 = _seg_dist(cu, cv, -0.34, 0.34, 0.34, -0.34)
    s1 = np.sqrt(np.clip(1.0 - (d1 / r_th) ** 2, 0.0, 1.0)) * (0.62 + 0.24 * np.sin(t1 * math.pi))
    s2 = np.sqrt(np.clip(1.0 - (d2 / r_th) ** 2, 0.0, 1.0)) * (0.76 + 0.24 * np.sin(t2 * math.pi))
    s1 *= 0.88 + 0.12 * np.sin((t1 * 4.0 + d1 * 6.0) * 2.0 * math.pi)
    s2 *= 0.88 + 0.12 * np.sin((t2 * 4.0 + d2 * 6.0) * 2.0 * math.pi)

    _h = (np.maximum(s1, s2) * band).astype(np.float32)
    return common.alpha_image(_h), _h


def generate(seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0) -> Image.Image:
    return _compute(seed=seed, palette=palette, size=size, variant=variant)[0]


def generate_height(seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0) -> np.ndarray:
    return _compute(seed=seed, palette=palette, size=size, variant=variant)[1]

SPEC = GeneratorSpec(
    name="cross-stitch-seam",
    kind="alpha",
    tileable=False,
    tile_axes="u",
    origin="p1_work/scripts/custom_generators.py (gen_cloth_alpha_custom kind=0)",
    default_palette="charcoal_flannel",
    description="Cross-stitch X-embroidery decorative seam band alpha (tiles horizontally).",
    render_fn=generate,
    normal_source="native",
    normal_rule="Native intersecting diagonal X-stitch thread relief along seam band.",
    normal_strength=6.0,
    height_fn=generate_height,
)
