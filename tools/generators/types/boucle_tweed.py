"""Heavy boucle tweed with irregular slub yarns and wool flecks."""

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
    threads = [18, 24, 32, 40][variant % 4]
    slub_u = common.periodic_fbm(u, v, threads, 3, 0.55, rng, freqy=4)
    slub_v = common.periodic_fbm(u, v, 4, 3, 0.55, rng, freqy=threads)

    # Organic weave warp distortion from thick boucle slub yarns
    tu_warp = (u + (slub_u - 0.5) * 0.008) % 1.0
    tv_warp = (v + (slub_v - 0.5) * 0.008) % 1.0
    plain_m = np.array([[1, 0], [0, 1]], dtype=np.int32)
    w_rad = np.clip(0.72 + 0.34 * slub_u, 0.55, 1.02)
    f_rad = np.clip(0.72 + 0.34 * slub_v, 0.55, 1.02)
    weave_h, _, _, _ = common.woven_fabric_field(
        tu_warp, tv_warp, threads, threads, plain_m, float_arch=0.38, ply_twist=0.18, warp_radius=w_rad, weft_radius=f_rad
    )

    # Curled boucle loop nubs and colorful wool neps along high-slub regions
    f1_nub, _, _, _, _ = common.periodic_lloyd_voronoi(u, v, threads // 2, 0.85, rng, lloyd_iters=1)
    nub_gate = np.clip((0.5 * slub_u + 0.5 * slub_v - 0.42) * 3.5, 0.0, 1.0)
    boucle_loop = np.exp(-(f1_nub ** 2) / 0.14) * nub_gate * 0.42
    fleck = (common.periodic_gradient_perlin(u, v, min(128, threads), min(128, threads), rng) > 0.76).astype(np.float32)

    h = common.normalize_field(weave_h + boucle_loop + 0.18 * fleck)
    c = common.normalize_field(0.58 * h + 0.24 * slub_u + 0.28 * fleck)
    stops = common.resolve_palette(palette, SPEC.default_palette, rng)
    albedo = common.apply_palette_5stop(c, slub_v, stops, contrast=1.18)
    return common.shade_cloth_surface(albedo, h, normal_strength=5.8, sheen_amt=14.0), np.asarray(h, dtype=np.float32)


def generate(seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0) -> Image.Image:
    return _compute(seed=seed, palette=palette, size=size, variant=variant)[0]


def generate_height(seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0) -> np.ndarray:
    return _compute(seed=seed, palette=palette, size=size, variant=variant)[1]

SPEC = GeneratorSpec(
    name="boucle-tweed",
    kind="texture",
    tileable=True,
    origin="p1_work/scripts/custom_generators.py (gen_cloth_boucle_slub_tweed)",
    default_palette="oatmeal_boucle",
    description="Heavy boucle tweed weave with slub warp/weft yarn thickness variation and wool flecks.",
    render_fn=generate,
    normal_source="native",
    normal_rule="Native physical surface height field (h) used by shade_cloth_surface.",
    normal_strength=5.8,
    height_fn=generate_height,
)
