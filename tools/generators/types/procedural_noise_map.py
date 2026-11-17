"""Tileable mathematical noise primitives: Swiss ridged, Gabor waves, cellular metrics, curl, and Turing."""

from __future__ import annotations

import math
import numpy as np
from PIL import Image

from .. import common
from ..interface import GeneratorSpec, PaletteArg


def _compute_field(seed: int = 1, size: int = 512, variant: int = 0) -> tuple[np.random.Generator, np.ndarray]:
    rng = np.random.default_rng(seed)
    u, v = common.make_uv_grid(size)
    style = variant % 5
    freq = 3 + (variant % 6) * 2

    if style == 0:
        field = common.periodic_fbm(u, v, freq, 6, 0.55, rng, mode="swiss")
    elif style == 1:
        # Lagae & Lefebvre (2009 SIGGRAPH) periodic sparse-convolution Gabor noise
        base_ang = (variant * 0.45) % math.pi
        field = common.periodic_gabor_noise(
            u, v, grid_cells=max(6, freq * 2), f0=3.5, omega0=base_ang, bandwidth=2.0, rng=rng, impulses_per_cell=4
        )
    elif style == 2:
        metric = "manhattan" if (variant % 2 == 0) else "chebyshev"
        f1, edge, _, _, _ = common.periodic_voronoi(u, v, freq, 0.85, rng, metric=metric)
        field = common.normalize_field(0.55 * edge + 0.45 * (1.0 - f1))
    elif style == 3:
        # Bridson et al. (2007 SIGGRAPH) resolution-independent divergence-free curl flow advection
        vx, vy = common.periodic_curl_noise(u, v, freq, 5, 0.55, rng)
        field = common.periodic_fbm((u + 0.08 * vx) % 1.0, (v + 0.08 * vy) % 1.0, freq, 4, 0.5, rng)
    else:
        # Pearson (1993) Gray-Scott PDE steady-state solution (solitons / coral)
        sim_res = 96 if size <= 256 else 128
        _, rd_v = common.gray_scott_rd(size, feed=0.042, kill=0.060, steps=420, rng=rng, sim_res=sim_res)
        field = rd_v
    return rng, field


def generate(seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0) -> Image.Image:
    rng, field = _compute_field(seed=seed, size=size, variant=variant)
    if palette is not None:
        stops = common.resolve_palette(palette, SPEC.default_palette, rng)
        return common.rgb_image(common.apply_palette_5stop(field, None, stops))
    return common.alpha_image(field)


def generate_height(seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0) -> np.ndarray:
    _ = palette
    return _compute_field(seed=seed, size=size, variant=variant)[1]


SPEC = GeneratorSpec(
    name="procedural-noise-map",
    kind="texture",
    tileable=True,
    origin="substance_painter_master_600_pack/procedural_texture_generator.py (Suite 5)",
    default_palette="charcoal_flannel",
    description="Pure periodic mathematical noises: Swiss (v0), Gabor (v1), Manhattan/Chebyshev (v2), curl flow (v3), Turing (v4).",
    render_fn=generate,
    normal_source="derived",
    normal_rule="Derived height directly from the dimensionless periodic noise scalar field.",
    normal_strength=5.5,
    height_fn=generate_height,
)
