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
        accum = np.zeros_like(u, dtype=np.float32)
        base_ang = (variant * 0.45) % math.pi
        for k in range(6):
            ang = base_ang + (k - 2.5) * 0.12
            kx = round(math.cos(ang) * freq)
            ky = round(math.sin(ang) * freq)
            if kx == 0 and ky == 0:
                kx = freq
            phase = (u * kx + v * ky) * 2.0 * math.pi
            env = common.periodic_gradient_perlin(u, v, max(2, freq // 2), max(2, freq // 2), rng)
            accum += np.cos(phase + env * 4.0)
        field = common.normalize_field(accum)
    elif style == 2:
        metric = "manhattan" if (variant % 2 == 0) else "chebyshev"
        f1, edge, _, _, _ = common.periodic_voronoi(u, v, freq, 0.85, rng, metric=metric)
        field = common.normalize_field(0.55 * edge + 0.45 * (1.0 - f1))
    elif style == 3:
        pot = common.periodic_fbm(u, v, freq, 5, 0.55, rng)
        du = np.roll(pot, -1, axis=1) - np.roll(pot, 1, axis=1)
        dv = np.roll(pot, -1, axis=0) - np.roll(pot, 1, axis=0)
        field = common.periodic_fbm((u + dv * 4.0) % 1.0, (v - du * 4.0) % 1.0, freq, 4, 0.5, rng)
    else:
        state = common.periodic_fbm(u, v, freq, 4, 0.55, rng)
        for _ in range(4):
            diff = common.periodic_box_blur(state, 1) - common.periodic_box_blur(state, 3)
            state = common.normalize_field(state + np.tanh(diff * 9.0) * 0.35)
        field = state
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
