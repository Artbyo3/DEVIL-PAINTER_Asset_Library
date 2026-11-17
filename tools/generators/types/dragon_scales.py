"""Overlapping keeled dragon, reptile, or koi fish armor scales."""

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
    cols = [5, 7, 9, 12][(variant + int(rng.integers(0, 4))) % 4]
    # Even row count guarantees the half-cell horizontal stagger wraps cleanly at v=0/1.
    rows = max(2, (int(round(cols * 1.2)) // 2) * 2)
    ry_i = np.floor(v * rows).astype(np.int32)
    su = ((u * cols) + (ry_i % 2) * 0.5) % 1.0 - 0.5
    sv = (v * rows) % 1.0 - 0.5
    r = np.sqrt(su * su + sv * sv * 0.85) * 2.0

    scale_dome = np.clip(1.0 - r * r, 0.0, 1.0)
    keel = np.exp(-(su * su) / 0.015) * scale_dome * 0.35
    micro = common.periodic_fbm(u, v, cols * 2, 3, 0.5, rng)

    h = common.normalize_field(scale_dome + keel + 0.10 * micro)
    c = common.normalize_field(0.70 * scale_dome + 0.20 * keel + 0.10 * micro)
    rough = np.clip(0.18 + 0.35 * (1.0 - scale_dome), 0.12, 0.75).astype(np.float32)

    stops = common.resolve_palette(palette, SPEC.default_palette, rng)
    albedo = common.apply_palette_5stop(c, micro, stops, contrast=1.20)
    return common.shade_pbr_surface(albedo, h, roughness=rough, normal_strength=7.0, spec_intensity=0.34), np.asarray(h, dtype=np.float32)


def generate(seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0) -> Image.Image:
    return _compute(seed=seed, palette=palette, size=size, variant=variant)[0]


def generate_height(seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0) -> np.ndarray:
    return _compute(seed=seed, palette=palette, size=size, variant=variant)[1]

SPEC = GeneratorSpec(
    name="dragon-scales",
    kind="texture",
    tileable=True,
    origin="p1_work/scripts/custom_generators.py (gen_v2_dragon_koi_scales; even-row seamless fix)",
    default_palette="imperial_emerald_jade",
    description="Overlapping staggered dragon/koi armor scales with central longitudinal keel ridge.",
    render_fn=generate,
    normal_source="native",
    normal_rule="Native physical surface height field (h) used by shade_pbr_surface.",
    normal_strength=7.0,
    height_fn=generate_height,
)
