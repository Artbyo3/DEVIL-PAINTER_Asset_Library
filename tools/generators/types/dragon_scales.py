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
    rows = max(2, (int(round(cols * 1.25)) // 2) * 2)
    scale_dome = np.zeros_like(u, dtype=np.float32)
    keel = np.zeros_like(u, dtype=np.float32)
    ry0 = np.floor(v * rows).astype(np.int32)
    # Imbrication: upper row (-1) overlaps lower row (0), producing true crescent scale pockets
    for dr in (0, -1):
        rr = ry0 + dr
        su = ((u * cols) + (rr % 2) * 0.5) % 1.0 - 0.5
        sv = (v * rows - rr) - 0.45
        # Pointed-shield scale contour: rounded tip downward (sv > 0)
        rx_s = su * 1.92
        ry_s = sv * 1.48 + 0.10
        r = np.hypot(rx_s, ry_s)
        inside = np.clip((1.05 - r) * 8.0, 0.0, 1.0)
        dome_r = np.sqrt(np.clip(1.0 - r * r * 0.88, 0.0, 1.0)) * inside
        # Anterior tilt: top rises smoothly over the next row's base
        tilt = np.clip(0.88 + 0.24 * sv, 0.65, 1.15)
        s_h = dome_r * tilt
        k_h = np.exp(-(su * su) / 0.012) * s_h * 0.34
        take = inside > 0.35
        scale_dome = np.where(take, np.maximum(scale_dome * (1.0 - inside * 0.7), s_h), scale_dome)
        keel = np.where(take, k_h, keel)

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
