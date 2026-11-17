"""Mulberry silk satin damask and jacquard floral brocade texture generator."""

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

    freq = 2 + (variant % 3) * 2
    warp = common.periodic_fbm(u, v, freq * 2, 4, 0.5, rng)
    # Symmetric ogee / damask floral medallion harmonic contour (4-fold periodic symmetry)
    du = (u * freq) % 1.0 - 0.5
    dv = (v * freq) % 1.0 - 0.5
    r = np.sqrt(du * du + dv * dv) * 2.0
    ang = np.arctan2(dv, du)
    ogee = (
        np.cos(du * 2.0 * math.pi)
        + np.cos(dv * 2.0 * math.pi)
        + 0.42 * np.cos(4.0 * ang) * np.exp(-r * 1.4)
        + 0.30 * (warp - 0.5)
    )
    motif = np.clip((ogee + 0.15) * 4.5, 0.0, 1.0)

    # 8-harness satin (7/1, move s=3 -> coprime(3,8)=1) vs orthogonal 8-harness sateen ground
    satin_8h = np.zeros((8, 8), dtype=np.int32)
    sateen_8h = np.ones((8, 8), dtype=np.int32)
    for row in range(8):
        col = (row * 3) % 8
        satin_8h[row, :] = 1
        satin_8h[row, col] = 0
        sateen_8h[row, :] = 0
        sateen_8h[row, col] = 1

    threads = 64
    h_satin, _, _, _ = common.woven_fabric_field(u, v, threads, threads, satin_8h, float_arch=0.30, ply_twist=0.06)
    h_sateen, _, _, _ = common.woven_fabric_field(u, v, threads, threads, sateen_8h, float_arch=0.30, ply_twist=0.06)

    h = common.normalize_field(motif * (0.38 + 0.62 * h_satin) + (1.0 - motif) * (0.08 + 0.42 * h_sateen))
    c = common.normalize_field(0.75 * motif + 0.25 * h)
    rough = np.where(motif > 0.5, 0.24, 0.14).astype(np.float32)
    stops = common.resolve_palette(palette, SPEC.default_palette, rng)
    albedo = common.apply_palette_5stop(c, warp, stops, contrast=1.20)
    return common.shade_pbr_surface(albedo, h, roughness=rough, normal_strength=5.2, spec_intensity=0.35), np.asarray(h, dtype=np.float32)


def generate(seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0) -> Image.Image:
    return _compute(seed=seed, palette=palette, size=size, variant=variant)[0]


def generate_height(seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0) -> np.ndarray:
    return _compute(seed=seed, palette=palette, size=size, variant=variant)[1]

SPEC = GeneratorSpec(
    name="silk-brocade",
    kind="texture",
    tileable=True,
    origin="cloth_apparel_master_600_pack/procedural_texture_generator.py (arch=7)",
    default_palette="crimson_mulberry_silk",
    description="Mulberry silk satin damask with raised floral brocade motif.",
    render_fn=generate,
    normal_source="native",
    normal_rule="Native physical surface height field (h) used by shade_pbr_surface.",
    normal_strength=5.2,
    height_fn=generate_height,
)
