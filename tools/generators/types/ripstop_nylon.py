"""Tactical crosshatch ripstop nylon texture generator."""

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
    slub = common.periodic_fbm(u, v, 4, 3, 0.5, rng, freqy=32)

    cells = 8 + (variant % 4) * 2
    sub = 8  # 8 threads per ripstop square (every 8th warp & weft thread is double-denier ripstop yarn)
    threads = cells * sub
    plain_m = np.array([[1, 0], [0, 1]], dtype=np.int32)
    weave_h, warp_over, warp_prof, weft_prof = common.woven_fabric_field(
        u, v, threads, threads, plain_m, float_arch=0.34, ply_twist=0.06, warp_radius=0.92, weft_radius=0.92
    )

    iu = np.floor(u * threads).astype(np.int32) % sub
    iv = np.floor(v * threads).astype(np.int32) % sub
    reinf_u = (iu == 0).astype(np.float32)
    reinf_v = (iv == 0).astype(np.float32)

    # Reinforcement yarns stand ~1.65x prouder where they interlace on top of perpendicular threads
    gu = (u * cells) % 1.0
    gv = (v * cells) % 1.0
    du = np.minimum(gu, 1.0 - gu) * sub
    dv = np.minimum(gv, 1.0 - gv) * sub
    rib_u = np.sqrt(np.clip(1.0 - (du / 0.65) ** 2, 0.0, 1.0)) * np.where(warp_over | (reinf_v == 0), 1.0, 0.42)
    rib_v = np.sqrt(np.clip(1.0 - (dv / 0.65) ** 2, 0.0, 1.0)) * np.where((~warp_over) | (reinf_u == 0), 1.0, 0.42)
    ripstop_grid = np.maximum(rib_u, rib_v)

    h = common.normalize_field(np.maximum(0.45 * weave_h, 0.95 * ripstop_grid) + 0.04 * slub)
    c = common.normalize_field(0.62 * h + 0.38 * slub)
    rough = np.clip(0.28 + 0.30 * (1.0 - ripstop_grid), 0.2, 0.65).astype(np.float32)
    stops = common.resolve_palette(palette, SPEC.default_palette, rng)
    albedo = common.apply_palette_5stop(c, slub, stops, contrast=1.18)
    return common.shade_pbr_surface(albedo, h, roughness=rough, normal_strength=5.4, spec_intensity=0.25), np.asarray(h, dtype=np.float32)


def generate(seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0) -> Image.Image:
    return _compute(seed=seed, palette=palette, size=size, variant=variant)[0]


def generate_height(seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0) -> np.ndarray:
    return _compute(seed=seed, palette=palette, size=size, variant=variant)[1]

SPEC = GeneratorSpec(
    name="ripstop-nylon",
    kind="texture",
    tileable=True,
    origin="cloth_apparel_master_600_pack/procedural_texture_generator.py (arch=9) + v2 ripstop",
    default_palette="tactical_coyote_ripstop",
    description="Tactical ripstop nylon with raised reinforcement grid over plain weave.",
    render_fn=generate,
    normal_source="native",
    normal_rule="Native physical surface height field (h) used by shade_pbr_surface.",
    normal_strength=5.4,
    height_fn=generate_height,
)
