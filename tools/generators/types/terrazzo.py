"""Dual-size Voronoi stone chip terrazzo flooring texture."""

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
    cells = [10, 14, 18, 24][(variant + int(rng.integers(0, 4))) % 4]
    f1, edge, cid, _, _ = common.periodic_voronoi(u, v, cells, 0.92, rng)
    f1_b, edge_b, cid_b, _, _ = common.periodic_voronoi((u + 0.37) % 1.0, (v + 0.61) % 1.0, cells + 6, 0.92, rng)
    base_noise = common.periodic_fbm(u, v, 6, 4, 0.5, rng)

    def _sstep(e0: float, e1: float, x: np.ndarray) -> np.ndarray:
        t = np.clip((x - e0) / (e1 - e0 + 1e-8), 0.0, 1.0)
        return t * t * (3.0 - 2.0 * t)

    chip1 = _sstep(0.32, 0.25, f1) * _sstep(0.03, 0.07, edge)
    chip2_raw = _sstep(0.25, 0.19, f1_b) * _sstep(0.028, 0.068, edge_b)
    chip2 = chip2_raw * (1.0 - chip1)
    chip_mask = np.clip(chip1 + chip2, 0.0, 1.0)

    vein = common.periodic_fbm(u, v, cells * 2, 3, 0.5, rng, mode="turbulence")
    chip_tone = chip1 * (0.20 + 0.68 * cid + 0.12 * vein) + chip2 * (0.14 + 0.64 * cid_b + 0.12 * vein)
    binder = 0.72 + 0.16 * base_noise
    c = common.normalize_field(chip_tone + (1.0 - chip_mask) * binder)
    h = common.normalize_field(0.68 + 0.10 * chip_mask + 0.14 * base_noise * (1.0 - 0.5 * chip_mask))

    stops = common.resolve_palette(palette, SPEC.default_palette, rng)
    albedo = common.apply_palette_5stop(c, base_noise, stops, contrast=1.18)
    return common.shade_pbr_surface(albedo, common.normalize_field(h), roughness=0.28, normal_strength=3.8, spec_intensity=0.20), np.asarray(common.normalize_field(h), dtype=np.float32)


def generate(seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0) -> Image.Image:
    return _compute(seed=seed, palette=palette, size=size, variant=variant)[0]


def generate_height(seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0) -> np.ndarray:
    return _compute(seed=seed, palette=palette, size=size, variant=variant)[1]

SPEC = GeneratorSpec(
    name="terrazzo",
    kind="texture",
    tileable=True,
    origin="p1_work/scripts/custom_generators.py (gen_v1_terrazzo)",
    default_palette="desert_travertine",
    description="Polished terrazzo stone aggregate chips embedded in cementitious binder.",
    render_fn=generate,
    normal_source="native",
    normal_rule="Native physical surface height field (common.normalize_field(h)) used by shade_pbr_surface.",
    normal_strength=3.8,
    height_fn=generate_height,
)
