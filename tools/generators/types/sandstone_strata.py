"""Layered sedimentary sandstone strata with undulating horizontal mineral bands."""

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

    bands = [8, 12, 16, 20][variant % 4]
    warp = common.periodic_fbm(u, v, 2, 4, 0.55, rng, freqy=3)
    cross = common.periodic_fbm((u + 0.41) % 1.0, (v + 0.19) % 1.0, 4, 3, 0.50, rng, freqy=6)
    grit = common.periodic_fbm(u, v, 32, 3, 0.52, rng)

    # Aeolian/fluvial cross-bedding: periodic bedset boundaries truncate inclined foreset laminae
    n_sets = 2 + (variant % 2)
    set_coord = (v * n_sets + 0.25 * (warp - 0.5)) % float(n_sets)
    set_idx = np.floor(set_coord).astype(np.int32) % n_sets
    set_dir = np.where(set_idx % 2 == 0, 1.0, -1.0)
    foreset_phase = (v * bands + u * set_dir * 2.0 + 1.45 * (warp - 0.5) + 0.40 * (cross - 0.5)) % 1.0
    # Asymmetric cemented ledge profile (gradual slope -> crisp eroded undercut)
    ledge = np.where(foreset_phase < 0.78, foreset_phase / 0.78, 1.0 - (foreset_phase - 0.78) / 0.22)
    strata_sub = 0.5 + 0.5 * np.sin((foreset_phase * 3.0 + 0.4 * cross) * 2.0 * math.pi)
    # Bounding surface groove between cross-bedding cosets
    set_frac = set_coord - np.floor(set_coord)
    bed_seam = np.clip(np.minimum(set_frac, 1.0 - set_frac) * 14.0, 0.25, 1.0)
    liesegang = 0.5 + 0.5 * np.sin((warp * 4.5 + cross * 2.0) * 2.0 * math.pi)

    h = common.normalize_field((0.60 * ledge + 0.24 * strata_sub + 0.16 * grit) * bed_seam)
    c = common.normalize_field(0.50 * ledge + 0.24 * strata_sub + 0.14 * liesegang + 0.12 * warp)
    rough = np.clip(0.76 + 0.16 * grit, 0.65, 0.95).astype(np.float32)

    stops = common.resolve_palette(palette, SPEC.default_palette, rng)
    albedo = common.apply_palette_5stop(c, grit, stops, contrast=1.18)
    return common.shade_pbr_surface(albedo, h, roughness=rough, normal_strength=5.8, spec_intensity=0.10), np.asarray(h, dtype=np.float32)


def generate(seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0) -> Image.Image:
    return _compute(seed=seed, palette=palette, size=size, variant=variant)[0]


def generate_height(seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0) -> np.ndarray:
    return _compute(seed=seed, palette=palette, size=size, variant=variant)[1]

SPEC = GeneratorSpec(
    name="sandstone-strata",
    kind="texture",
    tileable=True,
    origin="tools/generators/types/sandstone_strata.py",
    default_palette="layered_sandstone_strata",
    description="Layered sedimentary sandstone rock with undulating mineral strata bands and fine quartz grit.",
    render_fn=generate,
    normal_source="native",
    normal_rule="Native physical surface height field (h) used by shade_pbr_surface.",
    normal_strength=5.8,
    height_fn=generate_height,
)
