"""Deep forest cushion moss ground with organic rounded clumps and fine velvety frond texture."""

from __future__ import annotations

import numpy as np
from PIL import Image

from .. import common
from ..interface import GeneratorSpec, PaletteArg


def generate(seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0) -> Image.Image:
    rng = np.random.default_rng(seed)
    u, v = common.make_uv_grid(size)

    clumps = [6, 8, 10, 12][variant % 4]
    wx = common.periodic_fbm(u, v, 4, 3, 0.5, rng)
    wy = common.periodic_fbm((u + 0.31) % 1.0, (v + 0.59) % 1.0, 4, 3, 0.5, rng)
    uu = (u + 0.08 * (wx - 0.5)) % 1.0
    vv = (v + 0.08 * (wy - 0.5)) % 1.0

    _, edge, cid, _, _ = common.periodic_voronoi(uu, vv, clumps, 0.84, rng)
    cushion = np.power(np.clip(edge * 3.4, 0.0, 1.0), 0.45)
    velvet = common.periodic_fbm(u, v, 28, 4, 0.56, rng)
    micro_fuzz = common.periodic_fbm((u + 0.47) % 1.0, (v + 0.23) % 1.0, 48, 2, 0.5, rng)

    h = common.normalize_field(cushion * (0.68 + 0.22 * velvet + 0.10 * micro_fuzz))
    c = common.normalize_field(0.54 * cushion + 0.24 * velvet + 0.12 * cid + 0.10 * micro_fuzz)
    rough = np.clip(0.84 + 0.12 * (1.0 - cushion), 0.72, 0.96).astype(np.float32)

    stops = common.resolve_palette(palette, SPEC.default_palette, rng)
    albedo = common.apply_palette_5stop(c, micro_fuzz, stops, contrast=1.18)
    return common.shade_pbr_surface(albedo, h, roughness=rough, normal_strength=6.0, spec_intensity=0.07)


SPEC = GeneratorSpec(
    name="moss-ground",
    kind="texture",
    tileable=True,
    origin="tools/generators/types/moss_ground.py",
    default_palette="forest_moss_clumps",
    description="Deep forest green cushion moss ground with organic pillowy clumps and velvety micro-fronds.",
    render_fn=generate,
)
