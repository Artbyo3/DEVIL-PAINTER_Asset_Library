"""Deep forest cushion moss ground with organic rounded clumps and fine velvety frond texture."""

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

    clumps = [6, 8, 10, 12][variant % 4]
    wx = common.periodic_fbm(u, v, 4, 3, 0.5, rng)
    wy = common.periodic_fbm((u + 0.31) % 1.0, (v + 0.59) % 1.0, 4, 3, 0.5, rng)
    uu = (u + 0.07 * (wx - 0.5)) % 1.0
    vv = (v + 0.07 * (wy - 0.5)) % 1.0

    # Lloyd CVT (2 iters) models isotropic radial competition of cushion moss colonies
    _, edge, cid, _, _ = common.periodic_lloyd_voronoi(uu, vv, clumps, 0.84, rng, lloyd_iters=2)
    t_edge = np.clip(edge / 0.38, 0.0, 1.0)
    cushion = np.sqrt(np.clip(t_edge * t_edge * (3.0 - 2.0 * t_edge), 0.0, 1.0))

    # Velvety bryophyte phyllid tufts via isotropic periodic Gabor noise + micro-fuzz
    tuft_gabor = common.periodic_gabor_noise(
        u, v, grid_cells=max(24, clumps * 3), f0=2.6, omega0=0.0, bandwidth=1.2, rng=rng, impulses_per_cell=3, isotropic=True
    )
    velvet = common.periodic_fbm(u, v, 28, 4, 0.56, rng)
    micro_fuzz = common.periodic_fbm((u + 0.47) % 1.0, (v + 0.23) % 1.0, 48, 2, 0.5, rng)

    h = common.normalize_field(cushion * (0.62 + 0.16 * velvet + 0.14 * tuft_gabor + 0.08 * micro_fuzz))
    c = common.normalize_field(0.48 * cushion + 0.20 * velvet + 0.12 * tuft_gabor + 0.12 * cid + 0.08 * micro_fuzz)
    rough = np.clip(0.84 + 0.12 * (1.0 - cushion), 0.72, 0.96).astype(np.float32)

    stops = common.resolve_palette(palette, SPEC.default_palette, rng)
    albedo = common.apply_palette_5stop(c, micro_fuzz, stops, contrast=1.18)
    return common.shade_pbr_surface(albedo, h, roughness=rough, normal_strength=6.0, spec_intensity=0.07), np.asarray(h, dtype=np.float32)


def generate(seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0) -> Image.Image:
    return _compute(seed=seed, palette=palette, size=size, variant=variant)[0]


def generate_height(seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0) -> np.ndarray:
    return _compute(seed=seed, palette=palette, size=size, variant=variant)[1]

SPEC = GeneratorSpec(
    name="moss-ground",
    kind="texture",
    tileable=True,
    origin="tools/generators/types/moss_ground.py",
    default_palette="forest_moss_clumps",
    description="Deep forest green cushion moss ground with organic pillowy clumps and velvety micro-fronds.",
    render_fn=generate,
    normal_source="native",
    normal_rule="Native physical surface height field (h) used by shade_pbr_surface.",
    normal_strength=6.0,
    height_fn=generate_height,
)
