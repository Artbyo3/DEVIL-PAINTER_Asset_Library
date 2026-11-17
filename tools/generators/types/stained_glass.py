"""Leaded Voronoi stained glass window panels with glass ripple texture."""

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
    cells = [7, 10, 14, 18][(variant + int(rng.integers(0, 4))) % 4]
    _, edge, cid, _, _ = common.periodic_voronoi(u, v, cells, 0.85, rng)
    lead = np.clip((0.055 - edge) * 32.0, 0.0, 1.0)
    glass_ripple = common.periodic_fbm(u, v, cells * 2, 3, 0.5, rng)

    h = common.normalize_field(0.65 * lead + 0.35 * (1.0 - lead) * glass_ripple)
    c = common.normalize_field(np.where(lead > 0.5, 0.05, 0.25 + 0.65 * cid + 0.15 * (glass_ripple - 0.5)))
    rough = np.where(lead > 0.5, 0.75, 0.12).astype(np.float32)

    stops = common.resolve_palette(palette, SPEC.default_palette, rng)
    albedo = common.apply_palette_5stop(c, glass_ripple, stops, contrast=1.25)
    return common.shade_pbr_surface(albedo, h, roughness=rough, normal_strength=5.5, spec_intensity=0.35), np.asarray(h, dtype=np.float32)


def generate(seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0) -> Image.Image:
    return _compute(seed=seed, palette=palette, size=size, variant=variant)[0]


def generate_height(seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0) -> np.ndarray:
    return _compute(seed=seed, palette=palette, size=size, variant=variant)[1]

SPEC = GeneratorSpec(
    name="stained-glass",
    kind="texture",
    tileable=True,
    origin="p1_work/scripts/custom_generators.py (gen_v1_stained_glass)",
    default_palette="bismuth_iridescent",
    description="Cathedral Voronoi leaded stained glass panes with dark came borders and rippled glass relief.",
    render_fn=generate,
    normal_source="native",
    normal_rule="Native physical surface height field (h) used by shade_pbr_surface.",
    normal_strength=5.5,
    height_fn=generate_height,
)
