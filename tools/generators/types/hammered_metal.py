"""Hand-forged hammered metal dimples and subtle grain texture."""

from __future__ import annotations

import numpy as np
from PIL import Image

from .. import common
from ..interface import GeneratorSpec, PaletteArg


def generate(seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0) -> Image.Image:
    rng = np.random.default_rng(seed)
    u, v = common.make_uv_grid(size)
    cells = [8, 12, 16, 22][(variant + int(rng.integers(0, 4))) % 4]
    f1, _, cid, _, _ = common.periodic_voronoi(u, v, cells, 0.88, rng)
    micro = common.periodic_fbm(u, v, 12, 4, 0.5, rng)

    dimple = np.power(np.clip(f1 * 1.25, 0.0, 1.0), 1.6)
    h = common.normalize_field(0.85 * dimple + 0.15 * micro)
    c = common.normalize_field(0.55 * dimple + 0.25 * cid + 0.20 * micro)
    rough = np.clip(0.16 + 0.42 * (1.0 - dimple), 0.10, 0.75).astype(np.float32)

    stops = common.resolve_palette(palette, SPEC.default_palette, rng)
    albedo = common.apply_palette_5stop(c, micro, stops, contrast=1.18)
    return common.shade_pbr_surface(albedo, h, roughness=rough, normal_strength=7.5, spec_intensity=0.38)


SPEC = GeneratorSpec(
    name="hammered-metal",
    kind="texture",
    tileable=True,
    origin="p1_work/scripts/custom_generators.py (gen_v2_hammered_metal)",
    default_palette="forged_dwarven_gold",
    description="Hand-forged planished hammered metal armor/copper dimples via Voronoi concave facets.",
    render_fn=generate,
)
