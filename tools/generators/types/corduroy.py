"""Plush corduroy wales and ribbed velveteen texture generator."""

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
    fuzz = common.periodic_fbm(u, v, 8, 3, 0.5, rng)
    slub = common.periodic_fbm(u, v, 4, 3, 0.5, rng, freqy=32)

    wales = 14 + (variant % 6) * 4
    wu = (u * wales) % 1.0
    # Rounded cut-pile wale (~72% width) separated by flat exposed base-cloth channel (~28%)
    wale_half_w = 0.36
    du = np.abs(wu - 0.5) / wale_half_w
    rib = np.sqrt(np.clip(1.0 - du * du, 0.0, 1.0))

    # Ground 1/1 plain weave visible inside the recessed channels between wales
    plain_m = np.array([[1, 0], [0, 1]], dtype=np.int32)
    base_weave, _, _, _ = common.woven_fabric_field(u, v, wales * 4, wales * 4, plain_m, float_arch=0.35, ply_twist=0.08)

    # Anisotropic directional pile tuft fibers (fine horizontally, elongated along vertical nap direction)
    nap = common.periodic_fbm(u, v, min(128, wales * 4), 2, 0.55, rng, freqy=max(8, wales))
    tuft = common.periodic_gradient_perlin(u, v, min(128, wales * 3), max(8, wales), rng)

    pile_h = rib * (0.78 + 0.22 * (0.6 * tuft + 0.4 * nap))
    h = common.normalize_field(np.maximum(pile_h, base_weave * 0.24) + 0.07 * fuzz)
    c = common.normalize_field(0.76 * h + 0.24 * slub)
    stops = common.resolve_palette(palette, SPEC.default_palette, rng)
    albedo = common.apply_palette_5stop(c, nap, stops, contrast=1.18)
    return common.shade_cloth_surface(albedo, h, normal_strength=6.0, sheen_amt=22.0), np.asarray(h, dtype=np.float32)


def generate(seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0) -> Image.Image:
    return _compute(seed=seed, palette=palette, size=size, variant=variant)[0]


def generate_height(seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0) -> np.ndarray:
    return _compute(seed=seed, palette=palette, size=size, variant=variant)[1]

SPEC = GeneratorSpec(
    name="corduroy",
    kind="texture",
    tileable=True,
    origin="cloth_apparel_master_600_pack/procedural_texture_generator.py (arch=6)",
    default_palette="burgundy_velour",
    description="Plush corduroy wales and ribbed velveteen with plush pile tuft variation.",
    render_fn=generate,
    normal_source="native",
    normal_rule="Native physical surface height field (h) used by shade_cloth_surface.",
    normal_strength=6.0,
    height_fn=generate_height,
)
