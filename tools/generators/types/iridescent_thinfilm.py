"""Iridescent thin-film oil slick, bismuth oxide, and heat-anodized titanium interference."""

from __future__ import annotations

import math
import numpy as np
from PIL import Image

from .. import common
from ..interface import GeneratorSpec, PaletteArg


def generate(seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0) -> Image.Image:
    rng = np.random.default_rng(seed)
    u, v = common.make_uv_grid(size)
    _ = variant
    f1 = common.periodic_fbm(u, v, 3, 5, 0.58, rng)
    f2 = common.periodic_fbm((u + 0.22 * f1) % 1.0, (v + 0.22 * f1) % 1.0, 4, 5, 0.55, rng, mode="ridged")
    film = 0.5 + 0.5 * np.sin((f1 * 3.5 + f2 * 2.5) * 2.0 * math.pi)

    h = common.normalize_field(0.65 * f2 + 0.35 * f1)
    c = common.normalize_field(film)
    rough = np.clip(0.12 + 0.25 * (1.0 - f2), 0.08, 0.45).astype(np.float32)

    stops = common.resolve_palette(palette, SPEC.default_palette, rng)
    albedo = common.apply_palette_5stop(c, f1, stops, contrast=1.22)
    return common.shade_pbr_surface(albedo, h, roughness=rough, normal_strength=6.0, spec_intensity=0.42)


SPEC = GeneratorSpec(
    name="iridescent-thinfilm",
    kind="texture",
    tileable=True,
    origin="p1_work/scripts/custom_generators.py (gen_v2_iridescent_thinfilm)",
    default_palette="peacock_iridescent",
    description="Thin-film optical interference ripples for heat-treated titanium, bismuth, and oil slicks.",
    render_fn=generate,
)
