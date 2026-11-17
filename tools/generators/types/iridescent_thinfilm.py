"""Iridescent thin-film oil slick, bismuth oxide, and heat-anodized titanium interference."""

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
    _ = variant
    f1 = common.periodic_fbm(u, v, 3, 5, 0.58, rng)
    f2 = common.periodic_fbm((u + 0.22 * f1) % 1.0, (v + 0.22 * f1) % 1.0, 4, 5, 0.55, rng, mode="ridged")

    # Physical thin-film thickness d in nanometers [120nm .. 760nm], refractive index n2=1.46, cos(theta2)=0.92
    thickness_nm = 120.0 + 640.0 * np.clip(0.62 * f1 + 0.38 * f2, 0.0, 1.0)
    opd = 2.0 * 1.46 * thickness_nm * 0.92
    finesse = 0.65
    # Airy reflectance R(lambda) = (F * sin^2(beta)) / (1 + F * sin^2(beta)), beta = pi * OPD / lambda
    airy_rgb = []
    for lam in (635.0, 532.0, 465.0):
        s2 = np.sin(math.pi * opd / lam) ** 2
        airy_rgb.append((finesse * s2) / (1.0 + finesse * s2))
    r_spec = np.stack(airy_rgb, axis=-1).astype(np.float32)
    film_luma = 0.299 * r_spec[..., 0] + 0.587 * r_spec[..., 1] + 0.114 * r_spec[..., 2]

    h = common.normalize_field(0.65 * f2 + 0.35 * f1)
    c = common.normalize_field(0.55 * film_luma + 0.45 * common.normalize_field(opd))
    rough = np.clip(0.10 + 0.24 * (1.0 - f2), 0.06, 0.42).astype(np.float32)

    stops = common.resolve_palette(palette, SPEC.default_palette, rng)
    pal_rgb = common.apply_palette_5stop(c, f1, stops, contrast=1.20)
    # Blend physical Airy spectral chromatic dispersion into palette-mapped luminance
    spec_chroma = (r_spec - film_luma[..., None]) * 210.0
    albedo = np.clip(pal_rgb + spec_chroma * 0.42, 0.0, 255.0)
    return common.shade_pbr_surface(albedo, h, roughness=rough, normal_strength=6.0, spec_intensity=0.45), np.asarray(h, dtype=np.float32)


def generate(seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0) -> Image.Image:
    return _compute(seed=seed, palette=palette, size=size, variant=variant)[0]


def generate_height(seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0) -> np.ndarray:
    return _compute(seed=seed, palette=palette, size=size, variant=variant)[1]

SPEC = GeneratorSpec(
    name="iridescent-thinfilm",
    kind="texture",
    tileable=True,
    origin="p1_work/scripts/custom_generators.py (gen_v2_iridescent_thinfilm)",
    default_palette="peacock_iridescent",
    description="Thin-film optical interference ripples for heat-treated titanium, bismuth, and oil slicks.",
    render_fn=generate,
    normal_source="native",
    normal_rule="Native physical surface height field (h) used by shade_pbr_surface.",
    normal_strength=6.0,
    height_fn=generate_height,
)
