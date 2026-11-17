"""Woven geometric jacquard fabric with satin diamond, medallion, and zigzag motifs."""

from __future__ import annotations

import math
import numpy as np
from PIL import Image

from .. import common
from ..interface import GeneratorSpec, PaletteArg


def generate(seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0) -> Image.Image:
    rng = np.random.default_rng(seed)
    u, v = common.make_uv_grid(size)
    rep = [3, 4, 5, 6][int(rng.integers(0, 4))]
    kind = variant % 3
    gu = (u * rep) % 1.0
    gv = (v * rep) % 1.0

    if kind == 0:
        d = np.abs(np.abs(gu - 0.5) + np.abs(gv - 0.5) - 0.45)
        motif = np.clip((0.10 - d) * 14.0, 0.0, 1.0)
    elif kind == 1:
        r1 = np.sqrt((gu - 0.5) ** 2 + (gv - 0.5) ** 2)
        r2 = np.sqrt(((gu + 0.5) % 1.0 - 0.5) ** 2 + ((gv + 0.5) % 1.0 - 0.5) ** 2)
        motif = np.clip(1.0 - np.abs(np.minimum(r1, r2) - 0.36) * 14.0, 0.0, 1.0)
    else:
        zig = np.abs(((u * rep * 2.0) % 2.0) - 1.0)
        band = (v * rep * 2.0 + zig * 0.8) % 1.0
        motif = (band < 0.48).astype(np.float32)

    satin_u = 0.5 + 0.5 * np.sin(u * 48.0 * math.pi)
    satin_v = 0.5 + 0.5 * np.sin(v * 48.0 * math.pi)
    h = common.normalize_field(np.where(motif > 0.5, 0.65 + 0.35 * satin_u, 0.25 + 0.25 * satin_v))
    c = common.normalize_field(0.75 * motif + 0.25 * h)
    rough = np.where(motif > 0.5, 0.25, 0.65).astype(np.float32)

    micro = common.periodic_fbm(u, v, 10, 3, 0.5, rng)
    stops = common.resolve_palette(palette, SPEC.default_palette, rng)
    albedo = common.apply_palette_5stop(c, micro, stops, contrast=1.15)
    return common.shade_pbr_surface(albedo, h, roughness=rough, normal_strength=5.2, spec_intensity=0.22)


SPEC = GeneratorSpec(
    name="jacquard-geometric",
    kind="texture",
    tileable=True,
    origin="p1_work/scripts/custom_generators.py (gen_cloth_jacquard_geometric)",
    default_palette="teal_jacquard",
    description="Geometric jacquard upholstery/apparel fabric (diamond lattice, medallion, or chevron zigzag).",
    render_fn=generate,
)
