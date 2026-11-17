"""Heavy fisherman Aran cable-knit sweater texture generator."""

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

    cables = 4 + (variant % 3) * 2
    cu = (u * cables) % 1.0
    cv = (v * cables * 2) % 1.0
    in_cable = (cu > 0.18) & (cu < 0.82)

    norm_cu = (cu - 0.18) / 0.64
    strand_a = np.exp(-((norm_cu - (0.5 + 0.28 * np.sin(cv * 2.0 * math.pi))) ** 2) / 0.035)
    strand_b = np.exp(-((norm_cu - (0.5 - 0.28 * np.sin(cv * 2.0 * math.pi))) ** 2) / 0.035)
    over_a = np.cos(cv * 2.0 * math.pi) > 0
    braid = np.where(over_a, np.maximum(strand_a, strand_b * 0.72), np.maximum(strand_b, strand_a * 0.72))
    yarn_rib = 0.82 + 0.18 * np.sin((cv * 6.0 + norm_cu * 4.0) * 2.0 * math.pi)

    purl = 0.5 + 0.5 * np.sin(u * cables * 8.0 * math.pi) * np.sin(v * cables * 12.0 * math.pi)
    h = common.normalize_field(np.where(in_cable, braid * yarn_rib, purl * 0.45) + 0.10 * fuzz)
    c = common.normalize_field(0.80 * h + 0.20 * in_cable.astype(np.float32))
    stops = common.resolve_palette(palette, SPEC.default_palette, rng)
    albedo = common.apply_palette_5stop(c, fuzz, stops, contrast=1.18)
    return common.shade_cloth_surface(albedo, h, normal_strength=6.2, sheen_amt=14.0), np.asarray(h, dtype=np.float32)


def generate(seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0) -> Image.Image:
    return _compute(seed=seed, palette=palette, size=size, variant=variant)[0]


def generate_height(seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0) -> np.ndarray:
    return _compute(seed=seed, palette=palette, size=size, variant=variant)[1]

SPEC = GeneratorSpec(
    name="cable-knit",
    kind="texture",
    tileable=True,
    origin="cloth_apparel_master_600_pack/procedural_texture_generator.py (arch=2)",
    default_palette="aran_cream_cableknit",
    description="Fisherman Aran braided rope cable-knit with purl seed-stitch gutters.",
    render_fn=generate,
    normal_source="native",
    normal_rule="Native physical surface height field (h) used by shade_cloth_surface.",
    normal_strength=6.2,
    height_fn=generate_height,
)
