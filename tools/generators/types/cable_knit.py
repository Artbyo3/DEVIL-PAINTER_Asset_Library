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
    in_cable = (cu > 0.15) & (cu < 0.85)

    # 2x2 cable plait centerlines with cylindrical yarn bundle radius and knit sub-wale texture
    norm_cu = (cu - 0.15) / 0.70
    phase = cv * 2.0 * math.pi
    ca_center = 0.5 + 0.24 * np.sin(phase)
    cb_center = 0.5 - 0.24 * np.sin(phase)
    rad = 0.21

    da = (norm_cu - ca_center) / rad
    db = (norm_cu - cb_center) / rad
    prof_a = np.sqrt(np.clip(1.0 - da * da, 0.0, 1.0))
    prof_b = np.sqrt(np.clip(1.0 - db * db, 0.0, 1.0))

    # Sub-wale knit V-loop ridges oriented along cable curve plus ply twist
    knit_a = 0.82 + 0.18 * np.sin((cv * 6.0 + np.abs(da) * 0.5) * 2.0 * math.pi)
    knit_b = 0.82 + 0.18 * np.sin((cv * 6.0 + np.abs(db) * 0.5) * 2.0 * math.pi)

    # Over/under crossing elevation from cos(phase)
    za = 0.78 + 0.22 * np.cos(phase)
    zb = 0.78 - 0.22 * np.cos(phase)
    braid = np.maximum(prof_a * knit_a * za, prof_b * knit_b * zb)

    # True reverse-knit purl bumps (horizontal interlocking crescent arches) in side gutters
    gu = (u * cables * 6) % 1.0
    gv = (v * cables * 12) % 1.0
    purl_arc = np.sqrt(np.clip(1.0 - ((gv - 0.5 - 0.16 * np.cos((gu - 0.5) * math.pi)) * 2.4) ** 2, 0.0, 1.0))
    purl_seg = np.sqrt(np.clip(1.0 - ((gu - 0.5) * 2.05) ** 2, 0.0, 1.0))
    purl = purl_arc * purl_seg

    h = common.normalize_field(np.maximum(braid * in_cable.astype(np.float32), purl * 0.48) + 0.10 * fuzz)
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
