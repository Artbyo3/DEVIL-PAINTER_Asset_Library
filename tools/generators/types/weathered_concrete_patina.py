"""Weathered concrete, peeling paint, galvanized spangle, sedimentary patina, and roof tiles."""

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
    kind = variant % 6
    micro = common.periodic_fbm(u, v, 10, 3, 0.5, rng)

    if kind == 0:
        agg1 = common.periodic_voronoi(u, v, [16, 24, 32][int(rng.integers(0, 3))], 0.9, rng)[0]
        agg2 = common.periodic_voronoi((u + 0.31) % 1.0, (v + 0.57) % 1.0, [28, 36][int(rng.integers(0, 2))], 0.9, rng)[0]
        crack = common.periodic_lloyd_voronoi(u, v, [4, 6][int(rng.integers(0, 2))], 0.88, rng, lloyd_iters=1)[1]
        crack_line = np.clip((0.038 - crack) * 26.0, 0.0, 1.0)
        pebbles = np.clip(1.0 - np.minimum(agg1, agg2) * 2.2, 0.0, 1.0)
        h = common.normalize_field(0.52 * pebbles + 0.48 * micro - 0.52 * crack_line)
        c = common.normalize_field(0.45 * pebbles + 0.35 * micro + 0.20 * (1.0 - crack_line))
    elif kind == 1:
        base = common.periodic_fbm(u, v, [3, 5, 7][int(rng.integers(0, 3))], 5, 0.58, rng)
        peel = np.clip((base - float(rng.uniform(0.38, 0.58))) * 14.0, 0.0, 1.0)
        curl_edge = np.exp(-((base - 0.48) ** 2) / 0.0012)
        h = common.normalize_field(0.55 * peel + 0.30 * curl_edge + 0.15 * micro)
        c = common.normalize_field(0.65 * peel + 0.20 * micro + 0.15 * curl_edge)
    elif kind == 2:
        # Hot-dip galvanized zinc crystal spangle with per-grain dendritic feathering
        _, _, cid, rx, ry = common.periodic_lloyd_voronoi(u, v, [6, 10, 14, 18][int(rng.integers(0, 4))], 0.92, rng, lloyd_iters=1)
        ang = cid * 2.0 * math.pi
        dendrite = 0.5 + 0.5 * np.cos((rx * np.cos(ang) + ry * np.sin(ang)) * 180.0)
        spangle = 0.5 + 0.42 * (rx * np.cos(ang) + ry * np.sin(ang)) + 0.12 * dendrite
        patina = common.periodic_fbm(u, v, 5, 4, 0.55, rng, mode="turbulence")
        h = common.normalize_field(0.64 * spangle + 0.36 * patina)
        c = common.normalize_field(0.52 * cid + 0.28 * spangle + 0.20 * patina)
    elif kind == 3:
        warp = common.periodic_fbm(u, v, [2, 3, 5][int(rng.integers(0, 3))], 5, 0.55, rng)
        layers = [5, 8, 12, 16][int(rng.integers(0, 4))]
        # Asymmetric differential-weathering rock beds (steep cap ledges + gentle shale talus slopes)
        sp = (v * layers + warp * 2.4) % 1.0
        strata = np.where(sp < 0.75, sp / 0.75, 1.0 - (sp - 0.75) / 0.25)
        fissure = common.periodic_fbm(u, v, 6, 4, 0.5, rng, mode="ridged")
        h = common.normalize_field(0.60 * strata + 0.25 * fissure + 0.15 * micro)
        c = common.normalize_field(0.65 * strata + 0.35 * warp)
    elif kind == 4:
        cols = [5, 7, 9, 12][int(rng.integers(0, 4))]
        rows = [6, 8, 12][int(rng.integers(0, 3))]  # even rows wrap half-stagger cleanly
        ry_i = np.floor(v * rows).astype(np.int32)
        gu = ((u * cols) + (ry_i % 2) * 0.5) % 1.0
        cx_i = np.floor(u * cols + (ry_i % 2) * 0.5).astype(np.int64) % cols
        tile_id = (
            ((cx_i * 73856093) ^ ((ry_i.astype(np.int64) % rows) * 19349663) ^ np.int64((seed + variant) * 83492791))
            % 997
        ) / 997.0
        gv = (v * rows) % 1.0
        # Semicircular Spanish/Mission barrel roof tile profile tapering slightly up-course + crisp drip step
        taper_u = (gu - 0.5) / (0.46 - 0.06 * (1.0 - gv))
        arc = np.sqrt(np.clip(1.0 - taper_u * taper_u * 4.0, 0.0, 1.0))
        drip_step = np.clip((1.0 - gv) * 12.0, 0.0, 1.0)
        step = (0.35 + 0.65 * gv) * (0.35 + 0.65 * arc) * drip_step
        h = common.normalize_field(step + 0.14 * micro)
        c = common.normalize_field(0.45 * tile_id.astype(np.float32) + 0.35 * step + 0.20 * micro)
    else:
        n1 = common.periodic_fbm(u, v, [3, 5, 8][int(rng.integers(0, 3))], 5, 0.6, rng, mode="turbulence")
        n2 = common.periodic_fbm((u + 0.15 * n1) % 1.0, (v + 0.15 * n1) % 1.0, [4, 7][int(rng.integers(0, 2))], 4, 0.55, rng, mode="ridged")
        h = common.normalize_field(0.5 * n1 + 0.5 * n2)
        c = common.normalize_field(0.6 * n2 + 0.4 * micro)

    stops = common.resolve_palette(palette, SPEC.default_palette, rng)
    albedo = common.apply_palette_5stop(c, micro, stops, contrast=1.14)
    ao = 0.84 + 0.16 * h
    return common.rgb_image(albedo * ao[..., None]), h


def generate(seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0) -> Image.Image:
    return _compute(seed=seed, palette=palette, size=size, variant=variant)[0]


def generate_height(seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0) -> np.ndarray:
    return _compute(seed=seed, palette=palette, size=size, variant=variant)[1]


SPEC = GeneratorSpec(
    name="weathered-concrete-patina",
    kind="texture",
    tileable=True,
    origin="p1_work/scripts/custom_generators.py (gen_sp_texture_custom kinds 0-5)",
    default_palette="industrial_concrete",
    description="Aggregate concrete cracks (v0), peeling paint (v1), zinc spangle (v2), strata (v3), barrel roof tiles (v4), corrosion patina (v5).",
    render_fn=generate,
    normal_source="native",
    normal_rule="Native surface relief field h (aggregate cracks, paint curl, zinc spangle, strata, barrel tiles, corrosion patina).",
    normal_strength=5.5,
    height_fn=generate_height,
)
