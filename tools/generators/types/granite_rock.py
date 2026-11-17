"""Speckled crystalline granite rock with interlocking quartz, feldspar, and biotite mica grains."""

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

    coarse = [26, 32, 38, 44][variant % 4]
    fine = coarse * 2 - 4

    f1_a, edge_a, cid_a, rx_a, ry_a = common.periodic_lloyd_voronoi(u, v, coarse, 0.92, rng, lloyd_iters=1)
    f1_b, _, cid_b, _, _ = common.periodic_lloyd_voronoi(
        (u + 0.37) % 1.0, (v + 0.61) % 1.0, fine, 0.92, rng, lloyd_iters=1
    )
    fracture = common.periodic_fbm(u, v, 4, 4, 0.55, rng, mode="swiss")

    # IUGS QAPF Field 3 granite modal composition: ~12% dark biotite, ~25% plagioclase, ~35% K-feldspar, ~28% vitreous quartz
    def _classify(cid: np.ndarray) -> np.ndarray:
        return np.where(cid < 0.12, 0.06, np.where(cid < 0.37, 0.42, np.where(cid < 0.72, 0.68, 0.94)))

    mineral_a = _classify(cid_a)
    mineral_b = _classify(cid_b)
    grain = 0.64 * mineral_a + 0.36 * mineral_b + 0.12 * (cid_a - 0.5)
    cleavage = 0.5 + 0.35 * (rx_a * np.cos(cid_a * 6.2831853) + ry_a * np.sin(cid_a * 6.2831853))

    h = common.normalize_field(0.42 * fracture + 0.34 * cleavage * (1.0 - 0.5 * f1_a) + 0.24 * np.clip(edge_a * 3.5, 0.0, 1.0))
    c = common.normalize_field(0.78 * grain + 0.22 * fracture)
    rough = np.where(grain > 0.78, 0.22, 0.58 + 0.20 * (1.0 - edge_a)).astype(np.float32)

    stops = common.resolve_palette(palette, SPEC.default_palette, rng)
    albedo = common.apply_palette_5stop(c, 1.0 - f1_b, stops, contrast=1.22)
    return common.shade_pbr_surface(albedo, h, roughness=rough, normal_strength=5.6, spec_intensity=0.30), np.asarray(h, dtype=np.float32)


def generate(seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0) -> Image.Image:
    return _compute(seed=seed, palette=palette, size=size, variant=variant)[0]


def generate_height(seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0) -> np.ndarray:
    return _compute(seed=seed, palette=palette, size=size, variant=variant)[1]

SPEC = GeneratorSpec(
    name="granite-rock",
    kind="texture",
    tileable=True,
    origin="tools/generators/types/granite_rock.py",
    default_palette="speckled_granite",
    description="Speckled crystalline granite rock with interlocking quartz, feldspar, and dark biotite mica grains.",
    render_fn=generate,
    normal_source="native",
    normal_rule="Native physical surface height field (h) used by shade_pbr_surface.",
    normal_strength=5.6,
    height_fn=generate_height,
)
