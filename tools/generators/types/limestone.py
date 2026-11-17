"""Pale calcareous limestone with organic dissolution pits and subtle fossil specks."""

from __future__ import annotations

import numpy as np
from PIL import Image

from .. import common
from ..interface import GeneratorSpec, PaletteArg


def generate(seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0) -> Image.Image:
    rng = np.random.default_rng(seed)
    u, v = common.make_uv_grid(size)

    pit_cells = [14, 18, 22, 26][variant % 4]
    cloud = common.periodic_fbm(u, v, 3, 4, 0.52, rng)
    chalk = common.periodic_fbm(u, v, 24, 3, 0.5, rng)

    # Dissolution pits / small vugs
    f1_p, _, cid_p, _, _ = common.periodic_voronoi(u, v, pit_cells, 0.90, rng)
    pit_noise = common.periodic_fbm(u, v, pit_cells, 2, 0.5, rng)
    pit_score = (1.0 - f1_p) * (cid_p > 0.52).astype(np.float32) + 0.18 * (pit_noise - 0.5)
    pits = np.clip((pit_score - 0.68) * 14.0, 0.0, 1.0)

    # Fine dark/ochre fossil & shell fragment specks
    f1_f, _, cid_f, _, _ = common.periodic_voronoi((u + 0.43) % 1.0, (v + 0.19) % 1.0, pit_cells * 2, 0.94, rng)
    fossils = np.clip(((1.0 - f1_f) * (cid_f > 0.74).astype(np.float32) - 0.72) * 18.0, 0.0, 1.0)

    h = common.normalize_field(0.78 + 0.16 * cloud + 0.10 * chalk - 0.55 * pits - 0.12 * fossils)
    c = common.normalize_field(0.62 + 0.24 * cloud + 0.10 * chalk - 0.42 * pits - 0.28 * fossils)
    rough = np.clip(0.74 + 0.18 * pits + 0.08 * chalk, 0.65, 0.95).astype(np.float32)

    stops = common.resolve_palette(palette, SPEC.default_palette, rng)
    albedo = common.apply_palette_5stop(c, chalk, stops, contrast=1.12)
    return common.shade_pbr_surface(albedo, h, roughness=rough, normal_strength=5.4, spec_intensity=0.10)


SPEC = GeneratorSpec(
    name="limestone",
    kind="texture",
    tileable=True,
    origin="tools/generators/types/limestone.py",
    default_palette="pale_fossil_limestone",
    description="Pale porous limestone surface with pitted weathering vugs and fine fossil fragment specks.",
    render_fn=generate,
)
