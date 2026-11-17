"""Pale calcareous limestone with organic dissolution pits and subtle fossil specks."""

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

    pit_cells = [14, 18, 22, 26][variant % 4]
    cloud = common.periodic_fbm(u, v, 3, 4, 0.52, rng)
    chalk = common.periodic_fbm(u, v, 24, 3, 0.5, rng)

    # Karst dissolution joints / stylolite suture cracks
    _, edge_j, _, _, _ = common.periodic_lloyd_voronoi((u + 0.1 * cloud) % 1.0, v, 4 + (variant % 3), 0.85, rng, lloyd_iters=1)
    joints = np.exp(-edge_j * 16.0) * np.clip((cloud - 0.35) * 3.0, 0.0, 1.0)

    # Dissolution pits / small vugs
    f1_p, _, cid_p, _, _ = common.periodic_voronoi(u, v, pit_cells, 0.90, rng)
    pit_noise = common.periodic_fbm(u, v, pit_cells, 2, 0.5, rng)
    pit_score = (1.0 - f1_p) * (cid_p > 0.52).astype(np.float32) + 0.18 * (pit_noise - 0.5)
    t_p = np.clip((pit_score - 0.66) / 0.08, 0.0, 1.0)
    pits = t_p * t_p * (3.0 - 2.0 * t_p)

    # Fine dark/ochre fossil & bio-clast shell fragment specks
    f1_f, _, cid_f, _, _ = common.periodic_voronoi((u + 0.43) % 1.0, (v + 0.19) % 1.0, pit_cells * 2, 0.94, rng)
    t_f = np.clip(((1.0 - f1_f) * (cid_f > 0.74).astype(np.float32) - 0.70) / 0.06, 0.0, 1.0)
    fossils = t_f * t_f * (3.0 - 2.0 * t_f)

    h = common.normalize_field(0.80 + 0.16 * cloud + 0.10 * chalk - 0.50 * pits - 0.28 * joints - 0.12 * fossils)
    c = common.normalize_field(0.64 + 0.22 * cloud + 0.10 * chalk - 0.38 * pits - 0.26 * joints - 0.26 * fossils)
    rough = np.clip(0.74 + 0.16 * pits + 0.10 * joints + 0.08 * chalk, 0.64, 0.96).astype(np.float32)

    stops = common.resolve_palette(palette, SPEC.default_palette, rng)
    albedo = common.apply_palette_5stop(c, chalk, stops, contrast=1.12)
    return common.shade_pbr_surface(albedo, h, roughness=rough, normal_strength=5.4, spec_intensity=0.10), np.asarray(h, dtype=np.float32)


def generate(seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0) -> Image.Image:
    return _compute(seed=seed, palette=palette, size=size, variant=variant)[0]


def generate_height(seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0) -> np.ndarray:
    return _compute(seed=seed, palette=palette, size=size, variant=variant)[1]

SPEC = GeneratorSpec(
    name="limestone",
    kind="texture",
    tileable=True,
    origin="tools/generators/types/limestone.py",
    default_palette="pale_fossil_limestone",
    description="Pale porous limestone surface with pitted weathering vugs and fine fossil fragment specks.",
    render_fn=generate,
    normal_source="native",
    normal_rule="Native physical surface height field (h) used by shade_pbr_surface.",
    normal_strength=5.4,
    height_fn=generate_height,
)
