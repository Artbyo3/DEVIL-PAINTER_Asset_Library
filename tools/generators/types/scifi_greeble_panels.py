"""Multi-layered sci-fi hull plating, recessed cooling louvers, and rivet panels."""

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
    grid = 4 + (variant % 4) * 2
    wear = common.periodic_fbm(u, v, grid * 2, 4, 0.5, rng)

    # Periodic 2-level quad subdivision (macro hull plates OR 2x2 sub-panels)
    cx0 = np.floor(u * grid).astype(np.int32) % grid
    cy0 = np.floor(v * grid).astype(np.int32) % grid
    split_map = rng.random((grid, grid)) < 0.55
    do_split = split_map[cy0, cx0]

    lu0 = (u * grid) % 1.0
    lv0 = (v * grid) % 1.0
    lu1 = (u * grid * 2.0) % 1.0
    lv1 = (v * grid * 2.0) % 1.0
    lu = np.where(do_split, lu1, lu0)
    lv = np.where(do_split, lv1, lv0)

    sx = np.floor(u * grid * 2.0).astype(np.int32) % (grid * 2)
    sy = np.floor(v * grid * 2.0).astype(np.int32) % (grid * 2)
    hx = np.where(do_split, sx, cx0 * 2)
    hy = np.where(do_split, sy, cy0 * 2)

    plate_map = rng.random((grid * 2, grid * 2), dtype=np.float32)
    role_map = rng.integers(0, 3, size=(grid * 2, grid * 2))
    plate_h = plate_map[hy, hx]
    role = role_map[hy, hx]

    # Chamfered octagonal hull plate SDF (Inigo Quilez chamfered box boundary)
    ax = np.abs(lu - 0.5)
    ay = np.abs(lv - 0.5)
    chamfer = 0.08
    chamfer_d = np.maximum(np.maximum(ax, ay), (ax + ay + chamfer) * 0.70710678)
    edge_dist = 0.46 - chamfer_d
    t_s = np.clip(edge_dist / 0.028, 0.0, 1.0)
    seam = t_s * t_s * (3.0 - 2.0 * t_s)

    # Asymmetric sawtooth cooling louver slats recessed inside louver pocket
    louver_pocket = (ax < 0.32) & (ay < 0.32)
    saw_lv = (lv * 7.0) % 1.0
    louvers = np.where(louver_pocket, np.where(saw_lv < 0.72, saw_lv / 0.72, 1.0 - (saw_lv - 0.72) / 0.28) * 0.55, plate_h)
    # Recessed hex/circular countersunk fasteners at corners
    rivet_d = np.hypot(ax - 0.34, ay - 0.34)
    rivet = 1.0 - np.clip((rivet_d - 0.035) / 0.020, 0.0, 1.0)
    socket = 1.0 - np.clip((rivet_d - 0.015) / 0.010, 0.0, 1.0)
    fasteners = np.clip(rivet * 0.28 - socket * 0.20, -0.25, 0.30)

    feature = np.where(role == 1, louvers, np.where(role == 2, plate_h + fasteners, plate_h + 0.06 * (edge_dist > 0.12)))
    height = common.normalize_field(seam * (0.78 * np.clip(feature, 0.0, 1.2) + 0.22 * wear))
    color_field = common.normalize_field(0.65 * height + 0.20 * (role / 2.0) + 0.15 * wear)
    roughness = np.clip(0.24 + 0.44 * wear + 0.15 * (1.0 - seam), 0.15, 0.85).astype(np.float32)

    stops = common.resolve_palette(palette, SPEC.default_palette, rng)
    albedo = common.apply_palette_5stop(color_field, wear, stops, contrast=1.18)
    return common.shade_pbr_surface(albedo, height, roughness=roughness, normal_strength=6.2, spec_intensity=0.32), np.asarray(height, dtype=np.float32)


def generate(seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0) -> Image.Image:
    return _compute(seed=seed, palette=palette, size=size, variant=variant)[0]


def generate_height(seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0) -> np.ndarray:
    return _compute(seed=seed, palette=palette, size=size, variant=variant)[1]

SPEC = GeneratorSpec(
    name="scifi-greeble-panels",
    kind="texture",
    tileable=True,
    origin="procedural_textures_v2_300_pack (gen_hard_surface_scifi_greebles; merges V1 scifi_tech & SP arch=5)",
    default_palette="cyberpunk_hull_neon",
    description="Hard-surface sci-fi hull plating with panel seams, cooling louvers, and corner rivets.",
    render_fn=generate,
    normal_source="native",
    normal_rule="Native physical surface height field (height) used by shade_pbr_surface.",
    normal_strength=6.2,
    height_fn=generate_height,
)
