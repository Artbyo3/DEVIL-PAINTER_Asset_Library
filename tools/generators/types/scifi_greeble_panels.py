"""Multi-layered sci-fi hull plating, recessed cooling louvers, and rivet panels."""

from __future__ import annotations

import math
import numpy as np
from PIL import Image

from .. import common
from ..interface import GeneratorSpec, PaletteArg


def generate(seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0) -> Image.Image:
    rng = np.random.default_rng(seed)
    u, v = common.make_uv_grid(size)
    grid = 4 + (variant % 4) * 2
    wear = common.periodic_fbm(u, v, grid * 2, 4, 0.5, rng)

    cx = np.floor(u * grid).astype(np.int32) % grid
    cy = np.floor(v * grid).astype(np.int32) % grid
    lu = (u * grid) % 1.0
    lv = (v * grid) % 1.0

    plate_map = rng.random((grid, grid), dtype=np.float32)
    role_map = rng.integers(0, 3, size=(grid, grid))
    plate_h = plate_map[cy, cx]
    role = role_map[cy, cx]

    border = np.minimum(np.minimum(lu, 1.0 - lu), np.minimum(lv, 1.0 - lv))
    seam = np.clip((border - 0.04) / 0.02, 0.0, 1.0)

    louvers = 0.5 + 0.5 * np.sin(lv * 12.0 * math.pi)
    rivet_d = np.sqrt((np.abs(lu - 0.5) - 0.36) ** 2 + (np.abs(lv - 0.5) - 0.36) ** 2)
    rivets = np.clip(1.0 - rivet_d * 18.0, 0.0, 1.0)

    feature = np.where(role == 1, louvers * 0.4, np.where(role == 2, plate_h + rivets * 0.25, plate_h))
    height = common.normalize_field(seam * (0.78 * feature + 0.22 * wear))
    color_field = common.normalize_field(0.65 * height + 0.20 * (role / 2.0) + 0.15 * wear)
    roughness = np.clip(0.25 + 0.45 * wear, 0.15, 0.85).astype(np.float32)

    stops = common.resolve_palette(palette, SPEC.default_palette, rng)
    albedo = common.apply_palette_5stop(color_field, wear, stops, contrast=1.18)
    return common.shade_pbr_surface(albedo, height, roughness=roughness, normal_strength=6.2, spec_intensity=0.32)


SPEC = GeneratorSpec(
    name="scifi-greeble-panels",
    kind="texture",
    tileable=True,
    origin="procedural_textures_v2_300_pack (gen_hard_surface_scifi_greebles; merges V1 scifi_tech & SP arch=5)",
    default_palette="cyberpunk_hull_neon",
    description="Hard-surface sci-fi hull plating with panel seams, cooling louvers, and corner rivets.",
    render_fn=generate,
)
