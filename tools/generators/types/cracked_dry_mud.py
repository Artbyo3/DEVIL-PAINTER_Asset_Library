"""Sun-baked cracked dry mud flat with cupped polygonal clay plates and deep dark fissures."""

from __future__ import annotations

import numpy as np
from PIL import Image

from .. import common
from ..interface import GeneratorSpec, PaletteArg


def generate(seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0) -> Image.Image:
    rng = np.random.default_rng(seed)
    u, v = common.make_uv_grid(size)

    cells = [6, 8, 10, 7][variant % 4]
    wx = common.periodic_fbm(u, v, 6, 3, 0.5, rng)
    wy = common.periodic_fbm((u + 0.37) % 1.0, (v + 0.61) % 1.0, 6, 3, 0.5, rng)
    uu = (u + 0.045 * (wx - 0.5)) % 1.0
    vv = (v + 0.045 * (wy - 0.5)) % 1.0

    _, edge_p, cid_p, _, _ = common.periodic_voronoi(uu, vv, cells, 0.84, rng)
    _, edge_s, _, _, _ = common.periodic_voronoi((uu + 0.29) % 1.0, (vv + 0.43) % 1.0, cells * 2, 0.88, rng)

    main_crack = np.clip((edge_p - 0.025) / 0.025, 0.0, 1.0)
    sub_crack = np.clip((edge_s - 0.014) / 0.016, 0.0, 1.0)
    plate_mask = np.minimum(main_crack, 0.35 + 0.65 * sub_crack)

    # Slight upward cupping curl near plate rims before dropping into cracks
    cup_curl = np.clip(1.0 - edge_p * 2.8, 0.0, 1.0) * 0.16 * main_crack
    silt = common.periodic_fbm(u, v, 24, 3, 0.52, rng)

    h = common.normalize_field(plate_mask * (0.66 + 0.14 * silt + cup_curl))
    c = common.normalize_field(
        np.where(main_crack > 0.25, 0.48 + 0.22 * cid_p + 0.18 * silt + 0.12 * cup_curl, 0.05 + 0.08 * silt)
    )
    rough = np.clip(0.84 + 0.12 * (1.0 - plate_mask), 0.72, 0.96).astype(np.float32)

    stops = common.resolve_palette(palette, SPEC.default_palette, rng)
    albedo = common.apply_palette_5stop(c, silt, stops, contrast=1.18)
    return common.shade_pbr_surface(albedo, h, roughness=rough, normal_strength=7.0, spec_intensity=0.06)


SPEC = GeneratorSpec(
    name="cracked-dry-mud",
    kind="texture",
    tileable=True,
    origin="tools/generators/types/cracked_dry_mud.py",
    default_palette="cracked_dry_mud",
    description="Sun-baked dry mud earth divided into cupped polygonal clay plates by deep jagged shrinkage cracks.",
    render_fn=generate,
)
