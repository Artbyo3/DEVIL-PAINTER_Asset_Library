"""Sun-baked cracked dry mud flat with cupped polygonal clay plates and deep dark fissures."""

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

    cells = [6, 8, 10, 7][variant % 4]
    wx = common.periodic_fbm(u, v, 6, 3, 0.5, rng)
    wy = common.periodic_fbm((u + 0.37) % 1.0, (v + 0.61) % 1.0, 6, 3, 0.5, rng)
    uu = (u + 0.042 * (wx - 0.5)) % 1.0
    vv = (v + 0.042 * (wy - 0.5)) % 1.0

    # Goehring (2010, 2013): cyclic wetting-drying relaxes desiccation cracks toward a CVT (~120 deg Y-junctions)
    _, edge_p, cid_p, _, _ = common.periodic_lloyd_voronoi(uu, vv, cells, 0.84, rng, lloyd_iters=2)
    _, edge_s, _, _, _ = common.periodic_voronoi((uu + 0.29) % 1.0, (vv + 0.43) % 1.0, cells * 2, 0.88, rng)

    tp = np.clip((edge_p - 0.022) / 0.030, 0.0, 1.0)
    main_crack = tp * tp * (3.0 - 2.0 * tp)
    ts = np.clip((edge_s - 0.012) / 0.018, 0.0, 1.0)
    sub_crack = ts * ts * (3.0 - 2.0 * ts)
    plate_mask = np.minimum(main_crack, 0.38 + 0.62 * sub_crack)

    # Differential vertical shrinkage: concave upward plate-edge curl peaking just inside the fissure lip
    t_curl = np.clip((0.40 - edge_p) / 0.36, 0.0, 1.0)
    cup_curl = (t_curl * t_curl) * 0.22 * main_crack
    silt = common.periodic_fbm(u, v, 24, 3, 0.52, rng)

    h = common.normalize_field(plate_mask * (0.64 + 0.14 * silt + cup_curl))
    c = common.normalize_field(
        main_crack * (0.48 + 0.22 * cid_p + 0.18 * silt + 0.12 * cup_curl) + (1.0 - main_crack) * (0.04 + 0.07 * silt)
    )
    rough = np.clip(0.84 + 0.12 * (1.0 - plate_mask), 0.72, 0.96).astype(np.float32)

    stops = common.resolve_palette(palette, SPEC.default_palette, rng)
    albedo = common.apply_palette_5stop(c, silt, stops, contrast=1.18)
    return common.shade_pbr_surface(albedo, h, roughness=rough, normal_strength=7.0, spec_intensity=0.06), np.asarray(h, dtype=np.float32)


def generate(seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0) -> Image.Image:
    return _compute(seed=seed, palette=palette, size=size, variant=variant)[0]


def generate_height(seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0) -> np.ndarray:
    return _compute(seed=seed, palette=palette, size=size, variant=variant)[1]

SPEC = GeneratorSpec(
    name="cracked-dry-mud",
    kind="texture",
    tileable=True,
    origin="tools/generators/types/cracked_dry_mud.py",
    default_palette="cracked_dry_mud",
    description="Sun-baked dry mud earth divided into cupped polygonal clay plates by deep jagged shrinkage cracks.",
    render_fn=generate,
    normal_source="native",
    normal_rule="Native physical surface height field (h) used by shade_pbr_surface.",
    normal_strength=7.0,
    height_fn=generate_height,
)
