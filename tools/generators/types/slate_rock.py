"""Dark foliated slate rock with flat stepped cleavage sheets and sharp fracture lines."""

from __future__ import annotations

import math
import numpy as np
from PIL import Image

from .. import common
from ..interface import GeneratorSpec, PaletteArg


def generate(seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0) -> Image.Image:
    rng = np.random.default_rng(seed)
    u, v = common.make_uv_grid(size)

    sheets = [5, 7, 9, 12][variant % 4]
    tilt = [-0.45, 0.0, 0.45, 0.30][variant % 4]

    warp = common.periodic_fbm(u, v, 3, 3, 0.5, rng, freqy=5)
    # Periodic wave tilt + periodic warp keeps both frac and plate_id seamless in u and v
    coord = (v * sheets + tilt * np.sin(2.0 * math.pi * u) + 0.45 * (warp - 0.5)) % float(sheets)
    plate_vals = rng.uniform(0.0, 1.0, size=sheets).astype(np.float32)
    plate_id = plate_vals[np.floor(coord).astype(np.int32) % sheets]
    frac = coord - np.floor(coord)
    # Flat plate surface with crisp stepped edge drop
    plate_height = 0.72 + 0.22 * np.clip(frac * 12.0, 0.0, 1.0) * (1.0 - 0.18 * frac)

    # Cleavage fissures & diagonal cross-joints
    fissure = common.periodic_fbm(u, v, 4, 4, 0.55, rng, mode="ridged", freqy=10)
    crack_mask = np.clip((fissure - 0.76) * 8.0, 0.0, 1.0)
    fine_grain = common.periodic_fbm(u, v, 24, 3, 0.5, rng)

    h = common.normalize_field(0.68 * plate_height + 0.14 * plate_id - 0.35 * crack_mask + 0.10 * fine_grain)
    c = common.normalize_field(0.42 * plate_height + 0.30 * (1.0 - crack_mask) + 0.18 * plate_id + 0.10 * fine_grain)
    rough = np.clip(0.42 + 0.24 * crack_mask + 0.14 * fine_grain, 0.30, 0.85).astype(np.float32)

    stops = common.resolve_palette(palette, SPEC.default_palette, rng)
    albedo = common.apply_palette_5stop(c, fine_grain, stops, contrast=1.18)
    return common.shade_pbr_surface(albedo, h, roughness=rough, normal_strength=6.2, spec_intensity=0.22)


SPEC = GeneratorSpec(
    name="slate-rock",
    kind="texture",
    tileable=True,
    origin="tools/generators/types/slate_rock.py",
    default_palette="dark_cleavage_slate",
    description="Dark metamorphic slate rock featuring flat stepped cleavage sheets and sharp foliation lines.",
    render_fn=generate,
)
