"""Studio MatCap shading sphere texture for viewport previewing."""

from __future__ import annotations

import numpy as np
from PIL import Image

from .. import common
from ..interface import GeneratorSpec, PaletteArg


def _compute(
    seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0
) -> tuple[Image.Image, np.ndarray]:
    rng = np.random.default_rng(seed)
    _ = variant
    u, v = common.make_uv_grid(size)
    x = (u - 0.5) * 2.0
    y = (v - 0.5) * 2.0
    r2 = x * x + y * y
    r = np.sqrt(r2)
    mask = r2 <= 1.0
    sz = np.sqrt(np.clip(1.0 - r2, 0.0, 1.0))

    # Normalized key light L and Blinn-Phong half-vector H = normalize(L + V) with V=(0,0,1)
    import math
    lx, ly, lz = -0.45, -0.55, 0.703562
    ln = math.sqrt(lx * lx + ly * ly + lz * lz)
    lx, ly, lz = lx / ln, ly / ln, lz / ln
    hx, hy, hz = lx, ly, lz + 1.0
    hn = math.sqrt(hx * hx + hy * hy + hz * hz)
    hx, hy, hz = hx / hn, hy / hn, hz / hn

    stops = common.resolve_palette(palette, SPEC.default_palette, rng)
    ndotl = np.clip(lx * x + ly * y + lz * sz, 0.0, 1.0)
    # Schlick (1994) Fresnel approximation (F0=0.04 dielectric -> 1 at grazing angle)
    fresnel = 0.04 + 0.96 * np.power(1.0 - sz, 5.0)
    ndoth = np.clip(hx * x + hy * y + hz * sz, 0.0, 1.0)
    spec = np.power(ndoth, 40.0)

    field = np.clip(0.76 * ndotl + 0.24 * fresnel, 0.0, 1.0)
    rgb = common.apply_palette_5stop(field, None, stops)
    rgb = rgb + (spec * 225.0)[..., None] + (fresnel * 75.0)[..., None]
    aa = np.clip((1.004 - r) / 0.008, 0.0, 1.0)
    rgb = np.where(aa[..., None] > 0.0, np.clip(rgb, 0.0, 255.0) * aa[..., None] + 18.0 * (1.0 - aa[..., None]), 18.0)
    height = np.where(mask, sz, 0.0).astype(np.float32)
    return common.rgb_image(rgb), height


def generate(seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0) -> Image.Image:
    return _compute(seed=seed, palette=palette, size=size, variant=variant)[0]


def generate_height(seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0) -> np.ndarray:
    return _compute(seed=seed, palette=palette, size=size, variant=variant)[1]


SPEC = GeneratorSpec(
    name="matcap-sphere",
    kind="texture",
    tileable=False,
    tile_axes="none",
    origin="substance_painter_master_600_pack/procedural_texture_generator.py (Suite 6 style=3)",
    default_palette="anodized_titanium_heat",
    description="Non-tiling studio MatCap shading sphere for Blender/ZBrush/Substance material capture previews.",
    render_fn=generate,
    normal_source="derived",
    normal_rule="Derived hemisphere elevation z = sqrt(max(0, 1 - r^2)) within sphere disk mask (0 outside).",
    normal_strength=12.0,
    height_fn=generate_height,
)
