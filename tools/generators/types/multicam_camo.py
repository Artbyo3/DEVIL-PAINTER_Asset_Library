"""MultiCam camouflage with gradient background, soft elongated blobs, and fine light/dark speckles."""

from __future__ import annotations

import math
import numpy as np
from PIL import Image

from .. import common
from ..interface import GeneratorSpec, PaletteArg


def _smoothstep(x: np.ndarray, edge0: float, edge1: float) -> np.ndarray:
    t = np.clip((x - edge0) / max(edge1 - edge0, 1e-6), 0.0, 1.0)
    return (t * t * (3.0 - 2.0 * t)).astype(np.float32)


def _compute(
    seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0
) -> tuple[Image.Image, np.ndarray]:
    rng = np.random.default_rng(seed)
    u, v = common.make_uv_grid(size)

    stops = np.array(common.resolve_palette(palette, SPEC.default_palette, rng), dtype=np.float32)

    # 1. Soft multi-hue periodic gradient background (olive-sage <-> warm khaki <-> pale cream)
    grad = common.periodic_fbm(u, v, 2, 3, 0.5, rng, freqy=2)
    grad_b = common.periodic_fbm((u + 0.37) % 1.0, (v + 0.53) % 1.0, 2, 2, 0.5, rng, freqy=3)
    w_sage = _smoothstep(grad, 0.30, 0.65)
    w_cream = _smoothstep(grad_b, 0.48, 0.80) * 0.55
    albedo = stops[3] * (1.0 - w_sage[..., None]) + stops[2] * w_sage[..., None]
    albedo = albedo * (1.0 - w_cream[..., None]) + stops[4] * w_cream[..., None]

    # 2. Soft elongated organic blobs (brown, olive-sage, pale cream)
    wx = common.periodic_fbm(u, v, 2, 3, 0.5, rng, freqy=4)
    wy = common.periodic_fbm((u + 0.43) % 1.0, (v + 0.19) % 1.0, 2, 3, 0.5, rng, freqy=4)
    uu = (u + 0.18 * (wx - 0.5)) % 1.0
    vv = (v + 0.14 * (wy - 0.5)) % 1.0

    fx = 2 + (variant % 2)
    fy = 5 + ((variant // 2) % 2)
    b_sage = _smoothstep(common.periodic_fbm((uu + 0.11) % 1.0, (vv + 0.71) % 1.0, fx, 3, 0.48, rng, freqy=fy), 0.49, 0.57)
    b_brown = _smoothstep(common.periodic_fbm((uu + 0.61) % 1.0, (vv + 0.23) % 1.0, fx, 3, 0.48, rng, freqy=fy + 1), 0.53, 0.60)
    b_cream = _smoothstep(common.periodic_fbm((uu + 0.29) % 1.0, (vv + 0.47) % 1.0, fx + 1, 3, 0.46, rng, freqy=fy + 1), 0.60, 0.67)

    albedo = albedo * (1.0 - 0.82 * b_sage[..., None]) + stops[2] * (0.82 * b_sage[..., None])
    albedo = albedo * (1.0 - 0.88 * b_brown[..., None]) + stops[1] * (0.88 * b_brown[..., None])
    albedo = albedo * (1.0 - 0.72 * b_cream[..., None]) + stops[4] * (0.72 * b_cream[..., None])

    # 3. Fine light and dark twig/speckle accents (anisotropic stretched Voronoi + cluster gate)
    f1_l, _, cid_l, _, _ = common.periodic_voronoi(u, (v * 2.0) % 1.0, 18, 0.92, rng)
    cl_l = common.periodic_fbm(u, v, 3, 2, 0.5, rng, freqy=4)
    sp_light = _smoothstep((1.0 - f1_l) * (cid_l > 0.35).astype(np.float32) + 0.25 * (cl_l - 0.5), 0.72, 0.76)

    f1_d, _, cid_d, _, _ = common.periodic_voronoi((u + 0.47) % 1.0, ((v * 2.0) + 0.31) % 1.0, 20, 0.92, rng)
    cl_d = common.periodic_fbm((u + 0.53) % 1.0, (v + 0.19) % 1.0, 3, 2, 0.5, rng, freqy=4)
    sp_dark = _smoothstep((1.0 - f1_d) * (cid_d > 0.40).astype(np.float32) + 0.25 * (cl_d - 0.5), 0.73, 0.77)

    albedo = albedo * (1.0 - sp_light[..., None]) + stops[4] * sp_light[..., None]
    albedo = albedo * (1.0 - sp_dark[..., None]) + stops[0] * sp_dark[..., None]

    slub = common.periodic_fbm(u, v, 10, 2, 0.5, rng)
    albedo = albedo * (0.96 + 0.08 * slub[..., None])

    threads = max(24, min(96, size // 6))
    twill = 0.5 + 0.5 * np.sin(((u + v) * threads) * 2.0 * math.pi) * np.sin((u * threads * 2.0) * math.pi)
    h = common.normalize_field(0.75 * twill + 0.25 * slub)
    return common.shade_cloth_surface(albedo, h, normal_strength=2.0, sheen_amt=5.0), np.asarray(h, dtype=np.float32)


def generate(seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0) -> Image.Image:
    return _compute(seed=seed, palette=palette, size=size, variant=variant)[0]


def generate_height(seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0) -> np.ndarray:
    return _compute(seed=seed, palette=palette, size=size, variant=variant)[1]

SPEC = GeneratorSpec(
    name="multicam-camo",
    kind="texture",
    tileable=True,
    origin="tools/generators/types/multicam_camo.py (MultiCam)",
    default_palette="crye_multicam",
    description="MultiCam-style camouflage featuring a multi-tone gradient canvas, soft elongated blobs, and fine light and dark speckles.",
    render_fn=generate,
    normal_source="native",
    normal_rule="Native physical surface height field (h) used by shade_cloth_surface.",
    normal_strength=2.0,
    height_fn=generate_height,
)
