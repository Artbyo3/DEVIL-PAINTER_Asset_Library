"""Shared periodic procedural primitives, shaders, and image helpers.

Consolidates the best v2/SP/Cloth periodic gradient Perlin, quintic interpolation,
multi-mode fBm, 3-metric Voronoi, separable box blur, 5-stop palette mapping,
and PBR/cloth surface shading used across all generator modules.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from PIL import Image

try:
    from .palettes import PALETTES, PALETTE_NAMES, resolve_palette
except ImportError:
    from palettes import PALETTES, PALETTE_NAMES, resolve_palette  # type: ignore[no-redef]


def make_uv_grid(size: int) -> tuple[np.ndarray, np.ndarray]:
    lin = np.linspace(0.0, 1.0, size, endpoint=False, dtype=np.float32)
    return np.meshgrid(lin, lin)


def quintic(t: np.ndarray) -> np.ndarray:
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def normalize_field(arr: np.ndarray) -> np.ndarray:
    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)
    mn = float(arr.min())
    mx = float(arr.max())
    if mx - mn < 1e-7:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr - mn) / (mx - mn)


def periodic_gradient_perlin(
    u: np.ndarray,
    v: np.ndarray,
    freqx: int,
    freqy: int | None,
    rng: np.random.Generator,
) -> np.ndarray:
    fx = max(1, int(freqx))
    fy = max(1, int(freqy if freqy is not None else fx))
    angles = rng.uniform(0.0, 2.0 * math.pi, size=(fy, fx))
    gx = np.cos(angles).astype(np.float32)
    gy = np.sin(angles).astype(np.float32)

    x = (u * fx) % fx
    y = (v * fy) % fy
    x0 = np.floor(x).astype(np.int32) % fx
    y0 = np.floor(y).astype(np.int32) % fy
    x1 = (x0 + 1) % fx
    y1 = (y0 + 1) % fy

    dx = x - np.floor(x)
    dy = y - np.floor(y)

    n00 = gx[y0, x0] * dx + gy[y0, x0] * dy
    n10 = gx[y0, x1] * (dx - 1.0) + gy[y0, x1] * dy
    n01 = gx[y1, x0] * dx + gy[y1, x0] * (dy - 1.0)
    n11 = gx[y1, x1] * (dx - 1.0) + gy[y1, x1] * (dy - 1.0)

    sx = quintic(dx)
    sy = quintic(dy)
    nx0 = n00 + sx * (n10 - n00)
    nx1 = n01 + sx * (n11 - n01)
    n = nx0 + sy * (nx1 - nx0)
    return np.clip(n * 0.70710678 + 0.5, 0.0, 1.0).astype(np.float32)


def periodic_fbm(
    u: np.ndarray,
    v: np.ndarray,
    freqx: int,
    octaves: int,
    persistence: float,
    rng: np.random.Generator,
    mode: str = "fbm",
    freqy: int | None = None,
) -> np.ndarray:
    fy_base = max(1, int(freqy if freqy is not None else freqx))
    fx_base = max(1, int(freqx))
    total = np.zeros_like(u, dtype=np.float32)
    amp = 1.0
    amp_sum = 0.0
    weight = 1.0
    dx_acc = np.zeros_like(u, dtype=np.float32)
    dy_acc = np.zeros_like(v, dtype=np.float32)

    for i in range(max(1, int(octaves))):
        fx = min(fx_base * (1 << i), 128)
        fy = min(fy_base * (1 << i), 128)
        if mode == "swiss":
            uu = (u + 0.04 * dx_acc) % 1.0
            vv = (v + 0.04 * dy_acc) % 1.0
            n = periodic_gradient_perlin(uu, vv, fx, fy, rng)
            ridged = 1.0 - np.abs(n * 2.0 - 1.0)
            signal = ridged * ridged * weight
            weight = np.clip(signal * 1.8, 0.1, 1.0)
            dx_acc += (np.roll(n, -1, axis=1) - np.roll(n, 1, axis=1)) * 0.5
            dy_acc += (np.roll(n, -1, axis=0) - np.roll(n, 1, axis=0)) * 0.5
        else:
            n = periodic_gradient_perlin(u, v, fx, fy, rng)
            if mode == "turbulence":
                signal = np.abs(n * 2.0 - 1.0)
            elif mode == "billow":
                signal = 1.0 - np.abs(n * 2.0 - 1.0)
            elif mode == "ridged":
                signal = 1.0 - np.abs(n * 2.0 - 1.0)
                signal = signal * signal
            else:
                signal = n
        total += signal * amp
        amp_sum += amp
        amp *= persistence

    return normalize_field(total / max(amp_sum, 1e-6))


def periodic_voronoi(
    u: np.ndarray,
    v: np.ndarray,
    grid_cells: int,
    jitter: float,
    rng: np.random.Generator,
    metric: str = "euclidean",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = max(1, int(grid_cells))
    px = rng.uniform(0.5 - 0.5 * jitter, 0.5 + 0.5 * jitter, size=(n, n)).astype(np.float32)
    py = rng.uniform(0.5 - 0.5 * jitter, 0.5 + 0.5 * jitter, size=(n, n)).astype(np.float32)
    cid_map = rng.uniform(0.0, 1.0, size=(n, n)).astype(np.float32)

    gx = u * n
    gy = v * n
    ix = np.floor(gx).astype(np.int32)
    iy = np.floor(gy).astype(np.int32)

    f1 = np.full_like(u, 1e9, dtype=np.float32)
    f2 = np.full_like(u, 1e9, dtype=np.float32)
    cell_id = np.zeros_like(u, dtype=np.float32)
    best_dx = np.zeros_like(u, dtype=np.float32)
    best_dy = np.zeros_like(v, dtype=np.float32)

    for oy in (-1, 0, 1):
        for ox in (-1, 0, 1):
            cx = (ix + ox) % n
            cy = (iy + oy) % n
            fx = (ix + ox) + px[cy, cx] - gx
            fy = (iy + oy) + py[cy, cx] - gy
            if metric == "manhattan":
                dist = np.abs(fx) + np.abs(fy)
            elif metric == "chebyshev":
                dist = np.maximum(np.abs(fx), np.abs(fy))
            else:
                dist = np.sqrt(fx * fx + fy * fy)

            closer1 = dist < f1
            between = (~closer1) & (dist < f2)
            f2 = np.where(closer1, f1, np.where(between, dist, f2))
            f1 = np.where(closer1, dist, f1)
            cell_id = np.where(closer1, cid_map[cy, cx], cell_id)
            best_dx = np.where(closer1, fx, best_dx)
            best_dy = np.where(closer1, fy, best_dy)

    edge = np.clip(f2 - f1, 0.0, 1.0)
    return normalize_field(f1), normalize_field(edge), cell_id, best_dx, best_dy


def periodic_box_blur(arr: np.ndarray, radius: int) -> np.ndarray:
    r = max(1, int(radius))
    acc_x = np.zeros_like(arr, dtype=np.float32)
    for k in range(-r, r + 1):
        acc_x += np.roll(arr, k, axis=1)
    acc_x /= float(2 * r + 1)

    acc_y = np.zeros_like(arr, dtype=np.float32)
    for k in range(-r, r + 1):
        acc_y += np.roll(acc_x, k, axis=0)
    return acc_y / float(2 * r + 1)


def apply_palette_5stop(
    field: np.ndarray,
    micro_detail: np.ndarray | None,
    stops: Sequence[tuple[int, int, int]],
    contrast: float = 1.0,
    gamma: float = 1.0,
) -> np.ndarray:
    f = normalize_field(field)
    if micro_detail is not None:
        f = np.clip(f + (micro_detail - 0.5) * 0.14, 0.0, 1.0)
    if abs(gamma - 1.0) > 1e-4:
        f = np.power(np.clip(f, 0.0, 1.0), gamma)
    if abs(contrast - 1.0) > 1e-4:
        f = np.clip((f - 0.5) * contrast + 0.5, 0.0, 1.0)

    n_stops = len(stops)
    stop_pos = np.linspace(0.0, 1.0, n_stops, dtype=np.float32)
    out = np.empty((*f.shape, 3), dtype=np.float32)
    for ch in range(3):
        vals = np.array([float(s[ch]) for s in stops], dtype=np.float32)
        out[..., ch] = np.interp(f, stop_pos, vals)
    return out


def shade_pbr_surface(
    albedo_rgb: np.ndarray,
    height: np.ndarray,
    roughness: np.ndarray | float | None = None,
    normal_strength: float = 5.5,
    spec_intensity: float = 0.28,
    ao_strength: float = 0.30,
    cavity_strength: float = 0.14,
    light_dir: tuple[float, float, float] = (0.48, -0.58, 0.66),
) -> Image.Image:
    h = normalize_field(height)
    r = (
        np.full_like(h, 0.55 if roughness is None else float(roughness), dtype=np.float32)
        if not isinstance(roughness, np.ndarray)
        else np.clip(roughness, 0.05, 0.98).astype(np.float32)
    )

    dh_dx = (np.roll(h, -1, axis=1) - np.roll(h, 1, axis=1)) * 0.5 * normal_strength
    dh_dy = (np.roll(h, -1, axis=0) - np.roll(h, 1, axis=0)) * 0.5 * normal_strength
    nx, ny, nz = -dh_dx, -dh_dy, np.ones_like(h, dtype=np.float32)
    inv_len = 1.0 / np.sqrt(nx * nx + ny * ny + nz * nz + 1e-8)
    nx, ny, nz = nx * inv_len, ny * inv_len, nz * inv_len

    lx, ly, lz = light_dir
    ln = math.sqrt(lx * lx + ly * ly + lz * lz)
    lx, ly, lz = lx / ln, ly / ln, lz / ln

    ndotl = np.clip(nx * lx + ny * ly + nz * lz, 0.0, 1.0)
    hx, hy, hz = lx, ly, lz + 1.0
    hn = math.sqrt(hx * hx + hy * hy + hz * hz)
    hx, hy, hz = hx / hn, hy / hn, hz / hn
    ndoth = np.clip(nx * hx + ny * hy + nz * hz, 0.0, 1.0)

    shininess = np.clip((1.0 - r) ** 2 * 96.0 + 4.0, 4.0, 100.0)
    spec = np.power(ndoth, shininess) * spec_intensity * (1.0 - r * 0.85)

    blur_h = periodic_box_blur(h, 5)
    cavity = np.clip(blur_h - h, 0.0, 1.0)
    ao = 1.0 - np.clip((1.0 - h) * ao_strength + cavity * cavity_strength * 2.5, 0.0, 0.45)

    diffuse = (0.72 + 0.36 * ndotl) * ao
    lit = albedo_rgb * diffuse[..., None] + (spec * 255.0)[..., None]
    return Image.fromarray(np.clip(lit, 0.0, 255.0).astype(np.uint8), mode="RGB")


def shade_cloth_surface(
    albedo_rgb: np.ndarray,
    height: np.ndarray,
    normal_strength: float = 5.5,
    sheen_amt: float = 18.0,
) -> Image.Image:
    h = normalize_field(height)
    dh_dx = (np.roll(h, -1, axis=1) - np.roll(h, 1, axis=1)) * 0.5 * normal_strength
    dh_dy = (np.roll(h, -1, axis=0) - np.roll(h, 1, axis=0)) * 0.5 * normal_strength
    nx, ny, nz = -dh_dx, -dh_dy, np.ones_like(h, dtype=np.float32)
    inv_len = 1.0 / np.sqrt(nx * nx + ny * ny + nz * nz + 1e-8)
    nx, ny, nz = nx * inv_len, ny * inv_len, nz * inv_len

    ao = np.clip(0.55 + 0.45 * h, 0.0, 1.0)
    sheen = np.power(np.clip(1.0 - nz, 0.0, 1.0), 1.6)
    ndotl = np.clip(nx * 0.42 - ny * 0.52 + nz * 0.74, 0.0, 1.0)
    lit = albedo_rgb * (0.64 + 0.24 * ao + 0.16 * ndotl)[..., None] + (sheen * sheen_amt)[..., None]
    return Image.fromarray(np.clip(lit, 0.0, 255.0).astype(np.uint8), mode="RGB")


def compute_sobel_normals_and_baking(
    height: np.ndarray,
    normal_strength: float = 5.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    h = normalize_field(height)
    dx = (
        (np.roll(h, (-1, -1), axis=(0, 1)) - np.roll(h, (-1, 1), axis=(0, 1)))
        + 2.0 * (np.roll(h, -1, axis=1) - np.roll(h, 1, axis=1))
        + (np.roll(h, (1, -1), axis=(0, 1)) - np.roll(h, (1, 1), axis=(0, 1)))
    ) * 0.25 * normal_strength
    dy = (
        (np.roll(h, (-1, -1), axis=(0, 1)) - np.roll(h, (1, -1), axis=(0, 1)))
        + 2.0 * (np.roll(h, -1, axis=0) - np.roll(h, 1, axis=0))
        + (np.roll(h, (-1, 1), axis=(0, 1)) - np.roll(h, (1, 1), axis=(0, 1)))
    ) * 0.25 * normal_strength

    nx, ny, nz = -dx, -dy, np.ones_like(h, dtype=np.float32)
    inv_len = 1.0 / np.sqrt(nx * nx + ny * ny + nz * nz + 1e-8)
    nx, ny, nz = nx * inv_len, ny * inv_len, nz * inv_len

    normal_rgb = np.clip(np.stack([nx * 0.5 + 0.5, ny * 0.5 + 0.5, nz * 0.5 + 0.5], axis=-1) * 255.0, 0.0, 255.0)
    blur = periodic_box_blur(h, 6)
    ao = np.clip(0.55 + 0.45 * h - np.clip(blur - h, 0.0, 1.0) * 0.65, 0.15, 1.0)
    curv = np.clip(0.5 + (h - blur) * 3.2, 0.0, 1.0)
    return normal_rgb.astype(np.uint8), ao, curv, nx, ny


def alpha_image(field: np.ndarray) -> Image.Image:
    f = normalize_field(field)
    return Image.fromarray(np.clip(f * 255.0, 0.0, 255.0).astype(np.uint8), mode="L")


def grayscale_texture_image(field: np.ndarray) -> Image.Image:
    f = normalize_field(field)
    arr = np.clip(f * 255.0, 0.0, 255.0).astype(np.uint8)
    return Image.fromarray(np.stack([arr, arr, arr], axis=-1), mode="RGB")


def rgb_image(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(np.clip(arr, 0.0, 255.0).astype(np.uint8), mode="RGB")


def verify_edge_tiling(img: Image.Image, axes: str = "both") -> dict[str, float]:
    """Measure edge wrap continuity relative to interior adjacent pixel steps."""
    _ = axes
    arr = np.asarray(img, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[..., None]
    h, w, _ = arr.shape
    lr_wrap = float(np.mean(np.abs(arr[:, 0, :] - arr[:, w - 1, :])))
    tb_wrap = float(np.mean(np.abs(arr[0, :, :] - arr[h - 1, :, :])))
    lr_diffs = np.mean(np.abs(arr[:, 1:, :] - arr[:, :-1, :]), axis=(0, 2))
    tb_diffs = np.mean(np.abs(arr[1:, :, :] - arr[:-1, :, :]), axis=(1, 2))
    return {
        "lr_wrap": lr_wrap,
        "lr_interior": float(np.mean(lr_diffs)),
        "lr_interior_max": float(np.max(lr_diffs)),
        "tb_wrap": tb_wrap,
        "tb_interior": float(np.mean(tb_diffs)),
        "tb_interior_max": float(np.max(tb_diffs)),
    }
