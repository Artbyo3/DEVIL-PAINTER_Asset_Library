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


def _weave_run_tables(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Precompute continuous float length and phase offset per cell of a periodic (Q, P) weave matrix."""
    q, p = matrix.shape
    m = (matrix > 0).astype(np.int32)
    warp_len = np.ones((q, p), dtype=np.float32)
    warp_pos = np.zeros((q, p), dtype=np.float32)
    weft_len = np.ones((q, p), dtype=np.float32)
    weft_pos = np.zeros((q, p), dtype=np.float32)

    for col in range(p):
        for row in range(q):
            val = m[row, col]
            if val == 1:
                start = row
                steps = 0
                while m[(start - 1) % q, col] == 1 and steps < q:
                    start = (start - 1) % q
                    steps += 1
                length = 1
                while m[(start + length) % q, col] == 1 and length < q:
                    length += 1
                warp_len[row, col] = float(length)
                warp_pos[row, col] = float((row - start) % q)
            else:
                start = col
                steps = 0
                while m[row, (start - 1) % p] == 0 and steps < p:
                    start = (start - 1) % p
                    steps += 1
                length = 1
                while m[row, (start + length) % p] == 0 and length < p:
                    length += 1
                weft_len[row, col] = float(length)
                weft_pos[row, col] = float((col - start) % p)
    return warp_len, warp_pos, weft_len, weft_pos


def woven_fabric_field(
    u: np.ndarray,
    v: np.ndarray,
    threads_u: int,
    threads_v: int,
    matrix: np.ndarray,
    float_arch: float = 0.42,
    ply_twist: float = 0.14,
    warp_radius: np.ndarray | float = 0.94,
    weft_radius: np.ndarray | float = 0.94,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Exact periodic weave-matrix cloth elevation with continuous yarn floats (no sawtooth breaks).

    Args:
        matrix: (Q, P) binary array with 1 = warp over weft, 0 = weft over warp.
    Returns:
        (height, warp_over_mask, warp_profile, weft_profile)
    """
    m = (np.asarray(matrix) > 0).astype(np.int32)
    q, p = m.shape
    tu_n = max(p, ((int(threads_u) + p - 1) // p) * p)
    tv_n = max(q, ((int(threads_v) + q - 1) // q) * q)
    tu = u * tu_n
    tv = v * tv_n
    iu = np.floor(tu).astype(np.int32) % tu_n
    iv = np.floor(tv).astype(np.int32) % tv_n
    fu = tu - np.floor(tu)
    fv = tv - np.floor(tv)

    mi = iu % p
    mj = iv % q
    warp_over = m[mj, mi] == 1

    warp_len, warp_pos, weft_len, weft_pos = _weave_run_tables(m)
    wl = warp_len[mj, mi]
    wp = warp_pos[mj, mi]
    fl = weft_len[mj, mi]
    fp = weft_pos[mj, mi]

    # Longitudinal float phase s in [0, 1] across the entire multi-pick/multi-end float
    s_warp = np.clip((wp + fv) / np.maximum(wl, 1.0), 0.0, 1.0)
    s_weft = np.clip((fp + fu) / np.maximum(fl, 1.0), 0.0, 1.0)
    arch_warp = np.power(np.sin(s_warp * math.pi), float_arch)
    arch_weft = np.power(np.sin(s_weft * math.pi), float_arch)

    # Peirce-inspired elliptical cross-section cylinder with helical 2-ply yarn twist
    du = (fu - 0.5) * 2.0 / np.clip(warp_radius, 0.25, 1.05)
    dv = (fv - 0.5) * 2.0 / np.clip(weft_radius, 0.25, 1.05)
    cyl_u = np.sqrt(np.clip(1.0 - du * du, 0.0, 1.0))
    cyl_v = np.sqrt(np.clip(1.0 - dv * dv, 0.0, 1.0))

    twist_u = 1.0 + ply_twist * np.sin((fu * 2.5 + (wp + fv) * 3.0) * 2.0 * math.pi)
    twist_v = 1.0 + ply_twist * np.sin((fv * 2.5 - (fp + fu) * 3.0) * 2.0 * math.pi)

    top_u = cyl_u * (0.48 + 0.52 * arch_warp) * twist_u
    top_v = cyl_v * (0.48 + 0.52 * arch_weft) * twist_v
    under_u = cyl_u * 0.24
    under_v = cyl_v * 0.24

    height = np.where(warp_over, np.maximum(top_u, under_v), np.maximum(top_v, under_u))
    return height.astype(np.float32), warp_over, top_u.astype(np.float32), top_v.astype(np.float32)


def periodic_lloyd_voronoi(
    u: np.ndarray,
    v: np.ndarray,
    grid_cells: int,
    jitter: float,
    rng: np.random.Generator,
    lloyd_iters: int = 2,
    metric: str = "euclidean",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Periodic Centroidal Voronoi Tessellation via discrete toroidal Lloyd relaxation."""
    n = max(2, int(grid_cells))
    px = rng.uniform(0.5 - 0.5 * jitter, 0.5 + 0.5 * jitter, size=(n, n)).astype(np.float32)
    py = rng.uniform(0.5 - 0.5 * jitter, 0.5 + 0.5 * jitter, size=(n, n)).astype(np.float32)
    cid_map = rng.uniform(0.0, 1.0, size=(n, n)).astype(np.float32)

    if lloyd_iters > 0:
        res = min(max(64, n * 8), 160)
        lin = np.linspace(0.0, 1.0, res, endpoint=False, dtype=np.float32)
        lu, lv = np.meshgrid(lin, lin)
        gx_l = lu * n
        gy_l = lv * n
        ix_l = np.floor(gx_l).astype(np.int32)
        iy_l = np.floor(gy_l).astype(np.int32)

        for _ in range(int(lloyd_iters)):
            best_d = np.full_like(lu, 1e9, dtype=np.float32)
            own_cy = np.zeros_like(ix_l)
            own_cx = np.zeros_like(ix_l)
            rel_x = np.zeros_like(lu)
            rel_y = np.zeros_like(lv)
            for oy in (-1, 0, 1):
                for ox in (-1, 0, 1):
                    cx = (ix_l + ox) % n
                    cy = (iy_l + oy) % n
                    dx_s = (gx_l - ((ix_l + ox) + px[cy, cx]))
                    dy_s = (gy_l - ((iy_l + oy) + py[cy, cx]))
                    dist = dx_s * dx_s + dy_s * dy_s
                    closer = dist < best_d
                    best_d = np.where(closer, dist, best_d)
                    own_cx = np.where(closer, cx, own_cx)
                    own_cy = np.where(closer, cy, own_cy)
                    rel_x = np.where(closer, gx_l - (ix_l + ox), rel_x)
                    rel_y = np.where(closer, gy_l - (iy_l + oy), rel_y)

            flat_idx = own_cy.ravel() * n + own_cx.ravel()
            counts = np.bincount(flat_idx, minlength=n * n).astype(np.float32)
            sum_x = np.bincount(flat_idx, weights=rel_x.ravel(), minlength=n * n).astype(np.float32)
            sum_y = np.bincount(flat_idx, weights=rel_y.ravel(), minlength=n * n).astype(np.float32)
            mask = (counts > 0).reshape(n, n)
            new_px = np.where(mask, (sum_x / np.maximum(counts, 1.0)).reshape(n, n), px)
            new_py = np.where(mask, (sum_y / np.maximum(counts, 1.0)).reshape(n, n), py)
            px = np.clip(0.85 * new_px + 0.15 * px, 0.05, 0.95).astype(np.float32)
            py = np.clip(0.85 * new_py + 0.15 * py, 0.05, 0.95).astype(np.float32)

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


def periodic_gabor_noise(
    u: np.ndarray,
    v: np.ndarray,
    grid_cells: int,
    f0: float,
    omega0: float,
    bandwidth: float,
    rng: np.random.Generator,
    impulses_per_cell: int = 3,
    isotropic: bool = False,
) -> np.ndarray:
    """Periodic Lagae-Lefebvre (2009) sparse convolution Gabor noise on [0, 1)^2."""
    n = max(2, int(grid_cells))
    k_cnt = max(1, int(impulses_per_cell))
    gx = u * n
    gy = v * n
    ix = np.floor(gx).astype(np.int32)
    iy = np.floor(gy).astype(np.int32)

    acc = np.zeros_like(u, dtype=np.float32)
    a2 = float(bandwidth) ** 2
    two_pi = 2.0 * math.pi

    for k in range(k_cnt):
        px = rng.uniform(0.0, 1.0, size=(n, n)).astype(np.float32)
        py = rng.uniform(0.0, 1.0, size=(n, n)).astype(np.float32)
        amp = rng.uniform(-1.0, 1.0, size=(n, n)).astype(np.float32)
        phase = rng.uniform(0.0, two_pi, size=(n, n)).astype(np.float32)
        if isotropic:
            ang = rng.uniform(0.0, two_pi, size=(n, n)).astype(np.float32)
            ca_m, sa_m = np.cos(ang), np.sin(ang)
        else:
            ca_s, sa_s = math.cos(omega0), math.sin(omega0)

        for oy in (-1, 0, 1):
            for ox in (-1, 0, 1):
                cx = (ix + ox) % n
                cy = (iy + oy) % n
                dx = gx - ((ix + ox) + px[cy, cx])
                dy = gy - ((iy + oy) + py[cy, cx])
                env = np.exp(-math.pi * a2 * (dx * dx + dy * dy))
                if isotropic:
                    proj = dx * ca_m[cy, cx] + dy * sa_m[cy, cx]
                else:
                    proj = dx * ca_s + dy * sa_s
                acc += amp[cy, cx] * env * np.cos(two_pi * float(f0) * proj + phase[cy, cx])

    return normalize_field(acc)


def periodic_curl_noise(
    u: np.ndarray,
    v: np.ndarray,
    freq: int,
    octaves: int,
    persistence: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Bridson et al. (2007) divergence-free 2D curl velocity field (vx, vy) from periodic potential."""
    psi = periodic_fbm(u, v, freq, octaves, persistence, rng, mode="fbm")
    h, w = psi.shape
    dpsi_dv = (np.roll(psi, -1, axis=0) - np.roll(psi, 1, axis=0)) * (0.5 * h)
    dpsi_du = (np.roll(psi, -1, axis=1) - np.roll(psi, 1, axis=1)) * (0.5 * w)
    vx = dpsi_dv
    vy = -dpsi_du
    mag = np.sqrt(vx * vx + vy * vy + 1e-8)
    scale = 1.0 / (float(np.percentile(mag, 95)) + 1e-6)
    return (vx * scale).astype(np.float32), (vy * scale).astype(np.float32)


def gray_scott_rd(
    size: int,
    feed: float,
    kill: float,
    du: float = 0.16,
    dv: float = 0.08,
    steps: int = 420,
    rng: np.random.Generator | None = None,
    sim_res: int = 128,
) -> tuple[np.ndarray, np.ndarray]:
    """Periodic 9-point isotropic Laplacian Gray-Scott reaction-diffusion solver (Pearson 1993)."""
    if rng is None:
        rng = np.random.default_rng(1)
    n = min(max(64, int(sim_res)), int(size))
    U = np.ones((n, n), dtype=np.float32)
    V = np.zeros((n, n), dtype=np.float32)

    # Seed periodic stochastic localized perturbations to break uniform steady state
    n_seeds = max(12, (n * n) // 450)
    sx = rng.integers(0, n, size=n_seeds)
    sy = rng.integers(0, n, size=n_seeds)
    for x0, y0 in zip(sx, sy):
        rad = int(rng.integers(2, 5))
        for dy in range(-rad, rad + 1):
            for dx in range(-rad, rad + 1):
                if dx * dx + dy * dy <= rad * rad:
                    yy = (y0 + dy) % n
                    xx = (x0 + dx) % n
                    U[yy, xx] = 0.50
                    V[yy, xx] = 0.25 + float(rng.uniform(0.0, 0.08))
    noise = rng.uniform(-0.015, 0.015, size=(n, n)).astype(np.float32)
    V = np.clip(V + np.maximum(0.0, noise), 0.0, 1.0)

    f = float(feed)
    k = float(kill)
    fk = f + k
    dt = 1.0
    for _ in range(int(steps)):
        lap_u = (
            0.20 * (np.roll(U, 1, axis=0) + np.roll(U, -1, axis=0) + np.roll(U, 1, axis=1) + np.roll(U, -1, axis=1))
            + 0.05 * (np.roll(U, (1, 1), axis=(0, 1)) + np.roll(U, (1, -1), axis=(0, 1)) + np.roll(U, (-1, 1), axis=(0, 1)) + np.roll(U, (-1, -1), axis=(0, 1)))
            - U
        )
        lap_v = (
            0.20 * (np.roll(V, 1, axis=0) + np.roll(V, -1, axis=0) + np.roll(V, 1, axis=1) + np.roll(V, -1, axis=1))
            + 0.05 * (np.roll(V, (1, 1), axis=(0, 1)) + np.roll(V, (1, -1), axis=(0, 1)) + np.roll(V, (-1, 1), axis=(0, 1)) + np.roll(V, (-1, -1), axis=(0, 1)))
            - V
        )
        uvv = U * V * V
        U += (du * lap_u - uvv + f * (1.0 - U)) * dt
        V += (dv * lap_v + uvv - fk * V) * dt

    if n != size:
        # Periodic bicubic resize via tiled edge padding to preserve exact periodicity
        pad = 4
        Vp = np.pad(V, pad, mode="wrap")
        Up = np.pad(U, pad, mode="wrap")
        scale = size / float(n)
        out_n = int(round((n + 2 * pad) * scale))
        V_img = Image.fromarray(Vp, mode="F").resize((out_n, out_n), resample=Image.Resampling.BICUBIC)
        U_img = Image.fromarray(Up, mode="F").resize((out_n, out_n), resample=Image.Resampling.BICUBIC)
        off = int(round(pad * scale))
        V = np.asarray(V_img, dtype=np.float32)[off : off + size, off : off + size]
        U = np.asarray(U_img, dtype=np.float32)[off : off + size, off : off + size]

    return normalize_field(U), normalize_field(V)


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


def height_to_normal(height: np.ndarray, strength: float = 6.0) -> Image.Image:
    """Convert a 2D height field into a periodic OpenGL (+Y up, Blender) tangent-space normal map."""
    h = normalize_field(height)
    h_right = np.roll(h, -1, axis=1)
    h_left = np.roll(h, 1, axis=1)
    h_up = np.roll(h, 1, axis=0)
    h_down = np.roll(h, -1, axis=0)

    # 3x3 periodic Sobel derivatives: +X right, +Y up (row index decreases upward)
    dh_dx = (
        (np.roll(h_right, 1, axis=0) - np.roll(h_left, 1, axis=0))
        + 2.0 * (h_right - h_left)
        + (np.roll(h_right, -1, axis=0) - np.roll(h_left, -1, axis=0))
    ) * 0.125
    dh_dy_up = (
        (np.roll(h_up, 1, axis=1) - np.roll(h_down, 1, axis=1))
        + 2.0 * (h_up - h_down)
        + (np.roll(h_up, -1, axis=1) - np.roll(h_down, -1, axis=1))
    ) * 0.125

    s = float(strength)
    nx = -dh_dx * s
    ny = -dh_dy_up * s
    nz = np.ones_like(h, dtype=np.float32)
    inv_len = 1.0 / np.sqrt(nx * nx + ny * ny + nz * nz + 1e-12)
    nx = nx * inv_len
    ny = ny * inv_len
    nz = nz * inv_len

    rgb = np.stack([nx * 0.5 + 0.5, ny * 0.5 + 0.5, nz * 0.5 + 0.5], axis=-1)
    rgb_u8 = np.rint(np.clip(rgb * 255.0, 0.0, 255.0)).astype(np.uint8)
    return Image.fromarray(rgb_u8, mode="RGB")


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
