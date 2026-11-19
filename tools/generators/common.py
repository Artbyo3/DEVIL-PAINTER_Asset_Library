"""Shared periodic procedural primitives, shaders, and image helpers.

Provides exact toroidal [0, 1)^2 procedural math used across all generator modules:
- Resolution-independent periodic quintic gradient Perlin with analytic derivatives
  and Gaussian-integer toroidal octave rotation (zero Cartesian grid bias, zero seams).
- Nyquist-aware multi-mode fBm (`fbm`, `turbulence`, `billow`, `ridged`, `swiss`) with
  per-octave phase shifts, true analytic Swiss slope advection, and outlier-resistant
  percentile shaping.
- Periodic nested / curl domain-warped fBm (`periodic_domain_warp_fbm`).
- Periodic Worley Voronoi & Roberts-R2 free toroidal Lloyd CVT with exact Inigo Quilez
  perpendicular bisector edge distance (eliminates corner pinching and grid alignment).
- O(N) prefix-sum periodic box blur and exact FFT periodic isotropic Gaussian blur.
- Histogram / target-range shaping (`robust_normalize`, `match_histogram_range`, `s_curve`).
- Resolution-independent periodic 3x3 Scharr OpenGL (+Y up) normal derivation,
  microfacet/cavity roughness derivation, and multi-scale Gaussian AO PBR/cloth shaders.
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


# Precomputed 1024-entry unit-circle gradient LUT for fast vectorized periodic hashing.
_GRAD_LUT_SIZE = 1024
_GRAD_LUT_MASK = _GRAD_LUT_SIZE - 1
_GRAD_LUT_ANGLES = np.linspace(0.0, 2.0 * math.pi, _GRAD_LUT_SIZE, endpoint=False, dtype=np.float32)
_GRAD_LUT_GX = np.cos(_GRAD_LUT_ANGLES)
_GRAD_LUT_GY = np.sin(_GRAD_LUT_ANGLES)


def make_uv_grid(size: int) -> tuple[np.ndarray, np.ndarray]:
    lin = np.linspace(0.0, 1.0, size, endpoint=False, dtype=np.float32)
    return np.meshgrid(lin, lin)


def quintic(t: np.ndarray) -> np.ndarray:
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def quintic_deriv(t: np.ndarray) -> np.ndarray:
    """Analytic first derivative of Perlin's C2 quintic curve: 30 t^2 (t - 1)^2."""
    tm1 = t - 1.0
    return 30.0 * t * t * tm1 * tm1


def normalize_field(arr: np.ndarray) -> np.ndarray:
    """Exact min-max normalization to [0, 1] (preserves sparse/binary masks)."""
    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)
    mn = float(arr.min())
    mx = float(arr.max())
    if mx - mn < 1e-7:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr - mn) / (mx - mn)


def robust_normalize(
    arr: np.ndarray,
    low_pct: float = 0.2,
    high_pct: float = 99.8,
    soft: bool = True,
) -> np.ndarray:
    """Percentile-anchored normalization to [0, 1], immune to rare extreme-value spikes.

    At 2048x2048 (4.2M pixels), Gaussian-like stochastic sums have ~5-sigma outlier
    pixels that compress 99% of midtone contrast under naive min/max normalization.
    When `soft=True`, tail values outside [p_low, p_high] compress smoothly via a C1
    rational shoulder inside [0, 1] instead of hard-clipping (zero banding/flat plateaus).
    """
    a = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)
    lo, hi = np.percentile(a, [float(low_pct), float(high_pct)])
    span = float(hi - lo)
    if span < 1e-7:
        return normalize_field(a)
    x = (a - float(lo)) / span
    if not soft:
        return np.clip(x, 0.0, 1.0).astype(np.float32)
    # Smooth C1 soft-clip shoulder in [0.02, 0.98] -> [0, 1]
    margin = 0.04
    y = 0.5 + (x - 0.5) * (1.0 - 2.0 * margin)
    below = y < margin
    above = y > (1.0 - margin)
    # Rational shoulder: f(t) = margin / (1 + (margin - t) / margin) maps (-inf, margin] -> (0, margin]
    y = np.where(below, margin / (1.0 + np.maximum(margin - y, 0.0) / margin), y)
    y = np.where(above, 1.0 - margin / (1.0 + np.maximum(y - (1.0 - margin), 0.0) / margin), y)
    return np.clip(y, 0.0, 1.0).astype(np.float32)


def s_curve(arr: np.ndarray, gain: float = 1.0) -> np.ndarray:
    """Smooth symmetric S-curve contrast shaping around 0.5 without hard clipping."""
    x = np.clip(arr, 0.0, 1.0).astype(np.float32)
    g = float(gain)
    if abs(g - 1.0) < 1e-4:
        return x
    # Tunable rational S-curve (Schlick bias/gain)
    k = max(0.05, g)
    lo = 0.5 * np.power(np.clip(2.0 * x, 0.0, 1.0), k)
    hi = 1.0 - 0.5 * np.power(np.clip(2.0 * (1.0 - x), 0.0, 1.0), k)
    return np.where(x < 0.5, lo, hi).astype(np.float32)


def match_histogram_range(
    arr: np.ndarray,
    target_min: float,
    target_max: float,
    target_mean: float | None = None,
    contrast: float = 1.0,
    low_pct: float = 0.25,
    high_pct: float = 99.75,
) -> np.ndarray:
    """Shape a scalar field to a calibrated [target_min, target_max] range and optional mean.

    Uses robust percentile anchoring, optional smooth S-curve contrast, and a closed-form
    gamma adjustment when `target_mean` is specified so PBR roughness/albedo maps hit
    physically accurate material distributions without outlier washout or flat clipping.
    """
    x = robust_normalize(arr, low_pct=low_pct, high_pct=high_pct, soft=True)
    if abs(contrast - 1.0) > 1e-4:
        x = s_curve(x, gain=contrast)
    if target_mean is not None and target_max > target_min + 1e-6:
        desired = float(np.clip((target_mean - target_min) / (target_max - target_min), 0.05, 0.95))
        cur = float(np.clip(x.mean(), 0.02, 0.98))
        # Bisection solve for gamma > 0 such that mean(x ** gamma) == desired
        g_lo, g_hi = 0.15, 6.0
        for _ in range(14):
            g_mid = 0.5 * (g_lo + g_hi)
            m_mid = float(np.mean(np.power(np.clip(x, 1e-6, 1.0), g_mid)))
            if m_mid > desired:
                g_lo = g_mid
            else:
                g_hi = g_mid
        gamma = 0.5 * (g_lo + g_hi)
        if abs(cur - desired) > 1e-3:
            x = np.power(np.clip(x, 1e-6, 1.0), gamma).astype(np.float32)
    return np.clip(target_min + x * (target_max - target_min), 0.0, 1.0).astype(np.float32)


def _hash_gradient_lut(r1: np.ndarray, r2: np.ndarray, seed_u32: int) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic vectorized SplitMix/Murmur-style integer hash into unit gradient LUT."""
    h = (
        (r1.astype(np.uint32) * np.uint32(0x9E3779B1))
        ^ (r2.astype(np.uint32) * np.uint32(0x85EBCA6B))
        ^ np.uint32(seed_u32)
    )
    h ^= h >> np.uint32(16)
    h *= np.uint32(0x7FEB352D)
    h ^= h >> np.uint32(15)
    idx = h & np.uint32(_GRAD_LUT_MASK)
    return _GRAD_LUT_GX[idx], _GRAD_LUT_GY[idx]


def periodic_gradient_perlin_with_grad(
    u: np.ndarray,
    v: np.ndarray,
    freqx: int,
    freqy: int | None,
    rng: np.random.Generator,
    phase_u: float = 0.0,
    phase_v: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Periodic 2D quintic gradient Perlin returning `(value, dval_du_norm, dval_dv_norm)`.

    Derivatives are returned in lattice-normalized (O(1)) units — completely independent
    of image resolution and lattice frequency — enabling resolution-invariant Swiss
    advection and flow fields.
    """
    fx = max(1, int(freqx))
    fy = max(1, int(freqy if freqy is not None else fx))
    angles = rng.uniform(0.0, 2.0 * math.pi, size=(fy, fx))
    gx = np.cos(angles).astype(np.float32)
    gy = np.sin(angles).astype(np.float32)

    x = ((u + float(phase_u)) * fx) % fx
    y = ((v + float(phase_v)) * fy) % fy
    x0 = np.floor(x).astype(np.int32) % fx
    y0 = np.floor(y).astype(np.int32) % fy
    x1 = (x0 + 1) % fx
    y1 = (y0 + 1) % fy

    dx = (x - np.floor(x)).astype(np.float32)
    dy = (y - np.floor(y)).astype(np.float32)

    gx00, gy00 = gx[y0, x0], gy[y0, x0]
    gx10, gy10 = gx[y0, x1], gy[y0, x1]
    gx01, gy01 = gx[y1, x0], gy[y1, x0]
    gx11, gy11 = gx[y1, x1], gy[y1, x1]

    n00 = gx00 * dx + gy00 * dy
    n10 = gx10 * (dx - 1.0) + gy10 * dy
    n01 = gx01 * dx + gy01 * (dy - 1.0)
    n11 = gx11 * (dx - 1.0) + gy11 * (dy - 1.0)

    sx = quintic(dx)
    sy = quintic(dy)
    dsx = quintic_deriv(dx)
    dsy = quintic_deriv(dy)

    nx0 = n00 + sx * (n10 - n00)
    nx1 = n01 + sx * (n11 - n01)
    n = nx0 + sy * (nx1 - nx0)

    # Exact analytic gradient of quintic-interpolated Perlin noise in lattice space
    gx_x0 = gx00 + sx * (gx10 - gx00)
    gx_x1 = gx01 + sx * (gx11 - gx01)
    gy_x0 = gy00 + sx * (gy10 - gy00)
    gy_x1 = gy01 + sx * (gy11 - gy01)
    dn_dx = (gx_x0 + sy * (gx_x1 - gx_x0)) + dsx * ((n10 - n00) + sy * ((n11 - n01) - (n10 - n00)))
    dn_dy = (gy_x0 + sy * (gy_x1 - gy_x0)) + dsy * (nx1 - nx0)

    scale = np.float32(0.70710678)
    val = np.clip(n * scale + 0.5, 0.0, 1.0).astype(np.float32)
    return val, (dn_dx * scale).astype(np.float32), (dn_dy * scale).astype(np.float32)


def periodic_gradient_perlin(
    u: np.ndarray,
    v: np.ndarray,
    freqx: int,
    freqy: int | None,
    rng: np.random.Generator,
    phase_u: float = 0.0,
    phase_v: float = 0.0,
) -> np.ndarray:
    """Periodic 2D quintic gradient Perlin noise on [0, 1)^2 with optional phase shift."""
    fx = max(1, int(freqx))
    fy = max(1, int(freqy if freqy is not None else fx))
    angles = rng.uniform(0.0, 2.0 * math.pi, size=(fy, fx))
    gx = np.cos(angles).astype(np.float32)
    gy = np.sin(angles).astype(np.float32)

    x = ((u + float(phase_u)) * fx) % fx
    y = ((v + float(phase_v)) * fy) % fy
    x0 = np.floor(x).astype(np.int32) % fx
    y0 = np.floor(y).astype(np.int32) % fy
    x1 = (x0 + 1) % fx
    y1 = (y0 + 1) % fy

    dx = (x - np.floor(x)).astype(np.float32)
    dy = (y - np.floor(y)).astype(np.float32)

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


def periodic_rotated_perlin(
    u: np.ndarray,
    v: np.ndarray,
    freq: float,
    angle_rad: float,
    rng: np.random.Generator,
    phase_u: float = 0.0,
    phase_v: float = 0.0,
    return_grad: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exact orthogonally rotated periodic Perlin noise on T^2 via Gaussian integers.

    For integer vector (a, b) with D = a^2 + b^2, the rotated coordinates
        x = a * (u + phase_u) + b * (v + phase_v)
        y = -b * (u + phase_u) + a * (v + phase_v)
    shift lattice indices (ix, iy) by (a, -b) under u -> u+1 and by (b, a) under v -> v+1.
    The modular coordinates r1 = (a*ix - b*iy) % D and r2 = (b*ix + a*iy) % D are
    exact invariants of the unit torus Z^2, guaranteeing zero seam and zero distortion
    while breaking Perlin's Cartesian axis alignment across fBm octaves.
    """
    f = max(1.0, float(freq))
    a = int(round(f * math.cos(angle_rad)))
    b = int(round(f * math.sin(angle_rad)))
    if a == 0 and b == 0:
        a = max(1, int(round(f)))
    if b == 0 or a == 0:
        # Axis-aligned case: reuse fast direct lattice lookup
        ff = max(1, abs(a) + abs(b))
        if return_grad:
            return periodic_gradient_perlin_with_grad(u, v, ff, ff, rng, phase_u=phase_u, phase_v=phase_v)
        return periodic_gradient_perlin(u, v, ff, ff, rng, phase_u=phase_u, phase_v=phase_v)

    D = a * a + b * b
    seed_u32 = int(rng.integers(0, (1 << 31) - 1))
    uu = u + float(phase_u)
    vv = v + float(phase_v)
    x = a * uu + b * vv
    y = -b * uu + a * vv

    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    dx = (x - x0).astype(np.float32)
    dy = (y - y0).astype(np.float32)
    x1 = x0 + 1
    y1 = y0 + 1

    # Modular invariants under torus wrap: exact in int32 for D <= 10^6 (freq <= 1000)
    def _corner(ix: np.ndarray, iy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        r1 = (a * ix - b * iy) % D
        r2 = (b * ix + a * iy) % D
        return _hash_gradient_lut(r1, r2, seed_u32)

    gx00, gy00 = _corner(x0, y0)
    gx10, gy10 = _corner(x1, y0)
    gx01, gy01 = _corner(x0, y1)
    gx11, gy11 = _corner(x1, y1)

    n00 = gx00 * dx + gy00 * dy
    n10 = gx10 * (dx - 1.0) + gy10 * dy
    n01 = gx01 * dx + gy01 * (dy - 1.0)
    n11 = gx11 * (dx - 1.0) + gy11 * (dy - 1.0)

    sx = quintic(dx)
    sy = quintic(dy)
    nx0 = n00 + sx * (n10 - n00)
    nx1 = n01 + sx * (n11 - n01)
    n = nx0 + sy * (nx1 - nx0)

    scale = np.float32(0.70710678)
    val = np.clip(n * scale + 0.5, 0.0, 1.0).astype(np.float32)
    if not return_grad:
        return val

    dsx = quintic_deriv(dx)
    dsy = quintic_deriv(dy)
    gx_x0 = gx00 + sx * (gx10 - gx00)
    gx_x1 = gx01 + sx * (gx11 - gx01)
    gy_x0 = gy00 + sx * (gy10 - gy00)
    gy_x1 = gy01 + sx * (gy11 - gy01)
    dn_dx = (gx_x0 + sy * (gx_x1 - gx_x0)) + dsx * ((n10 - n00) + sy * ((n11 - n01) - (n10 - n00)))
    dn_dy = (gy_x0 + sy * (gy_x1 - gy_x0)) + dsy * (nx1 - nx0)

    # Rotate lattice-space gradient back to (u, v) unit-frequency direction
    inv_f = np.float32(1.0 / math.sqrt(D))
    du = (a * dn_dx - b * dn_dy) * inv_f * scale
    dv = (b * dn_dx + a * dn_dy) * inv_f * scale
    return val, du.astype(np.float32), dv.astype(np.float32)


def periodic_fbm(
    u: np.ndarray,
    v: np.ndarray,
    freqx: int,
    octaves: int,
    persistence: float,
    rng: np.random.Generator,
    mode: str = "fbm",
    freqy: int | None = None,
    lacunarity: float = 2.0,
    rotate_octaves: bool = True,
    robust: bool = True,
) -> np.ndarray:
    """Resolution-aware periodic multi-octave fBm on T^2.

    Improvements over naive octave summing:
    1. Per-octave random toroidal phase shifts (pu, pv) break shared integer lattice
       nodes where all octaves otherwise evaluate simultaneously to 0.5 (creating a
       visible grid of 1.0 spikes in ridged/billow and 0.0 pinholes in turbulence).
    2. Isotropic octaves use Gaussian-integer toroidal rotation (~55.6 deg per octave)
       to eliminate Cartesian Perlin axis bias without introducing tiling seams.
    3. Nyquist-aware band-limiting derived from `u.shape` replaces the old hardcoded
       128-cell cap, unlocking crisp sub-16px pores/grain at 1024-2048px while smoothly
       rolling off energy near Nyquist instead of stacking duplicate clamped octaves.
    4. `mode="swiss"` uses exact analytic lattice-normalized derivatives for
       resolution-independent hydraulic slope advection.
    """
    h_px, w_px = u.shape
    nyq_x = max(2, w_px // 2)
    nyq_y = max(2, h_px // 2)
    fx_base = max(1, int(freqx))
    fy_base = max(1, int(freqy if freqy is not None else fx_base))
    isotropic = (fx_base == fy_base) and rotate_octaves

    total = np.zeros_like(u, dtype=np.float32)
    amp = 1.0
    amp_sum = 0.0
    weight = 1.0
    dx_acc = np.zeros_like(u, dtype=np.float32)
    dy_acc = np.zeros_like(v, dtype=np.float32)

    # Golden-angle rotation step keeps successive octave lattices mutually incoherent
    angle_step = 0.61803398875 * (math.pi * 0.5)
    lac = max(1.25, float(lacunarity))
    n_oct = max(1, int(octaves))

    for i in range(n_oct):
        scale_i = lac ** i
        fx_f = fx_base * scale_i
        fy_f = fy_base * scale_i
        # Stop cleanly past Nyquist instead of stacking identical clamped frequencies
        if i > 0 and (fx_f >= nyq_x * 0.98 or fy_f >= nyq_y * 0.98):
            break

        # Smooth Nyquist rolloff in the top 25% band to prevent pixel aliasing
        fade_x = min(1.0, max(0.0, (nyq_x - fx_f) / max(1.0, 0.25 * nyq_x)))
        fade_y = min(1.0, max(0.0, (nyq_y - fy_f) / max(1.0, 0.25 * nyq_y)))
        nyq_fade = 1.0 if i == 0 else min(fade_x, fade_y)
        if nyq_fade <= 1e-4:
            break

        fx = min(max(1, int(round(fx_f))), nyq_x)
        fy = min(max(1, int(round(fy_f))), nyq_y)
        pu = float(rng.uniform(0.0, 1.0))
        pv = float(rng.uniform(0.0, 1.0))
        angle = (i * angle_step) if (isotropic and fx >= 2) else 0.0

        if mode == "swiss":
            # Resolution-independent analytic slope advection in UV space
            warp_scale = 0.075 / float(max(1, fx_base))
            uu = (u + warp_scale * dx_acc) % 1.0
            vv = (v + warp_scale * dy_acc) % 1.0
            if isotropic and fx >= 2:
                n, gdu, gdv = periodic_rotated_perlin(
                    uu, vv, fx_f, angle, rng, phase_u=pu, phase_v=pv, return_grad=True
                )
            else:
                n, gdu, gdv = periodic_gradient_perlin_with_grad(
                    uu, vv, fx, fy, rng, phase_u=pu, phase_v=pv
                )
            # Ridged signal weighted by running slope energy (Musgrave / Gaffer Swiss)
            s_raw = n * 2.0 - 1.0
            ridged = 1.0 - np.abs(s_raw)
            signal = ridged * ridged * weight
            # Derivative of ridged^2 w.r.t. u, v drives realistic downhill erosion gullies
            d_factor = -2.0 * ridged * np.sign(s_raw) * amp * weight
            dx_acc += gdu * d_factor
            dy_acc += gdv * d_factor
            weight = np.clip(signal * 1.65 + 0.12, 0.12, 1.0)
        else:
            if isotropic and fx >= 2:
                n = periodic_rotated_perlin(u, v, fx_f, angle, rng, phase_u=pu, phase_v=pv, return_grad=False)
            else:
                n = periodic_gradient_perlin(u, v, fx, fy, rng, phase_u=pu, phase_v=pv)

            if mode == "turbulence":
                signal = np.abs(n * 2.0 - 1.0)
            elif mode == "billow":
                signal = 1.0 - np.abs(n * 2.0 - 1.0)
            elif mode == "ridged":
                r = 1.0 - np.abs(n * 2.0 - 1.0)
                signal = r * r
            else:
                signal = n

        eff_amp = amp * nyq_fade
        total += signal * eff_amp
        amp_sum += eff_amp
        amp *= persistence

    out = total / max(amp_sum, 1e-6)
    if robust:
        return robust_normalize(out, low_pct=0.15, high_pct=99.85, soft=True)
    return normalize_field(out)


def periodic_domain_warp_fbm(
    u: np.ndarray,
    v: np.ndarray,
    freq: int,
    octaves: int,
    persistence: float,
    rng: np.random.Generator,
    warp_strength: float = 0.08,
    warp_freq: int | None = None,
    stages: int = 1,
    mode: str = "fbm",
    use_curl: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Periodic 1- or 2-stage domain-warped fBm (Inigo Quilez warp or Bridson curl advection).

    Returns `(warped_field, warp_u, warp_v)` on [0, 1)^2 with exact torus periodicity.
    """
    wf = max(1, int(warp_freq if warp_freq is not None else max(2, freq // 2)))
    ws = float(warp_strength)
    wu = np.zeros_like(u, dtype=np.float32)
    wv = np.zeros_like(v, dtype=np.float32)

    cur_u, cur_v = u, v
    for s in range(max(1, int(stages))):
        stage_scale = ws * (0.65 if s == 0 and stages > 1 else 1.0)
        if use_curl:
            vx, vy = periodic_curl_noise(cur_u, cur_v, wf * (1 << s), 3, 0.52, rng)
            dx, dy = vx * stage_scale, vy * stage_scale
        else:
            qx = periodic_fbm(cur_u, cur_v, wf * (1 << s), 4, 0.52, rng, mode="fbm")
            qy = periodic_fbm(cur_u, cur_v, wf * (1 << s), 4, 0.52, rng, mode="fbm")
            dx = (qx - 0.5) * 2.0 * stage_scale
            dy = (qy - 0.5) * 2.0 * stage_scale
        wu = wu + dx
        wv = wv + dy
        cur_u = (u + wu) % 1.0
        cur_v = (v + wv) % 1.0

    field = periodic_fbm(cur_u, cur_v, freq, octaves, persistence, rng, mode=mode)
    return field, wu.astype(np.float32), wv.astype(np.float32)


def periodic_voronoi(
    u: np.ndarray,
    v: np.ndarray,
    grid_cells: int,
    jitter: float,
    rng: np.random.Generator,
    metric: str = "euclidean",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Periodic Worley/Voronoi on T^2 with exact IQ perpendicular border distance.

    Improvements:
    - Random toroidal UV shift (when jitter > 0) eliminates cell-wall clustering on
      the u=0 / v=0 wrap seam.
    - For Euclidean metric, computes true perpendicular distance to the closest Voronoi
      bisector (Quilez) instead of `f2 - f1`, yielding uniform-width cracks/creases
      without vertex pinching.
    """
    n = max(1, int(grid_cells))
    j = float(jitter)
    px = rng.uniform(0.5 - 0.5 * j, 0.5 + 0.5 * j, size=(n, n)).astype(np.float32)
    py = rng.uniform(0.5 - 0.5 * j, 0.5 + 0.5 * j, size=(n, n)).astype(np.float32)
    cid_map = rng.uniform(0.0, 1.0, size=(n, n)).astype(np.float32)

    shift_u = float(rng.uniform(0.0, 1.0)) if j > 1e-4 else 0.0
    shift_v = float(rng.uniform(0.0, 1.0)) if j > 1e-4 else 0.0
    gx = ((u + shift_u) % 1.0) * n
    gy = ((v + shift_v) % 1.0) * n
    ix = np.floor(gx).astype(np.int32)
    iy = np.floor(gy).astype(np.int32)

    f1 = np.full_like(u, 1e9, dtype=np.float32)
    f2 = np.full_like(u, 1e9, dtype=np.float32)
    cell_id = np.zeros_like(u, dtype=np.float32)
    best_dx = np.zeros_like(u, dtype=np.float32)
    best_dy = np.zeros_like(v, dtype=np.float32)

    for oy in (-1, 0, 1):
        cy = (iy + oy) % n
        for ox in (-1, 0, 1):
            cx = (ix + ox) % n
            fx = (ix + ox) + px[cy, cx] - gx
            fy = (iy + oy) + py[cy, cx] - gy
            if metric == "manhattan":
                dist = np.abs(fx) + np.abs(fy)
            elif metric == "chebyshev":
                dist = np.maximum(np.abs(fx), np.abs(fy))
            else:
                dist = fx * fx + fy * fy

            closer1 = dist < f1
            between = (~closer1) & (dist < f2)
            f2 = np.where(closer1, f1, np.where(between, dist, f2))
            f1 = np.where(closer1, dist, f1)
            cell_id = np.where(closer1, cid_map[cy, cx], cell_id)
            best_dx = np.where(closer1, fx, best_dx)
            best_dy = np.where(closer1, fy, best_dy)

    if metric == "euclidean":
        f1 = np.sqrt(np.maximum(f1, 0.0))
        # Pass 2: exact perpendicular distance to closest Voronoi cell bisector (IQ)
        edge = np.full_like(u, 1e9, dtype=np.float32)
        for oy in (-1, 0, 1):
            cy = (iy + oy) % n
            for ox in (-1, 0, 1):
                cx = (ix + ox) % n
                fx = (ix + ox) + px[cy, cx] - gx
                fy = (iy + oy) + py[cy, cx] - gy
                ddx = fx - best_dx
                ddy = fy - best_dy
                d2 = ddx * ddx + ddy * ddy
                valid = d2 > 1e-6
                inv_d = np.where(valid, 1.0 / np.sqrt(np.where(valid, d2, 1.0)), 0.0)
                perp = 0.5 * ((best_dx + fx) * ddx + (best_dy + fy) * ddy) * inv_d
                edge = np.where(valid & (perp < edge), perp, edge)
        edge = np.clip(edge, 0.0, 10.0)
    else:
        edge = np.clip(f2 - f1, 0.0, 10.0)

    return normalize_field(f1), normalize_field(edge), cell_id, best_dx, best_dy


def _build_voronoi_buckets(
    sx: np.ndarray,
    sy: np.ndarray,
    cids: np.ndarray,
    m: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Bin N toroidal seeds (sx, sy) in [0, 1)^2 into an m x m spatial hash grid."""
    bx = np.floor(sx * m).astype(np.int32) % m
    by = np.floor(sy * m).astype(np.int32) % m
    flat = by * m + bx
    order = np.argsort(flat, kind="mergesort")
    flat_s = flat[order]
    counts = np.bincount(flat_s, minlength=m * m)
    max_k = max(1, int(counts.max()))
    starts = np.zeros(m * m, dtype=np.int32)
    if m * m > 1:
        np.cumsum(counts[:-1], out=starts[1:])

    px = np.full((max_k, m * m), 1e4, dtype=np.float32)
    py = np.full((max_k, m * m), 1e4, dtype=np.float32)
    pc = np.zeros((max_k, m * m), dtype=np.float32)
    pid = np.full((max_k, m * m), -1, dtype=np.int32)

    pos = np.arange(len(flat_s), dtype=np.int32) - np.repeat(starts, counts)
    px[pos, flat_s] = sx[order] * m - bx[order]
    py[pos, flat_s] = sy[order] * m - by[order]
    pc[pos, flat_s] = cids[order]
    pid[pos, flat_s] = order.astype(np.int32)

    return (
        px.reshape(max_k, m, m),
        py.reshape(max_k, m, m),
        pc.reshape(max_k, m, m),
        pid.reshape(max_k, m, m),
        max_k,
    )


def periodic_lloyd_voronoi(
    u: np.ndarray,
    v: np.ndarray,
    grid_cells: int,
    jitter: float,
    rng: np.random.Generator,
    lloyd_iters: int = 2,
    metric: str = "euclidean",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Periodic Centroidal Voronoi Tessellation via unconstrained toroidal Lloyd relaxation.

    Improvements over box-clamped grid Lloyd:
    1. Seeds initialize from Roberts (2018) R2 low-discrepancy plastic-constant sequence
       on T^2 with random phase and jitter, eliminating the Cartesian grid-frequency
       artifact and u=0/v=0 seam clustering caused by 1-seed-per-square-cell trapping.
    2. Seeds relax freely across cell boundaries on T^2 on a higher-resolution grid
       (up to 320px) into true isotropic 5/6/7-sided organic polygons.
    3. Euclidean border distance uses IQ exact perpendicular bisector distance.
    """
    n = max(2, int(grid_cells))
    j = float(jitter)
    N = n * n

    if j < 0.35:
        # Low-jitter callers explicitly requesting near-regular grid cells
        gx0, gy0 = np.meshgrid(np.arange(n, dtype=np.float32), np.arange(n, dtype=np.float32))
        sx = ((gx0 + 0.5 + rng.uniform(-0.5 * j, 0.5 * j, size=(n, n))) / n % 1.0).ravel().astype(np.float32)
        sy = ((gy0 + 0.5 + rng.uniform(-0.5 * j, 0.5 * j, size=(n, n))) / n % 1.0).ravel().astype(np.float32)
    else:
        # Roberts (2018) R2 plastic-constant blue-noise sequence on [0, 1)^2: zero row/col bias
        g = 1.3247179572447460
        a1 = 1.0 / g
        a2 = 1.0 / (g * g)
        idx = np.arange(N, dtype=np.float64)
        ox = float(rng.uniform(0.0, 1.0))
        oy = float(rng.uniform(0.0, 1.0))
        sx = ((ox + idx * a1) + rng.uniform(-0.36 * j, 0.36 * j, size=N) / n) % 1.0
        sy = ((oy + idx * a2) + rng.uniform(-0.36 * j, 0.36 * j, size=N) / n) % 1.0
        sx = sx.astype(np.float32)
        sy = sy.astype(np.float32)

    cids = rng.uniform(0.0, 1.0, size=N).astype(np.float32)
    # Slightly coarser bucket grid guarantees 3x3 neighborhood covers all nearest & bisector seeds
    m = max(2, int(round(n * 0.82))) if n >= 5 else n

    if lloyd_iters > 0:
        res = min(max(96, n * 6), 320)
        lin = np.linspace(0.0, 1.0, res, endpoint=False, dtype=np.float32)
        lu, lv = np.meshgrid(lin, lin)
        gx_l = lu * m
        gy_l = lv * m
        ix_l = np.floor(gx_l).astype(np.int32)
        iy_l = np.floor(gy_l).astype(np.int32)

        for _ in range(int(lloyd_iters)):
            px_b, py_b, _, pid_b, mk = _build_voronoi_buckets(sx, sy, cids, m)
            best_d = np.full_like(lu, 1e9, dtype=np.float32)
            best_id = np.zeros_like(ix_l)
            best_dx_l = np.zeros_like(lu, dtype=np.float32)
            best_dy_l = np.zeros_like(lv, dtype=np.float32)

            for oy_i in (-1, 0, 1):
                cy = (iy_l + oy_i) % m
                for ox_i in (-1, 0, 1):
                    cx = (ix_l + ox_i) % m
                    for k in range(mk):
                        fx = (ix_l + ox_i) + px_b[k, cy, cx] - gx_l
                        fy = (iy_l + oy_i) + py_b[k, cy, cx] - gy_l
                        dist = fx * fx + fy * fy
                        closer = dist < best_d
                        best_d = np.where(closer, dist, best_d)
                        best_id = np.where(closer, pid_b[k, cy, cx], best_id)
                        # Vector from seed to pixel is (-fx, -fy); mean gives toroidal centroid offset
                        best_dx_l = np.where(closer, -fx, best_dx_l)
                        best_dy_l = np.where(closer, -fy, best_dy_l)

            counts = np.bincount(best_id.ravel(), minlength=N).astype(np.float32)
            inv_c = 1.0 / np.maximum(counts, 1.0)
            mean_dx = np.bincount(best_id.ravel(), weights=best_dx_l.ravel(), minlength=N).astype(np.float32) * inv_c
            mean_dy = np.bincount(best_id.ravel(), weights=best_dy_l.ravel(), minlength=N).astype(np.float32) * inv_c
            sx = (sx + 0.88 * mean_dx / m) % 1.0
            sy = (sy + 0.88 * mean_dy / m) % 1.0

    px_b, py_b, pc_b, _, mk = _build_voronoi_buckets(sx, sy, cids, m)
    gx = u * m
    gy = v * m
    ix = np.floor(gx).astype(np.int32)
    iy = np.floor(gy).astype(np.int32)

    f1 = np.full_like(u, 1e9, dtype=np.float32)
    f2 = np.full_like(u, 1e9, dtype=np.float32)
    cell_id = np.zeros_like(u, dtype=np.float32)
    best_dx = np.zeros_like(u, dtype=np.float32)
    best_dy = np.zeros_like(v, dtype=np.float32)

    for oy_i in (-1, 0, 1):
        cy = (iy + oy_i) % m
        for ox_i in (-1, 0, 1):
            cx = (ix + ox_i) % m
            for k in range(mk):
                fx = (ix + ox_i) + px_b[k, cy, cx] - gx
                fy = (iy + oy_i) + py_b[k, cy, cx] - gy
                if metric == "manhattan":
                    dist = np.abs(fx) + np.abs(fy)
                elif metric == "chebyshev":
                    dist = np.maximum(np.abs(fx), np.abs(fy))
                else:
                    dist = fx * fx + fy * fy

                closer1 = dist < f1
                between = (~closer1) & (dist < f2)
                f2 = np.where(closer1, f1, np.where(between, dist, f2))
                f1 = np.where(closer1, dist, f1)
                cell_id = np.where(closer1, pc_b[k, cy, cx], cell_id)
                best_dx = np.where(closer1, fx, best_dx)
                best_dy = np.where(closer1, fy, best_dy)

    if metric == "euclidean":
        f1 = np.sqrt(np.maximum(f1, 0.0))
        edge = np.full_like(u, 1e9, dtype=np.float32)
        for oy_i in (-1, 0, 1):
            cy = (iy + oy_i) % m
            for ox_i in (-1, 0, 1):
                cx = (ix + ox_i) % m
                for k in range(mk):
                    fx = (ix + ox_i) + px_b[k, cy, cx] - gx
                    fy = (iy + oy_i) + py_b[k, cy, cx] - gy
                    ddx = fx - best_dx
                    ddy = fy - best_dy
                    d2 = ddx * ddx + ddy * ddy
                    valid = (d2 > 1e-6) & (fx < 5e3)
                    inv_d = np.where(valid, 1.0 / np.sqrt(np.where(valid, d2, 1.0)), 0.0)
                    perp = 0.5 * ((best_dx + fx) * ddx + (best_dy + fy) * ddy) * inv_d
                    edge = np.where(valid & (perp < edge), perp, edge)
        edge = np.clip(edge, 0.0, 10.0)
    else:
        edge = np.clip(f2 - f1, 0.0, 10.0)

    # Scale relative vectors back to n-cell units for exact caller compatibility
    scale_back = np.float32(float(n) / float(m))
    return (
        normalize_field(f1),
        normalize_field(edge),
        cell_id,
        (best_dx * scale_back).astype(np.float32),
        (best_dy * scale_back).astype(np.float32),
    )


def periodic_box_blur(arr: np.ndarray, radius: int) -> np.ndarray:
    """Exact O(N) separable periodic box blur using wrap-padded prefix sums."""
    r = max(1, int(radius))
    w = 2 * r + 1
    inv_w = 1.0 / float(w)
    px = np.pad(arr.astype(np.float32), ((0, 0), (r, r)), mode="wrap")
    csx = np.empty((px.shape[0], px.shape[1] + 1), dtype=np.float64)
    csx[:, 0] = 0.0
    np.cumsum(px, axis=1, out=csx[:, 1:])
    ax = ((csx[:, w:] - csx[:, :-w]) * inv_w).astype(np.float32)

    py = np.pad(ax, ((r, r), (0, 0)), mode="wrap")
    csy = np.empty((py.shape[0] + 1, py.shape[1]), dtype=np.float64)
    csy[0, :] = 0.0
    np.cumsum(py, axis=0, out=csy[1:, :])
    return ((csy[w:, :] - csy[:-w, :]) * inv_w).astype(np.float32)


def periodic_gaussian_blur(arr: np.ndarray, sigma: float) -> np.ndarray:
    """Exact isotropic periodic Gaussian blur via frequency-domain rfft2 multiplication."""
    s = float(sigma)
    if s <= 1e-4:
        return arr.astype(np.float32, copy=True)
    h, w = arr.shape
    fy = np.fft.fftfreq(h).astype(np.float32)[:, None]
    fx = np.fft.rfftfreq(w).astype(np.float32)[None, :]
    kernel = np.exp(-2.0 * (math.pi ** 2) * (s * s) * (fx * fx + fy * fy))
    out = np.fft.irfft2(np.fft.rfft2(arr.astype(np.float32)) * kernel, s=(h, w))
    return out.astype(np.float32)


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
    """Periodic Lagae-Lefebvre (2009) sparse convolution Gabor noise on [0, 1)^2.

    Uses robust percentile normalization so rare constructive impulse overlaps do not
    compress 98% of the field into a narrow washed-out midtone band.
    """
    n = max(2, int(grid_cells))
    k_cnt = max(1, int(impulses_per_cell))
    gx = u * n
    gy = v * n
    ix = np.floor(gx).astype(np.int32)
    iy = np.floor(gy).astype(np.int32)

    acc = np.zeros_like(u, dtype=np.float32)
    a2 = float(bandwidth) ** 2
    two_pi = 2.0 * math.pi

    for _ in range(k_cnt):
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
            cy = (iy + oy) % n
            for ox in (-1, 0, 1):
                cx = (ix + ox) % n
                dx = gx - ((ix + ox) + px[cy, cx])
                dy = gy - ((iy + oy) + py[cy, cx])
                env = np.exp(-math.pi * a2 * (dx * dx + dy * dy))
                if isotropic:
                    proj = dx * ca_m[cy, cx] + dy * sa_m[cy, cx]
                else:
                    proj = dx * ca_s + dy * sa_s
                acc += amp[cy, cx] * env * np.cos(two_pi * float(f0) * proj + phase[cy, cx])

    return robust_normalize(acc, low_pct=0.3, high_pct=99.7, soft=True)


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


def periodic_scharr_gradients(
    height: np.ndarray,
    reference_size: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute resolution-independent periodic 3x3 Scharr derivatives (dh_dx, dh_dy_up).

    - Uses Scharr weights [3, 10, 3] / 32 instead of Sobel [1, 2, 1] / 8, reducing
      rotational gradient anisotropy error from ~12% to <0.5% (critical for circular
      pores, pebbles, and organic wrinkles).
    - Scales by `res_scale = mean(H, W) / reference_size` so 1024px and 2048px renders
      preserve exact physical surface slopes instead of flattening 2x-4x relative to 512px.
    - Coordinate sign convention: +X right, +Y up (OpenGL / Blender).
    """
    h = height.astype(np.float32, copy=False)
    res_scale = (0.5 * (h.shape[0] + h.shape[1])) / float(max(16, int(reference_size)))

    h_right = np.roll(h, -1, axis=1)
    h_left = np.roll(h, 1, axis=1)
    h_up = np.roll(h, 1, axis=0)
    h_down = np.roll(h, -1, axis=0)

    # 3x3 Scharr: sum of weights = 3 + 10 + 3 = 16; central difference span = 2 -> / 32.0
    inv32 = np.float32(res_scale / 32.0)
    dh_dx = (
        3.0 * (np.roll(h_right, 1, axis=0) - np.roll(h_left, 1, axis=0))
        + 10.0 * (h_right - h_left)
        + 3.0 * (np.roll(h_right, -1, axis=0) - np.roll(h_left, -1, axis=0))
    ) * inv32
    dh_dy_up = (
        3.0 * (np.roll(h_up, 1, axis=1) - np.roll(h_down, 1, axis=1))
        + 10.0 * (h_up - h_down)
        + 3.0 * (np.roll(h_up, -1, axis=1) - np.roll(h_down, -1, axis=1))
    ) * inv32
    return dh_dx.astype(np.float32), dh_dy_up.astype(np.float32)


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
    reference_size: int = 512,
) -> Image.Image:
    """Resolution-independent PBR preview shading with Scharr normals, Fresnel & multi-scale AO."""
    h = normalize_field(height)
    r = (
        np.full_like(h, 0.55 if roughness is None else float(roughness), dtype=np.float32)
        if not isinstance(roughness, np.ndarray)
        else np.clip(roughness, 0.04, 0.98).astype(np.float32)
    )

    dh_dx, dh_dy_up = periodic_scharr_gradients(h, reference_size=reference_size)
    # Match existing light_dir coordinate sign (row-down Y in light_dir -> -dh_dy_up)
    nx = -dh_dx * float(normal_strength)
    ny = dh_dy_up * float(normal_strength)
    nz = np.ones_like(h, dtype=np.float32)
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

    # Microfacet-inspired specular with subtle Schlick Fresnel rim modulation
    shininess = np.clip((1.0 - r) ** 2 * 112.0 + 4.0, 4.0, 116.0)
    fresnel = 0.04 + 0.96 * np.power(np.clip(1.0 - nz, 0.0, 1.0), 4.0)
    spec = (
        np.power(ndoth, shininess) * spec_intensity * (1.0 - r * 0.85)
        + fresnel * spec_intensity * 0.35 * (1.0 - r) ** 2
    )

    # Resolution-scaled multi-scale periodic Gaussian cavity AO (fine crevices + broad valleys)
    res_scale = (0.5 * (h.shape[0] + h.shape[1])) / float(max(16, int(reference_size)))
    blur_fine = periodic_gaussian_blur(h, max(1.2, 2.2 * res_scale))
    blur_broad = periodic_gaussian_blur(h, max(3.5, 9.0 * res_scale))
    cavity_fine = np.clip(blur_fine - h, 0.0, 1.0)
    cavity_broad = np.clip(blur_broad - h, 0.0, 1.0)
    cavity = 0.60 * cavity_fine + 0.40 * cavity_broad
    ao = 1.0 - np.clip((1.0 - h) * ao_strength + cavity * cavity_strength * 3.0, 0.0, 0.50)

    diffuse = (0.70 + 0.38 * ndotl) * ao
    lit = albedo_rgb * diffuse[..., None] + (spec * 255.0)[..., None]
    return Image.fromarray(np.clip(lit, 0.0, 255.0).astype(np.uint8), mode="RGB")


def shade_cloth_surface(
    albedo_rgb: np.ndarray,
    height: np.ndarray,
    normal_strength: float = 5.5,
    sheen_amt: float = 18.0,
    reference_size: int = 512,
) -> Image.Image:
    """Resolution-independent cloth surface shading with Scharr normals and fiber sheen."""
    h = normalize_field(height)
    dh_dx, dh_dy_up = periodic_scharr_gradients(h, reference_size=reference_size)
    nx = -dh_dx * float(normal_strength)
    ny = dh_dy_up * float(normal_strength)
    nz = np.ones_like(h, dtype=np.float32)
    inv_len = 1.0 / np.sqrt(nx * nx + ny * ny + nz * nz + 1e-8)
    nx, ny, nz = nx * inv_len, ny * inv_len, nz * inv_len

    res_scale = (0.5 * (h.shape[0] + h.shape[1])) / float(max(16, int(reference_size)))
    blur_h = periodic_gaussian_blur(h, max(1.5, 3.0 * res_scale))
    cavity = np.clip(blur_h - h, 0.0, 1.0)
    ao = np.clip(0.56 + 0.44 * h - cavity * 0.45, 0.15, 1.0)
    sheen = np.power(np.clip(1.0 - nz, 0.0, 1.0), 1.6)
    ndotl = np.clip(nx * 0.42 - ny * 0.52 + nz * 0.74, 0.0, 1.0)
    lit = albedo_rgb * (0.64 + 0.24 * ao + 0.16 * ndotl)[..., None] + (sheen * sheen_amt)[..., None]
    return Image.fromarray(np.clip(lit, 0.0, 255.0).astype(np.uint8), mode="RGB")


def compute_sobel_normals_and_baking(
    height: np.ndarray,
    normal_strength: float = 5.5,
    reference_size: int = 512,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute resolution-independent Scharr normals, Gaussian AO, and curvature maps."""
    h = normalize_field(height)
    dh_dx, dh_dy_up = periodic_scharr_gradients(h, reference_size=reference_size)
    nx = -dh_dx * float(normal_strength)
    ny = dh_dy_up * float(normal_strength)
    nz = np.ones_like(h, dtype=np.float32)
    inv_len = 1.0 / np.sqrt(nx * nx + ny * ny + nz * nz + 1e-8)
    nx, ny, nz = nx * inv_len, ny * inv_len, nz * inv_len

    normal_rgb = np.clip(np.stack([nx * 0.5 + 0.5, ny * 0.5 + 0.5, nz * 0.5 + 0.5], axis=-1) * 255.0, 0.0, 255.0)
    res_scale = (0.5 * (h.shape[0] + h.shape[1])) / float(max(16, int(reference_size)))
    blur = periodic_gaussian_blur(h, max(2.0, 4.5 * res_scale))
    ao = np.clip(0.55 + 0.45 * h - np.clip(blur - h, 0.0, 1.0) * 0.65, 0.15, 1.0)
    curv = np.clip(0.5 + (h - blur) * 3.2, 0.0, 1.0)
    return normal_rgb.astype(np.uint8), ao, curv, nx, ny


def height_to_normal(
    height: np.ndarray,
    strength: float = 6.0,
    reference_size: int = 512,
) -> Image.Image:
    """Convert a 2D height field into a periodic OpenGL (+Y up, Blender) tangent-space normal map.

    Uses isotropic 3x3 periodic Scharr gradients scaled relative to `reference_size` so
    normal maps stay physically consistent at 512px, 1024px, and 2048px without flattening.
    Flat regions encode cleanly to (128, 128, 255).
    """
    h = normalize_field(height)
    dh_dx, dh_dy_up = periodic_scharr_gradients(h, reference_size=reference_size)

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


def height_to_roughness(
    height: np.ndarray,
    base_roughness: float = 0.55,
    spread: float = 0.22,
    cavity_boost: float = 0.14,
    micro_detail: np.ndarray | None = None,
    reference_size: int = 512,
) -> np.ndarray:
    """Derive a calibrated PBR roughness field in [0, 1] from height, slope, and cavities.

    Combines:
    - Microfacet slope magnitude (steeper micro-relief scatters more diffusely -> rougher)
    - Multi-scale Gaussian cavity occlusion (crevices/pores trap dust/shadow -> rougher)
    - Macro crown wear polishing (raised peaks slightly burnished -> smoother)
    and histogram-matches to `[base_roughness - spread/2, base_roughness + spread/2]`.
    """
    h = normalize_field(height)
    res_scale = (0.5 * (h.shape[0] + h.shape[1])) / float(max(16, int(reference_size)))
    dh_dx, dh_dy_up = periodic_scharr_gradients(h, reference_size=reference_size)
    slope = np.sqrt(dh_dx * dh_dx + dh_dy_up * dh_dy_up)
    slope_n = robust_normalize(slope, low_pct=1.0, high_pct=99.0, soft=True)

    blur_fine = periodic_gaussian_blur(h, max(1.2, 2.5 * res_scale))
    blur_broad = periodic_gaussian_blur(h, max(3.5, 8.5 * res_scale))
    cavity = np.clip(0.6 * (blur_fine - h) + 0.4 * (blur_broad - h), 0.0, 1.0)
    cavity_n = robust_normalize(cavity, low_pct=0.5, high_pct=99.5, soft=True)

    raw = 0.42 * (1.0 - h) + 0.34 * slope_n + (0.24 + float(cavity_boost)) * cavity_n
    if micro_detail is not None:
        raw = raw + 0.18 * (normalize_field(micro_detail) - 0.5)

    r_min = max(0.04, float(base_roughness) - 0.5 * float(spread))
    r_max = min(0.98, float(base_roughness) + 0.5 * float(spread))
    return match_histogram_range(raw, r_min, r_max, target_mean=float(base_roughness))


def roughness_image(roughness: np.ndarray) -> Image.Image:
    """Encode a calibrated float [0, 1] PBR roughness map to 8-bit RGB grayscale without re-normalizing."""
    r = np.clip(np.nan_to_num(roughness, nan=0.5, posinf=1.0, neginf=0.0), 0.0, 1.0)
    arr = np.rint(r * 255.0).astype(np.uint8)
    return Image.fromarray(np.stack([arr, arr, arr], axis=-1), mode="RGB")


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
