"""Printed circuit board copper traces, solder mask weave, and annular vias."""

from __future__ import annotations

import math
import numpy as np
from PIL import Image

from .. import common
from ..interface import GeneratorSpec, PaletteArg


def generate(seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0) -> Image.Image:
    rng = np.random.default_rng(seed)
    u, v = common.make_uv_grid(size)
    grid = [8, 12, 16, 20][(variant + int(rng.integers(0, 4))) % 4]
    gu = (u * grid) % 1.0
    gv = (v * grid) % 1.0
    iu = np.floor(u * grid).astype(np.int32) % grid
    iv = np.floor(v * grid).astype(np.int32) % grid
    cell_r = rng.integers(0, 4, size=(grid, grid))
    mode = cell_r[iv, iu]

    trace_w = 0.11
    t_h = np.abs(gv - 0.5) < trace_w
    t_v = np.abs(gu - 0.5) < trace_w
    t_d1 = np.abs((gu - 0.5) - (gv - 0.5)) < trace_w * 1.2
    t_d2 = np.abs((gu - 0.5) + (gv - 0.5)) < trace_w * 1.2
    traces = np.where(mode == 0, t_h, np.where(mode == 1, t_v, np.where(mode == 2, t_d1, t_d2))).astype(np.float32)

    rc = np.sqrt((gu - 0.5) ** 2 + (gv - 0.5) ** 2)
    via = np.clip((0.22 - rc) * 24.0, 0.0, 1.0) * np.clip((rc - 0.08) * 24.0, 0.0, 1.0)
    mask = np.clip(np.maximum(traces * 0.75, via), 0.0, 1.0)
    fiberglass = 0.5 + 0.5 * np.sin(u * 96.0 * math.pi) * np.sin(v * 96.0 * math.pi)

    h = common.normalize_field(mask * 0.8 + fiberglass * 0.2)
    c = common.normalize_field(0.18 + 0.78 * mask + 0.08 * fiberglass)
    rough = np.where(mask > 0.4, 0.22, 0.55).astype(np.float32)

    stops = common.resolve_palette(palette, SPEC.default_palette, rng)
    albedo = common.apply_palette_5stop(c, None, stops, contrast=1.22)
    return common.shade_pbr_surface(albedo, h, roughness=rough, normal_strength=6.0, spec_intensity=0.30)


SPEC = GeneratorSpec(
    name="pcb-circuit",
    kind="texture",
    tileable=True,
    origin="p1_work/scripts/custom_generators.py (gen_v1_pcb_circuit)",
    default_palette="solder_pcb_green",
    description="Printed circuit board copper routing traces, annular pad vias, and fiberglass weave substrate.",
    render_fn=generate,
)
