"""Common interface contract for all procedural texture and alpha generators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Sequence

import numpy as np
from PIL import Image

try:
    from .common import height_to_normal, height_to_roughness, roughness_image
except ImportError:
    from common import height_to_normal, height_to_roughness, roughness_image  # type: ignore[no-redef]

PaletteArg = str | Sequence[str] | Sequence[Sequence[int]] | None
KindType = Literal["texture", "alpha"]
NormalSource = Literal["native", "derived"]
RenderFn = Callable[[int, PaletteArg, int, int], Image.Image]
HeightFn = Callable[[int, PaletteArg, int, int], np.ndarray]


@dataclass(frozen=True)
class GeneratorSpec:
    name: str
    kind: KindType
    tileable: bool
    origin: str
    default_palette: str
    description: str
    render_fn: RenderFn
    tile_axes: str = "both"  # "both" | "u" | "v" | "none"
    normal_source: NormalSource = "derived"
    normal_rule: str = "Derived from generated luminance."
    normal_strength: float = 6.0
    height_fn: HeightFn | None = None
    normal_fn: RenderFn | None = None
    roughness_fn: HeightFn | None = None
    default_roughness: float = 0.55

    def generate(
        self,
        seed: int = 1,
        palette: PaletteArg = None,
        size: int = 512,
        variant: int = 0,
    ) -> Image.Image:
        if size < 16:
            raise ValueError(f"size must be >= 16, got {size}")
        return self.render_fn(int(seed), palette, int(size), int(variant))

    def generate_height(
        self,
        seed: int = 1,
        palette: PaletteArg = None,
        size: int = 512,
        variant: int = 0,
    ) -> np.ndarray:
        if size < 16:
            raise ValueError(f"size must be >= 16, got {size}")
        if self.height_fn is not None:
            return self.height_fn(int(seed), palette, int(size), int(variant))
        img = self.generate(seed=seed, palette=palette, size=size, variant=variant)
        return np.asarray(img.convert("L"), dtype=np.float32) / 255.0

    def generate_normal(
        self,
        seed: int = 1,
        palette: PaletteArg = None,
        size: int = 512,
        variant: int = 0,
        strength: float | None = None,
    ) -> Image.Image:
        if self.normal_fn is not None and strength is None:
            if size < 16:
                raise ValueError(f"size must be >= 16, got {size}")
            return self.normal_fn(int(seed), palette, int(size), int(variant))
        height = self.generate_height(seed=seed, palette=palette, size=size, variant=variant)
        eff_strength = self.normal_strength if strength is None else float(strength)
        return height_to_normal(height, strength=eff_strength)

    def generate_roughness(
        self,
        seed: int = 1,
        palette: PaletteArg = None,
        size: int = 512,
        variant: int = 0,
    ) -> Image.Image:
        if size < 16:
            raise ValueError(f"size must be >= 16, got {size}")
        if self.roughness_fn is not None:
            r_field = self.roughness_fn(int(seed), palette, int(size), int(variant))
        else:
            height = self.generate_height(seed=seed, palette=palette, size=size, variant=variant)
            r_field = height_to_roughness(height, base_roughness=self.default_roughness)
        return roughness_image(r_field)
