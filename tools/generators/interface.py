"""Common interface contract for all procedural texture and alpha generators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Sequence

from PIL import Image

PaletteArg = str | Sequence[str] | Sequence[Sequence[int]] | None
KindType = Literal["texture", "alpha"]
RenderFn = Callable[[int, PaletteArg, int, int], Image.Image]


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
