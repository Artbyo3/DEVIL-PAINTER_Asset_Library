# Canonical Procedural Texture & Alpha Generators (`tools/generators/`)

Pure local Python (`numpy` + `Pillow`) procedural texture and alpha generators consolidating every generator family from the original asset packs (`PACKS\`) and round-2 curation scripts (`p1_work\scripts\`).

## Structure

- `make_texture.py` — Canonical CLI entry point (`--list`, `--list-palettes`, `--normal`, single-file, and `--write-pack-json` batch/pack-ready modes).
- `interface.py` — Shared [`GeneratorSpec`](interface.py) contract (`name`, `kind`, `tileable`, `tile_axes`, `origin`, `default_palette`, `description`, `render_fn`, `normal_source`, `normal_rule`, `normal_strength`, `height_fn`). Every generator accepts `(seed, palette, size, variant)` and supports `generate()`, `generate_height()`, and `generate_normal()`.
- `registry.py` — Single canonical `REGISTERED_MODULES` tuple, `GENERATORS` map, and `NORMAL_SOURCES` (76 NATIVE, 5 DERIVED across 81 types).
- `common.py` — Shared periodic 2D gradient Perlin noise, multi-mode fBm (`fbm`, `turbulence`, `billow`, `ridged`, `swiss`), 3-metric periodic Voronoi, separable box blur, 5-stop palette mapping, Sobel OpenGL (+Y up) `height_to_normal()` converter, and PBR/cloth surface shaders.
- `palettes.py` — Consolidated 150-palette library (`PALETTES`) plus custom hex-stop parser/resampler (`resolve_palette`).
- `types/<module>.py` — Standalone generator modules (one module per registered type).

## Normal Map Convention & Classification

- **Coordinate Space:** OpenGL tangent-space normal maps (**+X right, +Y up, +Z out** — Blender convention), computed from each generator's height source via the shared periodic 3x3 Sobel filter `common.height_to_normal(height, strength)`. Flat areas encode to `(128, 128, 255)`.
- **NATIVE (76 types):** Exposes the real internal physical height/relief field already computed by the generator (cloth weaves/knits/wrinkles, seam/buckle/stamp/decal relief alphas, masonry/stone/rock/ground elevation, baking base surface).
- **DERIVED (5 types):** Generator is inherently a 2D color, opacity-cutout, dimensionless noise, or sphere-preview type; height is explicitly derived via its documented per-type rule.

## Registered Types (81: 76 NATIVE, 5 DERIVED)

| Type | Kind | Tiling | Normal Source | Strength | Height Source / Rule |
| :--- | :--- | :--- | :--- | ---: | :--- |
| `denim-twill` | texture | tileable | **NATIVE** | 5.6 | Native physical surface height field (h) used by shade_cloth_surface. |
| `jersey-knit` | texture | tileable | **NATIVE** | 5.8 | Native physical surface height field (h) used by shade_cloth_surface. |
| `cable-knit` | texture | tileable | **NATIVE** | 6.2 | Native physical surface height field (h) used by shade_cloth_surface. |
| `tartan-plaid` | texture | tileable | **NATIVE** | 5.4 | Native physical surface height field (h) used by shade_cloth_surface. |
| `houndstooth-wool` | texture | tileable | **NATIVE** | 5.2 | Native physical surface height field (h) used by shade_cloth_surface. |
| `quilted-puffer` | texture | tileable | **NATIVE** | 6.2 | Native physical surface height field (h) used by shade_pbr_surface. |
| `corduroy` | texture | tileable | **NATIVE** | 6 | Native physical surface height field (h) used by shade_cloth_surface. |
| `silk-brocade` | texture | tileable | **NATIVE** | 5.2 | Native physical surface height field (h) used by shade_pbr_surface. |
| `waffle-knit` | texture | tileable | **NATIVE** | 6 | Native physical surface height field (h) used by shade_cloth_surface. |
| `ripstop-nylon` | texture | tileable | **NATIVE** | 5.4 | Native physical surface height field (h) used by shade_pbr_surface. |
| `herringbone-weave` | texture | tileable | **NATIVE** | 5.6 | Native physical surface height field (h) used by shade_cloth_surface. |
| `rib-jersey-knit` | texture | tileable | **NATIVE** | 6 | Native physical surface height field (h) used by shade_cloth_surface. |
| `boucle-tweed` | texture | tileable | **NATIVE** | 5.8 | Native physical surface height field (h) used by shade_cloth_surface. |
| `jacquard-geometric` | texture | tileable | **NATIVE** | 5.2 | Native physical surface height field (h) used by shade_pbr_surface. |
| `seersucker-shibori` | texture | tileable | **NATIVE** | 4.8 | Native physical surface height field (h) used by shade_cloth_surface. |
| `pebbled-leather` | texture | tileable | **NATIVE** | 6 | Native physical surface height field (h) used by shade_pbr_surface. |
| `garment-wrinkles` | texture | tileable | **NATIVE** | 6.5 | Native garment fold, pleat, crumple, and drape elevation field h. |
| `double-topstitch-seam` | alpha | non-tiling (v) | **NATIVE** | 6 | Native folded seam step, fabric puckering, and twin stitch thread relief. |
| `zipper-teeth-trim` | alpha | non-tiling (v) | **NATIVE** | 6 | Native interlocking metallic zipper teeth, cord bead, and tape relief. |
| `overlock-zigzag-seam` | alpha | non-tiling (v) | **NATIVE** | 6 | Native zigzag overlock serger thread and dual border rail relief. |
| `four-hole-button` | alpha | non-tiling (none) | **NATIVE** | 6 | Native beveled button dish, raised rim, thread holes, and center X-stitch relief. |
| `ribbed-cuff-trim` | alpha | non-tiling (u) | **NATIVE** | 6 | Native vertical rib-knit wave and stitch relief band. |
| `cross-stitch-seam` | alpha | non-tiling (u) | **NATIVE** | 6 | Native intersecting diagonal X-stitch thread relief along seam band. |
| `rivet-grommet` | alpha | non-tiling (none) | **NATIVE** | 6 | Native beveled grommet ring, radial prongs, and fabric pucker relief. |
| `scalloped-lace-trim` | alpha | non-tiling (u) | **NATIVE** | 6 | Native scalloped edge profile, eyelet cutouts, and raised embroidery rim. |
| `lace-mesh-mask` | alpha | tileable | **DERIVED** | 4.5 | Derived thread relief from opacity/tear cutout alpha mask field. |
| `smocked-gather` | alpha | non-tiling (u) | **NATIVE** | 6 | Native pleated smocked fabric gather relief field. |
| `belt-buckle` | alpha | non-tiling (none) | **NATIVE** | 6 | Native beveled metallic buckle frame and pin relief field. |
| `blanket-stitch-seam` | alpha | non-tiling (v) | **NATIVE** | 6 | Native blanket-stitch thread spine, cross-bar, and seam ridge relief. |
| `domain-warped-marble` | texture | tileable | **DERIVED** | 5.5 | Derived relief height from domain-warped mineral vein scalar field (0.75 * color_field + 0.25 * r_u). |
| `swiss-erosion-rock` | texture | tileable | **NATIVE** | 6.2 | Native physical surface height field (height) used by shade_pbr_surface. |
| `voronoi-basalt-crackle` | texture | tileable | **NATIVE** | 6.8 | Native physical surface height field (height) used by shade_pbr_surface. |
| `hardwood-planks` | texture | tileable | **NATIVE** | 5.4 | Native physical surface height field (height) used by shade_pbr_surface. |
| `masonry-tiles` | texture | tileable | **NATIVE** | 6 | Native physical surface height field (height) used by shade_pbr_surface. |
| `damascus-carbon-metal` | texture | tileable | **NATIVE** | 6 | Native physical surface height field (height) used by shade_pbr_surface. |
| `scifi-greeble-panels` | texture | tileable | **NATIVE** | 6.2 | Native physical surface height field (height) used by shade_pbr_surface. |
| `biome-bark-foliage` | texture | tileable | **NATIVE** | 6.2 | Native physical surface height field (height) used by shade_pbr_surface. |
| `crystalline-geode-agate` | texture | tileable | **NATIVE** | 6.5 | Native physical surface height field (height) used by shade_pbr_surface. |
| `stylized-arcane-cel` | texture | tileable | **NATIVE** | 5 | Native physical surface height field (height) used by shade_pbr_surface. |
| `turing-reaction-diffusion` | texture | tileable | **NATIVE** | 6 | Native physical surface height field (height) used by shade_pbr_surface. |
| `terrazzo` | texture | tileable | **NATIVE** | 3.8 | Native physical surface height field (common.normalize_field(h)) used by shade_pbr_surface. |
| `hex-mosaic` | texture | tileable | **NATIVE** | 5.5 | Native physical surface height field (h) used by shade_pbr_surface. |
| `wind-dunes` | texture | tileable | **NATIVE** | 5 | Native physical surface height field (h) used by shade_pbr_surface. |
| `pcb-circuit` | texture | tileable | **NATIVE** | 6 | Native physical surface height field (h) used by shade_pbr_surface. |
| `stained-glass` | texture | tileable | **NATIVE** | 5.5 | Native physical surface height field (h) used by shade_pbr_surface. |
| `hammered-metal` | texture | tileable | **NATIVE** | 7.5 | Native physical surface height field (h) used by shade_pbr_surface. |
| `dragon-scales` | texture | tileable | **NATIVE** | 7 | Native physical surface height field (h) used by shade_pbr_surface. |
| `fluted-travertine` | texture | tileable | **NATIVE** | 6.8 | Native physical surface height field (h) used by shade_pbr_surface. |
| `cobblestone-paving` | texture | tileable | **NATIVE** | 7.2 | Native physical surface height field (h) used by shade_pbr_surface. |
| `iridescent-thinfilm` | texture | tileable | **NATIVE** | 6 | Native physical surface height field (h) used by shade_pbr_surface. |
| `weathered-concrete-patina` | texture | tileable | **NATIVE** | 5.5 | Native surface relief field h (aggregate cracks, paint curl, zinc spangle, strata, barrel tiles, corrosion patina). |
| `grunge-wear-mask` | texture | tileable | **DERIVED** | 5.5 | Derived relief height directly from the 2D grunge/scratch/drip/peel/crack wear mask. |
| `procedural-noise-map` | texture | tileable | **DERIVED** | 5.5 | Derived height directly from the dimensionless periodic noise scalar field. |
| `baking-utility-map` | texture | tileable | **NATIVE** | 6.5 | Native underlying surface height field h baked by _base_surface_fields. |
| `matcap-sphere` | texture | non-tiling (none) | **DERIVED** | 12 | Derived hemisphere elevation z = sqrt(max(0, 1 - r^2)) within sphere disk mask (0 outside). |
| `sculpt-stamp-alpha` | alpha | non-tiling (none) | **NATIVE** | 6.5 | Native sculpting brush height stamp relief (hex bolt, crater, rock chisel, weld bead, arcane sigil). |
| `mechanical-decal-alpha` | alpha | non-tiling (none) | **NATIVE** | 6.5 | Native hard-surface and sculpted decal relief field (gear, louver, scratch, shatter, rosette, scales, dents, reticle). |
| `woodland-camo` | texture | tileable | **NATIVE** | 2.2 | Native physical surface height field (h) used by shade_cloth_surface. |
| `desert-camo` | texture | tileable | **NATIVE** | 2 | Native physical surface height field (h) used by shade_cloth_surface. |
| `chocolate-chip-camo` | texture | tileable | **NATIVE** | 2.1 | Native physical surface height field (h) used by shade_cloth_surface. |
| `digital-woodland-camo` | texture | tileable | **NATIVE** | 1.9 | Native physical surface height field (h) used by shade_cloth_surface. |
| `digital-desert-camo` | texture | tileable | **NATIVE** | 1.9 | Native physical surface height field (h) used by shade_cloth_surface. |
| `tiger-stripe-camo` | texture | tileable | **NATIVE** | 2.2 | Native physical surface height field (h) used by shade_cloth_surface. |
| `flecktarn-camo` | texture | tileable | **NATIVE** | 2.1 | Native physical surface height field (h) used by shade_cloth_surface. |
| `multicam-camo` | texture | tileable | **NATIVE** | 2 | Native physical surface height field (h) used by shade_cloth_surface. |
| `urban-camo` | texture | tileable | **NATIVE** | 2.2 | Native physical surface height field (h) used by shade_cloth_surface. |
| `granite-rock` | texture | tileable | **NATIVE** | 5.6 | Native physical surface height field (h) used by shade_pbr_surface. |
| `sandstone-strata` | texture | tileable | **NATIVE** | 5.8 | Native physical surface height field (h) used by shade_pbr_surface. |
| `slate-rock` | texture | tileable | **NATIVE** | 6.2 | Native physical surface height field (h) used by shade_pbr_surface. |
| `limestone` | texture | tileable | **NATIVE** | 5.4 | Native physical surface height field (h) used by shade_pbr_surface. |
| `river-pebbles` | texture | tileable | **NATIVE** | 6.8 | Native physical surface height field (h) used by shade_pbr_surface. |
| `gravel` | texture | tileable | **NATIVE** | 6.5 | Native physical surface height field (h) used by shade_pbr_surface. |
| `grass-field` | texture | tileable | **NATIVE** | 5.6 | Native physical surface height field (h) used by shade_pbr_surface. |
| `grass-dirt-patch` | texture | tileable | **NATIVE** | 5.4 | Native physical surface height field (h) used by shade_pbr_surface. |
| `moss-ground` | texture | tileable | **NATIVE** | 6 | Native physical surface height field (h) used by shade_pbr_surface. |
| `dirt-soil` | texture | tileable | **NATIVE** | 5.8 | Native physical surface height field (h) used by shade_pbr_surface. |
| `cracked-dry-mud` | texture | tileable | **NATIVE** | 7 | Native physical surface height field (h) used by shade_pbr_surface. |
| `sand` | texture | tileable | **NATIVE** | 2.8 | Native physical surface height field (h) used by shade_pbr_surface. |
| `red-brick-wall` | texture | tileable | **NATIVE** | 6.2 | Native physical surface height field (h) used by shade_pbr_surface. |
| `stone-wall` | texture | tileable | **NATIVE** | 6.6 | Native physical surface height field (h) used by shade_pbr_surface. |
| `stucco-plaster-wall` | texture | tileable | **NATIVE** | 2.4 | Native physical surface height field (h) used by shade_pbr_surface. |

## Usage

```bash
# List all registered types (with normal source tag) and palettes
python tools/generators/make_texture.py --list
python tools/generators/make_texture.py --list-palettes

# Generate a single 512x512 texture (color only, or color + OpenGL normal)
python tools/generators/make_texture.py denim-twill --seed 42 --out denim-twill-01.png
python tools/generators/make_texture.py denim-twill --seed 42 --out denim-twill-01.png --normal

# Override size and palette (named palette or comma-separated hex stops)
python tools/generators/make_texture.py terrazzo --seed 7 --size 1024 --palette moroccan_zellige --out terrazzo-01.png
python tools/generators/make_texture.py masonry-tiles --seed 3 --palette "#1a1f24,#435260,#7b8d9e,#c2ceda,#eef2f6" --out tile-01.png --normal

# Generate a check_pack-ready pack folder (Textures/ or Alphas/ + pack.json)
python tools/generators/make_texture.py --types denim-twill,cable-knit,terrazzo --count-per-type 2 --seed 100 --out-dir ./New_Pack --write-pack-json --pack-name "New_Pack" --normal
python tools/check_pack.py ./New_Pack
```

## Adding a Future Generator Type

1. Drop a new module in `tools/generators/types/<new_type>.py` defining `generate(...)`, `generate_height(...)`, and `SPEC = GeneratorSpec(..., normal_source="native" | "derived", normal_rule=..., normal_strength=..., height_fn=generate_height)`.
2. Add `"<new_type>"` as one line inside `REGISTERED_MODULES` in `tools/generators/registry.py`.

Nothing else in the codebase changes.
