# Canonical Procedural Texture & Alpha Generators (`tools/generators/`)

Pure local Python (`numpy` + `Pillow`) procedural texture and alpha generators consolidating every generator family from the original asset packs (`PACKS\`) and round-2 curation scripts (`p1_work\scripts\`).

## Structure

- `make_texture.py` — Canonical CLI entry point (`--list`, `--list-palettes`, single-file, and `--write-pack-json` batch/pack-ready modes).
- `interface.py` — Shared [`GeneratorSpec`](interface.py) contract (`name`, `kind`, `tileable`, `tile_axes`, `origin`, `default_palette`, `description`, `render_fn`). Every generator accepts `(seed, palette, size, variant)` and outputs a `size x size` PNG (`512x512` default).
- `registry.py` — Single canonical `REGISTERED_MODULES` tuple and `GENERATORS` map.
- `common.py` — Shared periodic 2D gradient Perlin noise, multi-mode fBm (`fbm`, `turbulence`, `billow`, `ridged`, `swiss`), 3-metric periodic Voronoi, separable box blur, 5-stop palette mapping, Sobel normal/AO/curvature baking, and PBR/cloth surface shaders.
- `palettes.py` — Consolidated 150-palette library (`PALETTES`) plus custom hex-stop parser/resampler (`resolve_palette`).
- `types/<module>.py` — Standalone generator modules (one module per registered type).

## Registered Types (81)

- **Cloth & Apparel Textures (17):** `denim-twill`, `jersey-knit`, `cable-knit`, `tartan-plaid`, `houndstooth-wool`, `quilted-puffer`, `corduroy`, `silk-brocade`, `waffle-knit`, `ripstop-nylon`, `herringbone-weave`, `rib-jersey-knit`, `boucle-tweed`, `jacquard-geometric`, `seersucker-shibori`, `pebbled-leather`, `garment-wrinkles`
- **Cloth, Seam & Lace Alphas (12):** `double-topstitch-seam`, `zipper-teeth-trim`, `overlock-zigzag-seam`, `four-hole-button`, `ribbed-cuff-trim`, `cross-stitch-seam`, `rivet-grommet`, `scalloped-lace-trim`, `lace-mesh-mask`, `smocked-gather`, `belt-buckle`, `blanket-stitch-seam`
- **Consolidated Surface Material Textures (11):** `domain-warped-marble`, `swiss-erosion-rock`, `voronoi-basalt-crackle`, `hardwood-planks`, `masonry-tiles`, `damascus-carbon-metal`, `scifi-greeble-panels`, `biome-bark-foliage`, `crystalline-geode-agate`, `stylized-arcane-cel`, `turing-reaction-diffusion`
- **Specialized & Curated Surface Textures (11):** `terrazzo`, `hex-mosaic`, `wind-dunes`, `pcb-circuit`, `stained-glass`, `hammered-metal`, `dragon-scales`, `fluted-travertine`, `cobblestone-paving`, `iridescent-thinfilm`, `weathered-concrete-patina`
- **Substance Utility, Mask & Noise Maps (4):** `grunge-wear-mask`, `procedural-noise-map`, `baking-utility-map`, `matcap-sphere`
- **Sculpting & Mechanical Decal Alphas (2):** `sculpt-stamp-alpha`, `mechanical-decal-alpha`
- **Military Camouflage Textures (9):** `woodland-camo`, `desert-camo`, `chocolate-chip-camo`, `digital-woodland-camo`, `digital-desert-camo`, `tiger-stripe-camo`, `flecktarn-camo`, `multicam-camo`, `urban-camo`
- **Environment Surface Textures (15):** `granite-rock`, `sandstone-strata`, `slate-rock`, `limestone`, `river-pebbles`, `gravel`, `grass-field`, `grass-dirt-patch`, `moss-ground`, `dirt-soil`, `cracked-dry-mud`, `sand`, `red-brick-wall`, `stone-wall`, `stucco-plaster-wall`

## Usage

```bash
# List all registered types and palettes
python tools/generators/make_texture.py --list
python tools/generators/make_texture.py --list-palettes

# Generate a single 512x512 texture
python tools/generators/make_texture.py denim-twill --seed 42 --out denim-twill-01.png

# Override size and palette (named palette or comma-separated hex stops)
python tools/generators/make_texture.py terrazzo --seed 7 --size 1024 --palette moroccan_zellige --out terrazzo-01.png
python tools/generators/make_texture.py masonry-tiles --seed 3 --palette "#1a1f24,#435260,#7b8d9e,#c2ceda,#eef2f6" --out tile-01.png

# Generate a check_pack-ready pack folder (Textures/ or Alphas/ + pack.json)
python tools/generators/make_texture.py --types denim-twill,cable-knit,terrazzo --count-per-type 2 --seed 100 --out-dir ./New_Pack --write-pack-json --pack-name "New_Pack"
python tools/check_pack.py ./New_Pack
```

## Adding a Future Generator Type

1. Drop a new module in `tools/generators/types/<new_type>.py` defining `generate(seed=1, palette=None, size=512, variant=0) -> Image.Image` and `SPEC = GeneratorSpec(name="<new-type>", kind="texture" | "alpha", tileable=True | False, origin=..., default_palette=..., description=..., render_fn=generate)`.
2. Add `"<new-type>"` as one line inside `REGISTERED_MODULES` in `tools/generators/registry.py`.

Nothing else in the codebase changes.
