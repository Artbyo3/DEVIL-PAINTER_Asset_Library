"""Canonical CLI entry point for procedural texture and alpha generation.

Examples:
    python tools/generators/make_texture.py --list
    python tools/generators/make_texture.py --list-palettes
    python tools/generators/make_texture.py denim-twill --seed 42 --out denim-twill-01.png
    python tools/generators/make_texture.py terrazzo --seed 7 --palette moroccan_zellige --size 512 --out out/
    python tools/generators/make_texture.py --types denim-twill,terrazzo,masonry-tiles --count-per-type 2 --out-dir ./my_pack --write-pack-json --pack-name "Sample_Pack"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Support execution via both `python tools/generators/make_texture.py` and `-m tools.generators.make_texture`.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.generators.palettes import PALETTE_NAMES  # noqa: E402
from tools.generators.registry import GENERATORS, get_generator, list_generators  # noqa: E402

KIND_FOLDERS = {
    "texture": "Textures",
    "alpha": "Alphas",
}


def _slug_filename(type_name: str, index: int) -> str:
    return f"{type_name}-{index:02d}.png"


def _normal_path(base_path: Path) -> Path:
    return base_path.with_name(f"{base_path.stem}_normal{base_path.suffix}")


def _build_pack_json(
    pack_name: str,
    kind: str,
    types_used: list[str],
    total_files: int,
    description: str | None = None,
) -> dict:
    desc = description or (
        f"Procedural {kind} pack generated via tools/generators/make_texture.py "
        f"({', '.join(types_used[:6])}{'...' if len(types_used) > 6 else ''})."
    )
    tags = sorted({"procedural", kind, *{t.split("-")[0] for t in types_used}})
    return {
        "name": pack_name,
        "description": desc,
        "tags": tags,
        "sets": total_files,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="DEVIL-PAINTER Asset Library canonical procedural texture & alpha generator."
    )
    parser.add_argument(
        "type_name",
        nargs="?",
        help="Registered generator type name (kebab-case, e.g. denim-twill).",
    )
    parser.add_argument(
        "--types",
        help="Comma-separated list of generator types for batch / mixed-set generation.",
    )
    parser.add_argument("--seed", type=int, default=1, help="Base PRNG seed (default: 1).")
    parser.add_argument("--palette", default=None, help="Palette name, 'auto', or comma-separated hex stops.")
    parser.add_argument("--size", type=int, default=512, help="Square output dimension in pixels (default: 512).")
    parser.add_argument("--variant", type=int, default=0, help="Sub-style variant selector (default: 0).")
    parser.add_argument("--count", type=int, default=1, help="Number of textures to generate in single-type mode.")
    parser.add_argument("--count-per-type", type=int, default=1, help="Number of textures per type in --types mode.")
    parser.add_argument(
        "--normal",
        action="store_true",
        help="Also emit OpenGL (+Y up) tangent-space normal maps with the _normal suffix.",
    )
    parser.add_argument("--out", default=None, help="Output .png file path or directory (default: current directory).")
    parser.add_argument("--out-dir", default=None, help="Output directory for batch / pack-ready output.")
    parser.add_argument(
        "--pack-layout",
        action="store_true",
        help="Place generated files inside kind subfolder (Textures/ or Alphas/).",
    )
    parser.add_argument(
        "--write-pack-json",
        action="store_true",
        help="Enforce kind-purity, use Textures/ or Alphas/ layout, and write a check_pack-valid pack.json.",
    )
    parser.add_argument("--pack-name", default=None, help="Pack display name when --write-pack-json is used.")
    parser.add_argument("--description", default=None, help="Optional pack description for pack.json.")
    parser.add_argument("--list", action="store_true", help="List all registered generator types and exit.")
    parser.add_argument("--kind", choices=["texture", "alpha"], default=None, help="Filter --list by kind.")
    parser.add_argument("--list-palettes", action="store_true", help="List all consolidated palette names and exit.")

    args = parser.parse_args(argv)

    if args.list:
        specs = list_generators(args.kind)
        print(f"Registered generators ({len(specs)}):")
        for spec in specs:
            tile_label = "tileable" if spec.tileable else f"non-tiling({spec.tile_axes})"
            print(
                f"  {spec.name:<28} [{spec.kind:<7}] {tile_label:<18} "
                f"normal={spec.normal_source:<7} palette={spec.default_palette}"
            )
        return 0

    if args.list_palettes:
        print(f"Available palettes ({len(PALETTE_NAMES)}):")
        for name in PALETTE_NAMES:
            print(f"  {name}")
        return 0

    if not args.type_name and not args.types:
        parser.error("Provide a generator type_name, --types <list>, --list, or --list-palettes.")

    if args.types:
        type_list = [t.strip() for t in args.types.split(",") if t.strip()]
    else:
        type_list = [args.type_name]

    specs = [get_generator(t) for t in type_list]
    use_pack_layout = args.pack_layout or args.write_pack_json
    if args.write_pack_json:
        kinds = {s.kind for s in specs}
        if len(kinds) > 1:
            parser.error(
                f"--write-pack-json enforces pack kind-purity; cannot mix kinds {sorted(kinds)} in one pack folder."
            )

    # Single-file output shortcut when exactly 1 type and count == 1 and --out looks like a file
    is_single_file = (
        not args.types
        and args.count == 1
        and not use_pack_layout
        and args.out is not None
        and args.out.lower().endswith(".png")
        and args.out_dir is None
    )

    if is_single_file:
        out_file = Path(args.out)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        img = specs[0].generate(seed=args.seed, palette=args.palette, size=args.size, variant=args.variant)
        img.save(out_file, format="PNG", compress_level=4)
        print(f"Generated {out_file} ({specs[0].name}, {img.size[0]}x{img.size[1]}, {img.mode})")
        if args.normal:
            n_file = _normal_path(out_file)
            n_img = specs[0].generate_normal(
                seed=args.seed, palette=args.palette, size=args.size, variant=args.variant
            )
            n_img.save(n_file, format="PNG", compress_level=4)
            print(
                f"Generated {n_file} ({specs[0].name} normal [{specs[0].normal_source}], "
                f"{n_img.size[0]}x{n_img.size[1]}, {n_img.mode})"
            )
        return 0

    out_root = Path(args.out_dir or args.out or ".")
    out_root.mkdir(parents=True, exist_ok=True)

    per_type = args.count_per_type if args.types else args.count
    generated_paths: list[Path] = []
    seed_cursor = int(args.seed)

    for spec in specs:
        target_dir = out_root / KIND_FOLDERS[spec.kind] if use_pack_layout else out_root
        target_dir.mkdir(parents=True, exist_ok=True)
        for i in range(1, per_type + 1):
            filename = _slug_filename(spec.name, i)
            out_path = target_dir / filename
            variant_i = args.variant + (i - 1)
            img = spec.generate(
                seed=seed_cursor,
                palette=args.palette,
                size=args.size,
                variant=variant_i,
            )
            img.save(out_path, format="PNG", compress_level=4)
            generated_paths.append(out_path)
            if args.normal:
                n_path = _normal_path(out_path)
                n_img = spec.generate_normal(
                    seed=seed_cursor,
                    palette=args.palette,
                    size=args.size,
                    variant=variant_i,
                )
                n_img.save(n_path, format="PNG", compress_level=4)
                generated_paths.append(n_path)
            seed_cursor += 104729

    if args.write_pack_json:
        pack_kind = specs[0].kind
        pack_name = args.pack_name or out_root.resolve().name
        meta = _build_pack_json(
            pack_name=pack_name,
            kind=pack_kind,
            types_used=[s.name for s in specs],
            total_files=len(generated_paths),
            description=args.description,
        )
        pack_json_path = out_root / "pack.json"
        pack_json_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {pack_json_path} (kind={pack_kind}, sets={len(generated_paths)})")

    print(f"Generated {len(generated_paths)} file(s) in {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
