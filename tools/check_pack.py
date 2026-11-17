"""Validate a pack folder against DEVIL-PAINTER Asset Library standing rules.

Usage:
    python tools/check_pack.py <pack_folder> [--max 200]

Checks:
  1. No stray or hidden files (only pack.json and asset folders at pack root; no
     dotfiles, nested subdirectories, empty asset folders, or duplicate names).
  2. Kind-pure: exactly one asset kind derived from folder layout matching
     tools/build_pack_entry.py (Textures->texture, Alphas->alpha,
     Materials->material, Brushes->brush, Generators->generator, other->asset).
  3. At most 200 catalog files (excluding optional <stem>.preview.<img> files).
  4. pack.json present and valid with established fields (name, description,
     tags list, sets matching catalog file count).
  5. Every image file readable and valid (verified with Pillow).
  6. Non-image files follow the preview convention (optional <stem>.preview.<img>
     in the same folder, with no orphaned previews or bare images).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

try:
    import build_pack_entry
except ImportError:
    from tools import build_pack_entry  # type: ignore[no-redef]

IMAGE_KINDS = {"texture", "alpha"}
MAX_PACK_FILES = 200


@dataclass
class RuleResult:
    name: str
    passed: bool
    reason: str


@dataclass
class PackSummary:
    pack_dir: Path
    kind: str | None
    folders: list[tuple[Path, str]]
    catalog_files: list[tuple[Path, Path, str]]  # (file_path, folder, kind)
    preview_files: list[Path]
    meta: dict | None


def _split_preview_stem(filename: str) -> tuple[str, str] | None:
    stem, ext = os.path.splitext(filename)
    if ext.lower() not in build_pack_entry.IMAGE_EXTENSIONS:
        return None
    if not stem.lower().endswith(".preview"):
        return None
    asset_stem = stem[: -len(".preview")]
    if not asset_stem:
        return None
    return asset_stem, ext.lower()


def _verify_image_file(path: Path) -> str | None:
    try:
        if path.stat().st_size == 0:
            return "0-byte image file"
        with Image.open(path) as img:
            img.verify()
        with Image.open(path) as img:
            img.load()
        return None
    except Exception as exc:
        return str(exc) or exc.__class__.__name__


def validate_pack(
    pack_dir: Path,
    max_files: int = MAX_PACK_FILES,
) -> tuple[bool, list[RuleResult], PackSummary]:
    results: list[RuleResult] = []
    pack_dir = pack_dir.resolve()

    if not pack_dir.is_dir():
        results.append(RuleResult("pack-folder", False, f"Pack folder does not exist: {pack_dir}"))
        summary = PackSummary(pack_dir, None, [], [], [], None)
        return False, results, summary

    # Rule 1: No stray or hidden files
    stray_errors: list[str] = []
    hidden_entries: list[str] = []
    nested_dirs: list[str] = []
    empty_folders: list[str] = []

    root_entries = sorted(pack_dir.iterdir(), key=lambda p: p.name)
    for entry in root_entries:
        if entry.name.startswith("."):
            hidden_entries.append(entry.name)
            continue
        if entry.is_file():
            if entry.name != "pack.json":
                stray_errors.append(f"stray root file '{entry.name}'")
        elif entry.is_dir():
            sub_entries = sorted(entry.iterdir(), key=lambda p: p.name)
            has_non_hidden_file = False
            for sub in sub_entries:
                rel = f"{entry.name}/{sub.name}"
                if sub.name.startswith("."):
                    hidden_entries.append(rel)
                elif sub.is_dir():
                    nested_dirs.append(rel)
                elif sub.is_file():
                    has_non_hidden_file = True
                else:
                    stray_errors.append(f"non-regular entry '{rel}'")
            if not has_non_hidden_file:
                empty_folders.append(entry.name)
        else:
            stray_errors.append(f"non-regular root entry '{entry.name}'")

    folders = build_pack_entry.iter_pack_folders(pack_dir)
    catalog_files: list[tuple[Path, Path, str]] = []
    preview_files: list[Path] = []
    duplicate_names: list[str] = []
    seen_names: set[str] = set()

    for folder, kind in folders:
        for entry in sorted(folder.iterdir(), key=lambda p: p.name):
            if not entry.is_file() or entry.name.startswith("."):
                continue
            if build_pack_entry.is_preview_file(entry.name):
                preview_files.append(entry)
                continue
            if entry.name in seen_names:
                duplicate_names.append(entry.name)
            seen_names.add(entry.name)
            catalog_files.append((entry, folder, kind))

    rule1_issues: list[str] = []
    if hidden_entries:
        rule1_issues.append(f"hidden entries: {', '.join(hidden_entries[:5])}")
    if stray_errors:
        rule1_issues.append("; ".join(stray_errors[:5]))
    if nested_dirs:
        rule1_issues.append(f"nested directories not allowed: {', '.join(nested_dirs[:5])}")
    if empty_folders:
        rule1_issues.append(f"empty asset folder(s): {', '.join(empty_folders[:5])}")
    if duplicate_names:
        rule1_issues.append(f"duplicate catalog filenames across folders: {', '.join(duplicate_names[:5])}")

    if rule1_issues:
        results.append(RuleResult("no-stray-or-hidden", False, "; ".join(rule1_issues)))
    else:
        folder_list = ", ".join(f"{f.name}/" for f, _ in folders) or "none"
        results.append(
            RuleResult(
                "no-stray-or-hidden",
                True,
                f"No stray or hidden files ({len(catalog_files)} catalog files in {folder_list})",
            )
        )

    # Rule 2: Kind-pure (single asset kind derived from folder layout)
    kinds_present = sorted({kind for _, kind in folders})
    folder_kinds_desc = ", ".join(f"{folder.name} ({kind})" for folder, kind in folders)
    pack_kind: str | None = None
    if not folders:
        results.append(RuleResult("kind-pure", False, "No asset subfolders found inside pack folder"))
    elif len(kinds_present) > 1:
        results.append(
            RuleResult(
                "kind-pure",
                False,
                f"Multiple asset kinds mixed in one pack: {folder_kinds_desc}",
            )
        )
    else:
        pack_kind = kinds_present[0]
        if pack_kind in IMAGE_KINDS:
            non_images = [
                f"{folder.name}/{path.name}"
                for path, folder, _ in catalog_files
                if path.suffix.lower() not in build_pack_entry.IMAGE_EXTENSIONS
            ]
            if non_images:
                results.append(
                    RuleResult(
                        "kind-pure",
                        False,
                        f"Non-image file(s) inside image-kind folder ({pack_kind}): {', '.join(non_images[:5])}",
                    )
                )
            else:
                results.append(RuleResult("kind-pure", True, f"Single asset kind '{pack_kind}' ({folder_kinds_desc})"))
        else:
            bare_images = [
                f"{folder.name}/{path.name}"
                for path, folder, _ in catalog_files
                if path.suffix.lower() in build_pack_entry.IMAGE_EXTENSIONS
            ]
            if bare_images:
                results.append(
                    RuleResult(
                        "kind-pure",
                        False,
                        f"Bare image file(s) inside non-image folder ({pack_kind}) — "
                        f"mixes kinds or missing .preview.<img> suffix: {', '.join(bare_images[:5])}",
                    )
                )
            else:
                results.append(RuleResult("kind-pure", True, f"Single asset kind '{pack_kind}' ({folder_kinds_desc})"))

    # Rule 3: At most --max catalog files
    total_files = len(catalog_files)
    if total_files == 0:
        results.append(RuleResult("max-files", False, f"Pack contains 0 catalog files (expected 1..{max_files})"))
    elif total_files > max_files:
        results.append(
            RuleResult(
                "max-files",
                False,
                f"{total_files} catalog files exceeds cap of {max_files}",
            )
        )
    else:
        results.append(RuleResult("max-files", True, f"{total_files} / {max_files} catalog files"))

    # Rule 4: pack.json present and valid with established fields
    pack_json_path = pack_dir / "pack.json"
    meta: dict | None = None
    pack_json_errors: list[str] = []
    if not pack_json_path.is_file():
        pack_json_errors.append("pack.json is missing")
    else:
        try:
            loaded = json.loads(pack_json_path.read_text(encoding="utf-8"))
            if not isinstance(loaded, dict):
                pack_json_errors.append("pack.json root must be a JSON object")
            else:
                meta = loaded
                name = meta.get("name")
                desc = meta.get("description")
                tags = meta.get("tags")
                sets = meta.get("sets")

                if not isinstance(name, str) or not name.strip():
                    pack_json_errors.append("'name' must be a non-empty string")
                if not isinstance(desc, str) or not desc.strip():
                    pack_json_errors.append("'description' must be a non-empty string")
                if (
                    not isinstance(tags, list)
                    or not tags
                    or not all(isinstance(t, str) and t.strip() for t in tags)
                ):
                    pack_json_errors.append("'tags' must be a non-empty list of non-empty strings")
                if not isinstance(sets, int) or isinstance(sets, bool):
                    pack_json_errors.append("'sets' must be an integer")
                elif sets != total_files:
                    pack_json_errors.append(f"'sets' is {sets}, expected {total_files} matching catalog files")
        except Exception as exc:
            pack_json_errors.append(f"invalid JSON: {exc}")

    if pack_json_errors:
        results.append(RuleResult("pack-json", False, "; ".join(pack_json_errors)))
    else:
        assert meta is not None
        results.append(
            RuleResult(
                "pack-json",
                True,
                f"Valid pack.json ({meta['name']!r}, {len(meta['tags'])} tags, sets={meta['sets']})",
            )
        )

    # Rule 5: Every image file readable and valid
    image_paths: list[Path] = []
    for path, _, _ in catalog_files:
        if path.suffix.lower() in build_pack_entry.IMAGE_EXTENSIONS:
            image_paths.append(path)
    image_paths.extend(preview_files)

    bad_images: list[str] = []
    for img_path in image_paths:
        err = _verify_image_file(img_path)
        if err is not None:
            rel = img_path.relative_to(pack_dir).as_posix()
            bad_images.append(f"{rel} ({err})")

    if bad_images:
        results.append(
            RuleResult(
                "valid-images",
                False,
                f"{len(bad_images)} unreadable/invalid image(s): {', '.join(bad_images[:5])}",
            )
        )
    elif not image_paths:
        results.append(RuleResult("valid-images", True, "No image or preview files to verify (non-image pack)"))
    else:
        results.append(RuleResult("valid-images", True, f"All {len(image_paths)} image file(s) readable and valid"))

    # Rule 6: Non-image files follow preview convention
    preview_errors: list[str] = []
    for folder, kind in folders:
        folder_catalog = [p for p, f, _ in catalog_files if f == folder]
        folder_previews = [p for p in preview_files if p.parent == folder]

        if kind in IMAGE_KINDS:
            if folder_previews:
                names = ", ".join(p.name for p in folder_previews[:5])
                preview_errors.append(f"preview file(s) not allowed in image folder '{folder.name}': {names}")
            continue

        non_image_assets = [
            p for p in folder_catalog if p.suffix.lower() not in build_pack_entry.IMAGE_EXTENSIONS
        ]
        empty_assets = [p.name for p in non_image_assets if p.stat().st_size == 0]
        if empty_assets:
            preview_errors.append(f"0-byte non-image asset(s) in '{folder.name}': {', '.join(empty_assets[:5])}")

        asset_stems = {os.path.splitext(p.name)[0] for p in non_image_assets}
        stems_seen: dict[str, list[str]] = {}
        for prev in folder_previews:
            parsed = _split_preview_stem(prev.name)
            if parsed is None:
                preview_errors.append(f"malformed preview filename '{folder.name}/{prev.name}'")
                continue
            asset_stem, _ = parsed
            stems_seen.setdefault(asset_stem, []).append(prev.name)
            if asset_stem not in asset_stems:
                preview_errors.append(
                    f"orphaned preview '{folder.name}/{prev.name}' (no matching non-image asset '{asset_stem}.*')"
                )

        for asset_stem, pnames in stems_seen.items():
            if len(pnames) > 1:
                preview_errors.append(
                    f"multiple previews for '{asset_stem}' in '{folder.name}': {', '.join(pnames)}"
                )

    if preview_errors:
        results.append(RuleResult("preview-convention", False, "; ".join(preview_errors[:5])))
    elif pack_kind is None:
        non_img_count = sum(1 for _, _, k in catalog_files if k not in IMAGE_KINDS)
        results.append(
            RuleResult(
                "preview-convention",
                True,
                f"{non_img_count} non-image asset(s) follow preview convention "
                f"({len(preview_files)} optional .preview image(s))",
            )
        )
    elif pack_kind in IMAGE_KINDS:
        results.append(
            RuleResult(
                "preview-convention",
                True,
                f"Image-only pack ('{pack_kind}'); no preview files needed or misplaced",
            )
        )
    else:
        results.append(
            RuleResult(
                "preview-convention",
                True,
                f"{len(catalog_files)} non-image asset(s) follow preview convention "
                f"({len(preview_files)} optional .preview image(s))",
            )
        )

    all_passed = all(r.passed for r in results)
    summary = PackSummary(
        pack_dir=pack_dir,
        kind=pack_kind,
        folders=folders,
        catalog_files=catalog_files,
        preview_files=preview_files,
        meta=meta,
    )
    return all_passed, results, summary


def print_report(pack_dir: Path, all_passed: bool, results: list[RuleResult]) -> None:
    print(f"Checking pack: {pack_dir}")
    for res in results:
        tag = "PASS" if res.passed else "FAIL"
        print(f"[{tag}] {res.name}: {res.reason}")
    passed_count = sum(1 for r in results if r.passed)
    overall = "PASS" if all_passed else "FAIL"
    print(f"Result: {overall} ({passed_count}/{len(results)} rules passed)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate a pack folder against DEVIL-PAINTER Asset Library standing rules."
    )
    parser.add_argument("pack_folder", type=Path, help="Path to the pack folder to check")
    parser.add_argument(
        "--max",
        dest="max_files",
        type=int,
        default=MAX_PACK_FILES,
        help=f"Maximum catalog files allowed per pack (default: {MAX_PACK_FILES})",
    )
    args = parser.parse_args(argv)

    if args.max_files < 1:
        print("ERROR: --max must be at least 1", file=sys.stderr)
        return 2

    all_passed, results, _ = validate_pack(args.pack_folder, max_files=args.max_files)
    print_report(args.pack_folder, all_passed, results)
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
