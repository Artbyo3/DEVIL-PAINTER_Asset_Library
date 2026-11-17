"""Build or update a packs.json catalog entry from a pack folder, release tag, and zip file.

Usage:
    python tools/build_pack_entry.py <pack_dir> <release_tag> <zip_path> [--catalog packs.json]

Without --catalog, prints the JSON entry to stdout.
With --catalog, inserts or updates the matching entry (by releaseTag) in packs.json.

Kind convention:
    Each asset's kind is derived from its folder inside <pack_dir> and appended as
    the last entry of its fileTags list:
      - Image kinds: Textures -> "texture", Alphas -> "alpha"
      - Non-image kinds: Materials -> "material", Brushes -> "brush",
        Generators -> "generator", any other folder -> "asset"

Preview convention:
    Non-image assets may ship a sibling preview image in the same folder named
    <stem>.preview.<img_ext> (e.g. oak-wood.blend + oak-wood.preview.png), where
    <img_ext> is one of png, jpg, jpeg, webp, bmp, gif, tif, tiff. Preview files
    are never added to the catalog's files/fileTags and are never release assets.

Per-file hashes:
    Each catalog entry includes a 'fileHashes' object mapping every filename in
    'files' to the lowercase hex SHA-256 digest of that source file's bytes,
    computed directly from the pack folder files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from urllib.parse import quote

DEFAULT_REPO = "Artbyo3/DEVIL-PAINTER_Asset_Library"
KIND_FOLDERS = (
    ("Textures", "texture"),
    ("Alphas", "alpha"),
    ("Materials", "material"),
    ("Brushes", "brush"),
    ("Generators", "generator"),
)
KNOWN_FOLDER_NAMES = {folder_name for folder_name, _ in KIND_FOLDERS}
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff")
MAP_SUFFIXES = (
    "_normal",
    "_roughness",
    "_ao",
    "_height",
    "_metallic",
    "_basecolor",
    "_albedo",
    "_diffuse",
)


def is_preview_file(filename: str) -> bool:
    stem, ext = os.path.splitext(filename)
    return ext.lower() in IMAGE_EXTENSIONS and stem.lower().endswith(".preview")


def split_stem_and_role(filename: str) -> tuple[str, str]:
    stem = os.path.splitext(filename)[0]
    lower = stem.lower()
    for suffix in MAP_SUFFIXES:
        if lower.endswith(suffix) and len(stem) > len(suffix):
            return stem[: -len(suffix)], suffix[1:]
    return stem, "base"


def iter_pack_folders(pack_dir: Path) -> list[tuple[Path, str]]:
    folders: list[tuple[Path, str]] = []
    for folder_name, kind in KIND_FOLDERS:
        folder = pack_dir / folder_name
        if folder.is_dir():
            folders.append((folder, kind))
    extra_folders = sorted(
        (
            entry
            for entry in pack_dir.iterdir()
            if entry.is_dir()
            and not entry.name.startswith(".")
            and entry.name not in KNOWN_FOLDER_NAMES
        ),
        key=lambda entry: entry.name,
    )
    for folder in extra_folders:
        folders.append((folder, "asset"))
    return folders


def format_size(num_bytes: int) -> str:
    if num_bytes < 1024 * 1024:
        return f"{num_bytes / 1024:.1f} KB"
    if num_bytes < 1024 * 1024 * 1024:
        return f"{num_bytes / (1024 * 1024):.1f} MB"
    return f"{num_bytes / (1024 * 1024 * 1024):.2f} GB"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def build_file_tags(filename: str, kind: str) -> list[str]:
    base_stem, _ = split_stem_and_role(filename)
    slug = re.sub(r"-\d+$", "", base_stem.lower())
    words = [w for w in slug.split("-") if w]
    seen: set[str] = set()
    tags: list[str] = []
    for item in [slug, *words, kind]:
        if item and item not in seen:
            seen.add(item)
            tags.append(item)
    return tags


def build_entry(
    pack_dir: Path,
    release_tag: str,
    zip_path: Path,
    repo: str = DEFAULT_REPO,
) -> dict:
    pack_json_path = pack_dir / "pack.json"
    if not pack_json_path.is_file():
        raise FileNotFoundError(f"Missing pack.json in {pack_dir}")
    if not zip_path.is_file():
        raise FileNotFoundError(f"Missing zip file: {zip_path}")

    with pack_json_path.open("r", encoding="utf-8") as fh:
        meta = json.load(fh)

    files: list[str] = []
    file_tags: dict[str, list[str]] = {}
    file_hashes: dict[str, str] = {}
    materials: dict[str, dict[str, str]] = {}
    for folder, kind in iter_pack_folders(pack_dir):
        names = sorted(
            entry.name
            for entry in folder.iterdir()
            if entry.is_file()
            and not entry.name.startswith(".")
            and not is_preview_file(entry.name)
        )
        for name in names:
            if name in file_tags:
                raise ValueError(f"Duplicate filename across folders: {name}")
            files.append(name)
            file_tags[name] = build_file_tags(name, kind)
            file_hashes[name] = sha256_file(folder / name)
            base_stem, role = split_stem_and_role(name)
            mat_maps = materials.setdefault(base_stem, {})
            if role not in mat_maps:
                mat_maps[role] = name

    zip_bytes = zip_path.stat().st_size
    download_url = (
        f"https://github.com/{repo}/releases/download/"
        f"{quote(release_tag, safe='')}/{quote(zip_path.name, safe='')}"
    )

    return {
        "name": meta["name"],
        "description": meta["description"],
        "size": format_size(zip_bytes),
        "downloadUrl": download_url,
        "releaseTag": release_tag,
        "tags": meta["tags"],
        "files": files,
        "fileTags": file_tags,
        "fileHashes": file_hashes,
        "materials": materials,
        "sha256": sha256_file(zip_path),
    }


def update_catalog(catalog_path: Path, entry: dict) -> None:
    if catalog_path.is_file():
        with catalog_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    else:
        data = {"packs": []}

    packs = data.get("packs")
    if not isinstance(packs, list):
        raise ValueError(f"Expected 'packs' array in {catalog_path}")

    for idx, existing in enumerate(packs):
        if isinstance(existing, dict) and existing.get("releaseTag") == entry["releaseTag"]:
            merged = dict(entry)
            if "thumbs" in existing and "thumbs" not in merged:
                merged["thumbs"] = existing["thumbs"]
            packs[idx] = merged
            break
    else:
        packs.append(entry)

    with catalog_path.open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(data, fh, indent=2)
        fh.write("\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build or update a packs.json entry from a pack folder, release tag, and zip."
    )
    parser.add_argument("pack_dir", type=Path, help="Path to the pack folder")
    parser.add_argument("release_tag", help="GitHub release tag (e.g. cloth-apparel-vol1-v1.0.0)")
    parser.add_argument("zip_path", type=Path, help="Path to the pack zip archive")
    parser.add_argument(
        "--catalog",
        type=Path,
        default=None,
        help="Optional packs.json path to insert/update in place",
    )
    parser.add_argument(
        "--repo",
        default=DEFAULT_REPO,
        help=f"GitHub owner/repo for release download URLs (default: {DEFAULT_REPO})",
    )
    args = parser.parse_args(argv)

    entry = build_entry(args.pack_dir, args.release_tag, args.zip_path, repo=args.repo)
    if args.catalog is not None:
        update_catalog(args.catalog, entry)
    else:
        json.dump(entry, sys.stdout, indent=2)
        sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
