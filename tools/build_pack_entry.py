"""Build or update a packs.json catalog entry from a pack folder, release tag, and zip file.

Usage:
    python tools/build_pack_entry.py <pack_dir> <release_tag> <zip_path> [--catalog packs.json]

Without --catalog, prints the JSON entry to stdout.
With --catalog, inserts or updates the matching entry (by releaseTag) in packs.json.
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
KIND_FOLDERS = (("Textures", "texture"), ("Alphas", "alpha"))


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
    stem = os.path.splitext(filename)[0].lower()
    slug = re.sub(r"-\d+$", "", stem)
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
    for folder_name, kind in KIND_FOLDERS:
        folder = pack_dir / folder_name
        if not folder.is_dir():
            continue
        names = sorted(
            entry.name
            for entry in folder.iterdir()
            if entry.is_file() and not entry.name.startswith(".")
        )
        for name in names:
            if name in file_tags:
                raise ValueError(f"Duplicate filename across folders: {name}")
            files.append(name)
            file_tags[name] = build_file_tags(name, kind)

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
            packs[idx] = entry
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
