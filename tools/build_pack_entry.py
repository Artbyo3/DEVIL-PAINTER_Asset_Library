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
import tempfile
from pathlib import Path
from urllib.parse import quote

DEFAULT_REPO = "Artbyo3/DEVIL-PAINTER_Asset_Library"
SAFE_TAG_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
SAFE_REPO_RE = re.compile(r"^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$")
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


def validate_release_tag(release_tag: str) -> str:
    if (
        not isinstance(release_tag, str)
        or not SAFE_TAG_RE.match(release_tag)
        or ".." in release_tag
    ):
        raise ValueError(f"Unsafe release tag: {release_tag!r}")
    return release_tag


def validate_repo(repo: str) -> str:
    if not isinstance(repo, str) or not SAFE_REPO_RE.match(repo) or ".." in repo or repo != DEFAULT_REPO:
        raise ValueError(f"Refusing non-canonical repository slug {repo!r}; expected {DEFAULT_REPO!r}")
    return repo


def validate_safe_filename(name: str) -> str:
    if (
        not isinstance(name, str)
        or not name
        or name in (".", "..")
        or "/" in name
        or "\\" in name
        or "\x00" in name
        or any(ord(ch) < 32 for ch in name)
    ):
        raise ValueError(f"Unsafe filename: {name!r}")
    return name


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
    if pack_dir.is_symlink():
        raise ValueError(f"Symlink pack directory is not allowed: {pack_dir}")
    pack_root = pack_dir.resolve()
    folders: list[tuple[Path, str]] = []
    for folder_name, kind in KIND_FOLDERS:
        folder = pack_dir / folder_name
        if folder.is_symlink():
            raise ValueError(f"Symlink asset folder is not allowed: {folder}")
        if folder.is_dir():
            if not folder.resolve().is_relative_to(pack_root):
                raise ValueError(f"Asset folder escapes pack root: {folder}")
            folders.append((folder, kind))
    extra_folders = sorted(
        (
            entry
            for entry in pack_dir.iterdir()
            if not entry.name.startswith(".")
            and entry.name not in KNOWN_FOLDER_NAMES
            and (entry.is_symlink() or entry.is_dir())
        ),
        key=lambda entry: entry.name,
    )
    for folder in extra_folders:
        if folder.is_symlink():
            raise ValueError(f"Symlink asset folder is not allowed: {folder}")
        if not folder.resolve().is_relative_to(pack_root):
            raise ValueError(f"Asset folder escapes pack root: {folder}")
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
    validate_release_tag(release_tag)
    validate_repo(repo)

    if pack_dir.is_symlink():
        raise ValueError(f"Symlink pack directory is not allowed: {pack_dir}")
    pack_root = pack_dir.resolve()
    pack_json_path = pack_dir / "pack.json"
    if pack_json_path.is_symlink():
        raise ValueError(f"Symlink pack.json is not allowed: {pack_json_path}")
    if not pack_json_path.is_file():
        raise FileNotFoundError(f"Missing pack.json in {pack_dir}")
    if zip_path.is_symlink():
        raise ValueError(f"Symlink zip file is not allowed: {zip_path}")
    if not zip_path.is_file():
        raise FileNotFoundError(f"Missing zip file: {zip_path}")
    validate_safe_filename(zip_path.name)

    with pack_json_path.open("r", encoding="utf-8") as fh:
        meta = json.load(fh)
    if not isinstance(meta, dict):
        raise ValueError("pack.json root must be a JSON object")

    files: list[str] = []
    file_tags: dict[str, list[str]] = {}
    file_hashes: dict[str, str] = {}
    materials: dict[str, dict[str, str]] = {}
    for folder, kind in iter_pack_folders(pack_dir):
        entries = sorted(
            (e for e in folder.iterdir() if not e.name.startswith(".")),
            key=lambda e: e.name,
        )
        for entry in entries:
            if entry.is_symlink():
                raise ValueError(f"Symlink asset entry is not allowed: {entry}")
            if not entry.is_file():
                continue
            if not entry.resolve().is_relative_to(pack_root):
                raise ValueError(f"Asset entry escapes pack root: {entry}")
            name = validate_safe_filename(entry.name)
            if is_preview_file(name):
                continue
            if name in file_tags:
                raise ValueError(f"Duplicate filename across folders: {name}")
            files.append(name)
            file_tags[name] = build_file_tags(name, kind)
            file_hashes[name] = sha256_file(entry)
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


def _atomic_write_json(target_path: Path, data: dict) -> None:
    if target_path.is_symlink():
        raise ValueError(f"Refusing to write through symlink: {target_path}")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{target_path.name}.", suffix=".tmp", dir=str(target_path.parent))
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(data, fh, indent=2)
            fh.write("\n")
        os.replace(tmp_path, target_path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def update_catalog(catalog_path: Path, entry: dict) -> None:
    if catalog_path.is_symlink():
        raise ValueError(f"Refusing to read/write symlink catalog: {catalog_path}")
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

    _atomic_write_json(catalog_path, data)


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
