"""Build 160x160 WebP thumbnails for a pack and wire the thumbs map into packs.json.

Usage:
    python tools/build_thumbs.py <pack_dir> <release_tag> [--thumbs-dir thumbs] [--catalog packs.json]

Without --catalog, prints the filename -> relative thumb path JSON mapping to stdout.
With --catalog, sets the 'thumbs' field on the matching entry (by releaseTag) in packs.json.

Kind convention:
    Assets live inside subfolders of <pack_dir>:
      - Image kinds: Textures ("texture"), Alphas ("alpha")
      - Non-image kinds: Materials ("material"), Brushes ("brush"),
        Generators ("generator"), any other folder ("asset")

Preview convention:
    For image files (image kinds or known image extensions), thumbnails are
    generated from the file itself. For non-image files, if a sibling preview
    picture named <stem>.preview.<img_ext> exists in the same folder (e.g.
    oak-wood.blend with oak-wood.preview.png), the thumbnail is generated from
    the preview and mapped under the asset's filename in 'thumbs'. If no preview
    exists, the file gets no 'thumbs' entry. Preview files themselves are never
    treated as catalog assets.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from PIL import Image

KIND_FOLDERS = (
    ("Textures", "texture"),
    ("Alphas", "alpha"),
    ("Materials", "material"),
    ("Brushes", "brush"),
    ("Generators", "generator"),
)
KNOWN_FOLDER_NAMES = {folder_name for folder_name, _ in KIND_FOLDERS}
IMAGE_KINDS = {"texture", "alpha"}
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff")
MAX_SIZE = (160, 160)
WEBP_QUALITY = 72
JPEG_QUALITY = 75


def is_preview_file(filename: str) -> bool:
    stem, ext = os.path.splitext(filename)
    return ext.lower() in IMAGE_EXTENSIONS and stem.lower().endswith(".preview")


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


def find_preview_image(asset_path: Path) -> Path | None:
    expected_stem = f"{os.path.splitext(asset_path.name)[0]}.preview"
    candidates = sorted(
        (
            entry
            for entry in asset_path.parent.iterdir()
            if entry.is_file() and not entry.name.startswith(".")
        ),
        key=lambda entry: entry.name,
    )
    for candidate in candidates:
        cand_stem, cand_ext = os.path.splitext(candidate.name)
        if cand_stem == expected_stem and cand_ext.lower() in IMAGE_EXTENSIONS:
            return candidate
    return None


def _prepare_for_webp(img: Image.Image) -> Image.Image:
    if img.mode in ("RGB", "RGBA", "L", "LA"):
        return img
    if "A" in img.getbands() or img.info.get("transparency") is not None:
        return img.convert("RGBA")
    return img.convert("RGB")


def _prepare_for_jpeg(img: Image.Image) -> Image.Image:
    if img.mode in ("RGB", "L"):
        return img
    return img.convert("RGB")


def build_thumbs(
    pack_dir: Path,
    release_tag: str,
    thumbs_dir: Path = Path("thumbs"),
    site_prefix: str = "thumbs",
) -> dict[str, str]:
    out_dir = thumbs_dir / release_tag

    mapping: dict[str, str] = {}
    seen_assets: set[str] = set()
    for folder, kind in iter_pack_folders(pack_dir):
        entries = sorted(
            (
                e
                for e in folder.iterdir()
                if e.is_file()
                and not e.name.startswith(".")
                and not is_preview_file(e.name)
            ),
            key=lambda e: e.name,
        )
        for src_path in entries:
            if src_path.name in seen_assets:
                raise ValueError(f"Duplicate filename across folders: {src_path.name}")
            seen_assets.add(src_path.name)

            if kind in IMAGE_KINDS or src_path.suffix.lower() in IMAGE_EXTENSIONS:
                thumb_source: Path | None = src_path
            else:
                thumb_source = find_preview_image(src_path)
                if thumb_source is None:
                    continue

            out_dir.mkdir(parents=True, exist_ok=True)
            stem = os.path.splitext(src_path.name)[0]
            with Image.open(thumb_source) as raw:
                img = raw.copy()
            img.thumbnail(MAX_SIZE, Image.Resampling.LANCZOS)

            webp_name = f"{stem}.webp"
            webp_path = out_dir / webp_name
            try:
                _prepare_for_webp(img).save(webp_path, format="WEBP", quality=WEBP_QUALITY)
                rel_name = webp_name
            except Exception:
                if webp_path.exists():
                    webp_path.unlink()
                jpg_name = f"{stem}.jpg"
                jpg_path = out_dir / jpg_name
                _prepare_for_jpeg(img).save(jpg_path, format="JPEG", quality=JPEG_QUALITY)
                rel_name = jpg_name

            prefix = site_prefix.strip("/")
            mapping[src_path.name] = f"{prefix}/{release_tag}/{rel_name}" if prefix else f"{release_tag}/{rel_name}"

    return mapping


def update_catalog(catalog_path: Path, release_tag: str, thumbs_map: dict[str, str]) -> None:
    with catalog_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    packs = data.get("packs")
    if not isinstance(packs, list):
        raise ValueError(f"Expected 'packs' array in {catalog_path}")

    for entry in packs:
        if isinstance(entry, dict) and entry.get("releaseTag") == release_tag:
            entry["thumbs"] = thumbs_map
            break
    else:
        raise KeyError(f"No pack entry with releaseTag {release_tag!r} in {catalog_path}")

    with catalog_path.open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(data, fh, indent=2)
        fh.write("\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build 160x160 WebP thumbnails for a pack and optionally wire them into packs.json."
    )
    parser.add_argument("pack_dir", type=Path, help="Path to the pack folder")
    parser.add_argument("release_tag", help="GitHub release tag (e.g. cloth-textures-vol1-v1.0.0)")
    parser.add_argument(
        "--thumbs-dir",
        type=Path,
        default=Path("thumbs"),
        help="Root output directory for thumbnails (default: thumbs)",
    )
    parser.add_argument(
        "--catalog",
        type=Path,
        default=None,
        help="Optional packs.json path to update in place with the thumbs mapping",
    )
    args = parser.parse_args(argv)

    thumbs_map = build_thumbs(args.pack_dir, args.release_tag, thumbs_dir=args.thumbs_dir)
    if args.catalog is not None:
        update_catalog(args.catalog, args.release_tag, thumbs_map)
    else:
        json.dump(thumbs_map, sys.stdout, indent=2)
        sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
