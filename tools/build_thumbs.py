"""Build 160x160 WebP thumbnails for a pack and wire the thumbs map into packs.json.

Usage:
    python tools/build_thumbs.py <pack_dir> <release_tag> [--thumbs-dir thumbs] [--catalog packs.json]

Without --catalog, prints the filename -> relative thumb path JSON mapping to stdout.
With --catalog, sets the 'thumbs' field on the matching entry (by releaseTag) in packs.json.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from PIL import Image

KIND_FOLDERS = ("Textures", "Alphas")
MAX_SIZE = (160, 160)
WEBP_QUALITY = 72
JPEG_QUALITY = 75


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
    out_dir.mkdir(parents=True, exist_ok=True)

    mapping: dict[str, str] = {}
    for folder_name in KIND_FOLDERS:
        folder = pack_dir / folder_name
        if not folder.is_dir():
            continue
        entries = sorted(
            (e for e in folder.iterdir() if e.is_file() and not e.name.startswith(".")),
            key=lambda e: e.name,
        )
        for src_path in entries:
            if src_path.name in mapping:
                raise ValueError(f"Duplicate filename across folders: {src_path.name}")
            stem = os.path.splitext(src_path.name)[0]
            with Image.open(src_path) as raw:
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
