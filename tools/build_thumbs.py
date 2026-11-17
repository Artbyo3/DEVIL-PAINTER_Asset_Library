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
import warnings
from pathlib import Path

from PIL import Image

try:
    import build_pack_entry
except ImportError:
    from tools import build_pack_entry  # type: ignore[no-redef]

KIND_FOLDERS = build_pack_entry.KIND_FOLDERS
KNOWN_FOLDER_NAMES = build_pack_entry.KNOWN_FOLDER_NAMES
IMAGE_KINDS = {"texture", "alpha"}
IMAGE_EXTENSIONS = build_pack_entry.IMAGE_EXTENSIONS
MAX_SIZE = (160, 160)
WEBP_QUALITY = 72
JPEG_QUALITY = 75
MAX_IMAGE_PIXELS = 134_217_728  # 128 MP cap (allows up to 8K+ textures, blocks decompression bombs)
MAX_IMAGE_FILE_BYTES = 256 * 1024 * 1024


def is_preview_file(filename: str) -> bool:
    return build_pack_entry.is_preview_file(filename)


def iter_pack_folders(pack_dir: Path) -> list[tuple[Path, str]]:
    return build_pack_entry.iter_pack_folders(pack_dir)


def find_preview_image(asset_path: Path) -> Path | None:
    expected_stem = f"{os.path.splitext(asset_path.name)[0]}.preview"
    candidates = sorted(
        (
            entry
            for entry in asset_path.parent.iterdir()
            if not entry.name.startswith(".") and (entry.is_symlink() or entry.is_file())
        ),
        key=lambda entry: entry.name,
    )
    for candidate in candidates:
        cand_stem, cand_ext = os.path.splitext(candidate.name)
        if cand_stem == expected_stem and cand_ext.lower() in IMAGE_EXTENSIONS:
            if candidate.is_symlink():
                raise ValueError(f"Symlink preview file is not allowed: {candidate}")
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


def _load_safe_image(path: Path) -> Image.Image:
    if path.is_symlink():
        raise ValueError(f"Symlink image is not allowed: {path}")
    size_bytes = path.stat().st_size
    if size_bytes <= 0:
        raise ValueError(f"0-byte image file: {path}")
    if size_bytes > MAX_IMAGE_FILE_BYTES:
        raise ValueError(f"Image exceeds max file size ({MAX_IMAGE_FILE_BYTES} bytes): {path}")
    prev_max = Image.MAX_IMAGE_PIXELS
    Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(path) as raw:
                w, h = raw.size
                if w <= 0 or h <= 0 or w * h > MAX_IMAGE_PIXELS:
                    raise ValueError(f"Image dimensions {w}x{h} exceed pixel limit {MAX_IMAGE_PIXELS}")
                return raw.copy()
    finally:
        Image.MAX_IMAGE_PIXELS = prev_max


def build_thumbs(
    pack_dir: Path,
    release_tag: str,
    thumbs_dir: Path = Path("thumbs"),
    site_prefix: str = "thumbs",
) -> dict[str, str]:
    build_pack_entry.validate_release_tag(release_tag)
    if pack_dir.is_symlink():
        raise ValueError(f"Symlink pack directory is not allowed: {pack_dir}")
    if thumbs_dir.is_symlink():
        raise ValueError(f"Symlink thumbs directory is not allowed: {thumbs_dir}")

    prefix = site_prefix.strip("/")
    if prefix and (
        ".." in prefix
        or "\\" in prefix
        or ":" in prefix
        or "?" in prefix
        or "#" in prefix
        or any(ord(c) < 32 for c in prefix)
    ):
        raise ValueError(f"Unsafe site_prefix: {site_prefix!r}")

    pack_root = pack_dir.resolve()
    thumbs_root = thumbs_dir.resolve()
    out_dir = (thumbs_root / release_tag).resolve()
    if not out_dir.is_relative_to(thumbs_root) or out_dir == thumbs_root:
        raise ValueError(f"Thumbnail output directory escapes thumbs root: {out_dir}")
    if out_dir.exists() and out_dir.is_symlink():
        raise ValueError(f"Symlink thumbnail output directory is not allowed: {out_dir}")

    mapping: dict[str, str] = {}
    seen_assets: set[str] = set()
    for folder, kind in iter_pack_folders(pack_dir):
        entries = sorted(
            (
                e
                for e in folder.iterdir()
                if not e.name.startswith(".")
                and (e.is_symlink() or e.is_file())
            ),
            key=lambda e: e.name,
        )
        for src_path in entries:
            if src_path.is_symlink():
                raise ValueError(f"Symlink asset is not allowed: {src_path}")
            if not src_path.resolve().is_relative_to(pack_root):
                raise ValueError(f"Asset escapes pack root: {src_path}")
            name = build_pack_entry.validate_safe_filename(src_path.name)
            if is_preview_file(name):
                continue
            if name in seen_assets:
                raise ValueError(f"Duplicate filename across folders: {name}")
            seen_assets.add(name)

            if kind in IMAGE_KINDS or src_path.suffix.lower() in IMAGE_EXTENSIONS:
                thumb_source: Path | None = src_path
            else:
                thumb_source = find_preview_image(src_path)
                if thumb_source is None:
                    continue

            out_dir.mkdir(parents=True, exist_ok=True)
            stem = os.path.splitext(name)[0]
            img = _load_safe_image(thumb_source)
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

            mapping[name] = f"{prefix}/{release_tag}/{rel_name}" if prefix else f"{release_tag}/{rel_name}"

    return mapping


def update_catalog(catalog_path: Path, release_tag: str, thumbs_map: dict[str, str]) -> None:
    build_pack_entry.validate_release_tag(release_tag)
    if catalog_path.is_symlink():
        raise ValueError(f"Refusing to read/write symlink catalog: {catalog_path}")
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

    build_pack_entry._atomic_write_json(catalog_path, data)


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
