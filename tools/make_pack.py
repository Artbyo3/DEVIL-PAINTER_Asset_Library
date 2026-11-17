"""Validate a pack folder, build its release zip, and create a DRAFT GitHub release.

Usage:
    python tools/make_pack.py <pack_folder> [--repo Artbyo3/DEVIL-PAINTER_Asset_Library] [--version v1.0.0]

Steps performed:
  1. Run tools/check_pack.py validation rules; refuse to continue on failure.
  2. Build <PackFolder>.zip alongside <pack_folder> (containing <PackFolder>/pack.json
     and <PackFolder>/<KindFolder>/...).
  3. Create a DRAFT GitHub release using live naming/tag conventions
     (e.g. "Cloth Textures Vol. 1" -> tag "cloth-textures-vol1-v1.0.0") and upload
     the zip plus every individual catalog asset file (excluding .preview.* files).

Stops at the draft release. VirusTotal scanning, publishing, and packs.json /
thumbs updates belong strictly to the Release Scan Gate workflow.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

try:
    import build_pack_entry
    import check_pack
except ImportError:
    from tools import build_pack_entry, check_pack  # type: ignore[no-redef]

DEFAULT_REPO = build_pack_entry.DEFAULT_REPO
DEFAULT_VERSION = "v1.0.0"
UPLOAD_BATCH_SIZE = 25
CC0_FOOTER = (
    "Everything is CC0 - use it for any work, no credit needed. "
    "The download is an ordinary zip file."
)


def slugify_pack_name(name: str) -> str:
    s = name.strip().lower()
    s = re.sub(r"\bvol\.?\s*(\d+)\b", r"vol\1", s)
    s = re.sub(r"[^a-z0-9]+", "-", s)
    return s.strip("-")


def build_release_tag(pack_name: str, version: str = DEFAULT_VERSION) -> str:
    slug = slugify_pack_name(pack_name)
    if not slug:
        raise ValueError(f"Could not derive release tag slug from pack name: {pack_name!r}")
    ver = version.strip()
    if not ver.startswith("v"):
        ver = f"v{ver}"
    tag = f"{slug}-{ver}"
    return build_pack_entry.validate_release_tag(tag)


def build_release_notes(description: str) -> str:
    desc = description.strip()
    if "cc0" in desc.lower():
        return desc
    return f"{desc} {CC0_FOOTER}" if desc else CC0_FOOTER


def build_pack_zip(pack_dir: Path, zip_path: Path) -> Path:
    if pack_dir.is_symlink():
        raise ValueError(f"Symlink pack directory is not allowed: {pack_dir}")
    if zip_path.is_symlink():
        raise ValueError(f"Symlink zip destination is not allowed: {zip_path}")

    pack_dir = pack_dir.resolve()
    top = build_pack_entry.validate_safe_filename(pack_dir.name)
    build_pack_entry.validate_safe_filename(zip_path.name)
    zip_path.parent.mkdir(parents=True, exist_ok=True)

    pack_json = pack_dir / "pack.json"
    if pack_json.is_symlink():
        raise ValueError(f"Symlink pack.json is not allowed: {pack_json}")
    files_to_zip: list[Path] = [pack_json]
    for folder, _ in build_pack_entry.iter_pack_folders(pack_dir):
        for entry in sorted(folder.iterdir(), key=lambda p: p.name):
            if entry.is_symlink():
                raise ValueError(f"Symlink entry is not allowed: {entry}")
            if entry.is_file() and not entry.name.startswith("."):
                if not entry.resolve().is_relative_to(pack_dir):
                    raise ValueError(f"Entry escapes pack root: {entry}")
                build_pack_entry.validate_safe_filename(entry.name)
                files_to_zip.append(entry)

    fd, tmp_name = tempfile.mkstemp(prefix=f".{zip_path.name}.", suffix=".tmp", dir=str(zip_path.parent))
    os.close(fd)
    tmp_zip = Path(tmp_name)
    try:
        with zipfile.ZipFile(tmp_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for src in files_to_zip:
                rel = src.relative_to(pack_dir).as_posix()
                zf.write(src, arcname=f"{top}/{rel}")
        os.replace(tmp_zip, zip_path)
    except Exception:
        if tmp_zip.exists():
            tmp_zip.unlink()
        raise

    return zip_path


def _run_gh(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["gh", *args],
        check=False,
        text=True,
        capture_output=True,
    )


def _find_existing_release(repo: str, tag: str) -> dict | None:
    build_pack_entry.validate_repo(repo)
    build_pack_entry.validate_release_tag(tag)
    proc = _run_gh(["api", "--paginate", "--slurp", f"repos/{repo}/releases"])
    if proc.returncode != 0:
        raise RuntimeError(f"Failed to query GitHub releases via gh: {proc.stderr.strip()}")
    pages = json.loads(proc.stdout or "[]")
    releases = [rel for page in pages for rel in (page if isinstance(page, list) else [page])]
    for rel in releases:
        if isinstance(rel, dict) and rel.get("tag_name") == tag:
            return rel
    return None


def make_pack(
    pack_dir: Path,
    repo: str = DEFAULT_REPO,
    version: str = DEFAULT_VERSION,
    zip_path: Path | None = None,
    release_tag: str | None = None,
) -> int:
    build_pack_entry.validate_repo(repo)
    if pack_dir.is_symlink():
        print(f"ERROR: Symlink pack directory is not allowed: {pack_dir}", file=sys.stderr)
        return 1
    pack_dir = pack_dir.resolve()

    # Step 1: check_pack (refuse to continue if any rule fails)
    all_passed, results, summary = check_pack.validate_pack(pack_dir)
    check_pack.print_report(pack_dir, all_passed, results)
    if not all_passed:
        print("\nERROR: check_pack failed; refusing to build zip or create draft release.", file=sys.stderr)
        return 1

    assert summary.meta is not None
    pack_title = str(summary.meta["name"]).strip()
    description = str(summary.meta["description"]).strip()
    tag = (
        build_pack_entry.validate_release_tag(release_tag)
        if release_tag is not None
        else build_release_tag(pack_title, version=version)
    )
    notes = build_release_notes(description)
    out_zip = (zip_path or (pack_dir.parent / f"{pack_dir.name}.zip")).resolve()
    if out_zip.is_symlink():
        print(f"ERROR: Symlink zip destination is not allowed: {out_zip}", file=sys.stderr)
        return 1
    build_pack_entry.validate_safe_filename(out_zip.name)
    if out_zip.name.startswith("-"):
        print(f"ERROR: Unsafe zip filename starting with '-': {out_zip.name}", file=sys.stderr)
        return 1
    catalog_paths = [p.resolve() for p, _, _ in summary.catalog_files]
    for p in catalog_paths:
        build_pack_entry.validate_safe_filename(p.name)
        if p.name.startswith("-"):
            print(f"ERROR: Unsafe asset filename starting with '-': {p.name}", file=sys.stderr)
            return 1

    print(f"\nRelease plan:")
    print(f"  Repo:        {repo}")
    print(f"  Title:       {pack_title}")
    print(f"  Tag:         {tag}")
    print(f"  Zip:         {out_zip}")
    print(f"  Asset files: {len(catalog_paths)} individual file(s) + 1 zip")

    # Refuse to collide with an existing published or draft release with the same tag
    existing = _find_existing_release(repo, tag)
    if existing is not None:
        state = "draft" if existing.get("draft") else "published"
        print(
            f"ERROR: A {state} release with tag '{tag}' already exists on {repo} "
            f"(id={existing.get('id')}). Refusing to overwrite.",
            file=sys.stderr,
        )
        return 1

    # Step 2: Build the release zip
    build_pack_zip(pack_dir, out_zip)
    print(f"Built release zip: {out_zip} ({build_pack_entry.format_size(out_zip.stat().st_size)})")

    # Step 3: Create DRAFT release and upload zip + every individual catalog file
    create_proc = _run_gh(
        [
            "release",
            "create",
            tag,
            str(out_zip),
            "--draft",
            "--repo",
            repo,
            "--title",
            pack_title,
            "--notes",
            notes,
        ]
    )
    if create_proc.returncode != 0:
        print(f"ERROR: Failed to create draft release '{tag}':\n{create_proc.stderr.strip()}", file=sys.stderr)
        return 1

    total_batches = (len(catalog_paths) + UPLOAD_BATCH_SIZE - 1) // UPLOAD_BATCH_SIZE
    for batch_idx in range(total_batches):
        batch = catalog_paths[batch_idx * UPLOAD_BATCH_SIZE : (batch_idx + 1) * UPLOAD_BATCH_SIZE]
        up_proc = _run_gh(
            [
                "release",
                "upload",
                tag,
                "--repo",
                repo,
                "--clobber",
                "--",
                *[str(p) for p in batch],
            ]
        )
        if up_proc.returncode != 0:
            print(
                f"ERROR: Failed uploading asset batch {batch_idx + 1}/{total_batches} to draft '{tag}':\n"
                f"{up_proc.stderr.strip()}",
                file=sys.stderr,
            )
            return 1
        print(f"  Uploaded asset batch {batch_idx + 1}/{total_batches} ({len(batch)} file(s))")

    # Confirm final draft release state
    view_proc = _run_gh(
        [
            "release",
            "view",
            tag,
            "--repo",
            repo,
            "--json",
            "tagName,name,isDraft,url,assets",
        ]
    )
    if view_proc.returncode != 0:
        print(f"ERROR: Created draft release '{tag}' but verification query failed: {view_proc.stderr.strip()}", file=sys.stderr)
        return 1

    rel_info = json.loads(view_proc.stdout)
    asset_count = len(rel_info.get("assets", []))
    expected_count = 1 + len(catalog_paths)
    print(
        f"\nDraft release ready:\n"
        f"  Tag:    {rel_info['tagName']}\n"
        f"  Title:  {rel_info['name']}\n"
        f"  Draft:  {rel_info['isDraft']}\n"
        f"  Assets: {asset_count} (expected {expected_count})\n"
        f"  URL:    {rel_info['url']}\n"
        f"Next step: run the Release Scan Gate workflow (workflow_dispatch with release_tag={tag}) "
        f"to scan, update packs.json/thumbs, and publish."
    )
    if rel_info.get("isDraft") is not True or asset_count != expected_count:
        print("ERROR: Draft release verification mismatch.", file=sys.stderr)
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate a pack folder, build its zip, and create a DRAFT GitHub release."
    )
    parser.add_argument("pack_folder", type=Path, help="Path to the pack folder")
    parser.add_argument(
        "--repo",
        default=DEFAULT_REPO,
        help=f"Target GitHub repository (default: {DEFAULT_REPO})",
    )
    parser.add_argument(
        "--version",
        default=DEFAULT_VERSION,
        help=f"Release version suffix used in the tag (default: {DEFAULT_VERSION})",
    )
    parser.add_argument(
        "--tag",
        default=None,
        help="Optional explicit release tag (defaults to slugified pack.json name + version)",
    )
    parser.add_argument(
        "--zip-path",
        type=Path,
        default=None,
        help="Optional output path for the release zip (defaults to <parent>/<PackFolder>.zip)",
    )
    args = parser.parse_args(argv)

    return make_pack(
        args.pack_folder,
        repo=args.repo,
        version=args.version,
        zip_path=args.zip_path,
        release_tag=args.tag,
    )


if __name__ == "__main__":
    raise SystemExit(main())
