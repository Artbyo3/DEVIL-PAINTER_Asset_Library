"""Sync permanent shelf draft releases (shelf-<pack-slug>) to public releases, packs.json, thumbs/, and board/.

Modes:
  - report: Diff shelf assets against public release assets and packs.json; print full diff
            to stdout and GITHUB_STEP_SUMMARY without modifying anything.
  - apply:  For any pack with diffs:
              1. Scan new/changed shelf files via tools/vt_scan_gate.py (hash-first + cache + VIRUS_TOTAL_TOKEN).
                 Pure deletions skip VirusTotal scanning. Any detection aborts fail-closed.
              2. Assemble pack folder, validate via tools/check_pack.py, rebuild pack zip via tools/make_pack.py.
              3. Sync public release assets (add new, replace changed, delete removed, replace zip).
              4. Regenerate packs.json entry via tools/build_pack_entry.py and thumbnails via tools/build_thumbs.py.
              5. Regenerate board/ via tools/build_board.py.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import re
import shutil
import sys
import tempfile
import urllib.parse
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import build_board
    import build_pack_entry
    import build_thumbs
    import check_pack
    import make_pack
    import vt_scan_gate
except ImportError:
    from tools import (  # type: ignore[no-redef]
        build_board,
        build_pack_entry,
        build_thumbs,
        check_pack,
        make_pack,
        vt_scan_gate,
    )

DEFAULT_REPO = build_pack_entry.DEFAULT_REPO
SAFE_SLUG_RE = re.compile(r"^[a-z0-9]([a-z0-9-]{0,108}[a-z0-9])?$")
SHELF_PREFIX = "shelf-"
SHELF_META_RE = re.compile(r"<!--\s*shelf-meta\s*(\{.*?\})\s*-->", re.DOTALL)

KIND_TO_FOLDER = {kind: folder for folder, kind in build_pack_entry.KIND_FOLDERS}
FOLDER_TO_KIND = {folder: kind for folder, kind in build_pack_entry.KIND_FOLDERS}


@dataclass(frozen=True)
class AssetInfo:
    name: str
    asset_id: int
    api_url: str
    size: int
    sha256: str | None


@dataclass
class PackPlan:
    slug: str
    shelf_release: dict[str, Any]
    public_release: dict[str, Any] | None
    catalog_entry: dict[str, Any] | None
    meta: dict[str, Any]
    shelf_assets: dict[str, AssetInfo]
    public_assets: dict[str, AssetInfo]
    public_zip: AssetInfo | None
    adds: list[str]
    changes: list[str]
    deletes: list[str]
    drift: list[str]

    @property
    def has_changes(self) -> bool:
        return bool(self.adds or self.changes or self.deletes or self.drift)


def validate_pack_selector(pack_arg: str) -> str:
    val = pack_arg.strip()
    if val == "all":
        return "all"
    if not SAFE_SLUG_RE.match(val) or ".." in val or "--" in val:
        raise ValueError(f"Unsafe or invalid pack slug selector: {pack_arg!r}")
    return val


def validate_shelf_asset_name(name: str) -> str:
    safe = build_pack_entry.validate_safe_filename(name)
    if safe.startswith(".") or safe.startswith("-"):
        raise ValueError(f"Asset filename cannot start with '.' or '-': {safe!r}")
    if safe.lower().endswith(".zip") or safe in ("pack.json", "PACK-REPORT.txt"):
        raise ValueError(
            f"Shelf must hold loose master files only (disallowed artifact {safe!r})"
        )
    return safe


def _extract_sha256_digest(asset: dict[str, Any]) -> str | None:
    digest = asset.get("digest")
    if isinstance(digest, str) and digest.startswith("sha256:"):
        hex_part = digest[7:].strip().lower()
        if vt_scan_gate.SHA256_RE.match(hex_part):
            return hex_part
    return None


def list_all_releases(repo: str, gh_token: str) -> list[dict[str, Any]]:
    build_pack_entry.validate_repo(repo)
    releases: list[dict[str, Any]] = []
    page = 1
    while True:
        url = f"https://api.github.com/repos/{repo}/releases?per_page=100&page={page}"
        _, body, _ = vt_scan_gate.gh_api_request(url, gh_token)
        items = json.loads(body.decode("utf-8"))
        if not items:
            break
        for rel in items:
            if isinstance(rel, dict):
                releases.append(rel)
        page += 1
    return releases


def _parse_shelf_meta(body: str) -> tuple[dict[str, Any], str]:
    text = body or ""
    m = SHELF_META_RE.search(text)
    if not m:
        return {}, text.strip()
    meta_raw = json.loads(m.group(1))
    if not isinstance(meta_raw, dict):
        raise ValueError("shelf-meta block must be a JSON object")
    prose = (text[: m.start()] + text[m.end() :]).strip()
    return meta_raw, prose


def _infer_catalog_kind(catalog_entry: dict[str, Any] | None) -> str | None:
    if not catalog_entry:
        return None
    file_tags = catalog_entry.get("fileTags")
    if isinstance(file_tags, dict):
        for tags in file_tags.values():
            if isinstance(tags, list) and tags:
                last = str(tags[-1]).strip().lower()
                if last in KIND_TO_FOLDER:
                    return last
                if last == "asset":
                    return "asset"
    return None


def _match_catalog_entry(
    packs: list[dict[str, Any]],
    slug: str,
    explicit_tag: str | None,
) -> dict[str, Any] | None:
    for entry in packs:
        if explicit_tag and entry.get("releaseTag") == explicit_tag:
            return entry
    for entry in packs:
        name = str(entry.get("name", ""))
        if make_pack.slugify_pack_name(name) == slug:
            return entry
    for entry in packs:
        rel_tag = str(entry.get("releaseTag", ""))
        if rel_tag.startswith(f"{slug}-"):
            return entry
    return None


def _safe_zip_name(pack_name: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9]+", "_", pack_name).strip("_") or "Pack"
    zip_name = f"{stem}.zip"
    build_pack_entry.validate_safe_filename(zip_name)
    if zip_name.startswith("-") or zip_name.startswith("."):
        raise ValueError(f"Derived unsafe zip filename: {zip_name!r}")
    return zip_name


def resolve_pack_meta(
    slug: str,
    shelf_release: dict[str, Any],
    catalog_packs: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    shelf_meta, prose_body = _parse_shelf_meta(str(shelf_release.get("body") or ""))
    explicit_rel_tag = shelf_meta.get("releaseTag")
    if explicit_rel_tag is not None:
        explicit_rel_tag = build_pack_entry.validate_release_tag(str(explicit_rel_tag))

    catalog_entry = _match_catalog_entry(catalog_packs, slug, explicit_rel_tag)

    raw_title = str(shelf_release.get("name") or "").strip()
    if raw_title.lower().startswith("shelf:"):
        raw_title = raw_title[6:].strip()

    name = str(
        shelf_meta.get("name")
        or (catalog_entry.get("name") if catalog_entry else "")
        or raw_title
        or slug.replace("-", " ").title()
    ).strip()

    kind = str(
        shelf_meta.get("kind")
        or _infer_catalog_kind(catalog_entry)
        or (FOLDER_TO_KIND.get(str(shelf_meta.get("folder", ""))) if shelf_meta.get("folder") else "")
        or "texture"
    ).strip().lower()

    folder = str(
        shelf_meta.get("folder") or KIND_TO_FOLDER.get(kind, "Assets")
    ).strip()
    build_pack_entry.validate_safe_filename(folder)
    if folder.startswith(".") or folder.startswith("-"):
        raise ValueError(f"Unsafe pack folder name: {folder!r}")

    description = str(
        shelf_meta.get("description")
        or (catalog_entry.get("description") if catalog_entry else "")
        or prose_body
        or f"Curated {kind} assets for {name}."
    ).strip()

    release_tag = build_pack_entry.validate_release_tag(
        str(
            explicit_rel_tag
            or (catalog_entry.get("releaseTag") if catalog_entry else "")
            or make_pack.build_release_tag(name)
        )
    )
    if release_tag == f"{SHELF_PREFIX}{slug}":
        raise ValueError(f"Public releaseTag cannot collide with shelf draft tag: {release_tag!r}")

    catalog_zip = ""
    if catalog_entry and catalog_entry.get("downloadUrl"):
        catalog_zip = urllib.parse.unquote(str(catalog_entry["downloadUrl"]).rsplit("/", 1)[-1])

    zip_name = str(shelf_meta.get("zipName") or catalog_zip or _safe_zip_name(name)).strip()
    build_pack_entry.validate_safe_filename(zip_name)
    if not zip_name.lower().endswith(".zip") or zip_name.startswith(".") or zip_name.startswith("-"):
        raise ValueError(f"Invalid zip filename: {zip_name!r}")

    pack_dir_name = Path(zip_name).stem
    build_pack_entry.validate_safe_filename(pack_dir_name)

    raw_tags = shelf_meta.get("tags") or (catalog_entry.get("tags") if catalog_entry else None)
    if isinstance(raw_tags, list) and raw_tags and all(isinstance(t, str) and t.strip() for t in raw_tags):
        tags = [t.strip() for t in raw_tags]
    else:
        seen_tags: set[str] = set()
        tags = []
        for item in [slug, *slug.split("-"), kind]:
            if item and item not in seen_tags:
                seen_tags.add(item)
                tags.append(item)

    return (
        {
            "name": name,
            "description": description,
            "tags": tags,
            "kind": kind,
            "folder": folder,
            "releaseTag": release_tag,
            "zipName": zip_name,
            "packDirName": pack_dir_name,
        },
        catalog_entry,
    )


def _collect_shelf_assets(shelf_release: dict[str, Any]) -> dict[str, AssetInfo]:
    assets: dict[str, AssetInfo] = {}
    for raw in shelf_release.get("assets", []):
        name = validate_shelf_asset_name(str(raw.get("name", "")))
        if name in assets:
            raise ValueError(f"Duplicate shelf asset name: {name!r}")
        assets[name] = AssetInfo(
            name=name,
            asset_id=int(raw["id"]),
            api_url=str(raw["url"]),
            size=int(raw.get("size", 0)),
            sha256=_extract_sha256_digest(raw),
        )
    return assets


def _collect_public_assets(
    public_release: dict[str, Any] | None,
    zip_name: str,
) -> tuple[dict[str, AssetInfo], AssetInfo | None]:
    if public_release is None:
        return {}, None
    assets: dict[str, AssetInfo] = {}
    pub_zip: AssetInfo | None = None
    for raw in public_release.get("assets", []):
        name = build_pack_entry.validate_safe_filename(str(raw.get("name", "")))
        if name == "PACK-REPORT.txt":
            continue
        info = AssetInfo(
            name=name,
            asset_id=int(raw["id"]),
            api_url=str(raw["url"]),
            size=int(raw.get("size", 0)),
            sha256=_extract_sha256_digest(raw),
        )
        if name == zip_name:
            pub_zip = info
            continue
        if name in assets:
            raise ValueError(f"Duplicate public asset name: {name!r}")
        assets[name] = info
    return assets, pub_zip


def compute_pack_plan(
    slug: str,
    shelf_release: dict[str, Any],
    releases_by_tag: dict[str, dict[str, Any]],
    catalog_packs: list[dict[str, Any]],
    repo_root: Path,
) -> PackPlan:
    meta, catalog_entry = resolve_pack_meta(slug, shelf_release, catalog_packs)
    release_tag = meta["releaseTag"]
    zip_name = meta["zipName"]

    public_release = releases_by_tag.get(release_tag)
    if public_release is not None and public_release.get("draft") is True:
        raise RuntimeError(
            f"Public target tag {release_tag!r} is currently a draft release (not published); "
            "refusing to collide with scan-gate draft workflow."
        )

    shelf_assets = _collect_shelf_assets(shelf_release)
    public_assets, public_zip = _collect_public_assets(public_release, zip_name)

    cat_files = set(catalog_entry.get("files", [])) if catalog_entry else set()
    cat_hashes = dict(catalog_entry.get("fileHashes", {})) if catalog_entry else {}
    cat_thumbs = dict(catalog_entry.get("thumbs", {})) if catalog_entry else {}
    cat_zip_sha = str(catalog_entry.get("sha256", "")).lower() if catalog_entry else ""

    adds: list[str] = []
    changes: list[str] = []
    deletes: list[str] = []

    all_names = sorted(set(shelf_assets) | set(public_assets) | cat_files)
    for fname in all_names:
        s_info = shelf_assets.get(fname)
        p_info = public_assets.get(fname)
        in_cat = fname in cat_files
        c_sha = cat_hashes.get(fname)

        if s_info is not None:
            if p_info is None and not in_cat:
                adds.append(fname)
            elif (
                p_info is None
                or not in_cat
                or s_info.sha256 is None
                or p_info.sha256 != s_info.sha256
                or c_sha != s_info.sha256
            ):
                changes.append(fname)
        else:
            if p_info is not None or in_cat:
                deletes.append(fname)

    drift: list[str] = []
    if public_release is None:
        drift.append(f"missing public release ({release_tag})")
    if catalog_entry is None:
        drift.append("missing packs.json entry")
    if public_zip is None:
        drift.append(f"missing public zip ({zip_name})")
    elif cat_zip_sha and public_zip.sha256 and public_zip.sha256 != cat_zip_sha:
        drift.append(f"zip sha256 mismatch ({public_zip.sha256[:12]} != {cat_zip_sha[:12]})")

    if meta["kind"] in build_thumbs.IMAGE_KINDS:
        for fname in sorted(shelf_assets):
            if fname in cat_thumbs:
                t_rel = cat_thumbs[fname]
                if not (repo_root / t_rel).is_file():
                    drift.append(f"missing thumbnail file ({t_rel})")
                    break
            elif fname not in adds and fname not in changes:
                drift.append(f"missing thumbs map entry ({fname})")
                break

    if catalog_entry is not None and public_release is not None:
        if sorted(cat_files) != sorted(public_assets):
            drift.append("packs.json files list differs from public release assets")

    return PackPlan(
        slug=slug,
        shelf_release=shelf_release,
        public_release=public_release,
        catalog_entry=catalog_entry,
        meta=meta,
        shelf_assets=shelf_assets,
        public_assets=public_assets,
        public_zip=public_zip,
        adds=adds,
        changes=changes,
        deletes=deletes,
        drift=drift,
    )


def _upload_release_asset(
    release: dict[str, Any],
    gh_token: str,
    file_path: Path,
    asset_name: str,
) -> None:
    build_pack_entry.validate_safe_filename(asset_name)
    upload_base = str(release["upload_url"]).split("{", 1)[0]
    vt_scan_gate.validate_https_url(upload_base, vt_scan_gate.ALLOWED_GH_HOSTS)
    mime_type = mimetypes.guess_type(asset_name)[0] or "application/octet-stream"
    url = f"{upload_base}?name={urllib.parse.quote(asset_name, safe='')}"
    vt_scan_gate.gh_api_request(
        url,
        gh_token,
        method="POST",
        raw_body=file_path.read_bytes(),
        content_type=mime_type,
    )


def _delete_release_asset(asset: AssetInfo, gh_token: str) -> None:
    vt_scan_gate.gh_api_request(asset.api_url, gh_token, method="DELETE")


def _scan_files_with_vt(
    files: list[Path],
    vt_key: str,
    cache_path: Path,
) -> tuple[list[tuple[str, int, vt_scan_gate.ScanVerdict]], list[tuple[str, vt_scan_gate.ScanVerdict]], int]:
    """Reuse vt_scan_gate.py's cache and VirusTotalClient directly (no duplicated VT logic)."""
    cache = vt_scan_gate.load_verdict_cache(cache_path)
    if cache:
        print(f"Loaded {len(cache)} valid cached SHA-256 verdicts from {cache_path}")
    client = vt_scan_gate.VirusTotalClient(vt_key)

    records: list[tuple[str, int, vt_scan_gate.ScanVerdict]] = []
    detections: list[tuple[str, vt_scan_gate.ScanVerdict]] = []
    run_seen: dict[str, vt_scan_gate.ScanVerdict] = {}

    for local_path in sorted(files, key=lambda p: p.name):
        size_bytes = local_path.stat().st_size
        digest = vt_scan_gate.sha256_file(local_path)
        if digest in run_seen:
            verdict = vt_scan_gate.ScanVerdict(
                **{**run_seen[digest].__dict__, "source": "in-run-cache"}
            )
        elif digest in cache:
            verdict = vt_scan_gate.ScanVerdict.from_cache_dict(cache[digest])
        else:
            print(f"  Scanning {local_path.name} ({size_bytes} bytes, sha256={digest[:12]}...)...")
            verdict = client.scan_file(local_path, digest)
            if verdict.verdict == "clean":
                cache[digest] = verdict.to_cache_dict()

        run_seen[digest] = verdict
        records.append((local_path.name, size_bytes, verdict))
        print(f"    -> {local_path.name}: {verdict.verdict} ({verdict.source}) - {verdict.reason}")
        if verdict.verdict != "clean":
            detections.append((local_path.name, verdict))

    vt_scan_gate.save_verdict_cache(cache_path, cache)
    return records, detections, client.api_requests_made


def _reverify_shelf_unchanged(
    repo: str,
    gh_token: str,
    plan: PackPlan,
) -> dict[str, Any]:
    shelf_id = int(plan.shelf_release["id"])
    url = f"https://api.github.com/repos/{repo}/releases/{shelf_id}"
    _, body, _ = vt_scan_gate.gh_api_request(url, gh_token)
    refreshed = json.loads(body.decode("utf-8"))
    if refreshed.get("draft") is not True:
        raise RuntimeError(f"Shelf release shelf-{plan.slug} is no longer a draft.")
    current_assets = _collect_shelf_assets(refreshed)
    if set(current_assets) != set(plan.shelf_assets):
        raise RuntimeError(f"TOCTOU check failed: shelf-{plan.slug} asset set changed during sync.")
    for fname, orig in plan.shelf_assets.items():
        cur = current_assets[fname]
        if orig.sha256 and cur.sha256 and orig.sha256 != cur.sha256:
            raise RuntimeError(
                f"TOCTOU check failed: shelf-{plan.slug} asset {fname!r} digest changed during sync."
            )
    return refreshed


def apply_pack_plan(
    repo: str,
    gh_token: str,
    vt_key: str,
    cache_path: Path,
    repo_root: Path,
    plan: PackPlan,
) -> tuple[bool, list[str]]:
    """Apply a single pack diff. Returns (ok, summary_lines)."""
    if not plan.shelf_assets:
        msg = f"Shelf `shelf-{plan.slug}` has 0 assets; refusing to publish empty pack."
        print(f"ERROR: {msg}")
        return False, [f"- **{plan.meta['name']} (`{plan.slug}`)**: FAILED — {msg}"]

    if len(plan.shelf_assets) > check_pack.MAX_PACK_FILES:
        msg = (
            f"Shelf `shelf-{plan.slug}` has {len(plan.shelf_assets)} files "
            f"(exceeds max {check_pack.MAX_PACK_FILES})."
        )
        print(f"ERROR: {msg}")
        return False, [f"- **{plan.meta['name']} (`{plan.slug}`)**: FAILED — {msg}"]

    release_tag = plan.meta["releaseTag"]
    zip_name = plan.meta["zipName"]
    pack_dir_name = plan.meta["packDirName"]
    kind_folder = plan.meta["folder"]

    to_scan_names = sorted(set(plan.adds) | set(plan.changes))
    scan_records: list[tuple[str, int, vt_scan_gate.ScanVerdict]] = []
    vt_calls = 0

    with tempfile.TemporaryDirectory(prefix=f"sync_{plan.slug}_") as tmp:
        tmp_dir = Path(tmp)
        scan_dir = tmp_dir / "scan"
        scan_dir.mkdir(parents=True, exist_ok=True)

        downloaded: dict[str, Path] = {}
        if to_scan_names:
            if not vt_key:
                msg = (
                    "VIRUS_TOTAL_TOKEN secret is not set. Add VIRUS_TOTAL_TOKEN in repository "
                    "Settings -> Secrets and variables -> Actions. No changes were published."
                )
                print(f"ERROR: {msg}")
                return False, [f"- **{plan.meta['name']} (`{plan.slug}`)**: BLOCKED — {msg}"]

            print(
                f"[{plan.slug}] Downloading {len(to_scan_names)} new/changed shelf file(s) for VirusTotal scan..."
            )
            scan_paths: list[Path] = []
            for fname in to_scan_names:
                info = plan.shelf_assets[fname]
                local_p = vt_scan_gate.download_release_asset(
                    {"name": info.name, "url": info.api_url},
                    gh_token,
                    scan_dir,
                )
                downloaded[fname] = local_p
                scan_paths.append(local_p)

            scan_records, detections, vt_calls = _scan_files_with_vt(scan_paths, vt_key, cache_path)
            if detections:
                lines = [
                    f"### BLOCKED: `{plan.meta['name']}` (`shelf-{plan.slug}`)",
                    "",
                    "Public release, zip, `packs.json`, `thumbs/`, and `board/` were **NOT** modified.",
                    "",
                    "| File | SHA-256 | Reason | Engines | Report |",
                    "|---|---|---|---|---|",
                ]
                print(f"\nDETECTION(S) FOUND in shelf-{plan.slug} — ABORTING:")
                for fname, v in detections:
                    eng = "; ".join(v.engines[:5]) if v.engines else v.reason
                    print(f"  FAIL: {fname} sha256={v.sha256} reason={v.reason} engines={eng}")
                    lines.append(f"| `{fname}` | `{v.sha256}` | {v.reason} | {eng} | [VT]({v.vt_url}) |")
                return False, lines
        else:
            print(
                f"[{plan.slug}] Pure deletion / metadata sync (0 new/changed files) — skipping VirusTotal scan."
            )

        # Assemble pack folder from full shelf asset set
        pack_dir = tmp_dir / "pack" / pack_dir_name
        asset_folder = pack_dir / kind_folder
        asset_folder.mkdir(parents=True, exist_ok=True)

        print(f"[{plan.slug}] Assembling {len(plan.shelf_assets)} shelf asset(s) into {pack_dir_name}/{kind_folder}/...")
        for fname in sorted(plan.shelf_assets):
            info = plan.shelf_assets[fname]
            dest = asset_folder / fname
            if fname in downloaded:
                shutil.copy2(downloaded[fname], dest)
            else:
                vt_scan_gate.download_release_asset(
                    {"name": info.name, "url": info.api_url},
                    gh_token,
                    asset_folder,
                )
            if info.sha256:
                actual = vt_scan_gate.sha256_file(dest)
                if actual != info.sha256:
                    raise RuntimeError(
                        f"Downloaded shelf asset {fname!r} sha256 mismatch ({actual[:12]} != {info.sha256[:12]})"
                    )

        pack_json_data = {
            "name": plan.meta["name"],
            "description": plan.meta["description"],
            "tags": plan.meta["tags"],
            "sets": len(plan.shelf_assets),
        }
        (pack_dir / "pack.json").write_text(
            json.dumps(pack_json_data, indent=2) + "\n", encoding="utf-8"
        )

        # Validate pack folder via tools/check_pack.py
        all_passed, rule_results, _ = check_pack.validate_pack(pack_dir)
        check_pack.print_report(pack_dir, all_passed, rule_results)
        if not all_passed:
            failed_rules = "; ".join(f"{r.name}: {r.reason}" for r in rule_results if not r.passed)
            msg = f"check_pack failed: {failed_rules}"
            print(f"ERROR: {msg}")
            return False, [f"- **{plan.meta['name']} (`{plan.slug}`)**: BLOCKED — {msg}"]

        # Rebuild pack zip via tools/make_pack.py
        zip_path = tmp_dir / zip_name
        make_pack.build_pack_zip(pack_dir, zip_path)
        print(
            f"[{plan.slug}] Rebuilt zip {zip_name} ({build_pack_entry.format_size(zip_path.stat().st_size)})"
        )

        # Re-verify shelf hasn't changed before mutating public release / catalog (TOCTOU guard)
        _reverify_shelf_unchanged(repo, gh_token, plan)

        # Create or update public release
        pub_rel = plan.public_release
        if pub_rel is None:
            notes = make_pack.build_release_notes(plan.meta["description"])
            _, body, _ = vt_scan_gate.gh_api_request(
                f"https://api.github.com/repos/{repo}/releases",
                gh_token,
                method="POST",
                json_body={
                    "tag_name": release_tag,
                    "name": plan.meta["name"],
                    "body": notes,
                    "draft": False,
                    "prerelease": False,
                },
            )
            pub_rel = json.loads(body.decode("utf-8"))
            print(f"[{plan.slug}] Created public release {release_tag}")

        # Delete removed, replaced, old zip, and old PACK-REPORT.txt (when refreshing report)
        to_delete_names = set(plan.deletes) | set(plan.changes)
        for fname in sorted(to_delete_names):
            existing = plan.public_assets.get(fname)
            if existing is not None:
                _delete_release_asset(existing, gh_token)
                print(f"[{plan.slug}] Deleted public asset: {fname}")

        if plan.public_zip is not None:
            _delete_release_asset(plan.public_zip, gh_token)
            print(f"[{plan.slug}] Replaced old zip: {zip_name}")

        if scan_records:
            for raw in pub_rel.get("assets", []):
                if raw.get("name") == "PACK-REPORT.txt":
                    vt_scan_gate.gh_api_request(str(raw["url"]), gh_token, method="DELETE")

        # Upload missing/new/changed loose assets
        to_upload_names = sorted(
            fname
            for fname in plan.shelf_assets
            if fname in set(plan.adds) | set(plan.changes) or fname not in plan.public_assets
        )
        for fname in to_upload_names:
            _upload_release_asset(pub_rel, gh_token, asset_folder / fname, fname)
            print(f"[{plan.slug}] Uploaded public asset: {fname}")

        # Upload rebuilt pack zip
        _upload_release_asset(pub_rel, gh_token, zip_path, zip_name)
        print(f"[{plan.slug}] Uploaded rebuilt zip: {zip_name}")

        # Attach PACK-REPORT.txt and scan note when scans ran
        if scan_records:
            scanned_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            report_text = vt_scan_gate.format_scan_report(
                release_tag, plan.meta["name"], "virustotal-live", scanned_at, scan_records
            )
            report_path = tmp_dir / "PACK-REPORT.txt"
            report_path.write_text(report_text, encoding="utf-8")
            _upload_release_asset(pub_rel, gh_token, report_path, "PACK-REPORT.txt")

            notes_block = vt_scan_gate.format_release_notes_block(
                release_tag, "virustotal-live", scanned_at, scan_records
            )
            existing_body = (pub_rel.get("body") or "").strip()
            marker = "### VirusTotal Scan Gate"
            if marker in existing_body:
                existing_body = existing_body.split(marker, 1)[0].strip()
            new_body = f"{existing_body}\n\n{notes_block}".strip() if existing_body else notes_block
            vt_scan_gate.gh_api_request(
                f"https://api.github.com/repos/{repo}/releases/{pub_rel['id']}",
                gh_token,
                method="PATCH",
                json_body={"body": new_body},
            )

        # Regenerate packs.json entry via tools/build_pack_entry.py
        catalog_path = repo_root / "packs.json"
        entry = build_pack_entry.build_entry(pack_dir, release_tag, zip_path, repo=repo)
        build_pack_entry.update_catalog(catalog_path, entry)

        # Clean stale thumbnails in thumbs/<releaseTag>/ before regenerating via tools/build_thumbs.py
        thumbs_root = (repo_root / "thumbs").resolve()
        pack_thumbs_dir = (thumbs_root / release_tag).resolve()
        if pack_thumbs_dir.is_symlink():
            raise ValueError(f"Symlink pack thumbs directory is not allowed: {pack_thumbs_dir}")
        if pack_thumbs_dir.exists():
            shutil.rmtree(pack_thumbs_dir)

        thumbs_map = build_thumbs.build_thumbs(pack_dir, release_tag, thumbs_dir=thumbs_root)
        build_thumbs.update_catalog(catalog_path, release_tag, thumbs_map)

    summary_line = (
        f"- **{plan.meta['name']} (`{plan.slug}`)**: Synced (`+{len(plan.adds)} ~{len(plan.changes)} -{len(plan.deletes)}`; "
        f"{len(plan.shelf_assets)} files; VT scans={len(scan_records)}, VT API calls={vt_calls})"
    )
    return True, [summary_line]


def _format_plan_markdown(plans: list[PackPlan], mode: str) -> str:
    lines = [
        f"## Library Sync ({mode.upper()})",
        "",
        "| Pack | Slug | Shelf | Public Tag | Adds | Changes | Deletes | Drift | Status |",
        "|---|---|---|---|---:|---:|---:|---:|---|",
    ]
    for p in plans:
        status = "DIFF DETECTED" if p.has_changes else "IN SYNC (0 diff)"
        lines.append(
            f"| {p.meta['name']} | `{p.slug}` | `shelf-{p.slug}` | `{p.meta['releaseTag']}` | "
            f"{len(p.adds)} | {len(p.changes)} | {len(p.deletes)} | {len(p.drift)} | {status} |"
        )
    lines.append("")

    for p in plans:
        if not p.has_changes:
            continue
        lines.append(f"### Diff Details: {p.meta['name']} (`shelf-{p.slug}`)")
        if p.adds:
            lines.append("- **Added:** " + ", ".join(f"`{f}`" for f in p.adds))
        if p.changes:
            lines.append("- **Changed:** " + ", ".join(f"`{f}`" for f in p.changes))
        if p.deletes:
            lines.append("- **Deleted:** " + ", ".join(f"`{f}`" for f in p.deletes))
        if p.drift:
            lines.append("- **Drift:** " + "; ".join(p.drift))
        lines.append("")

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Diff and sync shelf drafts with public library state.")
    parser.add_argument("--mode", default=os.environ.get("INPUT_MODE", "report"))
    parser.add_argument("--pack", default=os.environ.get("INPUT_PACK", "all"))
    parser.add_argument("--repo", default=os.environ.get("GITHUB_REPOSITORY", DEFAULT_REPO))
    args = parser.parse_args(argv)

    mode = args.mode.strip().lower()
    if mode not in ("report", "apply"):
        print(f"ERROR: Invalid mode {args.mode!r}; expected 'report' or 'apply'.", file=sys.stderr)
        return 1

    pack_selector = validate_pack_selector(args.pack)
    repo = build_pack_entry.validate_repo(args.repo.strip())
    gh_token = os.environ.get("GH_TOKEN", "").strip()
    vt_key = os.environ.get("VT_API_KEY", "").strip()
    cache_path = Path(os.environ.get("VT_CACHE_FILE", ".vt-cache/verdicts.json"))

    if not gh_token:
        print("ERROR: GH_TOKEN is not set.", file=sys.stderr)
        return 1

    repo_root = Path.cwd()
    catalog_path = repo_root / "packs.json"
    catalog_data = json.loads(catalog_path.read_text(encoding="utf-8"))
    catalog_packs = list(catalog_data.get("packs", []))

    all_releases = list_all_releases(repo, gh_token)
    shelves_by_slug: dict[str, dict[str, Any]] = {}
    releases_by_tag: dict[str, dict[str, Any]] = {}

    for rel in all_releases:
        tag = str(rel.get("tag_name", ""))
        if not tag:
            continue
        is_versioned_public = bool(re.search(r"-v\d+(?:\.\d+)*$", tag))
        if tag.startswith(SHELF_PREFIX) and (rel.get("draft") is True or not is_versioned_public):
            if rel.get("draft") is not True:
                raise RuntimeError(
                    f"Shelf release {tag!r} is published (draft=false); shelves must remain drafts."
                )
            slug = tag[len(SHELF_PREFIX) :]
            validate_pack_selector(slug)
            if slug == "all":
                raise RuntimeError("Reserved pack slug 'all' is not allowed as a shelf name.")
            if slug in shelves_by_slug:
                raise RuntimeError(f"Duplicate shelf release found for slug {slug!r}")
            shelves_by_slug[slug] = rel
        else:
            releases_by_tag[tag] = rel

    if pack_selector == "all":
        target_slugs = sorted(shelves_by_slug)
    else:
        if pack_selector not in shelves_by_slug:
            msg = f"No draft shelf release found with tag `shelf-{pack_selector}`."
            print(f"ERROR: {msg}", file=sys.stderr)
            vt_scan_gate.append_summary(f"## Library Sync Error\n\n{msg}")
            vt_scan_gate.set_output("committed_changes", "false")
            return 1
        target_slugs = [pack_selector]

    if not target_slugs:
        msg = "No `shelf-*` draft releases found."
        print(msg)
        vt_scan_gate.append_summary(f"## Library Sync ({mode.upper()})\n\n{msg}")
        vt_scan_gate.set_output("has_changes", "false")
        vt_scan_gate.set_output("committed_changes", "false")
        return 0

    plans = [
        compute_pack_plan(slug, shelves_by_slug[slug], releases_by_tag, catalog_packs, repo_root)
        for slug in target_slugs
    ]

    report_md = _format_plan_markdown(plans, mode)
    print(report_md)
    vt_scan_gate.append_summary(report_md)

    any_changes = any(p.has_changes for p in plans)
    vt_scan_gate.set_output("has_changes", "true" if any_changes else "false")

    if mode == "report":
        vt_scan_gate.set_output("committed_changes", "false")
        return 0

    # Apply mode
    applied_notes: list[str] = []
    any_applied = False
    for plan in plans:
        if not plan.has_changes:
            applied_notes.append(f"- **{plan.meta['name']} (`{plan.slug}`)**: In sync (0 changes) — skipped")
            continue
        ok, notes = apply_pack_plan(repo, gh_token, vt_key, cache_path, repo_root, plan)
        applied_notes.extend(notes)
        if not ok:
            vt_scan_gate.append_summary("## Apply Result: FAILED\n\n" + "\n".join(applied_notes))
            vt_scan_gate.set_output("committed_changes", "false")
            return 1
        any_applied = True

    if any_applied:
        shelf_urls = {
            slug: str(rel.get("html_url", ""))
            for slug, rel in shelves_by_slug.items()
            if rel.get("html_url")
        }
        written_board = build_board.build_board(
            catalog_path=catalog_path,
            board_dir=repo_root / "board",
            thumbs_dir=repo_root / "thumbs",
            repo=repo,
            shelf_urls=shelf_urls,
        )
        applied_notes.append(f"- **Board**: Regenerated {len(written_board)} file(s) in `board/`")

    vt_scan_gate.append_summary("## Apply Result: SUCCEEDED\n\n" + "\n".join(applied_notes))
    vt_scan_gate.set_output("committed_changes", "true" if any_applied else "false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
