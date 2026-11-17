"""Split an oversized pack folder into balanced sibling pack folders by whole families.

Usage:
    python tools/split_pack.py <pack_folder> [--max 200]

Rules:
  - Pattern families (files sharing a family stem, e.g. boucle-tweed-01.png and
    boucle-tweed-02.png -> 'boucle-tweed') are never split apart.
  - Each output sibling pack has at most --max catalog files (excluding
    optional <stem>.preview.<img> sibling previews, which move with their asset).
  - Bucket sizes are balanced as evenly as whole-family constraints allow.
  - If any single family alone exceeds --max, stops without moving any files.
  - Moves files only (no copies left behind) and writes a valid pack.json in
    each sibling folder. Prints before/after counts and the full split map.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

try:
    import build_pack_entry
    import build_thumbs
except ImportError:
    from tools import build_pack_entry, build_thumbs  # type: ignore[no-redef]

DEFAULT_MAX_FILES = 200


@dataclass(frozen=True)
class AssetItem:
    asset_path: Path
    preview_path: Path | None
    folder_name: str
    kind: str
    family_stem: str


@dataclass
class FamilyGroup:
    stem: str
    items: list[AssetItem]

    @property
    def size(self) -> int:
        return len(self.items)


def extract_family_stem(filename: str) -> str:
    stem = os.path.splitext(filename)[0].lower()
    stripped = re.sub(r"[-_]v?\d+$", "", stem)
    return stripped or stem


def _load_source_meta(pack_dir: Path, kind: str, families: list[FamilyGroup]) -> dict:
    pack_json_path = pack_dir / "pack.json"
    if pack_json_path.is_file():
        loaded = json.loads(pack_json_path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError(f"pack.json must be a JSON object: {pack_json_path}")
        return loaded

    display_name = re.sub(r"[_-]+", " ", pack_dir.name).strip() or pack_dir.name
    seen_tags: set[str] = set()
    tags: list[str] = []
    for fam in families[:10]:
        for word in re.split(r"[-_]+", fam.stem):
            if word and word not in seen_tags:
                seen_tags.add(word)
                tags.append(word)
    if kind not in seen_tags:
        tags.append(kind)
    return {
        "name": display_name,
        "description": f"Curated {kind} assets.",
        "tags": tags or [kind],
    }


def _pack_families_into_buckets(
    families: list[FamilyGroup],
    num_buckets: int,
    max_files: int,
) -> list[list[FamilyGroup]] | None:
    ordered = sorted(families, key=lambda f: (-f.size, f.stem))
    buckets: list[list[FamilyGroup]] = [[] for _ in range(num_buckets)]
    totals = [0] * num_buckets

    for fam in ordered:
        candidates = [idx for idx in range(num_buckets) if totals[idx] + fam.size <= max_files]
        if not candidates:
            return None
        best_idx = min(candidates, key=lambda idx: (totals[idx], idx))
        buckets[best_idx].append(fam)
        totals[best_idx] += fam.size

    target = sum(totals) / num_buckets

    def score(bucket_totals: list[int]) -> tuple[int, float]:
        return (
            max(bucket_totals) - min(bucket_totals),
            sum((t - target) ** 2 for t in bucket_totals),
        )

    improved = True
    while improved:
        improved = False
        cur_score = score(totals)
        best_action: tuple[str, int, int, int, int] | None = None
        best_score = cur_score

        # Try moving one family from a heavier bucket to a lighter bucket
        for i in range(num_buckets):
            for j in range(num_buckets):
                if i == j or totals[i] <= totals[j]:
                    continue
                for fi, fam in enumerate(buckets[i]):
                    if totals[j] + fam.size > max_files:
                        continue
                    new_totals = list(totals)
                    new_totals[i] -= fam.size
                    new_totals[j] += fam.size
                    cand_score = score(new_totals)
                    if cand_score < best_score:
                        best_score = cand_score
                        best_action = ("move", i, j, fi, -1)

        # Try swapping two families across buckets
        for i in range(num_buckets):
            for j in range(i + 1, num_buckets):
                for fi, fam_a in enumerate(buckets[i]):
                    for fj, fam_b in enumerate(buckets[j]):
                        if fam_a.size == fam_b.size:
                            continue
                        new_i = totals[i] - fam_a.size + fam_b.size
                        new_j = totals[j] - fam_b.size + fam_a.size
                        if new_i > max_files or new_j > max_files:
                            continue
                        new_totals = list(totals)
                        new_totals[i] = new_i
                        new_totals[j] = new_j
                        cand_score = score(new_totals)
                        if cand_score < best_score:
                            best_score = cand_score
                            best_action = ("swap", i, j, fi, fj)

        if best_action is not None:
            kind_op, i, j, fi, fj = best_action
            if kind_op == "move":
                fam = buckets[i].pop(fi)
                buckets[j].append(fam)
                totals[i] -= fam.size
                totals[j] += fam.size
            else:
                fam_a = buckets[i][fi]
                fam_b = buckets[j][fj]
                buckets[i][fi] = fam_b
                buckets[j][fj] = fam_a
                totals[i] += fam_b.size - fam_a.size
                totals[j] += fam_a.size - fam_b.size
            improved = True

    if any(not b for b in buckets):
        return None

    for b in buckets:
        b.sort(key=lambda f: f.stem)
    buckets.sort(key=lambda b: b[0].stem)
    return buckets


def plan_balanced_split(families: list[FamilyGroup], max_files: int) -> list[list[FamilyGroup]]:
    total_files = sum(f.size for f in families)
    min_buckets = max(2, math.ceil(total_files / max_files))
    for num_buckets in range(min_buckets, len(families) + 1):
        buckets = _pack_families_into_buckets(families, num_buckets, max_files)
        if buckets is not None:
            return buckets
    raise RuntimeError("Unable to pack families within the requested max file cap")


def _choose_sibling_names(
    pack_dir: Path,
    base_display_name: str,
    count: int,
) -> list[tuple[Path, str]]:
    parent = pack_dir.parent
    use_part = bool(re.search(r"_Vol\d+$", pack_dir.name, flags=re.IGNORECASE))
    targets: list[tuple[Path, str]] = []
    if not use_part:
        clean_display = re.sub(r"\s+Vol\.?\s*\d+$", "", base_display_name, flags=re.IGNORECASE).strip()
        vol_paths = [parent / f"{pack_dir.name}_Vol{i}" for i in range(1, count + 1)]
        if not any(p.exists() for p in vol_paths):
            for i, path in enumerate(vol_paths, start=1):
                targets.append((path, f"{clean_display} Vol. {i}"))
            return targets

    for i in range(1, count + 1):
        path = parent / f"{pack_dir.name}_Part{i}"
        targets.append((path, f"{base_display_name} Part {i}"))
    return targets


def _format_split_description(orig_desc: str, count: int, display_name: str) -> str:
    if re.match(r"^\d+\b", orig_desc.strip()):
        return re.sub(r"^\d+\b", str(count), orig_desc.strip(), count=1)
    return orig_desc.strip() or f"{count} curated assets ({display_name})."


def split_pack(pack_dir: Path, max_files: int = DEFAULT_MAX_FILES) -> int:
    pack_dir = pack_dir.resolve()
    if not pack_dir.is_dir():
        print(f"ERROR: Pack folder does not exist: {pack_dir}", file=sys.stderr)
        return 1

    folders = build_pack_entry.iter_pack_folders(pack_dir)
    if not folders:
        print(f"ERROR: No asset subfolders found inside {pack_dir}", file=sys.stderr)
        return 1

    kinds = {kind for _, kind in folders}
    if len(kinds) > 1:
        print(
            f"ERROR: Pack mixes multiple asset kinds ({', '.join(sorted(kinds))}); "
            "split_pack only operates on kind-pure packs.",
            file=sys.stderr,
        )
        return 1
    pack_kind = next(iter(kinds))

    items: list[AssetItem] = []
    seen_names: set[str] = set()
    for folder, kind in folders:
        for entry in sorted(folder.iterdir(), key=lambda p: p.name):
            if not entry.is_file() or entry.name.startswith("."):
                continue
            if build_pack_entry.is_preview_file(entry.name):
                continue
            if entry.name in seen_names:
                print(f"ERROR: Duplicate catalog filename across folders: {entry.name}", file=sys.stderr)
                return 1
            seen_names.add(entry.name)
            preview = build_thumbs.find_preview_image(entry)
            items.append(
                AssetItem(
                    asset_path=entry,
                    preview_path=preview,
                    folder_name=folder.name,
                    kind=kind,
                    family_stem=extract_family_stem(entry.name),
                )
            )

    total_files = len(items)
    if total_files == 0:
        print(f"ERROR: Pack folder contains 0 catalog files: {pack_dir}", file=sys.stderr)
        return 1

    by_family: dict[str, list[AssetItem]] = {}
    for item in items:
        by_family.setdefault(item.family_stem, []).append(item)

    families = [
        FamilyGroup(stem=stem, items=sorted(fam_items, key=lambda x: x.asset_path.name))
        for stem, fam_items in sorted(by_family.items())
    ]

    if total_files <= max_files:
        print(
            f"No split needed: {pack_dir.name} has {total_files} catalog files "
            f"(<= --max {max_files}) across {len(families)} families."
        )
        return 0

    oversized = [fam for fam in families if fam.size > max_files]
    if oversized:
        print(
            f"ERROR: Cannot split {pack_dir.name} without breaking a whole pattern family "
            f"(cap --max {max_files}):",
            file=sys.stderr,
        )
        for fam in oversized:
            print(f"  - Family '{fam.stem}': {fam.size} files (> {max_files})", file=sys.stderr)
        print("Stopped without modifying or moving any files.", file=sys.stderr)
        return 1

    source_meta = _load_source_meta(pack_dir, pack_kind, families)
    orig_name = str(source_meta.get("name") or pack_dir.name).strip()
    orig_desc = str(source_meta.get("description") or "").strip()
    orig_tags = [str(t) for t in source_meta.get("tags", []) if str(t).strip()]
    if not orig_tags:
        orig_tags = [pack_kind]

    buckets = plan_balanced_split(families, max_files)
    targets = _choose_sibling_names(pack_dir, orig_name, len(buckets))

    existing_targets = [path for path, _ in targets if path.exists()]
    if existing_targets:
        print(
            "ERROR: Target sibling folder(s) already exist; refusing to overwrite: "
            + ", ".join(p.name for p in existing_targets),
            file=sys.stderr,
        )
        return 1

    print(f"Before: {pack_dir.name} -> {total_files} catalog files across {len(families)} families")

    moved_counts: list[tuple[Path, str, int, list[FamilyGroup]]] = []
    for (target_dir, display_name), bucket_families in zip(targets, buckets):
        target_dir.mkdir(parents=True, exist_ok=False)
        bucket_file_count = 0
        for fam in bucket_families:
            for item in fam.items:
                dest_folder = target_dir / item.folder_name
                dest_folder.mkdir(parents=True, exist_ok=True)
                shutil.move(str(item.asset_path), str(dest_folder / item.asset_path.name))
                if item.preview_path is not None and item.preview_path.exists():
                    shutil.move(str(item.preview_path), str(dest_folder / item.preview_path.name))
                bucket_file_count += 1

        new_meta = {
            "name": display_name,
            "description": _format_split_description(orig_desc, bucket_file_count, display_name),
            "tags": list(orig_tags),
            "sets": bucket_file_count,
        }
        (target_dir / "pack.json").write_text(
            json.dumps(new_meta, indent=2) + "\n",
            encoding="utf-8",
        )
        moved_counts.append((target_dir, display_name, bucket_file_count, bucket_families))

    # Remove emptied asset subfolders in source pack so moves leave 0 remaining catalog files
    remaining = 0
    for folder, _ in folders:
        leftover_files = [
            p for p in folder.iterdir()
            if p.is_file() and not p.name.startswith(".") and not build_pack_entry.is_preview_file(p.name)
        ]
        remaining += len(leftover_files)
        if not any(folder.iterdir()):
            folder.rmdir()

    after_total = sum(count for _, _, count, _ in moved_counts) + remaining
    print("After:")
    for target_dir, display_name, count, bucket_families in moved_counts:
        print(f"  - {target_dir.name} ({display_name}): {count} files, {len(bucket_families)} families")
    print(f"  - Remaining in {pack_dir.name}: {remaining} files")
    print(f"Total preserved: {total_files} -> {after_total}")
    print("Split map:")
    for target_dir, _, count, bucket_families in moved_counts:
        fam_summary = ", ".join(f"{fam.stem} ({fam.size})" for fam in bucket_families)
        print(f"  {target_dir.name} [{count}]: {fam_summary}")

    if after_total != total_files or remaining != 0:
        print("ERROR: Post-split count mismatch detected!", file=sys.stderr)
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Split an oversized pack folder into balanced sibling packs by whole pattern families."
    )
    parser.add_argument("pack_folder", type=Path, help="Path to the oversized pack folder")
    parser.add_argument(
        "--max",
        dest="max_files",
        type=int,
        default=DEFAULT_MAX_FILES,
        help=f"Maximum catalog files per output pack (default: {DEFAULT_MAX_FILES})",
    )
    args = parser.parse_args(argv)

    if args.max_files < 1:
        print("ERROR: --max must be at least 1", file=sys.stderr)
        return 2

    return split_pack(args.pack_folder, max_files=args.max_files)


if __name__ == "__main__":
    raise SystemExit(main())
