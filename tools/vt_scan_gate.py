"""VirusTotal scan-gate runner for draft GitHub releases.

Implements the release security gate specified in issue #1:
  - Scans every release asset (pack zip and individual files) on a draft release.
  - Computes SHA-256 first, checks local SHA-256 verdict cache, then queries
    VirusTotal v3 by hash with public API pacing (4 req/min, >=15.5s spacing).
  - Uploads files unknown to VirusTotal (using /files for <=32 MB and
    /files/upload_url for >32 MB) and polls analyses until completion.
  - Blocks publishing on any detection: keeps the release as a draft, makes no
    catalog changes, and writes the exact failing file(s) and reasons to the
    workflow summary and logs.
  - On a full clean pass: updates packs.json and thumbs/ via
    tools/build_pack_entry.py and tools/build_thumbs.py, uploads PACK-REPORT.txt
    to the release, writes the scan record into the release notes, and publishes
    the release (draft=false).

Supports simulate mode (INPUT_SIMULATE=true) using .github/vt-sim-fixtures.json
so the gate mechanics can be verified end-to-end without VIRUSTOTAL_API_KEY.
"""

from __future__ import annotations

import hashlib
import json
import mimetypes
import os
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import uuid
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MAX_DIRECT_UPLOAD_BYTES = 32 * 1024 * 1024  # 32 MB VirusTotal direct upload limit
VT_REQUEST_INTERVAL_SEC = 15.5  # Public API: 4 requests/minute
VT_POLL_MAX_ATTEMPTS = 120  # ~31 minutes max per uploaded file
VT_API_BASE = "https://www.virustotal.com/api/v3"


@dataclass
class ScanVerdict:
    sha256: str
    verdict: str  # "clean" | "malicious"
    malicious: int = 0
    suspicious: int = 0
    engines: list[str] = field(default_factory=list)
    reason: str = ""
    source: str = ""  # "cache" | "vt-hash" | "vt-upload" | "sim-fixture" | "sim-upload"
    vt_url: str = ""
    scanned_at: str = ""

    def to_cache_dict(self) -> dict[str, Any]:
        return {
            "sha256": self.sha256,
            "verdict": self.verdict,
            "malicious": self.malicious,
            "suspicious": self.suspicious,
            "engines": self.engines,
            "reason": self.reason,
            "vt_url": self.vt_url,
            "scanned_at": self.scanned_at,
        }

    @classmethod
    def from_cache_dict(cls, data: dict[str, Any]) -> "ScanVerdict":
        return cls(
            sha256=data["sha256"],
            verdict=data["verdict"],
            malicious=int(data.get("malicious", 0)),
            suspicious=int(data.get("suspicious", 0)),
            engines=list(data.get("engines", [])),
            reason=str(data.get("reason", "")),
            source="cache",
            vt_url=str(data.get("vt_url", "")),
            scanned_at=str(data.get("scanned_at", "")),
        )


def set_output(name: str, value: str) -> None:
    out_path = os.environ.get("GITHUB_OUTPUT")
    if out_path:
        with Path(out_path).open("a", encoding="utf-8") as fh:
            fh.write(f"{name}={value}\n")


def append_summary(markdown: str) -> None:
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with Path(summary_path).open("a", encoding="utf-8") as fh:
            fh.write(markdown.rstrip() + "\n\n")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def gh_api_request(
    url: str,
    token: str,
    method: str = "GET",
    json_body: dict[str, Any] | None = None,
    raw_body: bytes | None = None,
    content_type: str | None = None,
    accept: str = "application/vnd.github+json",
) -> tuple[int, bytes, dict[str, str]]:
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": accept,
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "DEVIL-PAINTER-Scan-Gate/1.0",
    }
    data: bytes | None = None
    if json_body is not None:
        data = json.dumps(json_body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    elif raw_body is not None:
        data = raw_body
        headers["Content-Type"] = content_type or "application/octet-stream"

    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return resp.status, resp.read(), dict(resp.headers.items())
    except urllib.error.HTTPError as exc:
        body = exc.read()
        raise RuntimeError(f"GitHub API {method} {url} failed ({exc.code}): {body.decode('utf-8', 'replace')}") from exc


def find_draft_release(repo: str, token: str, target_tag: str, release_id: str) -> dict[str, Any]:
    if release_id:
        url = f"https://api.github.com/repos/{repo}/releases/{release_id}"
        _, body, _ = gh_api_request(url, token)
        return json.loads(body.decode("utf-8"))

    page = 1
    drafts: list[dict[str, Any]] = []
    while True:
        url = f"https://api.github.com/repos/{repo}/releases?per_page=100&page={page}"
        _, body, _ = gh_api_request(url, token)
        items = json.loads(body.decode("utf-8"))
        if not items:
            break
        for rel in items:
            if target_tag and rel.get("tag_name") == target_tag:
                return rel
            if rel.get("draft") is True:
                drafts.append(rel)
        page += 1

    if target_tag:
        raise RuntimeError(f"Release with tag {target_tag!r} was not found in {repo}.")
    if not drafts:
        raise RuntimeError(f"No draft releases found in {repo}.")
    drafts.sort(key=lambda r: r.get("created_at", ""), reverse=True)
    return drafts[0]


def download_release_asset(asset: dict[str, Any], token: str, dest_dir: Path) -> Path:
    dest_path = dest_dir / asset["name"]
    _, data, _ = gh_api_request(
        asset["url"],
        token,
        accept="application/octet-stream",
    )
    dest_path.write_bytes(data)
    return dest_path


class VirusTotalClient:
    def __init__(self, api_key: str, interval_sec: float = VT_REQUEST_INTERVAL_SEC) -> None:
        self.api_key = api_key
        self.interval_sec = interval_sec
        self._last_request_ts = 0.0
        self.api_requests_made = 0

    def _pace(self) -> None:
        elapsed = time.monotonic() - self._last_request_ts
        if self._last_request_ts > 0 and elapsed < self.interval_sec:
            wait_for = self.interval_sec - elapsed
            print(f"  [VT pace] sleeping {wait_for:.1f}s (4 req/min limit)...")
            time.sleep(wait_for)

    def _request(
        self,
        url: str,
        method: str = "GET",
        data: bytes | None = None,
        headers: dict[str, str] | None = None,
    ) -> tuple[int, dict[str, Any]]:
        for attempt in range(4):
            self._pace()
            req_headers = {
                "x-apikey": self.api_key,
                "Accept": "application/json",
                "User-Agent": "DEVIL-PAINTER-Scan-Gate/1.0",
            }
            if headers:
                req_headers.update(headers)
            req = urllib.request.Request(url, data=data, method=method, headers=req_headers)
            self._last_request_ts = time.monotonic()
            self.api_requests_made += 1
            try:
                with urllib.request.urlopen(req, timeout=180) as resp:
                    payload = resp.read().decode("utf-8")
                    return resp.status, json.loads(payload) if payload else {}
            except urllib.error.HTTPError as exc:
                raw = exc.read().decode("utf-8", "replace")
                if exc.code == 404:
                    return 404, {}
                if exc.code == 429 and attempt < 3:
                    print("  [VT rate limit 429] backing off for 65s...")
                    time.sleep(65.0)
                    continue
                raise RuntimeError(f"VirusTotal {method} {url} failed ({exc.code}): {raw}") from exc
        raise RuntimeError(f"VirusTotal {method} {url} failed after rate-limit retries.")

    @staticmethod
    def _encode_multipart(file_path: Path) -> tuple[bytes, str]:
        boundary = f"----DEVILPAINTER{uuid.uuid4().hex}"
        mime_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
        file_bytes = file_path.read_bytes()
        parts = [
            f"--{boundary}\r\n".encode("utf-8"),
            f'Content-Disposition: form-data; name="file"; filename="{file_path.name}"\r\n'.encode("utf-8"),
            f"Content-Type: {mime_type}\r\n\r\n".encode("utf-8"),
            file_bytes,
            f"\r\n--{boundary}--\r\n".encode("utf-8"),
        ]
        return b"".join(parts), f"multipart/form-data; boundary={boundary}"

    @staticmethod
    def _parse_results(
        sha256: str,
        stats: dict[str, Any],
        results: dict[str, Any],
        source: str,
    ) -> ScanVerdict:
        malicious = int(stats.get("malicious", 0))
        suspicious = int(stats.get("suspicious", 0))
        flagged: list[str] = []
        for engine_name, detail in sorted(results.items()):
            if not isinstance(detail, dict):
                continue
            category = detail.get("category", "")
            if category in ("malicious", "suspicious"):
                result_label = detail.get("result") or category
                flagged.append(f"{engine_name}: {result_label}")
        is_clean = malicious == 0 and suspicious == 0
        reason = "clean" if is_clean else f"{malicious} malicious, {suspicious} suspicious"
        return ScanVerdict(
            sha256=sha256,
            verdict="clean" if is_clean else "malicious",
            malicious=malicious,
            suspicious=suspicious,
            engines=flagged,
            reason=reason,
            source=source,
            vt_url=f"https://www.virustotal.com/gui/file/{sha256}",
            scanned_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        )

    def scan_file(self, file_path: Path, sha256: str) -> ScanVerdict:
        status, payload = self._request(f"{VT_API_BASE}/files/{sha256}")
        if status == 200:
            attrs = payload.get("data", {}).get("attributes", {})
            return self._parse_results(
                sha256,
                attrs.get("last_analysis_stats", {}),
                attrs.get("last_analysis_results", {}),
                source="vt-hash",
            )

        # Unknown hash (404): upload file
        size_bytes = file_path.stat().st_size
        if size_bytes > MAX_DIRECT_UPLOAD_BYTES:
            _, up_payload = self._request(f"{VT_API_BASE}/files/upload_url")
            upload_url = up_payload["data"]
        else:
            upload_url = f"{VT_API_BASE}/files"

        body, content_type = self._encode_multipart(file_path)
        _, upload_resp = self._request(
            upload_url,
            method="POST",
            data=body,
            headers={"Content-Type": content_type},
        )
        analysis_id = upload_resp["data"]["id"]

        for _ in range(VT_POLL_MAX_ATTEMPTS):
            _, analysis_payload = self._request(f"{VT_API_BASE}/analyses/{analysis_id}")
            attrs = analysis_payload.get("data", {}).get("attributes", {})
            if attrs.get("status") == "completed":
                return self._parse_results(
                    sha256,
                    attrs.get("stats", {}),
                    attrs.get("results", {}),
                    source="vt-upload",
                )
        raise TimeoutError(f"VirusTotal analysis {analysis_id} for {file_path.name} did not complete in time.")


class SimulatedVirusTotalClient:
    def __init__(self, fixtures_path: Path) -> None:
        raw = json.loads(fixtures_path.read_text(encoding="utf-8"))
        self.default_unknown = raw.get("default_unknown_verdict", "clean")
        self.hashes: dict[str, dict[str, Any]] = raw.get("hashes", {})
        self.api_requests_made = 0

    def scan_file(self, file_path: Path, sha256: str) -> ScanVerdict:
        self.api_requests_made += 1
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        fixture = self.hashes.get(sha256)
        if fixture is not None:
            verdict = str(fixture.get("verdict", "clean"))
            return ScanVerdict(
                sha256=sha256,
                verdict=verdict,
                malicious=int(fixture.get("malicious", 1 if verdict != "clean" else 0)),
                suspicious=int(fixture.get("suspicious", 0)),
                engines=list(fixture.get("engines", [])),
                reason=str(fixture.get("reason", verdict)),
                source="sim-fixture",
                vt_url=f"https://www.virustotal.com/gui/file/{sha256}",
                scanned_at=now,
            )

        size_bytes = file_path.stat().st_size
        upload_route = "/files/upload_url" if size_bytes > MAX_DIRECT_UPLOAD_BYTES else "/files"
        verdict = str(self.default_unknown)
        return ScanVerdict(
            sha256=sha256,
            verdict=verdict,
            malicious=0 if verdict == "clean" else 1,
            suspicious=0,
            engines=[] if verdict == "clean" else ["SimDefault: Unknown"],
            reason=f"simulated upload via {upload_route} -> {verdict}",
            source="sim-upload",
            vt_url=f"https://www.virustotal.com/gui/file/{sha256}",
            scanned_at=now,
        )


def locate_pack_dir(zip_path: Path, extract_dir: Path) -> Path:
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)
    if (extract_dir / "pack.json").is_file():
        return extract_dir
    candidates = sorted(p.parent for p in extract_dir.rglob("pack.json") if p.is_file())
    if not candidates:
        raise FileNotFoundError(f"No pack.json found inside zip archive {zip_path.name}")
    return candidates[0]


def format_scan_report(
    release_tag: str,
    pack_name: str,
    mode: str,
    scanned_at: str,
    records: list[tuple[str, int, ScanVerdict]],
) -> str:
    total = len(records)
    clean_count = sum(1 for _, _, v in records if v.verdict == "clean")
    lines = [
        f"VIRUSTOTAL SCAN RECORD - {pack_name} ({release_tag})",
        "=" * 64,
        f"Date scanned (UTC): {scanned_at}",
        f"Gate mode: {mode}",
        f"Files scanned: {total}",
        f"Files clean: {clean_count}",
        f"Detections: {total - clean_count}",
        "",
        "FILES",
        "-----",
    ]
    for fname, size_bytes, verdict in records:
        lines.append(
            f"- {fname} | {size_bytes} bytes | sha256={verdict.sha256} | "
            f"verdict={verdict.verdict} ({verdict.source}) | {verdict.vt_url}"
        )
    return "\n".join(lines) + "\n"


def format_release_notes_block(
    release_tag: str,
    mode: str,
    scanned_at: str,
    records: list[tuple[str, int, ScanVerdict]],
) -> str:
    total = len(records)
    clean_count = sum(1 for _, _, v in records if v.verdict == "clean")
    lines = [
        "### VirusTotal Scan Gate",
        f"- **Date scanned (UTC):** {scanned_at}",
        f"- **Mode:** `{mode}`",
        f"- **Files scanned:** {total}",
        f"- **Files clean:** {clean_count}/{total}",
        f"- **Release tag:** `{release_tag}`",
        "",
        "Full per-file SHA-256 scan report is attached as `PACK-REPORT.txt`.",
    ]
    return "\n".join(lines)


def main() -> int:
    repo = os.environ.get("GITHUB_REPOSITORY", "Artbyo3/DEVIL-PAINTER_Asset_Library")
    gh_token = os.environ.get("GH_TOKEN", "").strip()
    vt_key = os.environ.get("VT_API_KEY", "").strip()
    event_name = os.environ.get("EVENT_NAME", "workflow_dispatch")
    event_release_id = os.environ.get("EVENT_RELEASE_ID", "").strip()
    event_release_tag = os.environ.get("EVENT_RELEASE_TAG", "").strip()
    input_release_tag = os.environ.get("INPUT_RELEASE_TAG", "").strip()
    simulate = os.environ.get("INPUT_SIMULATE", "false").strip().lower() == "true"
    cache_path = Path(os.environ.get("VT_CACHE_FILE", ".vt-cache/verdicts.json"))
    fixtures_path = Path(os.environ.get("VT_FIXTURES_FILE", ".github/vt-sim-fixtures.json"))

    if not simulate and not vt_key:
        msg = (
            "VIRUSTOTAL_API_KEY secret is not set. Add VIRUSTOTAL_API_KEY in repository "
            "Settings -> Secrets and variables -> Actions. Release remains a draft; no changes were made."
        )
        print(f"ERROR: {msg}")
        append_summary(f"## Release Scan Gate Blocked\n\n**Error:** {msg}")
        set_output("clean", "false")
        return 1

    if not gh_token:
        print("ERROR: GH_TOKEN is not set.")
        set_output("clean", "false")
        return 1

    target_tag = input_release_tag or event_release_tag
    release = find_draft_release(repo, gh_token, target_tag, event_release_id if event_name == "release" else "")
    release_id = release["id"]
    release_tag = release["tag_name"]

    if release.get("draft") is not True:
        msg = f"Release '{release_tag}' is already published (draft=false). Gate only processes draft releases."
        print(f"ERROR: {msg}")
        append_summary(f"## Release Scan Gate Stopped\n\n{msg}")
        set_output("clean", "false")
        return 1

    assets = [a for a in release.get("assets", []) if a.get("name") != "PACK-REPORT.txt"]
    if not assets:
        if event_name == "release":
            msg = (
                f"Draft release `{release_tag}` has 0 assets yet (uploads may still be in progress). "
                "Run **Release Scan Gate** manually via `workflow_dispatch` once all assets are uploaded."
            )
            print(msg)
            append_summary(f"## Release Scan Gate Waiting for Assets\n\n{msg}")
            set_output("clean", "false")
            return 0
        print(f"ERROR: Draft release '{release_tag}' has no assets to scan.")
        append_summary(f"## Release Scan Gate Failed\n\nDraft release `{release_tag}` has no assets to scan.")
        set_output("clean", "false")
        return 1

    mode_label = "simulate" if simulate else "virustotal-live"
    print(f"Scanning draft release {release_tag!r} (id={release_id}, assets={len(assets)}, mode={mode_label})")

    cache: dict[str, dict[str, Any]] = {}
    if not simulate and cache_path.is_file():
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
        print(f"Loaded {len(cache)} cached SHA-256 verdicts from {cache_path}")

    client: VirusTotalClient | SimulatedVirusTotalClient = (
        SimulatedVirusTotalClient(fixtures_path) if simulate else VirusTotalClient(vt_key)
    )

    with tempfile.TemporaryDirectory(prefix="vt_gate_") as tmp:
        tmp_dir = Path(tmp)
        assets_dir = tmp_dir / "assets"
        assets_dir.mkdir(parents=True, exist_ok=True)

        records: list[tuple[str, int, ScanVerdict]] = []
        run_seen: dict[str, ScanVerdict] = {}
        detections: list[tuple[str, ScanVerdict]] = []

        for asset in sorted(assets, key=lambda a: a["name"]):
            local_path = download_release_asset(asset, gh_token, assets_dir)
            size_bytes = local_path.stat().st_size
            digest = sha256_file(local_path)

            if digest in run_seen:
                verdict = ScanVerdict(**{**run_seen[digest].__dict__, "source": "in-run-cache"})
            elif not simulate and digest in cache:
                verdict = ScanVerdict.from_cache_dict(cache[digest])
            else:
                print(f"  Scanning {local_path.name} ({size_bytes} bytes, sha256={digest[:12]}...)...")
                verdict = client.scan_file(local_path, digest)
                if not simulate:
                    cache[digest] = verdict.to_cache_dict()

            run_seen[digest] = verdict
            records.append((local_path.name, size_bytes, verdict))
            print(f"    -> {local_path.name}: {verdict.verdict} ({verdict.source}) - {verdict.reason}")

            if verdict.verdict != "clean":
                detections.append((local_path.name, verdict))

        if not simulate:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(cache, indent=2) + "\n", encoding="utf-8")

        if detections:
            lines = [
                f"## Release Scan Gate BLOCKED (`{release_tag}`)",
                "",
                f"- **Mode:** `{mode_label}`",
                f"- **Release status:** Remains **DRAFT** (not published, no catalog change)",
                f"- **Files scanned:** {len(records)}",
                f"- **Blocked files:** {len(detections)}",
                "",
                "| File | SHA-256 | Reason | Engines | Report |",
                "|---|---|---|---|---|",
            ]
            print("\nDETECTION(S) FOUND - RELEASE REMAINS DRAFT:")
            for fname, v in detections:
                engines_str = "; ".join(v.engines[:5]) if v.engines else v.reason
                print(f"  FAIL: {fname} sha256={v.sha256} reason={v.reason} engines={engines_str}")
                lines.append(f"| `{fname}` | `{v.sha256}` | {v.reason} | {engines_str} | [VT]({v.vt_url}) |")
            append_summary("\n".join(lines))
            set_output("clean", "false")
            return 1

        # All files clean -> regenerate catalog entry + thumbs before publishing
        zip_files = [p for p in sorted(assets_dir.iterdir()) if p.suffix.lower() == ".zip"]
        if len(zip_files) != 1:
            raise RuntimeError(f"Expected exactly 1 pack .zip asset in {release_tag}, found {[z.name for z in zip_files]}")
        zip_path = zip_files[0]

        extract_dir = tmp_dir / "extracted"
        pack_dir = locate_pack_dir(zip_path, extract_dir)
        pack_meta = json.loads((pack_dir / "pack.json").read_text(encoding="utf-8"))
        pack_name = str(pack_meta.get("name") or release.get("name") or release_tag)

        repo_root = Path.cwd()
        subprocess.run(
            [
                sys.executable,
                str(repo_root / "tools" / "build_pack_entry.py"),
                str(pack_dir),
                release_tag,
                str(zip_path),
                "--catalog",
                str(repo_root / "packs.json"),
                "--repo",
                repo,
            ],
            check=True,
        )
        subprocess.run(
            [
                sys.executable,
                str(repo_root / "tools" / "build_thumbs.py"),
                str(pack_dir),
                release_tag,
                "--thumbs-dir",
                str(repo_root / "thumbs"),
                "--catalog",
                str(repo_root / "packs.json"),
            ],
            check=True,
        )

        scanned_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        report_text = format_scan_report(release_tag, pack_name, mode_label, scanned_at, records)

        # Delete any prior PACK-REPORT.txt asset if present, then upload new one
        for existing_asset in release.get("assets", []):
            if existing_asset.get("name") == "PACK-REPORT.txt":
                gh_api_request(existing_asset["url"], gh_token, method="DELETE")

        upload_base = release["upload_url"].split("{", 1)[0]
        gh_api_request(
            f"{upload_base}?name=PACK-REPORT.txt",
            gh_token,
            method="POST",
            raw_body=report_text.encode("utf-8"),
            content_type="text/plain; charset=utf-8",
        )

        notes_block = format_release_notes_block(release_tag, mode_label, scanned_at, records)
        existing_body = (release.get("body") or "").strip()
        marker = "### VirusTotal Scan Gate"
        if marker in existing_body:
            existing_body = existing_body.split(marker, 1)[0].strip()
        new_body = f"{existing_body}\n\n{notes_block}".strip() if existing_body else notes_block

        gh_api_request(
            f"https://api.github.com/repos/{repo}/releases/{release_id}",
            gh_token,
            method="PATCH",
            json_body={
                "draft": False,
                "body": new_body,
            },
        )

        summary = [
            f"## Release Scan Gate PASSED (`{release_tag}`)",
            "",
            f"- **Pack:** {pack_name}",
            f"- **Mode:** `{mode_label}`",
            f"- **Date scanned (UTC):** {scanned_at}",
            f"- **Files scanned:** {len(records)} ({len(records)} clean, 0 detections)",
            f"- **VT API calls made this run:** {client.api_requests_made}",
            "- **Release status:** Published (`draft: false`) with `PACK-REPORT.txt` attached",
        ]
        append_summary("\n".join(summary))
        set_output("clean", "true")
        set_output("release_tag", release_tag)
        set_output("pack_name", pack_name)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
