"""Official source probing and checksum helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import json

from .manifest import load_yaml, sha256_file
from .paths import REPRO_ROOT


@dataclass
class SourceProbe:
    source_id: str
    kind: str
    url: str
    checked_at: str
    ok: bool
    status: int | None
    final_url: str | None
    content_type: str | None
    content_length: str | None
    local_path: str | None
    local_exists: bool
    local_sha256: str | None
    error: str | None = None


def _utc_timestamp() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _probe_url(url: str, timeout_s: int) -> tuple[bool, int | None, str | None, str | None, str | None, str | None]:
    request = Request(url, method="HEAD", headers={"User-Agent": "YONOD-VJETHBKM-repro/0.1"})
    try:
        with urlopen(request, timeout=timeout_s) as response:
            return (
                True,
                int(response.status),
                response.geturl(),
                response.headers.get("Content-Type"),
                response.headers.get("Content-Length"),
                None,
            )
    except HTTPError as exc:
        return False, int(exc.code), exc.geturl(), exc.headers.get("Content-Type"), exc.headers.get("Content-Length"), str(exc)
    except URLError as exc:
        return False, None, None, None, None, str(exc.reason)
    except Exception as exc:
        return False, None, None, None, None, f"{type(exc).__name__}: {exc}"


def _local_status(local_path: str | None) -> tuple[bool, str | None]:
    if not local_path:
        return False, None
    path = (REPRO_ROOT / local_path).resolve()
    if path.is_dir():
        return True, None
    if path.exists():
        return True, sha256_file(path)
    return False, None


def probe_sources(timeout_s: int = 20) -> list[dict]:
    manifest = load_yaml(REPRO_ROOT / "data" / "manifest" / "sources.yaml")
    records: list[dict] = []
    for source in manifest.get("sources", []):
        url = source.get("url")
        local_path = source.get("local_path")
        local_exists, local_sha256 = _local_status(local_path)
        if not url:
            records.append(
                asdict(
                    SourceProbe(
                        source_id=source["id"],
                        kind=source["kind"],
                        url="",
                        checked_at=_utc_timestamp(),
                        ok=False,
                        status=None,
                        final_url=None,
                        content_type=None,
                        content_length=None,
                        local_path=local_path,
                        local_exists=local_exists,
                        local_sha256=local_sha256,
                        error="No URL configured; likely a local Zotero item.",
                    )
                )
            )
            continue
        ok, status, final_url, content_type, content_length, error = _probe_url(url, timeout_s)
        records.append(
            asdict(
                SourceProbe(
                    source_id=source["id"],
                    kind=source["kind"],
                    url=url,
                    checked_at=_utc_timestamp(),
                    ok=ok,
                    status=status,
                    final_url=final_url,
                    content_type=content_type,
                    content_length=content_length,
                    local_path=local_path,
                    local_exists=local_exists,
                    local_sha256=local_sha256,
                    error=error,
                )
            )
        )
    return records


def write_probe_manifest(records: list[dict]) -> Path:
    output_path = REPRO_ROOT / "data" / "manifest" / "source_access_probe.json"
    output_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path
