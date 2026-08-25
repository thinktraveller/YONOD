from __future__ import annotations

from pathlib import Path
import sys


REPRO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPRO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from vjethbkm_repro.sources import probe_sources


def test_probe_sources_returns_configured_records_without_network_for_local_item() -> None:
    records = probe_sources(timeout_s=1)
    ids = {record["source_id"] for record in records}
    assert "vjethbkm-zaihub" in ids
    local_record = next(record for record in records if record["source_id"] == "vjethbkm-zaihub")
    assert local_record["ok"] is False
    assert "local Zotero item" in local_record["error"]

