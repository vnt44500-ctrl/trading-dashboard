"""Small durable store for market-scan results and data-quality metadata."""
import json
import os
from datetime import datetime
from pathlib import Path

import pandas as pd

STORE_PATH = Path(__file__).resolve().parent.parent / "market_scan_history.json"

# Bump when the scanner's row schema changes; older persisted scans are then
# ignored instead of rendering tables with missing columns.
SCHEMA_VERSION = 2


def save_scan(results: dict, source: str = "Yahoo Finance") -> dict:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "scanned_at": datetime.now().isoformat(timespec="seconds"),
        "source": source,
        "groups": {},
    }
    for name, table in results.items():
        records = table.to_dict(orient="records") if isinstance(table, pd.DataFrame) else []
        payload["groups"][name] = {"rows": records, "count": len(records)}
    temporary = STORE_PATH.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, default=str, indent=2), encoding="utf-8")
    os.replace(temporary, STORE_PATH)
    return payload


def load_scan() -> tuple[dict, str] | None:
    if not STORE_PATH.exists():
        return None
    try:
        payload = json.loads(STORE_PATH.read_text(encoding="utf-8"))
        if payload.get("schema_version") != SCHEMA_VERSION:
            # Legacy layout (or older schema): stale columns would render
            # misleading tables, so treat it as no scan at all.
            return None
        results = {name: pd.DataFrame(group.get("rows", [])) for name, group in payload.get("groups", {}).items()}
        return results, payload.get("scanned_at", "unknown")
    except (OSError, ValueError, TypeError):
        return None
