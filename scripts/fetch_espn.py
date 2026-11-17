"""Download ESPN game summaries for every game in data/game_ids.csv.

Each summary is cached as data/raw/<game_id>.json, so re-running only fetches
games that are missing. Requests are spaced out to be polite to the endpoint.

Usage:
    python scripts/fetch_espn.py
"""

from __future__ import annotations

import csv
import json
import ssl
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
IDS_FILE = ROOT / "data" / "game_ids.csv"
URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/wnba/summary?event={gid}"
HEADERS = {"User-Agent": "wnba-agentic-rag/1.0 (educational project)"}
DELAY_SECONDS = 0.5


def _ssl_context() -> ssl.SSLContext:
    # python.org builds on macOS ship without system CA certs; certifi fixes that.
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


SSL_CTX = _ssl_context()


def fetch(game_id: str) -> dict:
    req = urllib.request.Request(URL.format(gid=game_id), headers=HEADERS)
    with urllib.request.urlopen(req, timeout=30, context=SSL_CTX) as resp:
        return json.load(resp)


def main() -> int:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    with IDS_FILE.open() as f:
        ids = [row["game_id"] for row in csv.DictReader(f)]

    failures = []
    for i, gid in enumerate(ids, 1):
        out = RAW_DIR / f"{gid}.json"
        if out.exists():
            continue
        for attempt in range(3):
            try:
                data = fetch(gid)
                out.write_text(json.dumps(data))
                break
            except Exception as exc:  # network hiccup: back off and retry
                if attempt == 2:
                    failures.append((gid, repr(exc)))
                time.sleep(2 * (attempt + 1))
        if i % 25 == 0:
            print(f"{i}/{len(ids)} games checked", flush=True)
        time.sleep(DELAY_SECONDS)

    print(f"Done. {len(ids) - len(failures)} cached, {len(failures)} failed.")
    for gid, err in failures:
        print("FAILED", gid, err)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
