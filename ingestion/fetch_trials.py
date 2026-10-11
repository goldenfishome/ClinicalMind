"""
Fetch clinical trial data from ClinicalTrials.gov v2 API.

Saves one JSON file per page to data/raw/ and returns the local paths.
Re-running is safe — existing files are skipped (idempotent).

Usage:
    python3 -m ingestion.fetch_trials
"""

import json
import time
from pathlib import Path

import httpx
import structlog

from agent.logging_config import setup_logging

setup_logging()
log = structlog.get_logger()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BASE_URL = "https://clinicaltrials.gov/api/v2/studies"
PAGE_SIZE = 100
MAX_RETRIES = 3

# Phase 1 scope: 3 conditions, active/completed trials only
FETCH_CONDITIONS = ["diabetes", "cardiac arrest", "hypertension"]
FETCH_STATUSES = ["COMPLETED", "RECRUITING", "ACTIVE_NOT_RECRUITING"]

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"


# ---------------------------------------------------------------------------
# Core fetch logic
# ---------------------------------------------------------------------------

def _fetch_page(client: httpx.Client, condition: str, page_token: str | None) -> dict:
    """Fetch one page with exponential backoff on failure."""
    params = {
        "query.cond": condition,
        "filter.overallStatus": ",".join(FETCH_STATUSES),
        "pageSize": PAGE_SIZE,
        "format": "json",
    }
    if page_token:
        params["pageToken"] = page_token

    last_exc: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.get(BASE_URL, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except (httpx.HTTPError, httpx.TimeoutException) as exc:
            last_exc = exc
            wait = 2 ** attempt
            log.warning("fetch_retry", attempt=attempt, wait_seconds=wait, error=str(exc))
            time.sleep(wait)

    raise RuntimeError(f"Failed after {MAX_RETRIES} retries") from last_exc


def fetch_condition(condition: str) -> list[Path]:
    """
    Fetch all pages for one condition and save each page to data/raw/.
    Returns list of saved file paths (skips pages already on disk).
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    page_token: str | None = None
    page_num = 1

    with httpx.Client() as client:
        while True:
            out_path = RAW_DIR / f"{condition.replace(' ', '_')}__page{page_num:04d}.json"

            if out_path.exists():
                log.info("skip_existing", file=out_path.name)
                # Read token from disk so pagination can continue
                data = json.loads(out_path.read_text())
                page_token = data.get("nextPageToken")
                saved.append(out_path)
            else:
                log.info("fetching", condition=condition, page=page_num)
                data = _fetch_page(client, condition, page_token)
                out_path.write_text(json.dumps(data, indent=2))
                log.info("saved", file=out_path.name, studies=len(data.get("studies", [])))
                saved.append(out_path)
                page_token = data.get("nextPageToken")

            if not page_token:
                break
            page_num += 1

    log.info("condition_done", condition=condition, pages=page_num)
    return saved


def fetch_all() -> dict[str, list[Path]]:
    """Fetch all configured conditions. Returns {condition: [paths]}."""
    results = {}
    for condition in FETCH_CONDITIONS:
        results[condition] = fetch_condition(condition)
    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    all_paths = fetch_all()
    total_pages = sum(len(v) for v in all_paths.values())
    log.info("fetch_complete", total_pages=total_pages)
