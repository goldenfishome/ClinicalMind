"""
Parse raw JSON files from data/raw/ into clean, flat trial records.

Each raw file contains one page (up to 100 trials) from the ClinicalTrials.gov
v2 API. This module extracts only the fields needed downstream for chunking,
embedding, and SQL lookup.

Usage:
    python3 -m ingestion.parse_trials
"""

import json
from pathlib import Path
from typing import Any

import structlog

from agent.logging_config import setup_logging

setup_logging()
log = structlog.get_logger()

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"


# ---------------------------------------------------------------------------
# Field extractors — each handles missing data with a safe default
# ---------------------------------------------------------------------------

def _get(d: dict, *keys: str, default: Any = None) -> Any:
    """Safely traverse nested dict keys; return default if any key is missing."""
    for key in keys:
        if not isinstance(d, dict):
            return default
        d = d.get(key, default)
        if d is None:
            return default
    return d


def _parse_trial(study: dict) -> dict | None:
    """
    Extract fields from one raw study object.
    Returns None if the NCT ID is missing (record is unusable).
    """
    proto = study.get("protocolSection", {})

    nct_id = _get(proto, "identificationModule", "nctId")
    if not nct_id:
        return None

    # Phase: API returns a list e.g. ["PHASE3"] — join for storage
    phases = _get(proto, "designModule", "phases", default=[])
    phase = ", ".join(phases) if phases else None

    # Conditions: list of condition strings
    conditions = _get(proto, "conditionsModule", "conditions", default=[])
    condition = ", ".join(conditions) if conditions else None

    return {
        "nct_id":               nct_id,
        "title":                _get(proto, "identificationModule", "briefTitle", default=""),
        "phase":                phase,
        "condition":            condition,
        "status":               _get(proto, "statusModule", "overallStatus"),
        "sponsor":              _get(proto, "sponsorCollaboratorsModule", "leadSponsor", "name"),
        "start_date":           _get(proto, "statusModule", "startDateStruct", "date"),
        "completion_date":      _get(proto, "statusModule", "completionDateStruct", "date"),
        # Free-text fields — handed to chunk_text.py as-is
        "brief_summary":        _get(proto, "descriptionModule", "briefSummary", default=""),
        "detailed_description": _get(proto, "descriptionModule", "detailedDescription", default=""),
        "eligibility_criteria": _get(proto, "eligibilityModule", "eligibilityCriteria", default=""),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_file(path: Path) -> list[dict]:
    """Parse one raw JSON page file. Returns list of clean trial dicts."""
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        log.error("parse_file_error", file=path.name, error=str(exc))
        return []

    studies = data.get("studies", [])
    records: list[dict] = []
    skipped = 0

    for study in studies:
        record = _parse_trial(study)
        if record is None:
            skipped += 1
        else:
            records.append(record)

    log.info("parsed_file", file=path.name, records=len(records), skipped=skipped)
    return records


def parse_all(raw_dir: Path = RAW_DIR) -> list[dict]:
    """
    Parse every JSON file in raw_dir.
    Deduplicates by NCT ID (last-write wins) so re-runs stay idempotent.
    Returns a list of unique trial dicts sorted by nct_id.
    """
    all_files = sorted(raw_dir.glob("*.json"))
    if not all_files:
        log.warning("no_raw_files", raw_dir=str(raw_dir))
        return []

    seen: dict[str, dict] = {}
    for path in all_files:
        for record in parse_file(path):
            seen[record["nct_id"]] = record

    trials = sorted(seen.values(), key=lambda r: r["nct_id"])
    log.info("parse_complete", total_trials=len(trials))
    return trials


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    trials = parse_all()
    if trials:
        sample = trials[0]
        print(f"\nSample record ({sample['nct_id']}):")
        for k, v in sample.items():
            preview = str(v)[:120].replace("\n", " ") if v else "—"
            print(f"  {k:<25} {preview}")
