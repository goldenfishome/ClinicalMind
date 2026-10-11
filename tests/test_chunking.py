"""Tests for ingestion/parse_trials.py and ingestion/chunk_text.py."""

import json
from pathlib import Path

import pytest

from ingestion.parse_trials import _parse_trial, parse_file, parse_all
from ingestion.chunk_text import (
    _count_tokens,
    _split_by_section_headers,
    _split_by_paragraphs,
    chunk_trial,
    chunk_all,
    MIN_TOKENS,
    MAX_TOKENS,
)


# ===========================================================================
# Helpers
# ===========================================================================

def _make_study(
    nct_id="NCT00000001",
    title="Test Trial",
    phase=None,
    conditions=None,
    status="COMPLETED",
    sponsor="NIH",
    brief_summary="A brief summary.",
    detailed_description="A detailed description.",
    eligibility="Inclusion Criteria:\n- Age >= 18",
) -> dict:
    return {
        "protocolSection": {
            "identificationModule": {"nctId": nct_id, "briefTitle": title},
            "statusModule": {
                "overallStatus": status,
                "startDateStruct": {"date": "2020-01"},
                "completionDateStruct": {"date": "2022-12"},
            },
            "sponsorCollaboratorsModule": {"leadSponsor": {"name": sponsor}},
            "conditionsModule": {"conditions": conditions or ["Diabetes"]},
            "designModule": {"phases": phase or ["PHASE3"]},
            "descriptionModule": {
                "briefSummary": brief_summary,
                "detailedDescription": detailed_description,
            },
            "eligibilityModule": {"eligibilityCriteria": eligibility},
        }
    }


def _make_trial(**kwargs) -> dict:
    return _parse_trial(_make_study(**kwargs))


# ===========================================================================
# parse_trials tests
# ===========================================================================

def test_parse_trial_extracts_all_fields():
    record = _make_trial()
    assert record["nct_id"] == "NCT00000001"
    assert record["title"] == "Test Trial"
    assert record["phase"] == "PHASE3"
    assert record["condition"] == "Diabetes"
    assert record["status"] == "COMPLETED"
    assert record["sponsor"] == "NIH"
    assert "brief_summary" in record
    assert "eligibility_criteria" in record


def test_parse_trial_returns_none_when_no_nct_id():
    study = {"protocolSection": {"identificationModule": {}}}
    assert _parse_trial(study) is None


def test_parse_trial_handles_missing_optional_fields():
    study = {
        "protocolSection": {
            "identificationModule": {"nctId": "NCT99999999", "briefTitle": "Minimal"},
            "statusModule": {"overallStatus": "RECRUITING"},
        }
    }
    record = _parse_trial(study)
    assert record is not None
    assert record["phase"] is None
    assert record["condition"] is None
    assert record["sponsor"] is None
    assert record["brief_summary"] == ""
    assert record["eligibility_criteria"] == ""


def test_parse_trial_joins_multiple_phases():
    record = _make_trial(phase=["PHASE2", "PHASE3"])
    assert record["phase"] == "PHASE2, PHASE3"


def test_parse_trial_joins_multiple_conditions():
    record = _make_trial(conditions=["Diabetes", "Hypertension"])
    assert record["condition"] == "Diabetes, Hypertension"


def test_parse_file_returns_records(tmp_path):
    page = {"studies": [_make_study("NCT00000001"), _make_study("NCT00000002")]}
    f = tmp_path / "page0001.json"
    f.write_text(json.dumps(page))
    records = parse_file(f)
    assert len(records) == 2


def test_parse_file_handles_bad_json(tmp_path):
    f = tmp_path / "bad.json"
    f.write_text("NOT JSON {{{{")
    assert parse_file(f) == []


def test_parse_file_skips_studies_without_nct_id(tmp_path):
    page = {
        "studies": [
            {"protocolSection": {"identificationModule": {}}},
            _make_study("NCT00000003"),
        ]
    }
    f = tmp_path / "page.json"
    f.write_text(json.dumps(page))
    records = parse_file(f)
    assert len(records) == 1


def test_parse_all_deduplicates_by_nct_id(tmp_path):
    page1 = {"studies": [_make_study("NCT00000001", title="Version A")]}
    page2 = {"studies": [_make_study("NCT00000001", title="Version B")]}
    (tmp_path / "page0001.json").write_text(json.dumps(page1))
    (tmp_path / "page0002.json").write_text(json.dumps(page2))
    records = parse_all(tmp_path)
    assert len(records) == 1


def test_parse_all_returns_empty_for_no_files(tmp_path):
    assert parse_all(tmp_path) == []


def test_parse_all_aggregates_across_files(tmp_path):
    (tmp_path / "page0001.json").write_text(json.dumps({"studies": [_make_study("NCT00000001")]}))
    (tmp_path / "page0002.json").write_text(json.dumps({"studies": [_make_study("NCT00000002")]}))
    records = parse_all(tmp_path)
    assert len(records) == 2


# ===========================================================================
# chunk_text tests
# ===========================================================================

# ---------------------------------------------------------------------------
# _split_by_section_headers
# ---------------------------------------------------------------------------

def test_split_finds_inclusion_exclusion():
    text = (
        "Inclusion Criteria:\n- Age >= 18\n- Diagnosed with diabetes\n\n"
        "Exclusion Criteria:\n- Pregnant\n- Severe renal impairment"
    )
    sections = _split_by_section_headers(text)
    labels = [s[0].lower() for s in sections]
    assert any("inclusion" in l for l in labels)
    assert any("exclusion" in l for l in labels)


def test_split_returns_full_text_when_no_headers():
    text = "This is plain text with no headers."
    sections = _split_by_section_headers(text)
    assert len(sections) == 1
    assert sections[0][1] == text


def test_split_handles_key_prefix():
    text = "Key Inclusion Criteria:\n- Item A\n\nKey Exclusion Criteria:\n- Item B"
    sections = _split_by_section_headers(text)
    assert len(sections) == 2


# ---------------------------------------------------------------------------
# _split_by_paragraphs
# ---------------------------------------------------------------------------

def test_paragraphs_respects_max_tokens():
    # Build text with many short paragraphs that together exceed MAX_TOKENS
    para = "This is a sentence that has some words. " * 5
    text = "\n\n".join([para] * 20)
    chunks = _split_by_paragraphs(text, MIN_TOKENS, MAX_TOKENS)
    for chunk in chunks:
        assert _count_tokens(chunk) <= MAX_TOKENS


def test_paragraphs_drops_short_chunks():
    # A single very short paragraph should be dropped
    text = "Hi."
    chunks = _split_by_paragraphs(text, MIN_TOKENS, MAX_TOKENS)
    assert chunks == []


# ---------------------------------------------------------------------------
# chunk_trial
# ---------------------------------------------------------------------------

def test_chunk_trial_returns_list_of_dicts():
    trial = _make_trial(
        brief_summary="A study on diabetes treatment over 3 years.",
        eligibility="Inclusion Criteria:\n- Age 18+\n\nExclusion Criteria:\n- Pregnant",
        detailed_description="Participants were randomised into two arms.",
    )
    chunks = chunk_trial(trial)
    assert isinstance(chunks, list)
    assert all(isinstance(c, dict) for c in chunks)


def test_chunk_trial_metadata_propagated():
    trial = _make_trial(nct_id="NCT12345678", phase=["PHASE2"], conditions=["Hypertension"])
    chunks = chunk_trial(trial)
    for chunk in chunks:
        assert chunk["nct_id"] == "NCT12345678"
        assert chunk["phase"] == "PHASE2"
        assert chunk["condition"] == "Hypertension"
        assert "source_field" in chunk
        assert "chunk_text" in chunk


def test_chunk_trial_source_fields_correct():
    trial = _make_trial(
        brief_summary="Brief text here with enough words to pass the token minimum threshold.",
        eligibility="Inclusion Criteria:\n- " + "criterion item detail. " * 30,
        detailed_description="Detail text. " * 20,
    )
    chunks = chunk_trial(trial)
    fields = {c["source_field"] for c in chunks}
    assert "brief_summary" in fields
    assert "eligibility_criteria" in fields
    assert "detailed_description" in fields


def test_chunk_trial_no_chunk_exceeds_max_tokens():
    long_text = "This is a long sentence with several words. " * 200
    trial = _make_trial(
        brief_summary=long_text,
        eligibility="Inclusion Criteria:\n" + long_text,
        detailed_description=long_text,
    )
    for chunk in chunk_trial(trial):
        assert _count_tokens(chunk["chunk_text"]) <= MAX_TOKENS


def test_chunk_trial_empty_fields_produce_no_chunks():
    trial = _make_trial(brief_summary="", eligibility="", detailed_description="")
    assert chunk_trial(trial) == []


def test_chunk_trial_eligibility_sections_split():
    """Inclusion and Exclusion should become separate chunks."""
    inclusion = "Inclusion Criteria:\n- " + "criterion item. " * 15
    exclusion = "Exclusion Criteria:\n- " + "exclusion item. " * 15
    trial = _make_trial(eligibility=f"{inclusion}\n\n{exclusion}", brief_summary="", detailed_description="")
    chunks = chunk_trial(trial)
    sections = [c["section"].lower() for c in chunks]
    assert any("inclusion" in s for s in sections)
    assert any("exclusion" in s for s in sections)


# ---------------------------------------------------------------------------
# chunk_all
# ---------------------------------------------------------------------------

def test_chunk_all_aggregates():
    trials = [
        _make_trial(nct_id="NCT00000001", brief_summary="Summary one. " * 10),
        _make_trial(nct_id="NCT00000002", brief_summary="Summary two. " * 10),
    ]
    chunks = chunk_all(trials)
    nct_ids = {c["nct_id"] for c in chunks}
    assert "NCT00000001" in nct_ids
    assert "NCT00000002" in nct_ids
