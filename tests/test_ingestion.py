"""Tests for the fetch_trials module."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ingestion.fetch_trials import _fetch_page, fetch_condition, RAW_DIR


# ---------------------------------------------------------------------------
# _fetch_page
# ---------------------------------------------------------------------------

def _mock_response(data: dict, status_code: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = data
    resp.raise_for_status = MagicMock()
    return resp


def test_fetch_page_returns_json():
    fake_data = {"studies": [{"protocolSection": {}}], "nextPageToken": None}
    with patch("ingestion.fetch_trials.httpx.Client") as mock_client_cls:
        mock_client = mock_client_cls.return_value.__enter__.return_value
        mock_client.get.return_value = _mock_response(fake_data)
        client = mock_client
        result = _fetch_page(client, "diabetes", None)
    assert result == fake_data


def test_fetch_page_retries_on_http_error():
    import httpx

    fake_data = {"studies": [], "nextPageToken": None}
    responses = [
        MagicMock(side_effect=httpx.TimeoutException("timeout")),
        _mock_response(fake_data),
    ]

    with patch("ingestion.fetch_trials.time.sleep"):  # don't actually sleep
        with patch("ingestion.fetch_trials.httpx.Client") as mock_client_cls:
            mock_client = mock_client_cls.return_value.__enter__.return_value
            mock_client.get.side_effect = [
                httpx.TimeoutException("timeout"),
                _mock_response(fake_data),
            ]
            result = _fetch_page(mock_client, "diabetes", None)

    assert result == fake_data


# ---------------------------------------------------------------------------
# fetch_condition
# ---------------------------------------------------------------------------

def test_fetch_condition_skips_existing(tmp_path, monkeypatch):
    """If a page file already exists, fetch_condition should skip the API call."""
    monkeypatch.setattr("ingestion.fetch_trials.RAW_DIR", tmp_path)

    existing_data = {"studies": [{}], "nextPageToken": None}
    existing_file = tmp_path / "diabetes__page0001.json"
    existing_file.write_text(json.dumps(existing_data))

    with patch("ingestion.fetch_trials.httpx.Client") as mock_client_cls:
        mock_client = mock_client_cls.return_value.__enter__.return_value
        saved = fetch_condition("diabetes")

    mock_client.get.assert_not_called()
    assert len(saved) == 1
    assert saved[0] == existing_file


def test_fetch_condition_saves_file(tmp_path, monkeypatch):
    """fetch_condition should save a JSON file for each page fetched."""
    monkeypatch.setattr("ingestion.fetch_trials.RAW_DIR", tmp_path)

    page_data = {"studies": [{"id": "NCT001"}], "nextPageToken": None}

    with patch("ingestion.fetch_trials._fetch_page", return_value=page_data):
        with patch("ingestion.fetch_trials.httpx.Client"):
            saved = fetch_condition("diabetes")

    assert len(saved) == 1
    assert saved[0].exists()
    assert json.loads(saved[0].read_text()) == page_data
