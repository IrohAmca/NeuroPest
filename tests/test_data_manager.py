import os
from pathlib import Path
import pytest

from neuropest import data_manager


def test_is_raw_data_present_false(tmp_path):
    assert not data_manager.is_raw_data_present(raw_dir=tmp_path)


def test_is_raw_data_present_true(tmp_path):
    for name in data_manager.RAW_SOURCES:
        p = tmp_path / name
        p.write_bytes(b"dummy_data_content")
    assert data_manager.is_raw_data_present(raw_dir=tmp_path)


def test_is_connectome_ready(tmp_path):
    cache = tmp_path / "cache.npz"
    assert not data_manager.is_connectome_ready(cache)
    cache.write_bytes(b"dummy_cache")
    assert data_manager.is_connectome_ready(cache)


def test_prompt_startup_data_in_test_environment(monkeypatch):
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "1")
    # In automated test mode, it should not open GUI and return bool based on ready status
    res = data_manager.prompt_startup_data()
    assert isinstance(res, bool)
