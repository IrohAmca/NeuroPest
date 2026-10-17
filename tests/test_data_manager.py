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


def _cache(path, meta: dict):
    import json

    import numpy as np

    np.savez(path, indptr=np.zeros(1, np.int64), meta_json=np.array(json.dumps(meta)))


def test_is_connectome_ready(tmp_path):
    cache = tmp_path / "cache.npz"
    assert not data_manager.is_connectome_ready(cache)
    _cache(cache, {"kind": "flywire", "version": data_manager.CACHE_VERSION})
    assert data_manager.is_connectome_ready(cache) and not data_manager.is_connectome_outdated(cache)


def test_a_cache_of_an_older_version_is_rebuilt_not_used_as_ready(tmp_path):
    """Its groups are inside the file: one built before the taste pathway has no SUGAR / MN9."""
    old, junk = tmp_path / "old.npz", tmp_path / "junk.npz"
    _cache(old, {"kind": "flywire"})                     # built before versions existed
    junk.write_bytes(b"dummy_cache")
    assert not data_manager.is_connectome_ready(old) and data_manager.is_connectome_outdated(old)
    assert not data_manager.is_connectome_ready(junk)
    assert data_manager.cache_version(old) == 0 and data_manager.cache_version(tmp_path / "none.npz") == 0


def test_prompt_startup_data_in_test_environment(monkeypatch):
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "1")
    # In automated test mode, it should not open GUI and return bool based on ready status
    res = data_manager.prompt_startup_data()
    assert isinstance(res, bool)
