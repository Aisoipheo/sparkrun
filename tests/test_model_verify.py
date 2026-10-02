"""Tests for model cache verification and the pre-flight distribution ladder."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

import sparkrun.models.distribute as distribute_mod
import sparkrun.models.download as download_mod
import sparkrun.models.verify as verify_mod
from sparkrun.models.verify import render_verify_script, verify_model_local
from sparkrun.orchestration.distribution import ModelDistributionPrefs, _distribute_single_model


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _make_cache(tmp_path: Path, corrupt: bool = False, weights: bool = True) -> Path:
    """Build a minimal HF-cache-shaped tree: snapshot entries are symlinks to
    content-addressed blobs (blobs/<sha256>), as huggingface_hub lays out."""
    hub = tmp_path / "hub" / "models--org--name"
    snap = hub / "snapshots" / "cafe1234"
    blobs = hub / "blobs"
    snap.mkdir(parents=True)
    blobs.mkdir()
    if weights:
        data = b"weights"
        blob = blobs / _sha(data)
        blob.write_bytes(b"corrupted" if corrupt else data)
        try:
            (snap / "model-00001-of-000002.safetensors").symlink_to(blob)
        except OSError:
            pytest.skip("symlinks unavailable on this filesystem")
    return tmp_path


# ---------------------------------------------------------------------------
# local verification
# ---------------------------------------------------------------------------


def test_verify_model_local_accepts_clean_cache(tmp_path):
    assert verify_model_local("org/name", cache_dir=str(_make_cache(tmp_path)), revision="cafe1234") == []


def test_verify_model_local_reports_corrupt_blob(tmp_path):
    bad = verify_model_local("org/name", cache_dir=str(_make_cache(tmp_path, corrupt=True)), revision="cafe1234")
    assert bad is not None and len(bad) == 1


def test_verify_model_local_missing_cache_is_none(tmp_path):
    assert verify_model_local("org/name", cache_dir=str(tmp_path), revision="cafe1234") is None


def test_verify_model_local_no_weights_is_none(tmp_path):
    cache = _make_cache(tmp_path, weights=False)
    assert verify_model_local("org/name", cache_dir=str(cache), revision="cafe1234") is None


def test_render_verify_script_has_no_stray_braces():
    """The script is consumed via str.format(); a stray brace must explode here, not on a node."""
    script = render_verify_script("org/name", "~/.cache/huggingface", "cafe1234")
    assert "{" not in script and "}" not in script
    assert "cafe1234" in script


# ---------------------------------------------------------------------------
# pre-flight ladder through _distribute_single_model
# ---------------------------------------------------------------------------


def _run_ladder(monkeypatch, tmp_path, verify_remote, verify_local=None, mode="local", targets=("h1", "h2"), download=None):
    """Drive _distribute_single_model with the verification layer mocked."""
    calls = {}

    monkeypatch.setattr(verify_mod, "verify_model_on_hosts", verify_remote)
    if verify_local is not None:
        monkeypatch.setattr(verify_mod, "verify_model_local", lambda *a, **k: verify_local)

    def fake_from_local(model, hosts, **kwargs):
        calls["from_local"] = list(hosts)
        return []

    def fake_from_head(model, hosts, **kwargs):
        calls["from_head"] = list(hosts)
        return []

    def fail_download(*a, **k):
        raise AssertionError("unexpected download_model call")

    monkeypatch.setattr(download_mod, "download_model", download or fail_download)
    monkeypatch.setattr(distribute_mod, "distribute_model_from_local", fake_from_local)
    monkeypatch.setattr(distribute_mod, "distribute_model_from_head", fake_from_head)

    failures = _distribute_single_model(
        "org/name",
        list(targets),
        list(targets),
        str(tmp_path),
        str(tmp_path),
        mode,
        None,
        None,
        {},
        "cafe1234",
        None,
        False,  # dry_run
        False,  # auto_delegated
        ModelDistributionPrefs(),
    )
    return failures, calls


def test_all_hosts_verified_skips_distribution(monkeypatch, tmp_path):
    def fail(*a, **k):
        raise AssertionError("distribution should have been skipped")

    monkeypatch.setattr(distribute_mod, "distribute_model_from_local", fail)
    monkeypatch.setattr(distribute_mod, "distribute_model_from_head", fail)
    failures, calls = _run_ladder(monkeypatch, tmp_path, verify_remote=lambda *a, **k: [])
    assert failures == []
    assert calls == {}


def test_only_bad_hosts_receive_the_rsync(monkeypatch, tmp_path):
    failures, calls = _run_ladder(
        monkeypatch,
        tmp_path,
        verify_remote=lambda *a, **k: ["h2"],
        verify_local=[],
    )
    assert failures == []
    assert calls == {"from_local": ["h2"]}


def test_corrupt_control_copy_is_purged_and_re_downloaded(monkeypatch, tmp_path):
    corrupt = tmp_path / "blob"
    corrupt.write_bytes(b"bad")
    failures, calls = _run_ladder(
        monkeypatch,
        tmp_path,
        verify_remote=lambda *a, **k: ["h2"],
        verify_local=[corrupt],
        download=lambda *a, **k: 0,
    )
    assert failures == []
    assert not corrupt.exists()
    assert calls == {"from_local": ["h2"]}


def test_control_without_copy_falls_back_to_head_fanout(monkeypatch, tmp_path):
    failures, calls = _run_ladder(
        monkeypatch,
        tmp_path,
        verify_remote=lambda *a, **k: ["h2"],
        verify_local=None,
        mode="local",
    )
    assert failures == []
    assert calls == {"from_head": ["h1", "h2"]}


def test_delegated_repairs_bad_head_before_fanout(monkeypatch, tmp_path):
    repaired = []

    def fake_repair(*a, **k):
        repaired.append(a)
        return True

    monkeypatch.setattr(verify_mod, "repair_model_on_host", fake_repair)
    failures, calls = _run_ladder(
        monkeypatch,
        tmp_path,
        verify_remote=lambda *a, **k: ["h1", "h2"],
        mode="delegated",
    )
    assert failures == []
    assert len(repaired) == 1
    assert calls == {"from_head": ["h1", "h2"]}


def test_kill_switch_disables_the_preflight(monkeypatch, tmp_path):
    monkeypatch.setenv("SPARKRUN_NO_MODEL_VERIFY", "1")

    def fail(*a, **k):
        raise AssertionError("pre-flight should have been switched off")

    monkeypatch.setattr(verify_mod, "verify_model_on_hosts", fail)
    failures, calls = _run_ladder(monkeypatch, tmp_path, verify_remote=fail)
    assert failures == []
    assert calls == {"from_local": ["h1", "h2"]}
