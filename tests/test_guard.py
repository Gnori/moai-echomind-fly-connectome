"""가드: 메모리 문턱 · 한 번에 하나 · 죽은 잠금 정리."""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from moai_echomind_fly import guard, paths


@pytest.fixture
def repo(tmp_path, monkeypatch):
    (tmp_path / "config").mkdir()
    (tmp_path / "pyproject.toml").write_text("")
    (tmp_path / "config" / "guard.yaml").write_text(
        "version: t\nmin_free_percent: 40\nqwen_weights_gb: 28\nlock_file: runs/.heavy.lock\n")
    monkeypatch.setenv("FLY_ROOT", str(tmp_path))
    monkeypatch.setattr(guard, "memory_free_percent", lambda: 80)
    return tmp_path


def test_low_memory_refused(repo, monkeypatch):
    monkeypatch.setattr(guard, "memory_free_percent", lambda: 30)
    with pytest.raises(guard.GuardRefused) as e:
        with guard.heavy("t"):
            pass
    assert e.value.code == "low_memory"


def test_extra_weights_counted(repo, monkeypatch):
    monkeypatch.setattr(guard, "memory_free_percent", lambda: 45)
    monkeypatch.setattr(guard, "_total_memory_gb", lambda: 128.0)
    with pytest.raises(guard.GuardRefused):
        with guard.heavy("qwen", extra_gb=28):   # 45 − 21.9 < 40
            pass


def test_busy_when_live_process_holds_lock(repo):
    lock = paths.root() / "runs" / ".heavy.lock"
    lock.parent.mkdir(parents=True)
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        lock.write_text(json.dumps({"pid": proc.pid, "kind": "other"}))
        with pytest.raises(guard.GuardRefused) as e:
            with guard.heavy("t"):
                pass
        assert e.value.code == "busy"
    finally:
        proc.kill()


def test_stale_lock_is_cleared_and_released(repo):
    lock = paths.root() / "runs" / ".heavy.lock"
    lock.parent.mkdir(parents=True)
    lock.write_text(json.dumps({"pid": 999_999, "kind": "dead"}))
    with guard.heavy("t") as report:
        assert json.loads(lock.read_text())["pid"] == os.getpid()
        assert report.free_percent == 80
    assert not lock.exists()
