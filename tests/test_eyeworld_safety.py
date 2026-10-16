"""안전(R-1177 · 설계 문서 I 2.9).

금지 포트 · 에코월드 잠금 · 우리 그룹만 종료 · 읽기 전용 확인.
"""

import os
import socket
import sys
import time

import pytest

from moai_echomind_fly.eyeworld import safety


def test_forbidden_ports_are_refused():
    for p in (9212, 9213, 8881, 9412):
        with pytest.raises(safety.SafetyError):
            safety.check_port(p)


def test_busy_port_is_refused():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        s.listen()
        with pytest.raises(safety.SafetyError):
            safety.check_port(s.getsockname()[1], forbidden=[])


def test_echomind_lock_rules(tmp_path):
    lock = tmp_path / "lock"
    assert safety.echomind_headless_busy(lock, 900) is None            # 없음
    lock.write_text(f"{os.getpid()} 12:00:00\n")
    assert safety.echomind_headless_busy(lock, 900) is not None        # 살아 있는 pid · 새것
    os.utime(lock, (time.time() - 1000, time.time() - 1000))
    assert safety.echomind_headless_busy(lock, 900) is None            # 오래됨
    lock.write_text("999999 12:00:00\n")
    assert safety.echomind_headless_busy(lock, 900) is None            # 죽은 pid


def test_owned_process_stops_only_its_group():
    other = safety.OwnedProcess([sys.executable, "-c", "import time; time.sleep(30)"])
    mine = safety.OwnedProcess([sys.executable, "-c",
                                "import subprocess,sys,time; subprocess.Popen([sys.executable,'-c',"
                                "'import time; time.sleep(30)']); time.sleep(30)"])
    other.start()
    mine.start()
    time.sleep(0.5)
    mine.stop(grace_s=2.0)
    assert not mine.alive() and other.alive()
    other.stop(grace_s=2.0)
    assert not other.alive()


def test_readonly_snapshot_detects_change(tmp_path):
    (tmp_path / "a.txt").write_text("x")
    snap = safety.readonly_snapshot([str(tmp_path)])
    assert safety.readonly_changed(snap) == []
    time.sleep(0.01)
    (tmp_path / "a.txt").write_text("y")
    os.utime(tmp_path / "a.txt", (time.time() + 5, time.time() + 5))
    assert safety.readonly_changed(snap) == [str(tmp_path / "a.txt")]
