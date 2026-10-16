"""안전 — 라이브 에코월드와 완전히 분리(설계 문서 I 2.9).

- 금지 포트(에코월드 라이브 · 눈 워커 · 상자 월드)를 쓰려 하면 거부 · 우리 포트가 비었나 확인.
- 에코월드 헤드리스 잠금이 살아 있으면 우리 크롬을 띄우지 않는다
  (그쪽 headless_guard 와 같은 규칙 · 읽기만).
- 우리가 띄운 프로세스는 제 세션(그룹)으로 띄우고 **그 그룹만** 내린다
  — 이름 패턴 종료는 하지 않는다.
- 읽기 전용 확인: 에코월드 저장소 표본 파일의 수정 시각을 실행 전후로 비교.
"""

from __future__ import annotations

import os
import signal
import socket
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

from .. import config, paths


class SafetyError(RuntimeError):
    """안전 규칙 위반 — 시작하지 않는다."""


def settings() -> dict:
    return config.load("eyeworld")["safety"]


def repo_path(rel: str) -> Path:
    """설정의 상대 경로(저장소 뿌리 기준) → 절대 경로."""
    p = Path(rel)
    return p if p.is_absolute() else (paths.root() / p).resolve()


def check_port(port: int, forbidden: list[int] | None = None) -> None:
    """금지 포트면 거부 · 이미 누가 쓰고 있으면 거부(127.0.0.1 에 묶어 봄)."""
    forbidden = list(settings()["forbidden_ports"]) if forbidden is None else forbidden
    if int(port) in {int(x) for x in forbidden}:
        raise SafetyError(f"포트 {port} 는 금지(에코월드 라이브 · 눈 워커 · 상자 월드)")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("127.0.0.1", int(port)))
        except OSError as e:
            raise SafetyError(f"포트 {port} 를 이미 누가 쓴다: {e}") from e


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def echomind_headless_busy(lock: Path | None = None, fresh_s: float | None = None) -> str | None:
    """에코월드 헤드리스 잠금이 살아 있으면 그 내용, 아니면 None(읽기만)."""
    cfg = settings()
    lock = Path(cfg["echomind_headless_lock"]) if lock is None else lock
    fresh_s = float(cfg["echomind_lock_fresh_s"]) if fresh_s is None else fresh_s
    if not lock.exists() or time.time() - lock.stat().st_mtime >= fresh_s:
        return None
    text = lock.read_text(errors="replace").strip()
    try:
        pid = int(text.split()[0])
    except (IndexError, ValueError):
        return None
    return text if _alive(pid) else None


@dataclass
class OwnedProcess:
    """우리가 띄운 프로세스 — 제 세션(그룹)으로 띄우고 그 그룹만 내린다."""

    cmd: list[str]
    log: Path | None = None
    proc: subprocess.Popen | None = field(default=None, repr=False)

    def start(self) -> int:
        out = open(self.log, "ab") if self.log else subprocess.DEVNULL  # noqa: SIM115
        self.proc = subprocess.Popen(self.cmd, stdout=out, stderr=subprocess.STDOUT,
                                     start_new_session=True)
        return self.proc.pid

    @property
    def pid(self) -> int | None:
        return self.proc.pid if self.proc else None

    def alive(self) -> bool:
        return bool(self.proc) and self.proc.poll() is None

    def stop(self, grace_s: float = 5.0) -> None:
        """우리 그룹(pgid = 우리 pid)만 TERM → 기다림 → 남으면 KILL."""
        if not self.proc:
            return
        pgid = self.proc.pid
        for sig, wait in ((signal.SIGTERM, grace_s), (signal.SIGKILL, 2.0)):
            try:
                os.killpg(pgid, sig)
            except ProcessLookupError:
                break
            try:
                self.proc.wait(timeout=wait)
                break
            except subprocess.TimeoutExpired:
                continue


def readonly_snapshot(roots: list[str] | None = None, per_root: int = 200) -> dict[str, float]:
    """읽기 전용이어야 할 저장소의 표본 파일 → 수정 시각(정렬된 앞 per_root 개 + 뿌리 바로 아래)."""
    roots = list(settings()["readonly_roots"]) if roots is None else roots
    snap: dict[str, float] = {}
    for r in roots:
        root = repo_path(r)
        if not root.exists():
            continue
        files = sorted(p for p in root.iterdir() if p.is_file())
        for sub in ("bj", "vendor/babylon", "assets/village"):
            d = root / sub
            if d.is_dir():
                files += sorted(p for p in d.iterdir() if p.is_file())[:per_root]
        for p in files:
            snap[str(p)] = p.stat().st_mtime
    return snap


def readonly_changed(before: dict[str, float]) -> list[str]:
    """실행 전 표본과 비교해 바뀐 파일 목록(없어야 한다)."""
    out = []
    for p, m in before.items():
        q = Path(p)
        if not q.exists() or q.stat().st_mtime != m:
            out.append(p)
    return out
