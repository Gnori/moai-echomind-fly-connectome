"""무거운 실행 가드 (R-3340 2.3 · I §2.3).

맥 한 대의 GPU·통합 메모리를 요구자 화면·다른 세션과 같이 쓴다. 무거운 실행은
① 메모리 여유가 문턱 이상일 때만 ② 한 번에 하나만 돈다. 통과 못 하면 `GuardRefused`.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from . import config, paths


class GuardRefused(RuntimeError):
    """가드가 실행을 막았다. `code` 는 I §2.2 의 에러 코드(low_memory · busy)."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


@dataclass(frozen=True)
class GuardReport:
    kind: str
    free_percent: int
    threshold: int
    lock_file: str


_FREE_RE = re.compile(r"System-wide memory free percentage:\s*(\d+)%")


def memory_free_percent() -> int:
    """`memory_pressure` 의 시스템 전체 여유 %."""
    out = subprocess.run(
        ["memory_pressure"], capture_output=True, text=True, check=True, timeout=30
    ).stdout
    match = _FREE_RE.search(out)
    if not match:
        raise RuntimeError(f"memory_pressure 출력에서 여유 %를 찾지 못했다: {out[-200:]!r}")
    return int(match.group(1))


_FOOT_RE = re.compile(r"phys_footprint:\s*([\d.]+)\s*([KMG])B")


def process_footprint_gb(pid: int | None = None) -> float:
    """프로세스 전체 메모리(phys_footprint · MLX 통합 메모리 포함) GB — `footprint` 명령."""
    out = subprocess.run(["footprint", "-p", str(pid or os.getpid())], capture_output=True,
                         text=True, timeout=60).stdout
    m = _FOOT_RE.search(out)
    if not m:
        raise RuntimeError(f"footprint 출력에서 phys_footprint 를 못 찾았다: {out[-200:]!r}")
    return float(m.group(1)) * {"K": 1e-6, "M": 1e-3, "G": 1.0}[m.group(2)]


class Watchdog:
    """긴 실행 중 감시 — 문턱을 넘으면 까닭을 돌려준다(부르는 쪽이 저장하고 멈춘다)."""

    def __init__(self) -> None:
        cfg = config.load("guard")
        self.every = int(cfg["training_check_every"])
        self.min_free = int(cfg["training_abort_free_percent"])
        self.max_gb = float(cfg["training_abort_footprint_gb"])
        self.last: dict[str, float] = {}

    def check(self, step: int) -> str | None:
        if step % self.every:
            return None
        free = memory_free_percent()
        foot = process_footprint_gb()
        self.last = {"step": step, "sys_free_percent": free, "footprint_gb": round(foot, 1)}
        if free < self.min_free:
            return f"시스템 메모리 여유 {free}% < {self.min_free}%"
        if foot > self.max_gb:
            return f"프로세스 메모리 {foot:.1f} GB > {self.max_gb} GB"
        return None


def _total_memory_gb() -> float:
    out = subprocess.run(
        ["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, check=True, timeout=10
    ).stdout
    return int(out.strip()) / 1024**3


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


@contextmanager
def heavy(kind: str, *, extra_gb: float = 0.0,
          min_free_percent: int | None = None) -> Generator[GuardReport]:
    """무거운 실행을 감싼다.

    Args:
        kind: 실행 종류(기록용 — "run" · "bench" · "qwen" …).
        extra_gb: 이 실행이 새로 올릴 메모리(GB). 여유에서 이만큼 빼고도 문턱을 넘어야 한다.
        min_free_percent: 문턱을 바꿀 때만(투영기 학습 — guard.yaml `training_min_free_percent`).

    Yields:
        검사 결과.

    Raises:
        GuardRefused: 메모리 부족(low_memory) 또는 다른 무거운 실행 중(busy).
    """
    cfg = config.load("guard")
    threshold = int(min_free_percent if min_free_percent is not None else cfg["min_free_percent"])
    free = memory_free_percent()
    after = free - (extra_gb / _total_memory_gb() * 100 if extra_gb else 0)
    if after < threshold:
        raise GuardRefused(
            "low_memory",
            f"메모리 여유 {free}%(이 실행 뒤 예상 {after:.0f}%) < 문턱 {threshold}%",
        )
    lock = paths.root() / cfg["lock_file"]
    lock.parent.mkdir(parents=True, exist_ok=True)
    report = GuardReport(kind=kind, free_percent=free, threshold=threshold, lock_file=str(lock))
    if _held_by_owner(lock):
        # 부모(FLY_GUARD_OWNER)가 쥔 잠금 안의 실행 — 예: Qwen 실험자가 띄운 MCP 서버의 실험.
        # 한 번에 하나라는 뜻은 그대로다(같은 무거운 작업의 일부). 잠금은 부모가 푼다.
        yield report
        return
    _acquire(lock, kind)
    try:
        yield report
    finally:
        lock.unlink(missing_ok=True)


OWNER_ENV = "FLY_GUARD_OWNER"


def _held_by_owner(lock: Path) -> bool:
    owner = int(os.environ.get(OWNER_ENV, "0") or 0)
    if not owner or not lock.exists():
        return False
    try:
        held = json.loads(lock.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return int(held.get("pid", 0)) == owner and _pid_alive(owner)


def _acquire(lock: Path, kind: str) -> None:
    record = json.dumps({"pid": os.getpid(), "kind": kind, "started": time.time()})
    for _ in range(2):
        try:
            with lock.open("x", encoding="utf-8") as fh:
                fh.write(record)
            return
        except FileExistsError:
            try:
                held = json.loads(lock.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                held = {}
            pid = int(held.get("pid", 0))
            if pid and pid != os.getpid() and _pid_alive(pid):
                raise GuardRefused("busy", f"다른 무거운 실행 중: {held}") from None
            lock.unlink(missing_ok=True)  # 죽은 프로세스가 남긴 잠금
    raise GuardRefused("busy", f"잠금을 얻지 못했다: {lock}")
