"""저장소 안의 자리들. `FLY_ROOT` 로 저장소 뿌리를 바꿀 수 있다(시험·다른 사본)."""

from __future__ import annotations

import os
from pathlib import Path


def root() -> Path:
    """저장소 뿌리. 기본은 이 파일에서 src/moai_echomind_fly 를 거슬러 올라간 곳."""
    env = os.environ.get("FLY_ROOT")
    if env:
        return Path(env).resolve()
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").is_file() and (parent / "config").is_dir():
            return parent
    raise RuntimeError(f"저장소 뿌리(pyproject.toml + config/)를 찾지 못했다: {here}")


def config_dir() -> Path:
    return root() / "config"


def raw_dir() -> Path:
    return root() / "data" / "raw"


def graph_dir() -> Path:
    return root() / "data" / "graph"


def runs_dir() -> Path:
    """실행 기록 자리. `FLY_RUNS_DIR` 로 바꿀 수 있다(시험)."""
    env = os.environ.get("FLY_RUNS_DIR")
    return Path(env).resolve() if env else root() / "runs"
