"""`config/*.yaml` 읽기.

모든 설정 파일은 `version` 칸을 가진다 — 실행 기록에 판을 남기기 위해서다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from . import paths


def load(name: str, directory: Path | None = None, version: str | None = None) -> dict[str, Any]:
    """`config/<name>.yaml` 을 읽는다. `version` 이 지금 판과 다르면 옛 판을 읽는다.

    옛 판은 `config/archive/<판>.yaml`(예: `archive/sim-v1.yaml`) — 판을 올릴 때 남긴다(I-7469 §1).

    Args:
        name: 확장자 없는 파일 이름(예: "sim").
        directory: 설정 폴더(기본은 저장소의 config/).
        version: 원하는 판(예: "sim-v1"). 없으면 지금 판.

    Returns:
        YAML 매핑.

    Raises:
        ValueError: 매핑이 아니거나 `version` 칸이 없을 때.
    """
    base = directory or paths.config_dir()
    path = base / f"{name}.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if version and isinstance(data, dict) and data.get("version") != version:
        path = base / "archive" / f"{version}.yaml"
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not data.get("version"):
        raise ValueError(f"{path}: 매핑이 아니거나 version 칸이 없다")
    return data


def current_dataset() -> str:
    """지금 쓰는 학습 자료 묶음(config/datasets.yaml current) — 명령마다 --dataset 기본값."""
    return str(load("datasets")["current"])
