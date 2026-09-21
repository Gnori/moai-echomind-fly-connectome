"""`config/*.yaml` 읽기.

모든 설정 파일은 `version` 칸을 가진다 — 실행 기록에 판을 남기기 위해서다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from . import paths


def load(name: str, directory: Path | None = None) -> dict[str, Any]:
    """`config/<name>.yaml` 을 읽는다.

    Args:
        name: 확장자 없는 파일 이름(예: "sim").
        directory: 설정 폴더(기본은 저장소의 config/).

    Returns:
        YAML 매핑.

    Raises:
        ValueError: 매핑이 아니거나 `version` 칸이 없을 때.
    """
    path = (directory or paths.config_dir()) / f"{name}.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not data.get("version"):
        raise ValueError(f"{path}: 매핑이 아니거나 version 칸이 없다")
    return data
