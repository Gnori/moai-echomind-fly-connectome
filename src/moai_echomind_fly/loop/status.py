"""상태 파일 — 만드는 쪽(학습)이 진행을 JSON 한 장으로 **직접** 쓴다(R-7347).

왜: 대시보드가 로그를 정규식으로 긁으면 문구가 바뀌거나 프로세스가 밖에서 끊길 때 틀린다
(2026-09-24 재배선 s0 를 끊은 뒤 "도는 중"으로 보였다). 요구자: *"DB 는 아니어도 데이터 파일로
결과를 가지고 있고 주기적으로 돌면서 업데이트"*.
어떻게: 칸을 합쳐 임시 파일에 쓰고 이름을 바꾼다(원자적 — 읽는 쪽이 반쯤 쓰인 파일을 보지 않는다) ·
쓸 때마다 `updated` 시각. 학습 규칙에는 아무 영향이 없다(기록만).
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _plain(x: Any) -> Any:
    """numpy 숫자 · Path 같은 것을 JSON 이 아는 값으로."""
    if hasattr(x, "item"):
        return x.item()
    return str(x)


class StatusFile:
    """한 실행의 상태 한 장 — `update(**칸)` 마다 파일이 새로 써진다."""

    def __init__(self, path: Path | str, **initial: Any) -> None:
        self.path = Path(path)
        self.data: dict[str, Any] = {"started": now(), "pid": os.getpid(), **initial}
        self.write()

    def update(self, **fields: Any) -> dict[str, Any]:
        self.data.update(fields)
        self.write()
        return self.data

    def write(self) -> None:
        self.data["updated"] = now()
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_text(json.dumps(self.data, ensure_ascii=False, indent=1, default=_plain),
                       encoding="utf-8")
        tmp.replace(self.path)


def read(path: Path | str) -> dict[str, Any] | None:
    """상태 파일을 읽는다 — 없거나 깨졌으면 None(읽는 쪽은 로그로 대체한다)."""
    p = Path(path)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return None
