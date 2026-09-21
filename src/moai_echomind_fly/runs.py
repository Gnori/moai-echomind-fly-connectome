"""실행 기록 (R-3340 F-007 · I §1.4) — `runs/<run_id>/`."""

from __future__ import annotations

import json
import secrets
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from . import paths
from .graph import Graph
from .sim import Engine


def new_run_id(kind: str) -> str:
    return f"{time.strftime('%Y%m%d-%H%M%S')}-{kind}-{secrets.token_hex(2)}"


class RunRecorder:
    """한 실행의 기록 폴더. 쓰는 순서와 상관없이 파일 하나씩 남긴다."""

    def __init__(self, kind: str, request: dict[str, Any], *, root: Path | None = None) -> None:
        self.run_id = new_run_id(kind)
        self.dir = (root or paths.runs_dir()) / self.run_id
        self.dir.mkdir(parents=True, exist_ok=False)
        self.t0 = time.time()
        self.write_json("request.json", request)
        self._log: list[str] = []

    def write_json(self, name: str, obj: Any) -> Path:
        path = self.dir / name
        path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=_jsonable),
                        encoding="utf-8")
        return path

    def log(self, line: str) -> None:
        # 표준 에러로 — MCP stdio 서버에서는 표준 출력이 프로토콜 통로다
        print(line, file=sys.stderr, flush=True)
        self._log.append(line)
        (self.dir / "stdout.log").write_text("\n".join(self._log) + "\n", encoding="utf-8")

    def record_setup(self, engine: Engine, graph: Graph, extra_versions: dict[str, str]) -> None:
        self.write_json("params.json", {
            "lif": engine.p.to_dict(), "versions": extra_versions, "device": str(engine.dev),
            "dtype": str(engine.dtype), "seed": engine.seed,
        })
        self.write_json("graph.json", {"name": graph.name, "meta": graph.meta})

    def record_spikes(self, engine: Engine, **extra: np.ndarray) -> Path:
        step, neuron = engine.spikes()
        path = self.dir / "spikes.npz"
        arrays: dict[str, Any] = {"step": step, "neuron": neuron, "dt_ms": engine.p.dt_ms}
        if engine.rec_idx.numel():
            arrays["v_idx"] = engine.rec_idx.cpu().numpy()
            arrays["voltage"] = engine.voltage().astype(np.float32)
        arrays.update(extra)
        np.savez_compressed(path, **arrays)
        return path

    def honesty(self, wiring: list[str], added: list[str]) -> None:
        """정직 목록 — 배선이 낸 것 / 코드가 더한 것."""
        lines = ["# 정직 목록", "", "## 배선이 낸 것", *[f"- {x}" for x in wiring], "",
                 "## 코드가 더한 것", *[f"- {x}" for x in added], ""]
        (self.dir / "honesty.md").write_text("\n".join(lines), encoding="utf-8")

    @property
    def wall_s(self) -> float:
        return time.time() - self.t0


def _jsonable(o: Any) -> Any:
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(type(o))
