"""세포 종류 단위 매개변수 — sim-v3 매개변수 파일 (R-7469 F-006 · I v2 §7.1).

무엇을 배우나: 범위(scope) 안 종류 t 마다 막 시간 상수 `tau_ms[t]` · 문턱 `v_th_mV[t]`,
그리고 범위 안 종류 쌍 p(t → u)마다 연결 배율 `alpha[p]`(≥ 0). 쌍은 연결마다 `pair_id[E]`
(쌍 밖은 −1)로 가리킨다. 연결 수 · 부호 · 상대 · `w_scale` · `ipsi_release` 는 **바꾸지 않는다**.

처음 값(`init`)은 sim-v2 와 같다 — `Engine(cell=init(...))` 은 `Engine()` 과 발화가 같다(시험).
범위 · 경계는 config/fit.yaml 에서 이름으로 잠근다(하드코딩 금지).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from . import config, paths
from .graph import Graph


@dataclass
class CellParams:
    types: np.ndarray               # str[T]
    tau_ms: np.ndarray              # float[T]
    v_th_mV: np.ndarray             # float[T]
    pairs: np.ndarray               # int[P, 2] — types 안 번호(출발 · 도착)
    alpha: np.ndarray               # float[P] ≥ 0
    pair_id: np.ndarray             # int32[E] — 쌍 밖 −1
    meta: dict[str, Any] = field(default_factory=dict)

    def neuron_type(self, graph: Graph) -> np.ndarray:
        """뉴런마다 범위 안 종류 번호(밖은 −1)."""
        pos = {t: i for i, t in enumerate(self.types)}
        return np.array([pos.get(t, -1) for t in graph.columns["type"]], dtype=np.int64)

    def per_neuron(self, graph: Graph, t_mbr_ms: float, v_th_mV: float
                   ) -> tuple[np.ndarray, np.ndarray]:
        """(막 시간 상수[N], 문턱[N]) — 범위 밖 뉴런은 sim 설정값 그대로."""
        k = self.neuron_type(graph)
        tau = np.full(graph.n, float(t_mbr_ms))
        vth = np.full(graph.n, float(v_th_mV))
        inside = k >= 0
        tau[inside] = self.tau_ms[k[inside]]
        vth[inside] = self.v_th_mV[k[inside]]
        return tau, vth

    def edge_scale(self) -> np.ndarray:
        """연결마다 곱할 배율 — 쌍 밖 1."""
        out = np.ones(len(self.pair_id))
        inside = self.pair_id >= 0
        out[inside] = self.alpha[self.pair_id[inside]]
        return out

    def check(self, graph: Graph) -> None:
        """정체성 검사(I v2 §7.1) — 모양 · α ≥ 0 · 쌍 번호 범위 · 종류가 그래프에 있음."""
        if len(self.pair_id) != len(graph.dst):
            raise ValueError(f"pair_id 길이 {len(self.pair_id)} ≠ 연결 수 {len(graph.dst)}")
        if (self.alpha < 0).any():
            raise ValueError("alpha 는 0 이상이어야 한다(부호는 바꾸지 않는다)")
        if self.pair_id.max(initial=-1) >= len(self.alpha):
            raise ValueError("pair_id 가 alpha 범위 밖")
        if not (len(self.tau_ms) == len(self.v_th_mV) == len(self.types)):
            raise ValueError("종류 수와 tau · v_th 길이가 다르다")
        if (self.tau_ms <= 0).any():
            raise ValueError("tau_ms 는 양수")
        missing = set(self.types) - set(graph.columns["type"])
        if missing:
            raise ValueError(f"그래프에 없는 종류 {sorted(missing)[:5]}")

    def save(self, path: Path) -> None:
        np.savez(path, types=self.types.astype(str), tau_ms=self.tau_ms, v_th_mV=self.v_th_mV,
                 pairs=self.pairs, alpha=self.alpha, pair_id=self.pair_id,
                 meta=np.array(json.dumps(self.meta, ensure_ascii=False)))

    @classmethod
    def load(cls, path: Path) -> CellParams:
        z = np.load(path)
        return cls(z["types"].astype(object), z["tau_ms"], z["v_th_mV"], z["pairs"], z["alpha"],
                   z["pair_id"].astype(np.int32), json.loads(str(z["meta"])))


def scope_types(scope: str, cfg: dict[str, Any] | None = None) -> list[str]:
    """범위 이름 → 종류 목록(config/fit.yaml)."""
    cfg = cfg or config.load("fit")
    sc = (cfg.get("scopes") or {}).get(scope)
    if sc is None:
        raise ValueError(f"범위 {scope!r} 는 아직 잠기지 않았다 — config/fit.yaml")
    types = list(config.load("loom")["amplitude"]) if sc.get("types_from_loom") else []
    return types + [t for t in sc.get("types", []) if t not in types]


def init(graph: Graph, types: list[str], t_mbr_ms: float, v_th_mV: float,
         pair_min_synapses: int, meta: dict[str, Any] | None = None,
         pair_dst: list[str] | None = None) -> CellParams:
    """처음 값 = sim 설정 그대로(α = 1). 쌍은 범위 안 종류끼리 시냅스 ≥ 문턱인 것만.

    `pair_dst` 가 있으면 도착 종류가 그 안인 쌍만(예: 하강 뉴런으로 가는 출력 쌍).
    """
    types_arr = np.array(types, dtype=object)
    pos = {t: i for i, t in enumerate(types)}
    k = np.array([pos.get(t, -1) for t in graph.columns["type"]], dtype=np.int64)
    ks, kd = k[graph.src()], k[graph.dst]
    inside = (ks >= 0) & (kd >= 0)
    if pair_dst is not None:
        inside &= np.isin(graph.columns["type"][graph.dst], list(pair_dst))
    code = np.where(inside, ks * len(types) + kd, -1)
    syn = np.bincount(code[inside], weights=graph.count[inside],
                      minlength=len(types) ** 2)
    keep = np.flatnonzero(syn >= pair_min_synapses)
    pairs = np.stack([keep // len(types), keep % len(types)], 1).astype(np.int64)
    lookup = np.full(len(types) ** 2, -1, dtype=np.int32)
    lookup[keep] = np.arange(len(keep), dtype=np.int32)
    pair_id = np.where(code >= 0, lookup[np.maximum(code, 0)], -1).astype(np.int32)
    t = len(types)
    return CellParams(types_arr, np.full(t, float(t_mbr_ms)), np.full(t, float(v_th_mV)),
                      pairs, np.ones(len(keep)), pair_id,
                      {"graph": graph.name, "pair_min_synapses": int(pair_min_synapses),
                       "pair_synapses": syn[keep].astype(int).tolist(), **(meta or {})})


def init_scope(graph: Graph, scope: str, t_mbr_ms: float, v_th_mV: float) -> CellParams:
    cfg = config.load("fit")
    sc = cfg["scopes"][scope]
    return init(graph, scope_types(scope, cfg), t_mbr_ms, v_th_mV,
                int(sc["pair_min_synapses"]), {"scope": scope, "version": cfg["version"]},
                pair_dst=sc.get("pair_dst"))


def path_for(graph_name: str, version: str, scope: str) -> Path:
    return paths.graph_dir() / f"{graph_name}.params-{version}-{scope}.npz"
