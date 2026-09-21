"""그래프 컴파일 (R-3340 F-002 · I §1.2).

공식 파일 → Traced 뉴런 · consensus_nt 부호 · 시냅스 수 세기 · 출발 뉴런 순 정렬(CSR).
섞은 배선(rewired)은 도착 배열만 무작위로 섞는다 — 뉴런마다 들어오고 나가는 연결 수가 원본과 같다.

저장:  data/graph/<name>.neurons.parquet  (뉴런 주석 + nt + sign)
       data/graph/<name>.edges.npz        (row_ptr · dst · count)
       data/graph/<name>.json             (메타데이터)
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.feather as feather
import pyarrow.ipc as ipc
import pyarrow.parquet as pq

from . import data, paths

# 부호 — 아세틸콜린 흥분 · GABA/글루탐산 억제(Liu & Wilson 2013) · 히스타민 억제(Hardie 1989) ·
# 조절 물질(도파민·세로토닌·옥토파민)과 미정은 0. 08 실측의 결론(consensus_nt)을 쓴다.
SIGNS = {"acetylcholine": 1, "gaba": -1, "glutamate": -1, "histamine": -1}
MISSING_NT = "missing"

# 주석에서 가져오는 칸 — 채널 표·대응표·요약이 이 칸들로 뉴런을 고른다.
ANNOTATION_COLUMNS = [
    "bodyId", "status", "type", "instance", "class", "subclass", "superclass", "somaSide",
    "entryNerve", "exitNerve", "somaNeuromere", "dimorphism", "fruDsx", "receptorType",
    "flywireType",
]


@dataclass
class Graph:
    """컴파일된 그래프. 인덱스 i 는 `body_ids[i]` 뉴런이다."""

    name: str
    columns: dict[str, np.ndarray]      # 주석 칸 → 문자열 배열(빈 값은 "")
    body_ids: np.ndarray                # int64[N] 오름차순
    sign: np.ndarray                    # int8[N]
    row_ptr: np.ndarray                 # int64[N+1]
    dst: np.ndarray                     # int32[E]
    count: np.ndarray                   # int32[E]
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.body_ids)

    @property
    def e(self) -> int:
        return len(self.dst)

    def src(self) -> np.ndarray:
        """연결마다 출발 인덱스(int32[E]) — 필요할 때만 펼친다."""
        return np.repeat(np.arange(self.n, dtype=np.int32), np.diff(self.row_ptr))

    def select(self, **filters: str | list[str]) -> np.ndarray:
        """주석 칸으로 뉴런 인덱스를 고른다. 값이 목록이면 그중 하나.

        예: `g.select(type="DNp01")` · `g.select(superclass="descending_neuron")`.
        칸 이름 `class` 는 파이썬 예약어라 `klass=` 로도 받는다.
        """
        mask = np.ones(self.n, dtype=bool)
        for key, want in filters.items():
            col = self.columns["class" if key == "klass" else key]
            wanted = [want] if isinstance(want, str) else list(want)
            mask &= np.isin(col, wanted)
        return np.flatnonzero(mask)

    def type_of(self, idx: np.ndarray | int) -> np.ndarray:
        return self.columns["type"][idx]


def _column_strings(table: pa.Table, name: str) -> np.ndarray:
    col = table.column(name).to_pylist()
    return np.array(["" if v is None else str(v) for v in col], dtype=object)


def _map_ids(ids: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    index = np.searchsorted(ids, values)
    valid = index < len(ids)
    valid[valid] &= ids[index[valid]] == values[valid]
    return index, valid


def _read_shas(raw: Path) -> dict[str, str]:
    sums = raw / "SHA256SUMS"
    if not sums.exists():
        return {}
    lines = [ln for ln in sums.read_text().split("\n") if ln]
    return {name: sha for sha, name in (ln.split("  ", 1) for ln in lines)}


def compile_graph(
    name: str = "traced-v1", *, raw: Path | None = None, out: Path | None = None, log=print
) -> Graph:
    """공식 파일에서 그래프를 만든다(뉴런 = status Traced).

    Returns:
        컴파일된 그래프(저장까지 끝난 것).
    """
    raw = raw or paths.raw_dir()
    files = {s.key: raw / s.name for s in data.sources()}

    ann = feather.read_table(files["annotations"])
    present = [c for c in ANNOTATION_COLUMNS if c in ann.column_names]
    missing_cols = [c for c in ANNOTATION_COLUMNS if c not in ann.column_names]
    status = _column_strings(ann, "status")
    keep = np.flatnonzero(status == "Traced")
    body_all = ann.column("bodyId").to_numpy()
    order = np.argsort(body_all[keep], kind="stable")
    rows = keep[order]
    body_ids = body_all[rows].astype(np.int64)
    if len(np.unique(body_ids)) != len(body_ids):
        raise ValueError("Traced 뉴런 bodyId 가 겹친다")
    columns = {c: _column_strings(ann, c)[rows] for c in present if c != "bodyId"}
    log(f"[graph] 주석 {ann.num_rows:,} 행 → Traced {len(body_ids):,}")

    nt_table = feather.read_table(files["neurotransmitters"], columns=["body", "consensus_nt"])
    nt_body = nt_table.column("body").to_numpy()
    nt_val = _column_strings(nt_table, "consensus_nt")
    idx, ok = _map_ids(body_ids, nt_body)
    nt = np.full(len(body_ids), MISSING_NT, dtype=object)
    nt[idx[ok]] = np.char.lower(nt_val[ok].astype(str)).astype(object)
    nt[nt == ""] = "unclear"
    sign = np.array([SIGNS.get(v, 0) for v in nt], dtype=np.int8)
    columns["nt"] = nt

    src_parts, dst_parts, cnt_parts, raw_rows = [], [], [], 0
    with pa.memory_map(str(files["weights"]), "r") as stream:
        reader = ipc.open_file(stream)
        for b in range(reader.num_record_batches):
            batch = reader.get_batch(b)
            pre = batch.column("body_pre").to_numpy()
            post = batch.column("body_post").to_numpy()
            weight = batch.column("weight").to_numpy()
            raw_rows += len(pre)
            s_idx, s_ok = _map_ids(body_ids, pre)
            d_idx, d_ok = _map_ids(body_ids, post)
            both = s_ok & d_ok
            if both.any():
                if (weight[both] <= 0).any():
                    raise ValueError("시냅스 수가 0 이하인 연결이 있다")
                src_parts.append(s_idx[both].astype(np.int32))
                dst_parts.append(d_idx[both].astype(np.int32))
                cnt_parts.append(weight[both].astype(np.int64))
    src = np.concatenate(src_parts)
    dst = np.concatenate(dst_parts)
    cnt = np.concatenate(cnt_parts)
    log(f"[graph] 연결 원본 {raw_rows:,} 행 → Traced 사이 {len(src):,}")

    order = np.lexsort((dst, src))
    src, dst, cnt = src[order], dst[order], cnt[order]
    key = src.astype(np.int64) * len(body_ids) + dst
    dup = np.flatnonzero(np.diff(key) == 0)
    merged = 0
    if len(dup):
        uniq, start = np.unique(key, return_index=True)
        cnt = np.add.reduceat(cnt, start)
        src, dst = src[start], dst[start]
        merged = len(key) - len(uniq)
        log(f"[graph] 같은 (출발, 도착) 쌍 {merged:,} 개를 합쳤다")
    if cnt.max() > np.iinfo(np.int32).max:
        raise ValueError("시냅스 수가 int32 를 넘는다")
    row_ptr = np.concatenate([[0], np.cumsum(np.bincount(src, minlength=len(body_ids)))]).astype(
        np.int64
    )

    kc = np.flatnonzero(columns.get("class", np.array([], dtype=object)) == "Kenyon_Cell")
    meta = {
        "name": name,
        "dataset": "male-cns:v1.0",
        "neuron_set": 'status == "Traced"',
        "neurons": int(len(body_ids)),
        "edges": int(len(dst)),
        "synapses": int(cnt.sum()),
        "raw_edge_rows": int(raw_rows),
        "merged_duplicate_pairs": int(merged),
        "self_loops": int((src == dst).sum()),
        "annotation_columns_missing": missing_cols,
        "nt_counts": dict(Counter(nt.tolist()).most_common()),
        "sign_counts": {str(k): int(v) for k, v in zip(*np.unique(sign, return_counts=True))},
        "sign_rule": {**SIGNS, "other (dopamine·serotonin·octopamine·unclear·missing)": 0},
        "kenyon_cells": {"n": int(len(kc)), "signs": dict(Counter(sign[kc].tolist()))},
        "source_sha256": _read_shas(raw),
    }
    graph = Graph(name, columns, body_ids, sign, row_ptr, dst.astype(np.int32),
                  cnt.astype(np.int32), meta)
    save(graph, out)
    return graph


def rewire(graph: Graph, seed: int) -> Graph:
    """섞은 배선: 도착 배열을 무작위 순열로 섞는다(출발·시냅스 수·출발 부호는 그대로)."""
    rng = np.random.default_rng(seed)
    dst = rng.permutation(graph.dst)
    meta = dict(graph.meta)
    meta.update({
        "name": f"{graph.name}.rewired-s{seed}",
        "rewired_from": graph.name,
        "rewire_seed": seed,
        "rewire_rule": "permute dst over all edges — in/out edge counts per neuron preserved",
        "self_loops": int((graph.src() == dst).sum()),
    })
    return Graph(meta["name"], graph.columns, graph.body_ids, graph.sign, graph.row_ptr,
                 dst.astype(np.int32), graph.count, meta)


def save(graph: Graph, out: Path | None = None) -> None:
    out = out or paths.graph_dir()
    out.mkdir(parents=True, exist_ok=True)
    table = pa.table({"bodyId": graph.body_ids, "sign": graph.sign,
                      **{k: pa.array(v.tolist(), pa.string()) for k, v in graph.columns.items()}})
    pq.write_table(table, out / f"{graph.name}.neurons.parquet")
    np.savez(out / f"{graph.name}.edges.npz", row_ptr=graph.row_ptr, dst=graph.dst,
             count=graph.count)
    (out / f"{graph.name}.json").write_text(
        json.dumps(graph.meta, ensure_ascii=False, indent=2), encoding="utf-8")


def load(name: str = "traced-v1", directory: Path | None = None) -> Graph:
    directory = directory or paths.graph_dir()
    table = pq.read_table(directory / f"{name}.neurons.parquet")
    names = [c for c in table.column_names if c not in ("bodyId", "sign")]
    columns = {c: _column_strings(table, c) for c in names}
    with np.load(directory / f"{name}.edges.npz") as z:
        row_ptr, dst, count = z["row_ptr"], z["dst"], z["count"]
    meta = json.loads((directory / f"{name}.json").read_text(encoding="utf-8"))
    return Graph(name, columns, table.column("bodyId").to_numpy().astype(np.int64),
                 table.column("sign").to_numpy().astype(np.int8), row_ptr, dst, count, meta)
