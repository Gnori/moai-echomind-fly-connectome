"""시야 자리 대응 retino-v2 (R-7469 F-008 2단계 · I v2 §9.2).

시각 투사 뉴런마다 고도 · 방위를 연결 지도로 계산한다.
축 규약(연구 노트 13 §4 · MaleCNS 시각 경로 논문 방법):
    h = hex1 − hex2(앞쪽 +) · v = hex1 + hex2(등쪽 +).
방위는 오른눈의 h 를 `azimuth_range_deg` 로 선형 대응하고 왼눈은 부호를 뒤집는다
(+ = 왼쪽 · task.yaml 방위 규약). 고도는 retino-v1 그대로(v 를 `elevation_range_deg` 로 선형).
규칙 · 근거 · 한계는 `config/retino.yaml`. 결과는 `data/graph/<그래프>.<판>.json`(I-7469 §3.3).
옛 판(v1 · 방위 없음) 설정으로도 돈다.
뒤섞은 대응표(대조군)는 `shuffled()` — 같은 종류 · 같은 눈 안에서 값만 섞는다.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.feather as feather

from . import config, data, paths
from . import graph as graph_mod

SHUFFLE_KEYS = {"azimuth": "azimuth_deg", "elevation": "elevation_deg"}
# 뒤섞는 묶음 — 기본은 종류 × 눈(좌우는 남음) · type 은 두 눈을 섞어 좌우까지 지운다
SHUFFLE_WITHIN = {"type_side": ("type", "side"), "type": ("type",)}
REPORT_TYPES = ("LC4", "LC6", "LC11", "LC12", "LC16", "LC17", "LPLC1", "LPLC2")


def column_hex(g: graph_mod.Graph, raw: Path | None = None) -> np.ndarray:
    """그래프 인덱스마다 공식 기둥 좌표 (hex1, hex2) — 없으면 NaN."""
    raw = raw or paths.raw_dir()
    name = {s.key: s.name for s in data.sources()}["annotations"]
    t = feather.read_table(raw / name, columns=["bodyId", "assignedOlHex1", "assignedOlHex2"])
    bid = t.column("bodyId").to_numpy()
    h = np.stack([np.array([np.nan if v is None else v for v in t.column(c).to_pylist()], float)
                  for c in ("assignedOlHex1", "assignedOlHex2")], axis=1)
    pos = np.searchsorted(g.body_ids, bid)
    ok = pos < g.n
    ok[ok] &= g.body_ids[pos[ok]] == bid[ok]
    out = np.full((g.n, 2), np.nan)
    out[pos[ok]] = h[ok]
    return out


def weighted_input_hex(g: graph_mod.Graph, targets: np.ndarray, known: np.ndarray
                       ) -> tuple[np.ndarray, np.ndarray]:
    """뉴런마다 좌표를 가진 입력의 시냅스 가중 평균 좌표 · 그 입력 시냅스 수."""
    src = g.src()
    m = np.isin(g.dst, targets) & ~np.isnan(known[src, 0])
    s, d, c = src[m], g.dst[m], g.count[m].astype(np.float64)
    order = {int(t): k for k, t in enumerate(targets)}
    k = np.array([order[int(x)] for x in d], dtype=np.int64)
    w = np.bincount(k, weights=c, minlength=len(targets))
    hx = np.stack([np.bincount(k, weights=c * known[s, j], minlength=len(targets))
                   for j in (0, 1)], axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return hx / w[:, None], w


def _linear(value: float, lo_in: float, hi_in: float, lo_out: float, hi_out: float) -> float:
    """[lo_in, hi_in] 을 [lo_out, hi_out] 으로 선형 대응(입력 폭이 0 이면 가운데)."""
    if hi_in == lo_in:
        return (lo_out + hi_out) / 2
    return lo_out + (value - lo_in) / (hi_in - lo_in) * (hi_out - lo_out)


def build(g: graph_mod.Graph, cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """시각 투사 뉴런마다 (h, v) → 고도 · (설정에 `azimuth_range_deg` 가 있으면) 방위.

    방위 규약: 오른눈은 h 최소(뒤) → 범위 아래 · h 최대(앞) → 범위 위 · 왼눈은 부호 반전(+ = 왼쪽).
    """
    from .loop.signals import side_of

    cfg = cfg or config.load("retino")
    hexes = column_hex(g)
    known = hexes.copy()
    ty = g.columns["type"].astype(str)
    two = np.flatnonzero(np.logical_or.reduce(
        [np.char.startswith(ty, p) for p in cfg["two_hop_types_prefix"]]))
    hx2, _ = weighted_input_hex(g, two, hexes)
    known[two] = hx2
    mask = np.ones(g.n, bool)
    for col, vals in cfg["select"].items():
        mask &= np.isin(g.columns[col], vals)
    targets = np.flatnonzero(mask)
    hx, w = weighted_input_hex(g, targets, known)
    cols = hexes[~np.isnan(hexes[:, 0])]
    vs, hs = cols.sum(1), cols[:, 0] - cols[:, 1]
    v_min, v_max = float(vs.min()), float(vs.max())
    h_min, h_max = float(hs.min()), float(hs.max())
    elo, ehi = cfg["elevation_range_deg"]
    az_range = cfg.get("azimuth_range_deg")
    side = side_of(g)
    neurons, missing = [], Counter()
    for i, t in enumerate(targets):
        if w[i] < cfg["min_input_synapses"] or np.isnan(hx[i, 0]):
            missing[str(g.columns["type"][t])] += 1
            continue
        v = float(hx[i].sum())
        h = float(hx[i, 0] - hx[i, 1])
        row = {"idx": int(t), "bodyId": int(g.body_ids[t]),
               "type": str(g.columns["type"][t]), "side": str(side[t]),
               "hex": [round(float(hx[i, 0]), 3), round(float(hx[i, 1]), 3)],
               "h": round(h, 3), "v": round(v, 3),
               "elevation_deg": round(_linear(v, v_min, v_max, elo, ehi), 2),
               "input_synapses": int(w[i])}
        if az_range is not None:
            az = _linear(h, h_min, h_max, float(az_range[0]), float(az_range[1]))
            row["azimuth_deg"] = round(-az if side[t] == "L" else az, 2)
        neurons.append(row)
    out = {"rule": cfg["version"], "graph": g.name, "when": time.strftime("%Y-%m-%d %H:%M:%S"),
           "source": "assignedOlHex1/2(공식 주석) · 입력 시냅스 가중 평균 · T4/T5 는 두 단계",
           "axes": {"h": "hex1 − hex2 · 앞쪽 +(MaleCNS 시각 경로 논문 방법 · 노트 13 §4)",
                    "v": "hex1 + hex2 · 등쪽 +(R7d/R8d · HS 로 확인)"},
           "y_range": [v_min, v_max], "v_range": [v_min, v_max], "h_range": [h_min, h_max],
           "elevation_range_deg": [elo, ehi]}
    if az_range is not None:
        out["azimuth_range_deg"] = [float(az_range[0]), float(az_range[1])]
        out["azimuth"] = ("오른눈 h → azimuth_range_deg 선형(h 최대 = 앞 → 범위 위) · "
                          "왼눈 부호 반전 · + = 왼쪽 · 0° = 정면(task.yaml 방위 규약) · "
                          "범위는 선언 · 선형 근사")
    else:
        out["azimuth"] = ("없음 — 이 설정 판에는 azimuth_range_deg 가 없다"
                          "(retino-v1: 눈 단위 좌/우만)")
    out["neurons"] = neurons
    out["missing"] = dict(missing)
    return out


def shuffled(res: dict[str, Any], what: str = "azimuth", seed: int = 0,
             within: str = "type_side") -> dict[str, Any]:
    """뒤섞은 대응표(대조군) — `what`(azimuth | elevation) 값을 묶음 안에서 뒤섞는다.

    `within="type_side"`(기본): 같은 종류 · 같은 눈 안 — 눈 안의 자리만 지우고 **좌우는 남는다**
    (방위를 섞어도 원본과 상관 0.78 · 연구 노트 13 §4.5).
    `within="type"`: 같은 종류 안에서 두 눈을 섞는다 — 좌우(어느 쪽 시야인가)까지 지운다.
    다른 칸(좌표 · 다른 각 · 시냅스 수)은 그대로. `rule` 은 `<판>.shuffled-<what>-s<seed>`
    (`type` 이면 `<판>.shuffled-<what>-type-s<seed>`)로 바뀌어 파일 이름 · 자료 메타에 남는다.
    """
    if what not in SHUFFLE_KEYS:
        raise ValueError(f"what 은 {sorted(SHUFFLE_KEYS)} 중 하나 — {what!r}")
    if within not in SHUFFLE_WITHIN:
        raise ValueError(f"within 은 {sorted(SHUFFLE_WITHIN)} 중 하나 — {within!r}")
    key = SHUFFLE_KEYS[what]
    if any(key not in n for n in res["neurons"]):
        raise ValueError(f"대응표에 {key} 가 없는 뉴런이 있다 — 판 {res.get('rule')}")
    rng = np.random.default_rng(seed)
    out = json.loads(json.dumps(res))
    groups: dict[tuple[str, ...], list[int]] = defaultdict(list)
    for k, n in enumerate(out["neurons"]):
        groups[tuple(n[c] for c in SHUFFLE_WITHIN[within])].append(k)
    for ids in groups.values():
        vals = [out["neurons"][k][key] for k in ids]
        for k, j in zip(ids, rng.permutation(len(ids))):
            out["neurons"][k][key] = vals[j]
    out["base_rule"] = res["rule"]
    tag = "" if within == "type_side" else f"-{within}"
    out["rule"] = f"{res['rule']}.shuffled-{what}{tag}-s{seed}"
    out["shuffle"] = {"what": what, "key": key, "seed": int(seed),
                      "within": list(SHUFFLE_WITHIN[within])}
    return out


def path_for(graph_name: str, rule: str) -> Path:
    return paths.graph_dir() / f"{graph_name}.{rule}.json"


def load(graph_name: str = "traced-v1", rule: str | None = None) -> dict[str, Any]:
    """대응표를 읽는다. `rule` 이 없으면 설정의 지금 판(retino-v2).

    뒤섞은 표는 `rule="<판>.shuffled-<what>-s<seed>"`.
    """
    ver = rule or str(config.load("retino")["version"])
    return json.loads(path_for(graph_name, ver).read_text(encoding="utf-8"))


def _pct(values: list[float]) -> str:
    return (f"중앙 {np.median(values):6.1f}° · 10~90 % {np.percentile(values, 10):6.1f}"
            f" ~ {np.percentile(values, 90):6.1f}°")


def summarize(res: dict[str, Any], types: tuple[str, ...] = REPORT_TYPES) -> list[str]:
    """종류마다 고도 · (있으면) 눈별 방위의 중앙 · 10~90 % 한 줄씩."""
    elev: dict[str, list[float]] = defaultdict(list)
    az: dict[tuple[str, str], list[float]] = defaultdict(list)
    for n in res["neurons"]:
        elev[n["type"]].append(n["elevation_deg"])
        if "azimuth_deg" in n:
            az[(n["type"], n["side"])].append(n["azimuth_deg"])
    lines = []
    for t in types:
        if not elev.get(t):
            continue
        line = f"  {t:<7} {len(elev[t]):>4} · 고도 {_pct(elev[t])}"
        for s in ("R", "L"):
            if az.get((t, s)):
                line += f" · 방위 {s}({len(az[(t, s)])}) {_pct(az[(t, s)])}"
        lines.append(line)
    return lines


def main(args: argparse.Namespace) -> int:
    g = graph_mod.load(args.graph)
    res = build(g)
    out = path_for(g.name, res["rule"])
    out.write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")
    n_miss = sum(res["missing"].values())
    az_range = res.get("azimuth_range_deg", "없음")
    print(f"[retino] {res['rule']} · 뉴런 {len(res['neurons'])} · 빠짐 {n_miss}"
          f" · v {res['v_range']} → 고도 {res['elevation_range_deg']}"
          f" · h {res['h_range']} → 방위 {az_range}(오른눈 · 왼눈 거울)")
    for line in summarize(res):
        print(line)
    print(f"[retino] 저장 {out}")
    unit = {"type_side": "종류 × 눈 안에서", "type": "종류 안에서 · 두 눈을 섞음"}[args.within]
    for what in args.shuffle or []:
        sh = shuffled(res, what, args.seed, args.within)
        p = path_for(g.name, sh["rule"])
        p.write_text(json.dumps(sh, ensure_ascii=False), encoding="utf-8")
        print(f"[retino] 뒤섞은 대응표({what} · 씨앗 {args.seed} · {unit}) 저장 {p}")
    return 0


def register(sub) -> None:
    p = sub.add_parser("retinotopy", help="시야 자리 대응(R-7469 F-008) — 가벼움 · CPU")
    rs = p.add_subparsers(dest="retino_cmd", required=True)
    b = rs.add_parser("build",
                      help="시각 투사 뉴런마다 고도 · 방위 → data/graph/<그래프>.<판>.json")
    b.add_argument("--graph", default="traced-v1")
    b.add_argument("--shuffle", action="append", choices=sorted(SHUFFLE_KEYS),
                   help="뒤섞은 대응표(대조군)도 저장 — 여러 번 줄 수 있다")
    b.add_argument("--seed", type=int, default=0, help="뒤섞기 씨앗")
    b.add_argument("--within", choices=sorted(SHUFFLE_WITHIN), default="type_side",
                   help="섞는 묶음 — type_side(기본 · 좌우는 남김)"
                        " | type(두 눈을 섞어 좌우까지 지움)")
    b.set_defaults(func=main)
