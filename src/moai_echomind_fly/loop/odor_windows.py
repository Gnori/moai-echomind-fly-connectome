"""냄새 쪽을 여러 창으로 읽기 — 연구 노트 14 §7 M2 (R-7469 F-003 다음 판).

한 결정 창(300 ms)에서는 입력부터 좌우 차가 잡음 아래다(ORN 이상적 관찰자 0.61 · 투사 뉴런 0.508).
읽는 시간을 늘리면 상한이 오른다(1 s 0.75 · 3 s 0.82 — 노트 14 §2) → 회로 신호로도 오르나 본다.
새 시뮬레이션 없이 이미 있는 자료의 냄새 채널(후각 투사 뉴런 · 종류 × 좌우) 특징으로:

| 특징 집합 | 뜻 |
|---|---|
| `win<K>` | 같은 에피소드의 **이번 결정과 직전 K−1 결정** 평균(지난 것만) |
| `delta` | 시간 차 — 총 냄새 채널 세기 변화 dS · 몸 회전 dθ · dS × dθ (직전 결정 대비) |
| `win<K>+delta` | 둘을 붙임 |

⚠️ dθ 는 참값(몸 방향)이다 — 파리가 자기 회전을 안다고 두는 것(운동 사본).
냄새 정답은 아니지만 **회로 밖 값**이라 `delta` 가 이기면 "회로가 좌우를 날랐다"가 아니라
"시간 비교 규칙이 된다"로 읽는다.
판정은 stats-v2 와 같다(재배선 앙상블 · 에피소드 부호 뒤집기 · Holm · 찍기).
에피소드 안 결정은 독립이 아니므로 에피소드 단위 검정이 곧 "에피소드당 한 표"다.
정답 안쪽 상관(ICC)도 함께 적는다(노트 14 §6.2 가 재지 않은 값).
"""

from __future__ import annotations

import json
import time
from typing import Any

import numpy as np

from .. import config, paths

ODOR_WINDOWS_RULE: dict[str, Any] = {
    "version": "odor-windows-v1",
    "channel": "smell",
    "field": "odor_side",
    "windows": [1, 3, 5, 10],
}


def _order(rows: list[dict]) -> list[list[int]]:
    """에피소드(씨앗)마다 결정 순서대로 행 번호."""
    eps: dict[int, list[int]] = {}
    for i, r in enumerate(rows):
        eps.setdefault(r["seed"], []).append(i)
    return [sorted(ix, key=lambda i: rows[i]["decision"]) for ix in eps.values()]


def window_mean(x: np.ndarray, rows: list[dict], k: int) -> np.ndarray:
    """결정마다 같은 에피소드의 자기와 직전 k−1 결정 평균(앞 결정이 모자라면 있는 만큼)."""
    out = np.empty_like(x, dtype=np.float64)
    for ix in _order(rows):
        c = np.cumsum(np.vstack([np.zeros(x.shape[1]), x[ix]]), axis=0)
        for j, i in enumerate(ix):
            lo = max(0, j + 1 - k)
            out[i] = (c[j + 1] - c[lo]) / (j + 1 - lo)
    return out


def _wrap(deg: float) -> float:
    return (deg + 180.0) % 360.0 - 180.0


def delta_features(x: np.ndarray, rows: list[dict]) -> np.ndarray:
    """[dS, dθ, dS·dθ] — 첫 결정은 0. dS 는 총 채널 세기의 직전 대비 상대 변화."""
    s = x.sum(1)
    out = np.zeros((len(rows), 3))
    for ix in _order(rows):
        for a, b in zip(ix, ix[1:]):
            ds = (s[b] - s[a]) / (abs(s[a]) + 1e-9)
            dth = _wrap(rows[b]["truth"]["heading_deg"] - rows[a]["truth"]["heading_deg"])
            out[b] = [ds, dth / 90.0, ds * dth / 90.0]
    return out


def icc(correct: np.ndarray, episodes: np.ndarray) -> float:
    """일원 ICC(1) — 맞힘(0/1)이 에피소드 안에서 얼마나 몰리나(1 이면 에피소드당 한 표와 같다)."""
    groups = [correct[episodes == e].astype(float) for e in np.unique(episodes)]
    n = np.array([len(g) for g in groups])
    k, grand = len(groups), correct.mean()
    msb = sum(len(g) * (g.mean() - grand) ** 2 for g in groups) / (k - 1)
    msw = sum(((g - g.mean()) ** 2).sum() for g in groups) / (n.sum() - k)
    n0 = (n.sum() - (n ** 2).sum() / n.sum()) / (k - 1)
    den = msb + (n0 - 1) * msw
    return float((msb - msw) / den) if den > 0 else 0.0


def run(dataset: str, rule: dict[str, Any] | None = None) -> dict[str, Any]:
    from .. import datasets
    from .linear_probe import FOLDS, ProbeSet, _public
    from .stats import finalize, judge_cell, rule_for_dataset
    from .train_cli import _sub, caption_fields

    rule = {**ODOR_WINDOWS_RULE, **(rule or {})}
    STATS_RULE = rule_for_dataset(dataset)
    brains = ["real", *datasets.rewired_brains(dataset)]
    ch, fld = rule["channel"], rule["field"]
    raw: dict[str, dict[str, dict[str, Any]]] = {}
    out: dict[str, Any] = {"dataset": dataset, "when": time.strftime("%Y-%m-%d %H:%M:%S"),
                           "rule": rule, "stats": STATS_RULE, "rewired_brains": brains[1:],
                           "method": f"선형(능선 회귀) · 규제 {FOLDS}겹 교차검증", "brains": {}}
    cut = rule.get("max_decision")   # task-v4 냄새 맡기 구간만(결정 < cut) · 창은 자르기 전에
    keep_te = np.array([cut is None or r["decision"] < cut
                        for r in datasets.load(dataset, "eval", "real")[1]])
    episodes = np.array([r["seed"] for r in datasets.load(dataset, "eval", "real")[1]])[keep_te]
    for brain in brains:
        ftr, rtr = datasets.load(dataset, "train", brain)
        fte, rte = datasets.load(dataset, "eval", brain)
        xtr, xte = ftr[ch].astype(np.float64), fte[ch].astype(np.float64)
        ytr = [_sub(caption_fields(r["caption"]))[fld] for r in rtr]
        yte = [_sub(caption_fields(r["caption"]))[fld] for r in rte]
        dtr, dte = delta_features(xtr, rtr), delta_features(xte, rte)
        sets: dict[str, tuple[np.ndarray, np.ndarray]] = {"delta": (dtr, dte)}
        for k in rule["windows"]:
            wtr, wte = window_mean(xtr, rtr, k), window_mean(xte, rte, k)
            sets[f"win{k}"] = (wtr, wte)
            sets[f"win{k}+delta"] = (np.hstack([wtr, dtr]), np.hstack([wte, dte]))
        ktr = np.array([cut is None or r["decision"] < cut for r in rtr])
        ytr = [y for y, k in zip(ytr, ktr) if k]
        yte = [y for y, k in zip(yte, keep_te) if k]
        raw[brain] = {name: ProbeSet(a[ktr], b[keep_te]).field(ytr, yte)
                      for name, (a, b) in sets.items()}
        out["brains"][brain] = {name: _public(r) for name, r in raw[brain].items()}
    cells: dict[str, dict[str, Any]] = {}
    for name, r in raw["real"].items():
        c = judge_cell(r["correct"], [raw[b][name]["correct"] for b in brains[1:]],
                       r["baseline_correct"], episodes, STATS_RULE)
        c["icc_real"] = icc(r["correct"], episodes)
        c["judge"] = True
        cells[f"{ch}-{name}|{fld}"] = c
    finalize(cells, STATS_RULE)
    out["cells"] = cells
    return out


def main(args) -> int:
    rule: dict[str, Any] = {"channel": args.channel,
                            "max_decision": args.max_decision or None}
    if args.windows:
        rule["windows"] = [int(k) for k in args.windows.split(",")]
    res = run(args.dataset, rule)
    path = paths.runs_dir() / f"odor-windows-{time.strftime('%Y%m%d-%H%M%S')}.json"
    path.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{res['rule']['version']} · {res['dataset']} · 재배선 {len(res['rewired_brains'])} 개 · "
          f"{res['stats']['version']}(순위 ≥ 95 % · Holm p < 0.05 · 찍기보다 높음)")
    print(f"{'특징|칸':<30}{'진짜':>7}{'재배선':>8}{'찍기':>7}{'순위%':>7}{'p':>8}{'Holm':>8}"
          f"{'찍기p':>8}{'ICC':>6}  판정")
    for key, c in res["cells"].items():
        print(f"{key:<30}{c['real']:>7.3f}{c['rewired']['mean']:>8.3f}{c['baseline']:>7.3f}"
              f"{c['rank_pct']:>7.0f}{c['p_block_perm']:>8.3f}{c['p_holm']:>8.3f}"
              f"{c['p_vs_baseline']:>8.3f}{c['icc_real']:>6.2f}  {'통과' if c['pass'] else '-'}")
    print(f"기록 {path}")
    return 0


def register(sub) -> None:
    p = sub.add_parser("odor-windows",
                       help="냄새 쪽을 여러 결정 누적 · 시간 차로 선형 검사(연구 노트 14 M2 · CPU)")
    p.add_argument("--dataset", default=config.current_dataset())
    p.add_argument("--windows", default="", help="누적 결정 수 목록(기본 1,3,5,10)")
    p.add_argument("--channel", default="smell", help="읽을 채널(channels-v3 의 smell_ln 등)")
    p.add_argument("--max-decision", type=int, default=0,
                   help="결정 번호 < N 만 채점(task-v4 냄새 맡기 구간 = 4) · 0 이면 전부")
    p.set_defaults(func=main)
