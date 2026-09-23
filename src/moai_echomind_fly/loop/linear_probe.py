"""신호에 정보가 있나 — Qwen 없이 채널 벡터만으로 캡션 칸을 맞혀 본다 (R-3340 정직 목록).

캡션 평가가 찍기 수준일 때 **어디가 문제인지** 가른다:

| 이 검사 | 뜻 |
|---|---|
| 찍기보다 높다 | 신호에는 정보가 있다 → 투영기·학습이 Qwen 에 전달하지 못한 것 |
| 찍기와 같다 | 신호(또는 그 특징 만드는 법)에 그 정보가 없다 → 앞쪽을 고쳐야 한다 |

분류기는 선형(능선 회귀 · 이중 형태)이다. 차원(3.2만)이 표본(581)보다 크므로 581×581 행렬만 푼다.
규제 세기는 학습 자료 안에서 5겹 교차검증으로 고르고, 정확도는 학습에 안 쓴 씨앗(eval)에서 잰다.
CPU·수 초. 진짜 배선과 섞은 배선을 같은 방법으로 잰다.
채널 전부를 한꺼번에 넣은 결과(`fields`)와 **채널 하나씩** 넣은 결과(`per_channel`)를 함께 남긴다 —
한꺼번에 넣으면 묻히는 차이가 채널 하나에서는 보일 수 있다
(2026-09-23 · 그전에는 채널별 값이 W 에만 있었다).
"""

from __future__ import annotations

import json
import time
from collections import Counter
from typing import Any, Callable

import numpy as np

from .. import paths

LAMBDAS = (1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3, 1e4)
FOLDS = 5


def _fit_predict(xtr: np.ndarray, ytr: np.ndarray, xte: np.ndarray, lam: float) -> np.ndarray:
    """능선 회귀 이중 형태 — 한쪽 열이 한 종류인 목표(one-hot)에 대해 점수를 낸다."""
    k = xtr @ xtr.T
    alpha = np.linalg.solve(k + lam * np.eye(len(k)), ytr)
    return (xte @ xtr.T) @ alpha


def _onehot(labels: list[str], classes: list[str]) -> np.ndarray:
    y = np.zeros((len(labels), len(classes)), dtype=np.float64)
    for i, v in enumerate(labels):
        y[i, classes.index(v)] = 1.0
    return y


def _cv_lambda(x: np.ndarray, labels: list[str], classes: list[str], seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(labels))
    folds = np.array_split(order, FOLDS)
    best, best_acc = LAMBDAS[0], -1.0
    for lam in LAMBDAS:
        hit = 0
        for f in folds:
            tr = np.setdiff1d(order, f)
            pred = _fit_predict(x[tr], _onehot([labels[i] for i in tr], classes), x[f], lam)
            hit += sum(classes[j] == labels[i] for i, j in zip(f, pred.argmax(1)))
        acc = hit / len(labels)
        if acc > best_acc:
            best, best_acc = lam, acc
    return best


def probe_field(xtr: np.ndarray, ytr: list[str], xte: np.ndarray, yte: list[str]) -> dict[str, Any]:
    """칸 하나의 선형 검사. 판정 통계를 위해 평가 결정마다 맞혔나(`correct`)도 돌려준다."""
    mean = xtr.mean(0)                     # 중심화 — 안 하면 같은 자료에서도 크게 떨어진다(실측)
    xtr, xte = xtr - mean, xte - mean
    classes = sorted(set(ytr) | set(yte))
    top = Counter(ytr).most_common(1)[0][0]
    base = np.array([t == top for t in yte])
    if len(classes) == 1:
        return {"accuracy": 1.0, "baseline": 1.0, "lambda": None, "classes": 1,
                "correct": np.ones(len(yte), bool), "baseline_correct": base}
    lam = _cv_lambda(xtr, ytr, classes)
    pred = _fit_predict(xtr, _onehot(ytr, classes), xte, lam).argmax(1)
    correct = np.array([classes[j] == t for j, t in zip(pred, yte)])
    return {"accuracy": float(correct.mean()), "baseline": float(base.mean()),
            "lambda": lam, "classes": len(classes), "correct": correct, "baseline_correct": base}


def _public(res: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in res.items() if k not in ("correct", "baseline_correct")}


def run(name: str, rule: dict[str, Any] | None = None,
        transform: Callable[[str, dict[str, np.ndarray]], dict[str, np.ndarray]] | None = None,
        ) -> dict[str, Any]:
    """진짜 배선과 묶음에 든 **모든 재배선 뇌**를 같은 방법으로 재고, 칸마다 판정(stats).

    `transform(brain, feats)` 가 있으면 특징을 바꾼 뒤 잰다(F-004 압축 관문).
    """
    from .. import datasets
    from .stats import STATS_RULE, finalize, judge_cell
    from .train_cli import _sub, caption_fields

    rule = rule or STATS_RULE
    brains = ["real", *datasets.rewired_brains(name)]
    out: dict[str, Any] = {"dataset": name, "when": time.strftime("%Y-%m-%d %H:%M:%S"),
                           "method": f"선형(능선 회귀) · 규제 {FOLDS}겹 교차검증",
                           "stats": rule, "rewired_brains": brains[1:], "brains": {}}
    raw: dict[str, dict[str, dict[str, Any]]] = {}      # 뇌 → "채널|칸" → 결과(배열 포함)
    episodes = None
    for brain in brains:
        ftr, rtr = datasets.load(name, "train", brain)
        fte, rte = datasets.load(name, "eval", brain)
        if transform is not None:
            ftr, fte = transform(brain, ftr), transform(brain, fte)
        episodes = np.array([r["seed"] for r in rte])
        names = list(ftr)
        ytr = [_sub(caption_fields(r["caption"])) for r in rtr]
        yte = [_sub(caption_fields(r["caption"])) for r in rte]

        def fields(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
            return {f: probe_field(a, [y[f] for y in ytr], b, [y[f] for y in yte])
                    for f in yte[0]}

        xtr = np.concatenate([ftr[n] for n in names], axis=1).astype(np.float64)
        xte = np.concatenate([fte[n] for n in names], axis=1).astype(np.float64)
        per = {"all": fields(xtr, xte)}
        for n in names:
            per[n] = fields(ftr[n].astype(np.float64), fte[n].astype(np.float64))
        raw[brain] = {f"{ch}|{f}": r for ch, fs in per.items() for f, r in fs.items()}
        out["brains"][brain] = {
            "samples": {"train": len(rtr), "eval": len(rte)}, "dims": int(xtr.shape[1]),
            "fields": {f: _public(r) for f, r in per["all"].items()},
            "per_channel": {n: {"dims": int(ftr[n].shape[1]),
                                "fields": {f: _public(r) for f, r in per[n].items()}}
                            for n in names},
        }
    excluded = set(rule["exclude_from_judgement"])
    cells: dict[str, dict[str, Any]] = {}
    for key, r in raw["real"].items():
        if r["classes"] == 1:
            continue
        ch = key.split("|", 1)[0]
        cell = judge_cell(r["correct"], [raw[b][key]["correct"] for b in brains[1:]],
                          r["baseline_correct"], episodes, rule)
        # 전 채널("all")은 판정에서 뺀 채널(운동)을 품고 있어 판정에 넣지 않는다
        field_out = key.split("|", 1)[1] in rule.get("exclude_fields_from_judgement", [])
        cell["judge"] = ch not in excluded and ch != "all" and not field_out
        cells[key] = cell
    finalize(cells, rule)
    out["cells"] = cells
    return out


def main(args) -> int:
    from .stats import STATS_RULE

    rule = {**STATS_RULE, **({"permutations": args.permutations} if args.permutations else {})}
    res = run(args.dataset, rule)
    path = paths.runs_dir() / f"linear-probe-{time.strftime('%Y%m%d-%H%M%S')}.json"
    path.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    n_rew = len(res["rewired_brains"])
    print(f"재배선 {n_rew} 개 · {res['stats']['version']} · 판정 = 앙상블 순위 ≥ "
          f"{rule['rank_pct']:.0f} % · Holm p < {rule['alpha']} · 찍기보다 높음")
    print(f"{'채널|칸':<26}{'진짜':>7}{'재배선':>8}{'찍기':>7}{'순위%':>7}"
          f"{'p':>8}{'Holm':>8}  판정")
    for key, c in res["cells"].items():
        if args.judge_only and not c["judge"]:
            continue
        holm_p = "-" if c["p_holm"] is None else f"{c['p_holm']:.3f}"
        verdict = {True: "통과", False: "-", None: "(판정 밖)"}[c["pass"]]
        print(f"{key:<26}{c['real']:>7.3f}{c['rewired']['mean']:>8.3f}{c['baseline']:>7.3f}"
              f"{c['rank_pct']:>7.0f}{c['p_block_perm']:>8.3f}{holm_p:>8}  {verdict}")
    passed = [k for k, c in res["cells"].items() if c["pass"]]
    print(f"통과 {len(passed)} 칸: {passed}")
    print(f"기록 {path}")
    return 0


def register(sub) -> None:
    p = sub.add_parser("linear-probe",
                       help="Qwen 없이 채널 벡터만으로 캡션 칸 맞히기 — 신호에 정보가 있나(CPU)")
    p.add_argument("--dataset", default="fly-loop-v1")
    p.add_argument("--judge-only", action="store_true", help="판정 대상 칸만 출력")
    p.add_argument("--permutations", type=int, default=0,
                   help="에피소드가 많을 때 몬테카를로 횟수(기본 = stats-v1)")
    p.set_defaults(func=main)
