"""신호에 정보가 있나 — Qwen 없이 채널 벡터만으로 캡션 칸을 맞혀 본다 (R-3340 정직 목록).

캡션 평가가 찍기 수준일 때 **어디가 문제인지** 가른다:

| 이 검사 | 뜻 |
|---|---|
| 찍기보다 높다 | 신호에는 정보가 있다 → 투영기·학습이 Qwen 에 전달하지 못한 것 |
| 찍기와 같다 | 신호(또는 그 특징 만드는 법)에 그 정보가 없다 → 앞쪽을 고쳐야 한다 |

분류기는 선형(능선 회귀 · 이중 형태)이다. 차원(3.2만)이 표본(581)보다 크므로 581×581 행렬만 푼다.
규제 세기는 학습 자료 안에서 5겹 교차검증으로 고르고, 정확도는 학습에 안 쓴 씨앗(eval)에서 잰다.
CPU·수 초. 진짜 배선과 섞은 배선을 같은 방법으로 잰다.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from typing import Any

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
    mean = xtr.mean(0)                     # 중심화 — 안 하면 같은 자료에서도 크게 떨어진다(실측)
    xtr, xte = xtr - mean, xte - mean
    classes = sorted(set(ytr) | set(yte))
    if len(classes) == 1:
        return {"accuracy": 1.0, "baseline": 1.0, "lambda": None, "classes": 1}
    lam = _cv_lambda(xtr, ytr, classes)
    pred = _fit_predict(xtr, _onehot(ytr, classes), xte, lam).argmax(1)
    top = Counter(ytr).most_common(1)[0][0]
    return {"accuracy": float(np.mean([classes[j] == t for j, t in zip(pred, yte)])),
            "baseline": float(np.mean([t == top for t in yte])),
            "lambda": lam, "classes": len(classes)}


def run(name: str) -> dict[str, Any]:
    from .. import datasets
    from .train_cli import _sub, caption_fields

    out: dict[str, Any] = {"dataset": name, "when": time.strftime("%Y-%m-%d %H:%M:%S"),
                           "method": f"선형(능선 회귀) · 규제 {FOLDS}겹 교차검증", "brains": {}}
    for brain in ("real", "rewired"):
        ftr, rtr = datasets.load(name, "train", brain)
        fte, rte = datasets.load(name, "eval", brain)
        names = list(ftr)
        xtr = np.concatenate([ftr[n] for n in names], axis=1).astype(np.float64)
        xte = np.concatenate([fte[n] for n in names], axis=1).astype(np.float64)
        ytr = [_sub(caption_fields(r["caption"])) for r in rtr]
        yte = [_sub(caption_fields(r["caption"])) for r in rte]
        out["brains"][brain] = {
            "samples": {"train": len(rtr), "eval": len(rte)}, "dims": int(xtr.shape[1]),
            "fields": {f: probe_field(xtr, [y[f] for y in ytr], xte, [y[f] for y in yte])
                       for f in yte[0]},
        }
    return out


def main(args) -> int:
    res = run(args.dataset)
    path = paths.runs_dir() / f"linear-probe-{time.strftime('%Y%m%d-%H%M%S')}.json"
    path.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    fields = list(res["brains"]["real"]["fields"])
    print(f"{'칸':<14}{'진짜 배선':<12}{'섞은 배선':<12}찍기")
    for f in fields:
        r, w = (res["brains"][b]["fields"][f] for b in ("real", "rewired"))
        print(f"{f:<14}{r['accuracy']:<12.3f}{w['accuracy']:<12.3f}{r['baseline']:.3f}")
    print(f"기록 {path}")
    return 0


def register(sub) -> None:
    p = sub.add_parser("linear-probe",
                       help="Qwen 없이 채널 벡터만으로 캡션 칸 맞히기 — 신호에 정보가 있나(CPU)")
    p.add_argument("--dataset", default="fly-loop-v1")
    p.set_defaults(func=main)
