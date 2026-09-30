"""2층 빠른 정책 (R-1027 F-002 · P loop §1.3) — 초파리 신호에서 행동을 바로 내는 선형 읽기 장치.

조향 하강 뉴런(DNa02)이 이 모델에서 침묵해(R-7469 M1) 회로 혼자서는 먹이 쪽으로 돌지 못한다.
그 사이를 LLM 없이 메우는 가장 가벼운 장치 — 행렬 곱 한 번.

- 입력: 투영기와 **같은 압축 입력**(관문 통과 채널 · `compress` 판 · 뇌마다 자기 압축기).
- 정답: 교사 행동에서 세 머리 — 돌기(left · none · right) · 멈춤(예 · 아니오)
  · 위험(교사가 위협 회피 분기였나 = 교사 규칙에서 걷기 1.0 · 돌기 세기 1.0 · `teacher.action`).
  참값이 아니라 **학습 정답**으로만 쓴다.
- 방법: 능선 회귀(원초 형태 · 특징 수가 작다) · λ 는 **에피소드 단위** 5 겹 교차검증
  (한 에피소드의 결정은 서로 닮아서).
- 확신도: 돌기 머리의 두 높은 점수 차 · 위험 점수: 위험 열의 예측값(0~1 로 자름).
- 놀람 문턱용: 학습 나눔 에피소드 안 직전 결정 대비 특징 변화량의 분위수(0.50~0.99)를
  meta 에 남긴다.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .. import config, datasets, paths

FAST_RULE: dict[str, Any] = {
    "version": "fast-v1",
    "compress": "compress-v3",
    "lambdas": [1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3, 1e4],
    "folds": 5,
    "seed": 0,
    "quantiles": [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99],
    "walk_straight": 0.8, "walk_turn": 0.6,          # teacher.action 과 같은 값
}
TURNS = ["left", "none", "right"]


def labels(ta: dict) -> tuple[str, bool, bool]:
    """교사 행동 → (돌기, 멈춤, 위험 분기)."""
    threat = float(ta.get("walk", 0.0)) == 1.0 and float(ta.get("turn_strength", 0.0)) == 1.0
    return str(ta.get("turn", "none")), bool(ta.get("stop", False)), bool(threat)


def featurize(feats: dict[str, np.ndarray], names: list[str]) -> np.ndarray:
    """채널 압축 특징을 정해진 순서로 이어 붙인다(한 결정이면 1 차원 · 여럿이면 2 차원)."""
    return np.concatenate([np.asarray(feats[n], dtype=np.float64) for n in names], axis=-1)


def _onehot(y: list, classes: list) -> np.ndarray:
    return np.array([[1.0 if v == c else 0.0 for c in classes] for v in y])


def _ridge(x: np.ndarray, y: np.ndarray, lam: float) -> tuple[np.ndarray, np.ndarray]:
    mu, ym = x.mean(0), y.mean(0)
    xc = x - mu
    w = np.linalg.solve(xc.T @ xc + lam * np.eye(x.shape[1]), xc.T @ (y - ym))
    return w, ym - mu @ w


def _group_folds(groups: np.ndarray, k: int, seed: int) -> list[np.ndarray]:
    uniq = np.unique(groups)
    order = np.random.default_rng(seed).permutation(uniq)
    return [np.isin(groups, part) for part in np.array_split(order, k)]


def _fit_head(x: np.ndarray, y: list, classes: list, groups: np.ndarray,
              rule: dict) -> tuple[np.ndarray, np.ndarray, float]:
    """λ 를 에피소드 단위 교차검증으로 고른 뒤 전체로 다시 맞춘다."""
    yo = _onehot(y, classes)
    best, best_acc = rule["lambdas"][0], -1.0
    folds = _group_folds(groups, rule["folds"], rule["seed"])
    ya = np.array([classes.index(v) for v in y])
    for lam in rule["lambdas"]:
        hit = 0
        for te in folds:
            w, b = _ridge(x[~te], yo[~te], lam)
            hit += int(((x[te] @ w + b).argmax(1) == ya[te]).sum())
        acc = hit / len(y)
        if acc > best_acc:
            best, best_acc = lam, acc
    w, b = _ridge(x, yo, best)
    return w, b, float(best)


@dataclass
class FastOut:
    action: dict
    turn: str
    turn_conf: float
    threat: float
    stop: bool
    threat_action: dict = field(default_factory=dict)


class FastPolicy:
    """선형 세 머리 — `predict(z)` 한 번이 결정 하나."""

    def __init__(self, heads: dict[str, dict[str, np.ndarray]], names: list[str],
                 strength: float, meta: dict[str, Any]) -> None:
        self.heads, self.names, self.strength, self.meta = heads, names, strength, meta

    def _scores(self, head: str, z: np.ndarray) -> np.ndarray:
        h = self.heads[head]
        return z @ h["w"] + h["b"]

    def predict(self, z: np.ndarray) -> FastOut:
        z = np.asarray(z, dtype=np.float64)
        st = self._scores("turn", z)
        order = np.argsort(st)[::-1]
        turn = TURNS[int(order[0])]
        conf = float(st[order[0]] - st[order[1]])
        stop = bool(self._scores("stop", z).argmax() == 1)
        threat = float(np.clip(self._scores("threat", z)[1], 0.0, 1.0))
        r = FAST_RULE
        if stop:
            action = {"walk": 0.0, "turn": "none", "turn_strength": 0.0, "stop": True,
                      "groom": False}
        elif turn == "none":
            action = {"walk": r["walk_straight"], "turn": "none", "turn_strength": 0.0,
                      "stop": False, "groom": False}
        else:
            action = {"walk": r["walk_turn"], "turn": turn, "turn_strength": self.strength,
                      "stop": False, "groom": False}
        away = turn if turn != "none" else "left"
        threat_action = {"walk": 1.0, "turn": away, "turn_strength": 1.0, "stop": False,
                         "groom": False}
        return FastOut(action, turn, conf, threat, stop, threat_action)

    def save(self, out: Path) -> None:
        out.mkdir(parents=True, exist_ok=True)
        np.savez(out / "fast.npz", **{f"{h}.{k}": v for h, d in self.heads.items()
                                      for k, v in d.items()})
        (out / "meta.json").write_text(json.dumps(
            {**self.meta, "names": self.names, "strength": self.strength},
            ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, out: Path) -> FastPolicy:
        meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
        z = np.load(out / "fast.npz")
        heads: dict[str, dict[str, np.ndarray]] = {}
        for key in z.files:
            h, k = key.split(".", 1)
            heads.setdefault(h, {})[k] = z[key]
        return cls(heads, meta["names"], float(meta["strength"]), meta)


def out_dir(dataset: str, brain: str, version: str | None = None) -> Path:
    return paths.runs_dir() / f"fast-policy-{dataset}-{brain}-{version or FAST_RULE['version']}"


def _compressed(dataset: str, split: str, brain: str, rule: dict):
    from . import compress

    gate = json.loads((compress.out_dir(dataset, version=rule["compress"]) / "gate.json")
                      .read_text(encoding="utf-8"))
    names = list(gate["passed_channels"])
    model = compress.load(dataset, brain, version=rule["compress"])
    feats, rows = datasets.load(dataset, split, brain)
    squeezed = compress.apply({c: model[c] for c in names}, feats)
    return featurize(squeezed, names), rows, names


def _delta_quantiles(x: np.ndarray, rows: list[dict], qs: list[float]) -> dict[str, float]:
    eps: dict[int, list[int]] = {}
    for i, r in enumerate(rows):
        eps.setdefault(r["seed"], []).append(i)
    d = []
    for ix in eps.values():
        ix = sorted(ix, key=lambda i: rows[i]["decision"])
        d += [float(np.linalg.norm(x[b] - x[a])) for a, b in zip(ix, ix[1:])]
    return {f"{q:.2f}": float(np.quantile(d, q)) for q in qs}


def fit(dataset: str, brain: str, rule: dict | None = None) -> tuple[FastPolicy, dict]:
    """학습 나눔으로 맞추고 평가 나눔으로 잰다(평가는 보고용 · 문턱에 쓰지 않는다)."""
    rule = {**FAST_RULE, **(rule or {})}
    x, rows, names = _compressed(dataset, "train", brain, rule)
    lab = [labels(r["teacher_action"]) for r in rows]
    groups = np.array([r["seed"] for r in rows])
    heads, lams = {}, {}
    for h, i, classes in (("turn", 0, TURNS), ("stop", 1, [False, True]),
                          ("threat", 2, [False, True])):
        w, b, lam = _fit_head(x, [t[i] for t in lab], classes, groups, rule)
        heads[h], lams[h] = {"w": w, "b": b}, lam
    strengths = [float(r["teacher_action"].get("turn_strength", 0.0)) for r, t in zip(rows, lab)
                 if t[0] != "none" and not t[2]]
    meta = {"rule": rule, "dataset": dataset, "brain": brain, "lambda": lams,
            "train_samples": len(rows), "delta_quantiles": _delta_quantiles(x, rows,
                                                                            rule["quantiles"])}
    pol = FastPolicy(heads, names, float(np.median(strengths)) if strengths else 0.5, meta)
    return pol, evaluate(pol, dataset, brain, rule)


def evaluate(pol: FastPolicy, dataset: str, brain: str, rule: dict) -> dict:
    """평가 나눔 — 머리마다 정확도 · 찍기 · 돌기 확신도 구간별 정확도 · 추론 시간."""
    x, rows, _ = _compressed(dataset, "eval", brain, rule)
    lab = [labels(r["teacher_action"]) for r in rows]
    outs = []
    t0 = time.perf_counter()
    for z in x:
        outs.append(pol.predict(z))
    per_ms = 1000.0 * (time.perf_counter() - t0) / max(len(x), 1)

    def acc(pred: list, truth: list) -> dict:
        maj = max(set(truth), key=truth.count)
        return {"accuracy": round(float(np.mean([p == t for p, t in zip(pred, truth)])), 4),
                "baseline": round(float(np.mean([t == maj for t in truth])), 4)}

    turns = [t[0] for t in lab]
    conf = np.array([o.turn_conf for o in outs])
    ok = np.array([o.turn == t for o, t in zip(outs, turns)])
    bins = np.quantile(conf, [0, 0.25, 0.5, 0.75, 1.0])
    calib = [{"conf_from": round(float(lo), 4), "conf_to": round(float(hi), 4),
              "accuracy": round(float(ok[(conf >= lo) & (conf <= hi)].mean()), 4)}
             for lo, hi in zip(bins, bins[1:])]
    return {"turn": acc([o.turn for o in outs], turns),
            "stop": acc([o.stop for o in outs], [t[1] for t in lab]),
            "threat": acc([o.threat >= 0.5 for o in outs], [t[2] for t in lab]),
            "turn_calibration": calib, "predict_ms": round(per_ms, 4),
            "eval_samples": len(rows)}


def main_fit(args: argparse.Namespace) -> int:
    brains = [b for b in args.brains.split(",") if b]
    for brain in brains:
        pol, rep = fit(args.dataset, brain)
        out = out_dir(args.dataset, brain)
        pol.save(out)
        (out / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2),
                                         encoding="utf-8")
        print(f"[fast-policy] {brain:11} 돌기 {rep['turn']['accuracy']:.3f}/찍기 "
              f"{rep['turn']['baseline']:.3f} · 멈춤 {rep['stop']['accuracy']:.3f}/"
              f"{rep['stop']['baseline']:.3f} · 위험 {rep['threat']['accuracy']:.3f}/"
              f"{rep['threat']['baseline']:.3f} · 결정당 {rep['predict_ms']:.3f} ms · {out}",
              flush=True)
    return 0


def register(sub) -> None:
    p = sub.add_parser("fast-policy",
                       help="2층 빠른 정책(R-1027) — 초파리 신호 → 행동 선형 읽기 장치(CPU)")
    s = p.add_subparsers(dest="fp_cmd", required=True)
    f = s.add_parser("fit", help="학습 나눔으로 맞추고 평가 나눔으로 잰다")
    f.add_argument("--dataset", default=config.current_dataset())
    f.add_argument("--brains", default="real,rewired-s0,rewired-s1,rewired-s2")
    f.set_defaults(func=main_fit)
