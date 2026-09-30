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
- fast-v2(2026-10-01 · 연구 노트 15 §5): 학습 나눔을 **에피소드 단위로 맞춤 · 보정**으로 나눈다.
  맞춤 나눔으로 세 머리와 **전방 모형**(ẑ_t = A·z_{t−1} + B·a_{t−1} · 능선)을 맞추고,
  보정 나눔에서 호출 신호 셋(놀람 = 전방 오차² · 애매함 = 돌기 확신도 · 충돌 = 위험 × 먹이 쪽
  확신도)의 분위수를 meta 에 남긴다 → 문턱은 **분위수로** 잠가 뇌마다 같은 절차가 된다.
- `calibrate`: 보정 나눔에서 게이트를 결정 순서대로 흉내 내 문턱 격자마다 호출률 · 교사 일치율
  (부른 결정 = 교사 행동으로 대신 · 그렇게 적음)을 재고, 예산 안에서 가장 나은 조합을 **제안**만
  한다(잠금은 사람이 config/control.yaml 에 적고 커밋 · 연구 노트 15 §5.2).
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
    "version": "fast-v2",
    "compress": "compress-v3",
    "lambdas": [1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3, 1e4],
    "folds": 5,
    "seed": 0,
    "quantiles": [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99],
    "calib_frac": 0.3,                                # 학습 나눔 중 보정 몫(에피소드 단위)
    "walk_straight": 0.8, "walk_turn": 0.6,          # teacher.action 과 같은 값
}
TURNS = ["left", "none", "right"]
SIGNALS = ("surprise", "confidence", "conflict")


def act_vec(a: dict | None) -> np.ndarray:
    """전방 모형의 행동 입력 — 돌기 셋 한 겹 · 멈춤 · 걷기(없으면 0)."""
    if not a:
        return np.zeros(5)
    t = str(a.get("turn", "none"))
    return np.array([t == "left", t == "none", t == "right", bool(a.get("stop", False)),
                     float(a.get("walk", 0.0))], dtype=np.float64)


def split_episodes(seeds: list[int], frac: float, seed: int) -> set[int]:
    """보정 나눔에 갈 에피소드(씨앗) — 에피소드 단위 · 정해진 난수."""
    uniq = sorted(set(seeds))
    k = max(1, int(round(frac * len(uniq))))
    return {int(v) for v in np.random.default_rng(seed).permutation(uniq)[:k]}


def episodes(rows: list[dict]) -> dict[int, list[int]]:
    """씨앗 → 결정 순서의 행 번호."""
    eps: dict[int, list[int]] = {}
    for i, r in enumerate(rows):
        eps.setdefault(int(r["seed"]), []).append(i)
    return {k: sorted(v, key=lambda i: rows[i]["decision"]) for k, v in eps.items()}


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

    @property
    def conflict(self) -> float:
        """충돌 = 위험(회피) × 먹이 쪽 돌기 확신도(접근) — Botvinick 2001 의 곱(노트 15 §5.1)."""
        return 0.0 if self.turn == "none" else self.threat * float(np.clip(self.turn_conf, 0, 1))


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

    def surprise(self, z_prev: np.ndarray | None, a_prev: dict | None, z: np.ndarray) -> float:
        """놀람 = 전방 모형 오차² ‖z_t − (A·z_{t−1} + B·a_{t−1})‖² · 직전이 없으면 0."""
        if z_prev is None or "forward" not in self.heads:
            return 0.0
        h = self.heads["forward"]
        x = np.concatenate([np.asarray(z_prev, dtype=np.float64), act_vec(a_prev)])
        return float(np.sum((np.asarray(z, dtype=np.float64) - (x @ h["w"] + h["b"])) ** 2))

    def quantile(self, signal: str, q: float) -> float:
        """보정 나눔 신호 분위수(meta) — 문턱은 분위수로 잠근다."""
        table = self.meta.get("signal_quantiles", {}).get(signal, {})
        key = f"{float(q):.2f}"
        if key not in table:
            raise ValueError(f"{signal} 분위수 {key} 가 빠른 정책 meta 에 없다 — 판 "
                             f"{self.meta.get('rule', {}).get('version')}")
        return float(table[key])

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


def make_featurizer(pol: FastPolicy, table, brain: str):
    """성적표 · 수집용 — 결정의 채널 벡터 → 빠른 정책 입력(학습 때와 같은 압축 · 같은 순서)."""
    from . import compress

    model = compress.load(pol.meta["dataset"], brain, version=pol.meta["rule"]["compress"])
    names = list(pol.names)
    sub = {c: model[c] for c in names}

    def featurize_vectors(vectors: dict) -> np.ndarray:
        v = vectors[brain]
        raw = {n: v[n].features(table.rate_scale_hz, table.depol_scale_mV)[None] for n in names}
        sq = compress.apply(sub, raw)
        return featurize({n: sq[n][0] for n in names}, names)

    return featurize_vectors


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


def _pairs(rows: list[dict]) -> list[tuple[int, int]]:
    """(직전, 지금) 결정 쌍 — 한 에피소드 안에서만."""
    return [(a, b) for ix in episodes(rows).values() for a, b in zip(ix, ix[1:])]


def _fit_forward(x: np.ndarray, rows: list[dict], rule: dict) -> tuple[dict, float]:
    """전방 모형 ẑ_t = [z_{t−1}, a_{t−1}] → z_t · λ 는 에피소드 단위 교차검증(오차² 최소)."""
    pr = _pairs(rows)
    xin = np.array([np.concatenate([x[a], act_vec(rows[a].get("action"))]) for a, _ in pr])
    yout = np.array([x[b] for _, b in pr])
    groups = np.array([rows[a]["seed"] for a, _ in pr])
    folds = _group_folds(groups, rule["folds"], rule["seed"])
    best, best_err = rule["lambdas"][0], np.inf
    for lam in rule["lambdas"]:
        err = 0.0
        for te in folds:
            w, b = _ridge(xin[~te], yout[~te], lam)
            err += float(np.sum((xin[te] @ w + b - yout[te]) ** 2))
        if err < best_err:
            best, best_err = lam, err
    w, b = _ridge(xin, yout, best)
    return {"w": w, "b": b}, float(best)


def signal_series(pol: FastPolicy, x: np.ndarray, rows: list[dict]) -> dict[str, np.ndarray]:
    """결정마다 호출 신호 셋 + 빠른 정책 출력(행 순서). 놀람의 직전 행동 = 기록된 실행 행동."""
    outs = [pol.predict(z) for z in x]
    sur = np.zeros(len(rows))
    for a, b in _pairs(rows):
        sur[b] = pol.surprise(x[a], rows[a].get("action"), x[b])
    return {"surprise": sur, "confidence": np.array([o.turn_conf for o in outs]),
            "conflict": np.array([o.conflict for o in outs]), "outs": np.array(outs, dtype=object)}


def _signal_quantiles(sig: dict[str, np.ndarray], rows: list[dict],
                      qs: list[float]) -> dict[str, dict[str, float]]:
    first = {ix[0] for ix in episodes(rows).values()}
    keep = np.array([i not in first for i in range(len(rows))])   # 놀람은 첫 결정에 없다
    out = {}
    for name in SIGNALS:
        v = sig[name][keep] if name == "surprise" else sig[name]
        out[name] = {f"{q:.2f}": float(np.quantile(v, q)) for q in qs}
    return out


def fit(dataset: str, brain: str, rule: dict | None = None) -> tuple[FastPolicy, dict]:
    """학습 나눔(맞춤 몫)으로 맞추고 · 보정 몫으로 신호 분위수 · 평가 나눔은 보고용만."""
    rule = {**FAST_RULE, **(rule or {})}
    x_all, rows_all, names = _compressed(dataset, "train", brain, rule)
    calib = split_episodes([r["seed"] for r in rows_all], rule["calib_frac"], rule["seed"])
    is_cal = np.array([int(r["seed"]) in calib for r in rows_all])
    x = x_all[~is_cal]
    rows = [r for r, c in zip(rows_all, is_cal) if not c]
    lab = [labels(r["teacher_action"]) for r in rows]
    groups = np.array([r["seed"] for r in rows])
    heads, lams = {}, {}
    for h, i, classes in (("turn", 0, TURNS), ("stop", 1, [False, True]),
                          ("threat", 2, [False, True])):
        w, b, lam = _fit_head(x, [t[i] for t in lab], classes, groups, rule)
        heads[h], lams[h] = {"w": w, "b": b}, lam
    strengths = [float(r["teacher_action"].get("turn_strength", 0.0)) for r, t in zip(rows, lab)
                 if t[0] != "none" and not t[2]]
    heads["forward"], lams["forward"] = _fit_forward(x, rows, rule)
    meta = {"rule": rule, "dataset": dataset, "brain": brain, "lambda": lams,
            "train_samples": len(rows), "calib_seeds": sorted(calib),
            "calib_samples": int(is_cal.sum())}
    pol = FastPolicy(heads, names, float(np.median(strengths)) if strengths else 0.5, meta)
    rows_cal = [r for r, c in zip(rows_all, is_cal) if c]
    sig = signal_series(pol, x_all[is_cal], rows_cal)
    pol.meta["signal_quantiles"] = _signal_quantiles(sig, rows_cal, rule["quantiles"])
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


def simulate(pol: FastPolicy, rows: list[dict], sig: dict[str, np.ndarray],
             gate_cfg: dict[str, Any], sniff: int) -> dict[str, Any]:
    """게이트를 결정 순서대로 흉내 낸다(보정 나눔 · 오프라인).

    부른 결정의 행동 = 교사 행동으로 대신한다(Qwen 기록이 없어서 · 연구 노트 15 §5.2 ③).
    일치 = (돌기 · 멈춤) 이 교사와 같음. 냄새 맡기 결정은 행동이 STOP 으로 덮여 세지 않는다.
    ⚠️ 특징은 교사가 몬 기록에서 온다 — 실제 성적표에서는 행동이 달라 신호도 달라진다.
    """
    from .tiered import Gate

    gate = Gate(gate_cfg, pol, sniff_decisions=sniff)
    n = calls = agree = fast_ok = 0
    why_n: dict[str, int] = {}
    for seed, ix in episodes(rows).items():
        gate.reset(seed)
        for i in ix:
            d = int(rows[i]["decision"])
            fo = sig["outs"][i]
            why = gate.trigger(fo, d, float(sig["surprise"][i]))
            if d < sniff:
                continue
            n += 1
            t = labels(rows[i]["teacher_action"])[:2]
            ok = (fo.turn, fo.stop) == t
            fast_ok += int(ok)
            if why is not None:
                gate.called(d)
                calls += 1
                agree += 1
                why_n[why] = why_n.get(why, 0) + 1
            else:
                agree += int(ok)
    goal = why_n.get("goal", 0)
    rate_rest = (calls - goal) / max(n - goal, 1)
    fast_acc = fast_ok / max(n, 1)
    # 같은 호출률의 무작위라면(기댓값): 목표 호출 + 나머지 결정에서 rate_rest 만큼 무작위로 고친다
    random_exp = (fast_ok + goal * (1 - fast_acc) + (n - goal) * rate_rest * (1 - fast_acc)) / n
    return {"decisions": n, "calls": calls, "call_rate": round(calls / max(n, 1), 4),
            "agreement": round(agree / max(n, 1), 4), "fast_only": round(fast_acc, 4),
            "random_matched_expected": round(float(random_exp), 4), "triggers": why_n}


def calibrate(dataset: str, brain: str, control: dict[str, Any], sniff: int,
              version: str | None = None) -> dict[str, Any]:
    """문턱 격자(config/control.yaml `calibration`)에서 예산 안 최선 조합을 **제안**한다."""
    import itertools

    pol = FastPolicy.load(out_dir(dataset, brain, version))
    rule = pol.meta["rule"]
    x_all, rows_all, _ = _compressed(dataset, "train", brain, rule)
    cal = set(pol.meta["calib_seeds"])
    keep = [i for i, r in enumerate(rows_all) if int(r["seed"]) in cal]
    rows = [rows_all[i] for i in keep]
    sig = signal_series(pol, x_all[keep], rows)
    c = control["calibration"]
    grid = c["grid"]
    keys = list(grid)
    table = []
    for combo in itertools.product(*(grid[k] for k in keys)):
        th = dict(zip(keys, combo))
        res = simulate(pol, rows, sig, {"mode": "gated", "thresholds": th,
                                        "conditions": control.get("conditions")}, sniff)
        table.append({"thresholds": th, **res})
    ok = [t for t in table if t["call_rate"] <= float(c["budget"])]
    best = (max(ok, key=lambda t: (t["agreement"], -t["call_rate"])) if ok else None)
    return {"dataset": dataset, "brain": brain, "fast_policy": rule["version"],
            "control": control["version"], "budget": c["budget"], "sniff_decisions": sniff,
            "calib_seeds": len(cal), "best": best, "grid_size": len(table),
            "within_budget": len(ok), "table": table,
            "note": "부른 결정 = 교사 행동으로 대신 · 제안만 — 잠금은 control.yaml 에 적고 커밋"}


def main_calibrate(args: argparse.Namespace) -> int:
    control = config.load("control")
    sniff = int(config.load("task")["episode"].get("sniff_decisions", 0))
    for brain in [b for b in args.brains.split(",") if b]:
        rep = calibrate(args.dataset, brain, control, sniff)
        out = out_dir(args.dataset, brain) / f"calibration-{control['version']}.json"
        out.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
        b = rep["best"]
        if b is None:
            print(f"[calibrate] {brain:11} 예산 {rep['budget']} 안의 조합 없음 · {out}")
            continue
        print(f"[calibrate] {brain:11} 호출률 {b['call_rate']:.3f} · 일치 {b['agreement']:.3f} "
              f"(빠른 정책만 {b['fast_only']:.3f} · 같은 호출률 무작위 기댓값 "
              f"{b['random_matched_expected']:.3f}) · {b['thresholds']} · {out}", flush=True)
    return 0


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
    c = s.add_parser("calibrate", help="보정 나눔 문턱 격자 → 예산 안 최선 조합 제안")
    c.add_argument("--dataset", default=config.current_dataset())
    c.add_argument("--brains", default="real")
    c.set_defaults(func=main_calibrate)
