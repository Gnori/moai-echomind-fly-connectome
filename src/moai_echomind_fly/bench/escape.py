"""도망 반사 채점 (R-3340 F-004): 다가옴 → LC/LPLC → 거대섬유(GF = DNp01).

채점 기준은 생리 기록에서 온 것이다(von Reyn et al. 2014 Nat Neurosci 17:962 등) —
IONOFIELD/FLYCNS 의 benchmarks/gf_escape.py 머리말이 정리한 기준을 원 출처와 함께
옮겼다(코드는 쓰지 않음).
TTMn 1:1 항목은 전기 시냅스 모형이 있어야 해서 여기에는 없다(F-006).
"""

from __future__ import annotations

import argparse
import math
from dataclasses import asdict, dataclass

import numpy as np

from .. import config
from .. import graph as graph_mod
from ..guard import GuardRefused, heavy
from ..protocols import LoomSpec, run_loom
from ..runs import RunRecorder
from ..sim import Engine, LIFParams

WINDOW_AFTER_BURST_MS = 100.0   # FLYCNS score(): burst_ms + window_ms(100)

CRITERIA = {
    "gf_spikes_per_response_1_to_2":
        "반응한 시행에서 반응한 GF 한 개당 발화 1~2 회 (von Reyn 2014: 단일 GF 기록)",
    "gf_response_prob_0.5_to_1": "GF 반응 확률 0.5~1.0 (다가옴 대부분이 문턱 아래 — von Reyn 2014)",
    "gf_latency_10_60ms": "첫 GF 발화 잠복 10~60 ms",
    "cns_quiet": "자극받지 않은 뉴런 전체 평균 발화율 < 0.05 Hz",
    "null_silent": "섞은 배선 대조군의 GF 반응 확률 < 0.1",
}


@dataclass
class ArmResult:
    arm: str
    graph: str
    trials: int
    gf_hit: float
    gf_hit_ci95: tuple[float, float]
    gf_per_hit: float
    gf_latency_ms: float
    pop_rate_hz: float
    total_spikes: int
    stim_neurons: int
    missing_types: list[str]
    wall_s: float
    run_id: str


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, (c - h) / d), min(1.0, (c + h) / d))


def score_trials(step: np.ndarray, neuron: np.ndarray, dt_ms: float, gf: np.ndarray,
                 onsets: list[float], burst_ms: float) -> dict[str, float]:
    t_ms = step * dt_ms
    is_gf = np.isin(neuron, gf)
    hits, per_hit, lats = 0, [], []
    for on in onsets:
        m = is_gf & (t_ms >= on) & (t_ms < on + burst_ms + WINDOW_AFTER_BURST_MS)
        if m.any():
            hits += 1
            per_hit.append(m.sum() / len(np.unique(neuron[m])))
            lats.append(float(t_ms[m].min() - on))
    n = len(onsets)
    return {
        "hits": hits,
        "gf_hit": hits / n if n else 0.0,
        "gf_per_hit": float(np.mean(per_hit)) if per_hit else 0.0,
        "gf_latency_ms": float(np.mean(lats)) if lats else float("nan"),
    }


def run_arm(arm: str, g: graph_mod.Graph, params: LIFParams, spec: LoomSpec, args) -> ArmResult:
    rec = RunRecorder("escape", {"arm": arm, "graph": g.name, "trials": args.trials,
                                 "seed": args.seed, "device": args.device,
                                 "w_scale": params.w_scale, "peak_hz": spec.peak_hz})
    eng = Engine(g, params, device=args.device, seed=args.seed)
    rec.record_setup(eng, g, {"sim": params.version, "loom": spec.version})
    gf = g.select(type="DNp01")
    trials = run_loom(eng, spec, args.trials, seed=args.seed)
    step, neuron = eng.spikes()
    s = score_trials(step, neuron, params.dt_ms, gf, trials.onsets_ms, spec.burst_ms)
    stim = np.zeros(g.n, bool)
    stim[trials.stim_idx] = True
    nonstim_spikes = int((~stim[neuron]).sum())
    pop = nonstim_spikes / (g.n - stim.sum()) / (eng.t_ms / 1000.0)
    result = ArmResult(
        arm=arm, graph=g.name, trials=args.trials, gf_hit=s["gf_hit"],
        gf_hit_ci95=wilson(int(s["hits"]), args.trials), gf_per_hit=s["gf_per_hit"],
        gf_latency_ms=s["gf_latency_ms"], pop_rate_hz=pop, total_spikes=int(len(step)),
        stim_neurons=int(stim.sum()), missing_types=trials.missing_types, wall_s=rec.wall_s,
        run_id=rec.run_id)
    rec.record_spikes(eng, onsets_ms=np.array(trials.onsets_ms), gains=np.array(trials.gains))
    rec.write_json("summary.json", asdict(result))
    rec.honesty(
        wiring=["자극 뉴런(LC/LPLC) → GF 까지의 신호 전파 · GF 발화 여부 · 잠복"],
        added=[f"자극 = LC/LPLC 13종 포아송(상대 세기 Turner 2022 · "
               f"척도 peak_hz={spec.peak_hz} 선언된 자유 매개변수)",
               f"시냅스 세기 배율 w_scale={params.w_scale} ({params.version})",
               *[f"적응 {a}" for a in params.adaptation]])
    rec.log(f"[{arm}] GF 반응 {result.gf_hit:.2f} (95% {result.gf_hit_ci95[0]:.2f}–"
            f"{result.gf_hit_ci95[1]:.2f}) · 반응당 {result.gf_per_hit:.2f} · 잠복 "
            f"{result.gf_latency_ms:.1f} ms · 뇌 전체 {pop:.4f} Hz · 발화 {len(step):,} · "
            f"{result.wall_s:.0f} s · {rec.run_id}")
    return result


def replay(run_dir) -> tuple[np.ndarray, np.ndarray]:
    """F-007: 기록(request.json · params.json · graph.json)만 보고 같은 실행을 다시 돌린다."""
    import json
    from pathlib import Path

    run_dir = Path(run_dir)
    req = json.loads((run_dir / "request.json").read_text())
    prm = json.loads((run_dir / "params.json").read_text())
    gname = json.loads((run_dir / "graph.json").read_text())["name"]
    lif = dict(prm["lif"])
    lif["adaptation"] = tuple(lif.get("adaptation") or ())
    params = LIFParams(**lif)
    spec = LoomSpec.from_config(peak_hz=req["peak_hz"])
    if spec.version != prm["versions"]["loom"]:
        raise ValueError(f"loom 판이 다르다: 기록 {prm['versions']['loom']} · 지금 {spec.version}")
    base, _, seed = gname.partition(".rewired-s")
    g = graph_mod.load(base)
    if seed:
        g = graph_mod.rewire(g, int(seed))
    eng = Engine(g, params, device=req["device"], seed=prm["seed"])
    run_loom(eng, spec, req["trials"], seed=req["seed"])
    return eng.spikes()


def checks(real: ArmResult, null: ArmResult | None) -> dict[str, bool | None]:
    return {
        "gf_spikes_per_response_1_to_2": 1.0 <= real.gf_per_hit <= 2.0,
        "gf_response_prob_0.5_to_1": 0.5 <= real.gf_hit <= 1.0,
        "gf_latency_10_60ms": 10.0 <= real.gf_latency_ms <= 60.0,
        "cns_quiet": real.pop_rate_hz < 0.05,
        "null_silent": (null.gf_hit < 0.1) if null else None,
    }


# w_scale 맞추기 규칙 — 결과를 보기 전에 적어 둔 것(2026-09-22 · R-3340 1.3-4). 바꾸면 판을 올린다.
FIT_RULE = {
    "version": "fit-v1",
    "w_scale_grid": [0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.75, 1.0],
    "peak_hz": 5.0,
    "trials": 20,
    "pick": "실제 배선 네 항목(발화 1~2 · 반응 0.5~1 · 잠복 10~60 ms · 뇌 전체 < 0.05 Hz)을 "
            "모두 통과한 값 중 이어진 통과 구간의 가운데 값(둘이면 작은 쪽). "
            "고른 값은 40 시행 + 섞은 배선 대조로 확인 뒤 sim-v1 로 잠근다.",
}


def fit(args: argparse.Namespace) -> int:
    spec = LoomSpec.from_config(peak_hz=FIT_RULE["peak_hz"])
    args.trials = FIT_RULE["trials"]
    rows = []
    try:
        with heavy("bench-fit") as guard:
            print(f"[guard] 메모리 여유 {guard.free_percent}% · 규칙 {FIT_RULE['version']}")
            g = graph_mod.load(args.graph)
            for w in FIT_RULE["w_scale_grid"]:
                params = LIFParams.from_config(w_scale=w)
                r = run_arm(f"fit-w{w}", g, params, spec, args)
                c = checks(r, None)
                passed = all(v for k, v in c.items() if k != "null_silent")
                rows.append({"w_scale": w, "passed": passed, **{k: v for k, v in c.items()
                                                                if k != "null_silent"},
                             "gf_hit": r.gf_hit, "gf_per_hit": r.gf_per_hit,
                             "gf_latency_ms": r.gf_latency_ms, "pop_rate_hz": r.pop_rate_hz,
                             "run_id": r.run_id})
    except GuardRefused as err:
        print(f"[guard] 막음 — {err}")
        return 2
    ok = [r["w_scale"] for r in rows if r["passed"]]
    runs_: list[list[float]] = []
    for r in rows:
        if r["passed"] and runs_ and runs_[-1][-1] == rows[rows.index(r) - 1]["w_scale"] \
                and rows[rows.index(r) - 1]["passed"]:
            runs_[-1].append(r["w_scale"])
        elif r["passed"]:
            runs_.append([r["w_scale"]])
    best = max(runs_, key=len) if runs_ else []
    pick = best[(len(best) - 1) // 2] if best else None
    print("\nw_scale | 통과 | GF 반응 | 반응당 | 잠복 ms | 뇌 전체 Hz")
    for r in rows:
        print(f"{r['w_scale']:7} | {'O' if r['passed'] else 'X':4} | {r['gf_hit']:.2f} | "
              f"{r['gf_per_hit']:.2f} | {r['gf_latency_ms']:.1f} | {r['pop_rate_hz']:.4f}")
    print(f"\n통과: {ok} · 가장 긴 이어진 구간: {best} · 고른 값: {pick}")
    return 0 if pick is not None else 1


def main(args: argparse.Namespace) -> int:
    params = LIFParams.from_config(**({"w_scale": args.w_scale} if args.w_scale else {}))
    spec = LoomSpec.from_config(peak_hz=args.peak_hz)
    try:
        with heavy("bench-escape") as guard:
            print(f"[guard] 메모리 여유 {guard.free_percent}% ≥ {guard.threshold}% · "
                  f"잠금 {guard.lock_file}")
            real = run_arm("real", graph_mod.load(args.graph), params, spec, args)
            null = None
            if not args.no_null:
                g = graph_mod.load(args.graph)
                rewired = graph_mod.rewire(g, args.rewired_seed)
                null = run_arm("rewired", rewired, params, spec, args)
    except GuardRefused as err:
        print(f"[guard] 막음 — {err}")
        return 2
    result = checks(real, null)
    print("\n채점 (기준: " + config.load("sim")["version"] + " · " + spec.version + ")")
    for key, ok in result.items():
        mark = "통과" if ok else ("건너뜀" if ok is None else "실패")
        print(f"  {mark:4}  {key} — {CRITERIA[key]}")
    return 0 if all(v is not False for v in result.values()) else 1
