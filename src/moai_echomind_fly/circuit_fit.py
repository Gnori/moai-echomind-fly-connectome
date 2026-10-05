"""회로 학습 1 단계 — 기울기 없음 CMA-ES (R-7469 F-006 · I v2 §7.2 · 착수는 요구자 확인 뒤).

배우는 것: `cellparams.CellParams`(범위 안 종류의 막 시간 상수 · 문턱 · 종류 쌍 배율 α).
θ = [log tau (T) · log(문턱 − 쉬는 전위) (T) · log α (P)] — 경계(config/fit.yaml bounds)로 자른다.
손실 = physiology-v1 의 도망 항목(범위 목표 · 안이면 0) + λ‖θ − θ₀‖². 평가 한 번 = 다가옴 시행을
학습하는 그래프에서 한 번 + 같은 매개변수를 그 그래프를 섞은 배선에 옮겨 한 번(대조 항).

⚠️ 무거운 실행(뇌 전체 LIF × 개체 수 × 세대). `fit-probe` 로 평가 한 번의 시간 · 메모리를
두 시행 수로 재어 무엇에 비례하는지 본 뒤에만 `fit` 을 돌린다(CLAUDE.md 무거운 실행 규약).
CMA-ES 는 Hansen(2016, "The CMA Evolution Strategy: A Tutorial", arXiv:1604.00772)의 식으로
새로 적었다.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import replace
from typing import Any, Callable

import numpy as np

from . import cellparams as cp
from . import config, paths
from .graph import Graph


# ---------------------------------------------------------------- 매개변수 ↔ θ
def bounds_log(cell: cp.CellParams, bounds: dict[str, list[float]]
               ) -> tuple[np.ndarray, np.ndarray]:
    t, p = len(cell.types), len(cell.alpha)

    def rep(key: str, n: int) -> tuple[np.ndarray, np.ndarray]:
        lo, hi = bounds[key]
        return np.full(n, math.log(lo)), np.full(n, math.log(hi))

    parts = [rep("tau_ms", t), rep("v_th_above_rest_mV", t), rep("alpha", p)]
    return np.concatenate([a for a, _ in parts]), np.concatenate([b for _, b in parts])


def to_theta(cell: cp.CellParams, v0_mV: float) -> np.ndarray:
    return np.concatenate([np.log(cell.tau_ms), np.log(cell.v_th_mV - v0_mV),
                           np.log(np.maximum(cell.alpha, 1e-12))])


def from_theta(theta: np.ndarray, cell: cp.CellParams, v0_mV: float,
               lo: np.ndarray, hi: np.ndarray) -> cp.CellParams:
    """θ → 새 CellParams(경계로 자름 · 모양 · 쌍 · pair_id 는 그대로)."""
    th = np.clip(theta, lo, hi)
    t = len(cell.types)
    return replace(cell, tau_ms=np.exp(th[:t]), v_th_mV=v0_mV + np.exp(th[t:2 * t]),
                   alpha=np.exp(th[2 * t:]), meta=dict(cell.meta))


def transfer(cell: cp.CellParams, graph: Graph, pair_dst: list[str] | None,
             groups: list[dict[str, Any]] | None = None) -> cp.CellParams:
    """같은 종류 매개변수를 다른 그래프(섞은 배선)에 옮긴다 — 쌍 이름이 같으면 같은 α, 없으면 1."""
    out = cp.init(graph, list(cell.types), 0.0, 0.0,
                  1 if cell.meta.get("pair_min_synapses") is not None else None,
                  dict(cell.meta), pair_dst=pair_dst, groups=groups)
    out.tau_ms, out.v_th_mV = cell.tau_ms.copy(), cell.v_th_mV.copy()
    alpha = {(int(a), int(b)): float(x) for (a, b), x in zip(cell.pairs, cell.alpha)}
    out.alpha = np.array([alpha.get((int(a), int(b)), 1.0) for a, b in out.pairs])
    return out


# ---------------------------------------------------------------- 손실
def item_loss(value: float, target: list[float], tol: float, nan_tol: float) -> float:
    """범위 목표 [lo, hi] — 안이면 0 · 밖이면 (거리 / 허용폭)² · 측정 없음(NaN)이면 nan_tol²."""
    if value is None or not np.isfinite(value):
        return float(nan_tol ** 2)
    lo, hi = float(target[0]), float(target[1])
    d = lo - value if value < lo else (value - hi if value > hi else 0.0)
    return float((d / tol) ** 2)


def physiology_loss(measures: dict[str, float], scope: str, phys: dict[str, Any],
                    nan_tol: float) -> tuple[float, dict[str, float]]:
    """physiology 판의 항목 중 이 범위 것만(측정 이름 `<묶음>.<칸>` 의 칸으로 찾음).

    없는 측정은 건너뛴다(기록) — 범위마다 평가 함수가 내는 칸이 다르다(escape · mb).
    """
    total, parts = 0.0, {}
    for it in phys["items"]:
        if scope not in it["scope"]:
            continue
        key = it["measure"].split(".", 1)[1]
        if key not in measures:
            continue
        v = float(it["weight"]) * item_loss(measures[key], it["target"], float(it["tolerance"]),
                                            nan_tol)
        parts[it["name"]] = round(v, 6)
        total += v
    return total, parts


# ---------------------------------------------------------------- CMA-ES
class CMAES:
    """(μ/μ_w, λ)-CMA-ES — 순위 하나 · 순위 μ 갱신 · 걸음 크기는 누적 경로 길이(Hansen 2016 식)."""

    def __init__(self, x0: np.ndarray, sigma0: float, popsize: int | None = None,
                 seed: int = 0) -> None:
        n = len(x0)
        self.n, self.mean, self.sigma = n, np.asarray(x0, dtype=np.float64).copy(), float(sigma0)
        self.lam = int(popsize or 4 + math.floor(3 * math.log(n)))
        mu = self.lam // 2
        w = math.log((self.lam + 1) / 2) - np.log(np.arange(1, mu + 1))
        self.w = w / w.sum()
        self.mu, self.mueff = mu, 1.0 / float(np.sum(self.w ** 2))
        me = self.mueff
        self.cc = (4 + me / n) / (n + 4 + 2 * me / n)
        self.cs = (me + 2) / (n + me + 5)
        self.c1 = 2 / ((n + 1.3) ** 2 + me)
        self.cmu = min(1 - self.c1, 2 * (me - 2 + 1 / me) / ((n + 2) ** 2 + me))
        self.damps = 1 + 2 * max(0.0, math.sqrt((me - 1) / (n + 1)) - 1) + self.cs
        self.chin = math.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n * n))
        self.pc, self.ps = np.zeros(n), np.zeros(n)
        self.C = np.eye(n)
        self.B, self.D = np.eye(n), np.ones(n)
        self.gen = 0
        self.rng = np.random.default_rng(seed)
        self._z: np.ndarray | None = None

    def ask(self) -> np.ndarray:
        self._z = self.rng.standard_normal((self.lam, self.n))
        y = self._z @ (self.B * self.D).T
        return self.mean + self.sigma * y

    def tell(self, xs: np.ndarray, fit: np.ndarray) -> None:
        order = np.argsort(fit)[: self.mu]
        y = (xs[order] - self.mean) / self.sigma
        yw = self.w @ y
        self.mean = self.mean + self.sigma * yw
        c_inv_sqrt = self.B @ np.diag(1 / self.D) @ self.B.T
        self.ps = ((1 - self.cs) * self.ps
                   + math.sqrt(self.cs * (2 - self.cs) * self.mueff) * (c_inv_sqrt @ yw))
        self.gen += 1
        hsig = (np.linalg.norm(self.ps) / math.sqrt(1 - (1 - self.cs) ** (2 * self.gen))
                < (1.4 + 2 / (self.n + 1)) * self.chin)
        self.pc = (1 - self.cc) * self.pc + hsig * math.sqrt(
            self.cc * (2 - self.cc) * self.mueff) * yw
        rank_mu = (y.T * self.w) @ y
        self.C = ((1 - self.c1 - self.cmu) * self.C
                  + self.c1 * (np.outer(self.pc, self.pc)
                               + (1 - hsig) * self.cc * (2 - self.cc) * self.C)
                  + self.cmu * rank_mu)
        self.sigma *= math.exp((self.cs / self.damps)
                               * (np.linalg.norm(self.ps) / self.chin - 1))
        self.C = np.triu(self.C) + np.triu(self.C, 1).T
        d2, self.B = np.linalg.eigh(self.C)
        self.D = np.sqrt(np.maximum(d2, 1e-20))


# ---------------------------------------------------------------- 평가(무거움)
def measure_escape(graph: Graph, lif, cell: cp.CellParams | None, trials: int, seed: int,
                   device: str) -> dict[str, float]:
    """다가옴 시행 → GF 반응 · 반응당 · 잠복 · 뇌 전체 발화율(bench/escape.py 채점 그대로)."""
    from .bench.escape import score_trials
    from .protocols import LoomSpec, run_loom
    from .sim import Engine

    spec = LoomSpec.from_config()
    eng = Engine(graph, lif, device=device, seed=seed, cell=cell)
    tr = run_loom(eng, spec, trials, seed=seed)
    step, neuron = eng.spikes()
    s = score_trials(step, neuron, lif.dt_ms, graph.select(type="DNp01"), tr.onsets_ms,
                     spec.burst_ms)
    stim = np.zeros(graph.n, bool)
    stim[tr.stim_idx] = True
    pop = int((~stim[neuron]).sum()) / (graph.n - stim.sum()) / (eng.t_ms / 1000.0)
    return {"gf_hit": s["gf_hit"], "gf_per_hit": s["gf_per_hit"],
            "gf_latency_ms": s["gf_latency_ms"], "pop_rate_hz": float(pop)}


def make_objective(graph: Graph, null_graph: Graph, lif, template: cp.CellParams, scope: str,
                   rule: dict[str, Any], pair_dst: list[str] | None,
                   measure: Callable = measure_escape,
                   groups: list[dict[str, Any]] | None = None) -> Callable:
    """θ → (손실, 기록). `measure` 는 시험에서 바꿔 끼운다."""
    phys = config.load("physiology")
    lo, hi = bounds_log(template, config.load("fit")["bounds"])
    theta0 = to_theta(template, lif.v0_mV)
    # 섞은 배선 측정은 손실에 그것을 쓰는 항목(null_*)이 있을 때만 — 없으면 같은 절차를 재배선 뇌에
    # 따로 돌려 나란히 둔다(I v2 §7.4 · KC 범위는 평가 시간이 절반)
    needs_null = any(scope in it["scope"] and ".null_" in str(it["measure"])
                     for it in phys["items"])

    def objective(theta: np.ndarray) -> tuple[float, dict[str, Any]]:
        cell = from_theta(theta, template, lif.v0_mV, lo, hi)
        m = measure(graph, lif, cell, rule["trials"], rule["seed"], rule["device"])
        if needs_null:
            null = measure(null_graph, lif, transfer(cell, null_graph, pair_dst, groups),
                           rule["trials"], rule["seed"], rule["device"])
            m.update({f"null_{k}": v for k, v in null.items()})  # 대조 항(escape.null_gf_hit)
        phys_l, parts = physiology_loss(m, scope, phys, float(rule["nan_tolerances"]))
        reg = float(rule["reg_lambda"]) * float(np.sum((np.clip(theta, lo, hi) - theta0) ** 2))
        return phys_l + reg, {"measures": m, "parts": parts, "reg": round(reg, 6)}

    return objective


def run_cmaes(objective: Callable, theta0: np.ndarray, rule: dict[str, Any], *,
              on_gen: Callable | None = None, stop: Callable | None = None
              ) -> tuple[np.ndarray, float, list[dict]]:
    es = CMAES(theta0, float(rule["sigma0"]), rule.get("popsize"), int(rule["seed"]))
    best, best_f, log = theta0.copy(), math.inf, []
    for g in range(int(rule["generations"])):
        xs = es.ask()
        res = [objective(x) for x in xs]
        f = np.array([r[0] for r in res])
        es.tell(xs, f)
        k = int(np.argmin(f))
        if f[k] < best_f:
            best, best_f = xs[k].copy(), float(f[k])
        row = {"gen": g, "best": best_f, "gen_best": float(f[k]),
               "gen_median": float(np.median(f)), "sigma": es.sigma, "detail": res[k][1]}
        log.append(row)
        if on_gen:
            on_gen(row)
        if stop and stop(g):
            break
    return best, best_f, log


# ---------------------------------------------------------------- 명령
def measure_for(sc: dict[str, Any]) -> Callable:
    """범위 설정 `measure`(escape | mb) → 평가 함수."""
    kind = sc.get("measure", "escape")
    if kind == "escape":
        return measure_escape
    if kind == "mb":
        from .mb_probe import measure_kc

        if sc.get("kc_probe"):                # 범위별 덧붙임(예: kc-mbon-b 의 extra_odors)
            from functools import partial

            return partial(measure_kc, cfg={**config.load("fit")["kc_probe"], **sc["kc_probe"]})
        return measure_kc
    raise ValueError(f"평가 종류 {kind!r} — escape | mb")


def _graphs(brain: str, rule: dict[str, Any]) -> tuple[Graph, Graph]:
    from . import graph as graph_mod

    g = graph_mod.load()
    if brain != "real":
        g = graph_mod.rewire(g, int(brain.rsplit("-s", 1)[1]))
    return g, graph_mod.rewire(g, int(rule["null_seed"]))


def probe_sgbptt(g: Graph, lif, cell: cp.CellParams, window_steps: int, sg: dict[str, Any],
                 device: str, seed: int = 0) -> dict[str, Any]:
    """2 단계 한 창: 다가옴 자극(최대 세기) → GF 발화 수 → 역전파. 시간 · 메모리 · 기울기 크기."""
    import torch

    from .circuit_grad import FuncLIF
    from .protocols import LoomSpec, loom_targets

    spec = LoomSpec.from_config()
    idx, amp, _ = loom_targets(g, spec)
    scope = np.flatnonzero(cell.neuron_type(g) >= 0)
    dtype = getattr(torch, str(sg["dtype"]))
    f = FuncLIF(g, lif, cell, stim_idx=idx, readout_idx=g.select(type="DNp01"),
                grad_scope=scope, beta=float(sg["surrogate_beta"]), dtype=dtype, device=device)
    prob = np.asarray(amp) * spec.peak_hz * lif.dt_ms / 1000.0
    kicks = np.random.default_rng(seed).random((window_steps, len(idx))) < prob
    t0 = time.time()
    _, counts, _ = f.run(f.initial_state(), kicks,
                         checkpoint_steps=int(sg["checkpoint_steps"]))
    t_fwd = time.time() - t0
    if counts.requires_grad:
        counts.sum().backward()
    grads = {n: float(t.grad.abs().sum()) if t.grad is not None else 0.0
             for n, t in zip(("tau", "vth", "alpha"), f.params())}
    return {"window_steps": window_steps, "forward_s": round(t_fwd, 2),
            "total_s": round(time.time() - t0, 2), "scope_neurons": int(len(scope)),
            "gf_spikes": float(counts.detach().sum()), "grad_abs_sum": grads,
            "footprint_gb": _footprint_gb(), "free_percent_after": _free_percent()}


def main_probe(args: argparse.Namespace) -> int:
    """평가 한 번(학습 그래프 + 대조)의 시간 · 메모리를 시행 수 둘로 잰다 — 저장 없음(기록만).

    `--method sgbptt` 면 2 단계 한 창(대리 기울기 · 역전파)을 `--window-ms` 두 길이로 잰다.
    """
    from .guard import GuardRefused, heavy
    from .sim import LIFParams

    fit_cfg = config.load("fit")
    rule = {**fit_cfg["cmaes"], **(fit_cfg["scopes"][args.scope].get("cmaes") or {}),
            "device": args.device}
    try:
        with heavy("fit-probe") as guard:
            print(f"[guard] 메모리 여유 {guard.free_percent}%", flush=True)
            lif = LIFParams.from_config()
            g, null = _graphs(args.brain, rule)
            sc = fit_cfg["scopes"][args.scope]
            cell = cp.init_scope(g, args.scope, lif.t_mbr_ms, lif.v_th_mV)
            n_theta = 2 * len(cell.types) + len(cell.alpha)
            rows = []
            if args.method == "sgbptt":
                sg = fit_cfg["sgbptt"]
                for ms in [float(x) for x in args.window_ms.split(",")]:
                    r = probe_sgbptt(g, lif, cell, int(round(ms / lif.dt_ms)), sg, args.device)
                    rows.append({"window_ms": ms, **r})
                    print(f"[fit-probe sgbptt] 창 {ms} ms · {r['total_s']} s · 프로세스 "
                          f"{r['footprint_gb']} GB · GF {r['gf_spikes']} · {r['grad_abs_sum']}",
                          flush=True)
            for n in ([] if args.method == "sgbptt" else
                      [int(x) for x in args.trials.split(",")]):
                obj = make_objective(g, null, lif, cell, args.scope, {**rule, "trials": n},
                                     sc.get("pair_dst"),
                                     measure=measure_for(sc), groups=sc.get("groups"))
                t0 = time.time()
                loss, info = obj(to_theta(cell, lif.v0_mV))
                rows.append({"trials": n, "wall_s": round(time.time() - t0, 1),
                             "free_percent_after": _free_percent(),
                             "footprint_gb": _footprint_gb(), "loss_at_init": loss,
                             **info})
                print(f"[fit-probe] 시행 {n} · {rows[-1]['wall_s']} s · 처음 손실 {loss:.3f} · "
                      f"{info['measures']}", flush=True)
    except GuardRefused as err:
        print(f"[guard] 막음 — {err}")
        return 2
    lam = int(rule.get("popsize") or 4 + math.floor(3 * math.log(n_theta)))
    per_trial = (rows[-1]["wall_s"] / rows[-1]["trials"]) if rows and "trials" in rows[-1] \
        else None
    est_h = (per_trial * int(rule["trials"]) * lam * int(rule["generations"]) / 3600
             if per_trial else None)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = paths.runs_dir() / f"fit-probe-{args.scope}-{args.brain}-{stamp}.json"
    out.write_text(json.dumps({"scope": args.scope, "brain": args.brain, "params": n_theta,
                               "method": args.method,
                               "popsize": lam, "rows": rows, "rule": rule,
                               "estimate_hours": est_h}, ensure_ascii=False, indent=2,
                              default=float), encoding="utf-8")
    print(f"[fit-probe] 매개변수 {n_theta} · 개체 {lam} · 추정 {est_h:.1f} h"
          f"(시행 수에 비례 가정 — 두 줄로 확인) · {out}" if est_h else f"[fit-probe] {out}")
    return 0


def _free_percent() -> int:
    from .guard import memory_free_percent

    return memory_free_percent()


def _footprint_gb() -> float:
    from .guard import process_footprint_gb

    return round(process_footprint_gb(), 2)


def graph_hash(g: Graph) -> str:
    """정체성 검사용 — 연결(row_ptr · dst · count)과 부호의 sha256."""
    import hashlib

    h = hashlib.sha256()
    for a in (g.row_ptr, g.dst, g.count, g.sign):
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def main_fit(args: argparse.Namespace) -> int:
    from .guard import GuardRefused, Watchdog, heavy
    from .sim import LIFParams

    if args.method != "cmaes":
        raise SystemExit("2 단계(sgbptt)는 아직 없다 — I v2 §7.2")
    fit_cfg = config.load("fit")
    rule = {**fit_cfg["cmaes"], **(fit_cfg["scopes"][args.scope].get("cmaes") or {}),
            "device": args.device}
    if args.generations:
        rule["generations"] = args.generations
    out = paths.runs_dir() / (f"fit-{args.scope}-{fit_cfg['version']}-{args.brain}-"
                              f"{time.strftime('%Y%m%d-%H%M%S')}")
    try:
        with heavy("fit") as guard:
            print(f"[guard] 메모리 여유 {guard.free_percent}%", flush=True)
            out.mkdir(parents=True)
            lif = LIFParams.from_config()
            g, null = _graphs(args.brain, rule)
            h0 = graph_hash(g)
            sc = fit_cfg["scopes"][args.scope]
            cell = cp.init_scope(g, args.scope, lif.t_mbr_ms, lif.v_th_mV)
            obj = make_objective(g, null, lif, cell, args.scope, rule, sc.get("pair_dst"),
                                 measure=measure_for(sc), groups=sc.get("groups"))
            dog = Watchdog("fit")
            why: list[str] = []
            log_f = (out / "loss.jsonl").open("w", encoding="utf-8")

            def on_gen(row: dict) -> None:
                log_f.write(json.dumps(row, ensure_ascii=False, default=float) + "\n")
                log_f.flush()
                print(f"[fit] 세대 {row['gen']} · 최선 {row['best']:.4f} · σ {row['sigma']:.3f}",
                      flush=True)

            def stop(gen: int) -> bool:
                r = dog.check(gen + 1)
                if r:
                    why.append(r)
                return bool(r)

            best, best_f, _ = run_cmaes(obj, to_theta(cell, lif.v0_mV), rule, on_gen=on_gen,
                                        stop=stop)
            log_f.close()
            lo, hi = bounds_log(cell, fit_cfg["bounds"])
            fitted = from_theta(best, cell, lif.v0_mV, lo, hi)
            fitted.meta.update({"method": "cmaes", "brain": args.brain,
                                "targets": config.load("physiology")["version"],
                                "seed": rule["seed"]})
            fitted.check(g)
            fitted.save(out / "params.npz")
            (out / "meta.json").write_text(json.dumps(
                {"scope": args.scope, "brain": args.brain, "rule": rule, "best_loss": best_f,
                 "stopped": why or None, "graph_sha256": h0,
                 "graph_unchanged": graph_hash(g) == h0, "footprint_gb": _footprint_gb(),
                 "physiology": config.load("physiology")["version"]},
                ensure_ascii=False, indent=2, default=float), encoding="utf-8")
            print(f"[fit] 끝 · 최선 손실 {best_f:.4f} · {out}", flush=True)
    except GuardRefused as err:
        print(f"[guard] 막음 — {err}")
        return 2
    return 0


def register(sub) -> None:
    p = sub.add_parser("fit-probe", help="회로 학습 평가 한 번의 시간 · 메모리(무거움 · F-006)")
    p.add_argument("--scope", default="escape-small")
    p.add_argument("--brain", default="real")
    p.add_argument("--trials", default="2,4", help="시행 수 둘 — 무엇에 비례하는지(cmaes)")
    p.add_argument("--method", default="cmaes", choices=["cmaes", "sgbptt"])
    p.add_argument("--window-ms", default="10,30", help="창 길이 둘(sgbptt · I v2 §7.2)")
    p.add_argument("--device", default="cpu", choices=["cpu", "mps"])
    p.set_defaults(func=main_probe)
    f = sub.add_parser("fit", help="회로 학습 1 단계 CMA-ES(무거움 · 요구자 확인 뒤 · F-006)")
    f.add_argument("--scope", default="escape-small")
    f.add_argument("--method", default="cmaes", choices=["cmaes", "sgbptt"])
    f.add_argument("--brain", default="real")
    f.add_argument("--generations", type=int, default=0, help="0 = config/fit.yaml")
    f.add_argument("--device", default="cpu", choices=["cpu", "mps"])
    f.set_defaults(func=main_fit)
