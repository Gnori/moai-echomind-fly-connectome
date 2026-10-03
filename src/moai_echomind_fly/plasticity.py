"""버섯체 가소성 plasticity-v1 — KC → MBON 배율의 3요소 규칙 (R-7469 F-007 · I v2 §8).

dm_ij/dt = −η · D_c(j) · ē_i · m_ij — 곱셈형 · 약화만(앞 짝짓기) · m ∈ [m_min, m_max].
- ē_i: KC i 의 발화 흔적(τ_e · 발화 하나에 +1)
- D_c: 구획 c 의 DAN 신호(τ_D · 그 구획 DAN 뉴런당 발화 수)
- 구획 = MBON 종류마다 시냅스를 가장 많이 주는 DAN 종류(그래프에서 · 문헌 대조표와 맞춰 본다)
- 바꾸는 것은 KC → MBON 연결의 **세기 배율**뿐(존재 · 시냅스 수 · 부호 불변) · DAN 부호 0 그대로

엔진과는 창 단위로 붙는다: `run(ms)` → `update(발화)` → `apply(engine)`. 값은 config/plasticity.yaml
(결과 전에 적은 판 · 바꾸면 판을 올린다). η 는 값이 아니라 목표로 보정한다(`plasticity-probe`).
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Any

import numpy as np

from . import config, paths
from .graph import Graph


def _starts(types: np.ndarray, prefixes: list[str]) -> np.ndarray:
    t = types.astype(str)
    return np.any([np.char.startswith(t, p) for p in prefixes], axis=0)


def compartments(graph: Graph, cfg: dict[str, Any]) -> dict[str, Any]:
    """MBON 종류 → 시냅스를 가장 많이 주는 DAN 종류 + 문헌 대조(config `compartments.check`)."""
    tg = cfg["targets"]
    types = graph.columns["type"]
    is_dan = _starts(types, tg["dan_type_prefix"])
    is_mbon = np.isin(graph.columns["class"], tg["post_class"])
    src = graph.src()
    m = is_dan[src] & is_mbon[graph.dst]
    syn: dict[tuple[str, str], int] = {}
    for a, b, c in zip(types[src[m]], types[graph.dst[m]], graph.count[m]):
        syn[(b, a)] = syn.get((b, a), 0) + int(c)
    best: dict[str, tuple[str, int]] = {}
    for (mbon, dan), c in syn.items():
        if mbon not in best or c > best[mbon][1]:
            best[mbon] = (dan, c)
    table = {mb: dan for mb, (dan, _) in sorted(best.items())}
    check = {dan: mb for dan, mb in (cfg["compartments"].get("check") or {}).items()}
    agree = {f"{dan}→{mb}": table.get(mb) == dan for dan, mb in check.items()}
    return {"table": table, "synapses": {mb: c for mb, (_, c) in best.items()}, "check": agree}


class ThreeFactor:
    """KC → MBON 배율 m 과 흔적 상태. `update` 는 창 하나의 발화로 상태를 앞으로 민다."""

    def __init__(self, graph: Graph, cfg: dict[str, Any], eta: float,
                 table: dict[str, str] | None = None) -> None:
        tg, r = cfg["targets"], cfg["rule"]
        if r.get("backward_pairing") not in (None, "none"):
            raise ValueError("plasticity-v1 은 뒤 짝짓기를 두지 않는다")
        self.g, self.eta = graph, float(eta)
        self.tau_e = float(r["kc_trace_tau_s"]) * 1000.0
        self.tau_d = float(r["dan_signal_tau_s"]) * 1000.0
        self.m_min, self.m_max = float(r["m_min"]), float(r["m_max"])
        cls, types = graph.columns["class"], graph.columns["type"]
        src = graph.src()
        is_kc = np.isin(cls, tg["pre_class"])
        is_mbon = np.isin(cls, tg["post_class"])
        self.edges = np.flatnonzero(is_kc[src] & is_mbon[graph.dst])
        self.kc = np.flatnonzero(is_kc)
        kc_pos = np.full(graph.n, -1)
        kc_pos[self.kc] = np.arange(len(self.kc))
        self.edge_kc = kc_pos[src[self.edges]]
        table = table if table is not None else compartments(graph, cfg)["table"]
        self.comp_names = sorted(set(table.values()))
        cpos = {d: i for i, d in enumerate(self.comp_names)}
        mbon_type = types[graph.dst[self.edges]]
        self.edge_comp = np.array([cpos.get(table.get(t, ""), -1) for t in mbon_type])
        self.dan_of_comp = [np.flatnonzero(types == d) for d in self.comp_names]
        self.e = np.zeros(len(self.kc))
        self.d = np.zeros(len(self.comp_names))
        self.m = np.ones(len(self.edges))
        self.w0: np.ndarray | None = None

    def update(self, steps: np.ndarray, neurons: np.ndarray, window_ms: float) -> None:
        """창 하나(길이 window_ms)의 발화 → 흔적 갱신 → m 곱셈 갱신(창 끝 값으로 적분)."""
        dec_e, dec_d = np.exp(-window_ms / self.tau_e), np.exp(-window_ms / self.tau_d)
        kc_cnt = np.bincount(np.searchsorted(self.kc, neurons[np.isin(neurons, self.kc)]),
                             minlength=len(self.kc))
        self.e = self.e * dec_e + kc_cnt
        cnt = np.bincount(neurons, minlength=self.g.n)
        dan = np.array([cnt[ix].sum() / max(len(ix), 1) for ix in self.dan_of_comp])
        self.d = self.d * dec_d + dan
        ok = self.edge_comp >= 0
        rate = np.zeros(len(self.edges))
        rate[ok] = self.d[self.edge_comp[ok]] * self.e[self.edge_kc[ok]]
        self.m = np.clip(self.m * np.exp(-self.eta * rate * window_ms / 1000.0),
                         self.m_min, self.m_max)

    def apply(self, engine) -> None:
        """엔진의 KC → MBON 연결 세기 = 처음 세기 × m."""
        if self.w0 is None:
            self.w0 = engine.w[self.edges].detach().cpu().numpy().copy()
        engine.set_edge_weights(self.edges, self.w0 * self.m)

    def state(self) -> dict[str, Any]:
        return {"edges": int(len(self.edges)), "m_mean": float(self.m.mean()),
                "m_min_seen": float(self.m.min()), "changed": int((self.m < 1.0).sum())}


# ---------------------------------------------------------------- 먼저 잴 것(무거움 · 뇌 전체)
def _window(eng, stim: list[tuple[np.ndarray, float]] | None, ms: float):
    """stim = [(뉴런 묶음, 발화율 Hz)] 로 ms 만큼 달리고 그 창의 발화를 돌려준다."""
    step0 = eng.step
    if stim:
        idx = np.concatenate([k for k, _ in stim])
        rate = np.concatenate([np.full(len(k), v) for k, v in stim])
        eng.set_poisson(idx, rate)
    else:
        eng.set_poisson(np.zeros(0, np.int64), 0.0)
    eng.run(ms)
    return eng.spikes_since(step0)


def probe(graph: Graph, lif, cfg: dict[str, Any], eta: float, odor_types: list[str],
          odor_hz: float, window_ms: float, seed: int, device: str,
          cell=None) -> dict[str, Any]:
    """(i) KC 발화 비율 (ii) MBON 이 KC 로 발화하나 (iii) 짝짓기 뒤 구획 MBON 의 CS+ 반응 변화."""
    from .sim import Engine

    comp = compartments(graph, cfg)
    tf = ThreeFactor(graph, cfg, eta, comp["table"])
    eng = Engine(graph, lif, device=device, seed=seed, cell=cell)  # 회로 학습 매개변수
    odor = graph.select(type=odor_types)
    mbon = np.flatnonzero(np.isin(graph.columns["class"], cfg["targets"]["post_class"]))
    pair = cfg["eta"]["pairing"]
    teach = cfg["teaching"]["reward"]["dan_types"]
    dan = graph.select(type=teach)
    reward_mbon = np.isin(graph.columns["type"][mbon],
                          [mb for mb, d in comp["table"].items() if d in teach])

    def test_odor() -> tuple[float, np.ndarray]:
        eng.reset()
        _, n = _window(eng, [(odor, odor_hz)], window_ms)
        cnt = np.bincount(n, minlength=graph.n)
        return float((cnt[tf.kc] > 0).mean()), cnt[mbon]

    kc_frac, mb_before = test_odor()
    eng.reset()
    t0 = eng.step
    for _ in range(int(round(pair["odor_s"] * 1000 / window_ms))):
        _, n = _window(eng, [(odor, odor_hz), (dan, float(cfg["teaching"]["rate_hz"]))],
                       window_ms)
        tf.update(*eng.spikes_since(t0), window_ms)
        t0 = eng.step
        tf.apply(eng)
    _, mb_after = test_odor()
    b, a = mb_before[reward_mbon].sum(), mb_after[reward_mbon].sum()
    return {"eta": eta, "kc_active_fraction": kc_frac,
            "mbon_spikes_before": int(mb_before.sum()), "mbon_fire_from_kc": bool(mb_before.sum()),
            "reward_compartment_mbon_before": int(b), "reward_compartment_mbon_after": int(a),
            "cs_plus_drop": (float(1 - a / b) if b else None), **tf.state(),
            "compartment_check": comp["check"]}


def main_probe(args: argparse.Namespace) -> int:
    from . import graph as graph_mod
    from .guard import GuardRefused, heavy
    from .sim import LIFParams

    cfg = config.load("plasticity")
    rows = []
    try:
        with heavy("plasticity-probe") as guard:
            print(f"[guard] 메모리 여유 {guard.free_percent}%", flush=True)
            g = graph_mod.load()
            lif = LIFParams.from_config()
            cell = None
            if args.cell_from:
                from pathlib import Path

                from . import cellparams as cp

                cell = cp.CellParams.load(Path(args.cell_from) / "params.npz")
            for eta in [float(x) for x in args.eta.split(",")]:
                r = probe(g, lif, cfg, eta, args.odor.split(","), args.odor_hz, args.window_ms,
                          args.seed, args.device, cell)
                rows.append(r)
                print(f"[plasticity-probe] η {eta:g} · KC 발화 {r['kc_active_fraction']:.3f} · "
                      f"MBON {r['mbon_spikes_before']} · 보상 구획 MBON "
                      f"{r['reward_compartment_mbon_before']} → "
                      f"{r['reward_compartment_mbon_after']} · m 평균 {r['m_mean']:.3f}",
                      flush=True)
    except GuardRefused as err:
        print(f"[guard] 막음 — {err}")
        return 2
    out = paths.runs_dir() / f"plasticity-probe-{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.write_text(json.dumps({"rule": cfg["version"], "rows": rows,
                               "args": {k: v for k, v in vars(args).items() if k != "func"}},
                              ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"[plasticity-probe] {out}")
    return 0


def register(sub) -> None:
    p = sub.add_parser("plasticity-probe",
                       help="버섯체 가소성 먼저 잴 것(무거움 · 뇌 전체 · F-007 · 요구자 확인 뒤)")
    p.add_argument("--eta", default="0.001,0.01,0.1", help="보정 후보(목표: CS+ 50~90 %% 감소)")
    p.add_argument("--odor", default="ORN_DM1,ORN_VA2", help="CS+ 냄새 ORN 종류")
    p.add_argument("--odor-hz", type=float, default=150.0)
    p.add_argument("--window-ms", type=float, default=100.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu", choices=["cpu", "mps"])
    p.add_argument("--cell-from", default="", help="회로 학습 결과 폴더(params.npz) — 진짜 뇌만")
    p.set_defaults(func=main_probe)
