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


def cs_minus_types(graph: Graph, cs_plus: list[str], n: int, seed: int) -> list[str]:
    """짝 안 지은 냄새 — CS+ 를 뺀 ORN 종류에서 씨앗으로 n 개(결과 전 선언 · config)."""
    orn = sorted({t for t in graph.columns["type"].astype(str)
                  if t.startswith("ORN_") and t not in cs_plus})
    return sorted(np.random.default_rng(seed).choice(orn, size=n, replace=False).tolist())


def shuffled_table(table: dict[str, str], seed: int) -> dict[str, str]:
    """구획 표 섞기 대조 — MBON 종류마다 짝지은 DAN 종류를 씨앗으로 뒤섞는다."""
    keys = sorted(table)
    vals = np.random.default_rng(seed).permutation([table[k] for k in keys])
    return dict(zip(keys, vals.tolist()))


def drive_split(tf: ThreeFactor, graph: Graph, reward_mbon: np.ndarray, kc_plus: np.ndarray,
                kc_minus: np.ndarray, kc_paired: np.ndarray) -> dict[str, Any]:
    """보상 구획 MBON 에 닿는 KC 입력을 CS+ 와 겹치는 몫으로 나눈다(F-007 · 10-06 탐색).

    kc_plus · kc_minus = 짝짓기 전 시험 창의 KC 발화 수(tf.kc 순서) · kc_paired = 짝짓기 동안
    발화한 KC(불) · 입력 = 발화 수 × 처음 세기(tf.w0)의 합. 남은 몫 = 같은 발화에 짝짓기 뒤
    배율 m 을 곱한 입력 / 처음 입력 — 발화 수가 그대로라는 가정의 예측이다(실제 MBON 발화
    변화와 견준다).
    """
    edge = np.isin(graph.dst[tf.edges], reward_mbon) & (tf.edge_comp >= 0)
    k, w0, m = tf.edge_kc[edge], tf.w0[edge], tf.m[edge]
    to_reward = np.zeros(len(tf.kc), bool)
    to_reward[k] = True

    def jac(a: np.ndarray, b: np.ndarray) -> float:
        return round(float((a & b).sum() / max((a | b).sum(), 1)), 4)

    out: dict[str, Any] = {
        "kc_to_reward": int(to_reward.sum()),
        "jaccard_all": jac(kc_plus > 0, kc_minus > 0),
        "jaccard_reward": jac((kc_plus > 0) & to_reward, (kc_minus > 0) & to_reward)}
    for name, cnt in (("cs_plus", kc_plus), ("cs_minus", kc_minus)):
        d = cnt[k] * w0
        tot = float(d.sum())
        on = int(((cnt > 0) & to_reward).sum())
        out[name] = {
            "kc_on_reward": on, "drive": round(tot, 3),
            "share_with_cs_plus": round(float(d[kc_plus[k] > 0].sum()) / tot, 4) if tot else None,
            "share_paired_kc": round(float(d[kc_paired[k]].sum()) / tot, 4) if tot else None,
            "predicted_remaining": round(float((d * m).sum()) / tot, 4) if tot else None}
    return out


def controls_sets(g: Graph, cfg: dict[str, Any], cs_plus: list[str], cs_minus: list[str]
                  ) -> dict[str, Any]:
    """대조에 쓰는 뉴런 묶음 — 구획 표 · 보상 · 처벌 구획 MBON · DAN · CS+ · CS− ORN."""
    comp = compartments(g, cfg)
    types = g.columns["type"]
    mbon = np.flatnonzero(np.isin(g.columns["class"], cfg["targets"]["post_class"]))
    rew, pun = (cfg["teaching"][k]["dan_types"] for k in ("reward", "punishment"))
    return {"table": comp["table"], "mbon": mbon,
            "reward_mbon": mbon[np.isin(types[mbon], [m for m, d in comp["table"].items()
                                                      if d in rew])],
            "punish_mbon": mbon[np.isin(types[mbon], [m for m, d in comp["table"].items()
                                                      if d in pun])],
            "reward_dan": g.select(type=rew), "punish_dan": g.select(type=pun),
            "cs_plus": g.select(type=cs_plus), "cs_minus": g.select(type=cs_minus)}


def run_condition(graph: Graph, lif, cfg: dict[str, Any], cell, cond: str, seed: int,
                  device: str, sets: dict[str, Any], detail: bool = False) -> dict[str, Any]:
    """조건 하나(paired · cs_minus 는 paired 안에서 · unpaired · punish · shuffled · off).

    짝짓기 전 · 뒤에 CS+ 와 CS− 를 시험 창만큼 주고 보상 · 처벌 구획 MBON 발화를 센다(I v2 §8.3).
    detail 이면 보상 구획 KC 입력의 겹침 분해(`drive_split`)를 함께 적는다.
    """
    from .sim import Engine

    pc = cfg["probe_controls"]
    eta = 0.0 if cond == "off" else float(pc["eta"])
    table = (shuffled_table(sets["table"], int(pc["shuffle_seed"]) + seed)
             if cond == "shuffled" else sets["table"])
    tf = ThreeFactor(graph, cfg, eta, table)
    eng = Engine(graph, lif, device=device, seed=seed, cell=cell)
    win, hz = float(pc["test_ms"]), float(pc["odor_hz"])
    step_ms = float(pc["update_ms"])
    dan_hz = float(cfg["teaching"]["rate_hz"])
    dan = sets["punish_dan"] if cond == "punish" else sets["reward_dan"]

    def test(odor: np.ndarray) -> dict[str, Any]:
        eng.reset()
        _, n = _window(eng, [(odor, hz)], win)
        cnt = np.bincount(n, minlength=graph.n)
        return {"reward": int(cnt[sets["reward_mbon"]].sum()),
                "punish": int(cnt[sets["punish_mbon"]].sum()),
                "all": int(cnt[sets["mbon"]].sum()),
                "kc_on": np.flatnonzero(cnt[tf.kc] > 0), "kc_cnt": cnt[tf.kc],
                "mbon_cnt": cnt[sets["mbon"]]}

    before = {"cs_plus": test(sets["cs_plus"]), "cs_minus": test(sets["cs_minus"])}
    eng.reset()
    t0 = eng.step
    kc_paired = np.zeros(len(tf.kc), bool)

    def phase(stim: list[tuple[np.ndarray, float]], ms: float) -> None:
        nonlocal t0
        for _ in range(int(round(ms / step_ms))):
            _window(eng, stim, step_ms)
            st, ne = eng.spikes_since(t0)
            kc_paired[np.searchsorted(tf.kc, ne[np.isin(ne, tf.kc)])] = True
            tf.update(st, ne, step_ms)
            t0 = eng.step
            tf.apply(eng)

    odor_ms = float(cfg["eta"]["pairing"]["odor_s"]) * 1000
    if cond == "unpaired":                    # 보상 먼저 → 쉼 → 냄새(시간이 어긋난 보상)
        phase([(dan, dan_hz)], odor_ms)
        phase([], float(pc["unpaired_gap_s"]) * 1000)
        phase([(sets["cs_plus"], hz)], odor_ms)
    else:
        phase([(sets["cs_plus"], hz), (dan, dan_hz)], odor_ms)
    after = {"cs_plus": test(sets["cs_plus"]), "cs_minus": test(sets["cs_minus"])}
    out: dict[str, Any] = {"cond": cond, "seed": seed, "eta": eta, **tf.state()}
    for k in ("cs_plus", "cs_minus"):
        for g in ("reward", "punish", "all"):
            out[f"{k}.{g}"] = [before[k][g], after[k][g]]
    a, b = set(before["cs_plus"]["kc_on"].tolist()), set(before["cs_minus"]["kc_on"].tolist())
    out["kc_overlap_cs"] = round(len(a & b) / max(len(a | b), 1), 4)
    if detail:                                # MBON 뉴런마다 발화(가치 읽기 · valence_readout)
        out["mbon_counts"] = {k: [before[k]["mbon_cnt"].tolist(), after[k]["mbon_cnt"].tolist()]
                              for k in ("cs_plus", "cs_minus")}
    if detail and tf.w0 is not None:
        out["split"] = drive_split(tf, graph, sets["reward_mbon"], before["cs_plus"]["kc_cnt"],
                                   before["cs_minus"]["kc_cnt"], kc_paired)
    return out


def change(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    """씨앗 합 전 → 뒤와 변화율(뒤/전 − 1) · 씨앗마다 감소했는지."""
    b = sum(r[key][0] for r in rows)
    a = sum(r[key][1] for r in rows)
    return {"before": b, "after": a, "change": (round(a / b - 1, 4) if b else None),
            "seeds_down": sum(r[key][1] < r[key][0] for r in rows)}


def verdict_controls(summary: dict[str, dict[str, Any]], pc: dict[str, Any]) -> dict[str, Any]:
    """결과 전 선언한 기준(config `probe_controls.pass`)으로 조건마다 통과 여부."""
    c = pc["pass"]

    def drop(cond: str, key: str) -> float | None:
        ch = summary[cond][key]["change"]
        return None if ch is None else -ch

    lo, hi = c["paired_drop"]
    out = {}
    d = drop("paired", "cs_plus.reward")
    out["paired"] = d is not None and lo <= d <= hi
    ch = summary["paired"]["cs_minus.reward"]["change"]
    out["cs_minus"] = None if ch is None else abs(ch) <= c["control_change_max"]
    for cond in ("unpaired", "shuffled", "off"):
        d = drop(cond, "cs_plus.reward")
        out[cond] = None if d is None else d <= c["control_change_max"]
    dr, dp = drop("punish", "cs_plus.reward"), drop("punish", "cs_plus.punish")
    out["punish"] = (None if dr is None or dp is None
                     else dr <= c["control_change_max"] and dp >= c["punish_drop_min"])
    return out


def cs_minus_ok(summary: dict[str, dict[str, Any]], pc: dict[str, Any]) -> bool | None:
    """짝 안 지은 냄새만 다시 잴 때의 판정(기준은 verdict_controls 와 같음)."""
    ch = summary["paired"]["cs_minus.reward"]["change"]
    return None if ch is None else abs(ch) <= pc["pass"]["control_change_max"]


def screen_cs_minus(graph: Graph, lif, cfg: dict[str, Any], cell, device: str
                    ) -> dict[str, Any]:
    """짝 안 지은 냄새 고르기(짝짓기 전 반응만 · config `probe_controls.cs_minus_screen`).

    후보마다 시험 창 한 번: 보상 구획 MBON 발화 · CS+ 와 켜진 KC 겹침(Jaccard).
    """
    from .sim import Engine

    pc = cfg["probe_controls"]
    sc = pc["cs_minus_screen"]
    comp = compartments(graph, cfg)
    types = graph.columns["type"]
    mbon = np.flatnonzero(np.isin(graph.columns["class"], cfg["targets"]["post_class"]))
    rew = cfg["teaching"]["reward"]["dan_types"]
    reward_mbon = mbon[np.isin(types[mbon], [m for m, d in comp["table"].items() if d in rew])]
    kc = np.flatnonzero(np.isin(graph.columns["class"], cfg["targets"]["pre_class"]))
    eng = Engine(graph, lif, device=device, seed=int(sc["sim_seed"]), cell=cell)

    def resp(odor_types: list[str]) -> tuple[int, set]:
        eng.reset()
        _, n = _window(eng, [(graph.select(type=odor_types), float(pc["odor_hz"]))],
                       float(pc["test_ms"]))
        cnt = np.bincount(n, minlength=graph.n)
        return int(cnt[reward_mbon].sum()), set(kc[cnt[kc] > 0].tolist())

    plus_r, plus_kc = resp(list(pc["cs_plus"]))
    rows = []
    for s in range(int(sc["first_seed"]), int(sc["first_seed"]) + int(sc["candidates"])):
        cand = cs_minus_types(graph, list(pc["cs_plus"]), int(pc["cs_minus_n"]), s)
        r, k = resp(cand)
        ov = len(k & plus_kc) / max(len(k | plus_kc), 1)
        rows.append({"seed": s, "types": cand, "reward_spikes": r,
                     "rel": round(r / plus_r, 4) if plus_r else None,
                     "kc_overlap": round(ov, 4), "kc_on": len(k)})
    ok = [r for r in rows if r["rel"] is not None and r["rel"] >= sc["min_rel_response"]
          and r["kc_overlap"] <= sc["max_kc_overlap"]]
    pick = max(ok, key=lambda r: r["reward_spikes"]) if ok else None
    return {"cs_plus_reward_spikes": plus_r, "cs_plus_kc_on": len(plus_kc), "rows": rows,
            "pick": pick}


def main_screen(args: argparse.Namespace) -> int:
    from pathlib import Path

    from . import graph as graph_mod
    from .cellparams import CellParams
    from .guard import GuardRefused, heavy
    from .sim import LIFParams

    cfg = config.load("plasticity")
    try:
        with heavy("plasticity-cs-screen") as guard:
            print(f"[guard] 메모리 여유 {guard.free_percent}%", flush=True)
            g = graph_mod.load()
            res = screen_cs_minus(g, LIFParams.from_config(), cfg,
                                  CellParams.load(Path(args.cell_from) / "params.npz"),
                                  args.device)
    except GuardRefused as err:
        print(f"[guard] 막음 — {err}")
        return 2
    for r in res["rows"]:
        print(f"[cs-screen] 씨앗 {r['seed']:2} 보상 구획 {r['reward_spikes']:3} "
              f"(CS+ 의 {r['rel']}) · KC 겹침 {r['kc_overlap']} · {','.join(r['types'])}",
              flush=True)
    print(f"[cs-screen] CS+ 보상 구획 {res['cs_plus_reward_spikes']} · 고름 {res['pick']}",
          flush=True)
    out = paths.runs_dir() / f"plasticity-cs-screen-{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.write_text(json.dumps({"screen": cfg["probe_controls"]["cs_minus_screen"],
                               "cell_from": args.cell_from, **res}, ensure_ascii=False,
                              indent=2), encoding="utf-8")
    print(f"[cs-screen] {out}")
    return 0


def main_controls(args: argparse.Namespace) -> int:
    from . import graph as graph_mod
    from .cellparams import CellParams
    from .guard import GuardRefused, heavy
    from .sim import LIFParams

    cfg = config.load("plasticity")
    if args.eta:                               # 보정한 η(결과 전 잠근 값 · eta_calibration)
        cfg = {**cfg, "probe_controls": {**cfg["probe_controls"], "eta": float(args.eta)}}
    pc = cfg["probe_controls"]
    rows: list[dict[str, Any]] = []
    try:
        with heavy("plasticity-controls") as guard:
            print(f"[guard] 메모리 여유 {guard.free_percent}%", flush=True)
            g = graph_mod.load()
            g = graph_mod.brain_graph(g, args.brain)    # real · rewired-s<K> · pnkc/kcmbon-s<K>
            lif = LIFParams.from_config()
            from pathlib import Path

            cell = (None if args.cell_from == "none"               # 회로 학습 전 뇌
                    else CellParams.load(Path(args.cell_from) / "params.npz"))
            cs_plus = list(pc["cs_plus"])
            cs_minus = (args.cs_minus.split(",") if args.cs_minus else
                        cs_minus_types(g, cs_plus, int(pc["cs_minus_n"]), int(pc["cs_minus_seed"])))
            sets = controls_sets(g, cfg, cs_plus, cs_minus)
            conds = args.conditions.split(",") if args.conditions else list(pc["conditions"])
            for seed in [int(x) for x in args.seeds.split(",")]:
                for cond in conds:
                    r = run_condition(g, lif, cfg, cell, cond, seed, args.device, sets,
                                      detail=args.detail)
                    rows.append(r)
                    if "split" in r:
                        sp = r["split"]
                        print(f"[plasticity-controls] 씨앗 {seed} 겹침 KC 전체 "
                              f"{sp['jaccard_all']} · 보상 구획 KC {sp['jaccard_reward']} · "
                              f"CS− 입력 중 CS+ KC 몫 {sp['cs_minus']['share_with_cs_plus']} · "
                              f"짝짓기 KC 몫 {sp['cs_minus']['share_paired_kc']} · 남을 입력 "
                              f"CS+ {sp['cs_plus']['predicted_remaining']} CS− "
                              f"{sp['cs_minus']['predicted_remaining']}", flush=True)
                    print(f"[plasticity-controls] 씨앗 {seed} {cond:9} CS+ 보상 "
                          f"{r['cs_plus.reward']} 처벌 {r['cs_plus.punish']} · CS− 보상 "
                          f"{r['cs_minus.reward']}", flush=True)
    except GuardRefused as err:
        print(f"[guard] 막음 — {err}")
        return 2
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = paths.runs_dir() / f"plasticity-controls-{args.brain}-{stamp}.json"
    out.write_text(json.dumps({"rows": rows}, default=str), encoding="utf-8")   # 판정 전 먼저 저장
    summary = {cond: {k: change([r for r in rows if r["cond"] == cond], k)
                      for k in ("cs_plus.reward", "cs_plus.punish", "cs_minus.reward",
                                "cs_plus.all")}
               for cond in conds}
    passed = (verdict_controls(summary, pc) if set(conds) >= set(pc["conditions"])
              else {"cs_minus": cs_minus_ok(summary, pc)} if "paired" in conds else {})
    for cond, v in summary.items():
        print(f"[plasticity-controls] {cond:9} CS+ 보상 {v['cs_plus.reward']} · CS+ 처벌 "
              f"{v['cs_plus.punish']} · CS− 보상 {v['cs_minus.reward']}", flush=True)
    print(f"[plasticity-controls] 판정 {passed}", flush=True)
    out.write_text(json.dumps({"rule": cfg["version"], "controls": pc, "cs_minus": cs_minus,
                               "mbon_types": g.columns["type"][sets["mbon"]].astype(str).tolist(),
                               "mbon_nt": g.columns["nt"][sets["mbon"]].astype(str).tolist(),
                               "brain": args.brain, "cell_from": args.cell_from,
                               "summary": summary, "pass": passed, "rows": rows},
                              ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"[plasticity-controls] {out}")
    return 0


def calibrate_eta(g: Graph, lif, cfg: dict[str, Any], cell, device: str) -> dict[str, Any]:
    """η 다시 보정(config eta_calibration) — 짝짓기 조건 · CS+ 보상 구획만 본다.

    CS− 자리에도 CS+ 를 넣어 짝 안 지은 냄새 정보가 계산되지 않게 한다. 씨앗은 판정과 다른 것.
    고름 = 씨앗 합 CS+ 보상 구획 감소가 target 에 가장 가까운 η.
    """
    ec = cfg["eta_calibration"]
    cs_plus = list(cfg["probe_controls"]["cs_plus"])
    sets = controls_sets(g, cfg, cs_plus, cs_plus)
    rows = []
    for eta in ec["grid"]:
        c2 = {**cfg, "probe_controls": {**cfg["probe_controls"], "eta": float(eta)}}
        rs = [run_condition(g, lif, c2, cell, "paired", int(sd), device, sets)
              for sd in ec["seeds"]]
        ch = change(rs, "cs_plus.reward")
        rows.append({"eta": float(eta), "cs_plus_reward": ch})
        print(f"[eta-calibrate] η {eta:g} · CS+ 보상 구획 {ch['before']} → {ch['after']} · "
              f"감소 {(-ch['change']) if ch['change'] is not None else None}", flush=True)
    ok = [r for r in rows if r["cs_plus_reward"]["change"] is not None]
    pick = min(ok, key=lambda r: abs(-r["cs_plus_reward"]["change"] - float(ec["target_drop"])))
    return {"rule": ec, "rows": rows, "pick": pick["eta"]}


def main_calibrate(args: argparse.Namespace) -> int:
    from pathlib import Path

    from . import graph as graph_mod
    from .cellparams import CellParams
    from .guard import GuardRefused, heavy
    from .sim import LIFParams

    cfg = config.load("plasticity")
    try:
        with heavy("plasticity-eta-calibrate") as guard:
            print(f"[guard] 메모리 여유 {guard.free_percent}%", flush=True)
            g = graph_mod.load()
            res = calibrate_eta(g, LIFParams.from_config(), cfg,
                                CellParams.load(Path(args.cell_from) / "params.npz"), args.device)
    except GuardRefused as err:
        print(f"[guard] 막음 — {err}")
        return 2
    out = paths.runs_dir() / f"plasticity-eta-calibrate-{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.write_text(json.dumps({"cell_from": args.cell_from, **res}, ensure_ascii=False,
                              indent=2, default=str), encoding="utf-8")
    print(f"[eta-calibrate] 고름 η {res['pick']:g} · {out}", flush=True)
    return 0


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
            g = graph_mod.brain_graph(g, args.brain)    # 대조 — 섞은 배선(전체 · 부분)
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
    p.add_argument("--brain", default="real", help="real | rewired-s<씨앗>(대조)")
    p.add_argument("--cell-from", default="", help="회로 학습 폴더(params.npz · 같은 뇌)")
    p.set_defaults(func=main_probe)
    c = sub.add_parser("plasticity-controls",
                       help="가소성 대조 다섯(F-007 · I v2 §8.3 · 무거움 · 짧음)")
    c.add_argument("--cell-from", required=True,
                   help="회로 학습 폴더(params.npz · 같은 뇌) · none = 회로 학습 전")
    c.add_argument("--brain", default="real", help="real | rewired-s<씨앗>")
    c.add_argument("--seeds", default="0,1,2,3,4")
    c.add_argument("--device", default="cpu", choices=["cpu", "mps"])
    c.add_argument("--cs-minus", default="", help="CS− ORN 종류(쉼표) · 비우면 설정")
    c.add_argument("--conditions", default="", help="조건 일부만(쉼표) — 비우면 전부")
    c.add_argument("--eta", type=float, default=0.0,
                   help="η 덮어쓰기(0 = probe_controls.eta) — eta_calibration 으로 고른 값만")
    c.add_argument("--detail", action="store_true",
                   help="보상 구획 KC 입력의 CS+ · CS− 겹침 분해도 적는다(탐색)")
    c.set_defaults(func=main_controls)
    e = sub.add_parser("plasticity-eta-calibrate",
                       help="η 다시 보정 — 짝짓기 CS+ 만 · 판정과 다른 씨앗(eta_calibration)")
    e.add_argument("--cell-from", required=True)
    e.add_argument("--device", default="cpu", choices=["cpu", "mps"])
    e.set_defaults(func=main_calibrate)
    s = sub.add_parser("plasticity-cs-screen",
                       help="짝 안 지은 냄새 고르기 — 짝짓기 전 반응만(무거움 · 짧음)")
    s.add_argument("--cell-from", required=True)
    s.add_argument("--device", default="cpu", choices=["cpu", "mps"])
    s.set_defaults(func=main_screen)
