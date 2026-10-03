"""KC 희소성 M0 — 학습 없이 지금 매개변수로 냄새 자극의 KC · PN · APL 반응을 잰다(연구 노트 16 §6).

자극 묶음(식초 6 · DM1+VA2 · 무작위 10 사구체 · 없음) × 포아송 씨앗 × 뇌(진짜 · 재배선)마다 한 번
가장 긴 창만큼 돌리고, 앞 구간(100 · 300 · 500 ms)을 잘라 센다:
KC 주요 종류 중 한 번 이상 발화한 비율 · 반응 KC 스파이크 수 중앙 · 자극 PN 평균 발화율(종류별) ·
켜진 PN 짝 수별 KC 반응률(발톱 진단) · APL 발화율 · 나머지 뇌 발화율(KC · APL · 자극 · PN 제외).
값은 config/fit.yaml `kc_probe`(결과 전 선언). 무거움(뇌 전체 · 짧음) — scripts/20-odor/kc-probe.sh.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from typing import Any

import numpy as np

from . import config, paths
from .graph import Graph


def stimulus_sets(graph: Graph, cfg: dict[str, Any]) -> dict[str, list[str]]:
    sets = {k: list(v) for k, v in cfg["stimuli"].items()}
    orn = sorted({t for t in graph.columns["type"].astype(str) if t.startswith("ORN_")})
    rng = np.random.default_rng(int(cfg["random_seed"]))
    for i in range(int(cfg["random_sets"])):
        sets[f"random{cfg['random_size']}-{i}"] = sorted(
            rng.choice(orn, size=int(cfg["random_size"]), replace=False).tolist())
    return sets


def stim_pns(graph: Graph, orn_idx: np.ndarray, cfg: dict[str, Any]) -> np.ndarray:
    """자극 ORN 에서 받은 시냅스 비율 ≥ pn_input_min 인 PN."""
    pn = graph.columns["class"] == cfg["pn_class"]
    src = graph.src()
    into_pn = pn[graph.dst]
    tot = np.bincount(graph.dst[into_pn], weights=graph.count[into_pn], minlength=graph.n)
    from_orn = into_pn & np.isin(src, orn_idx)
    got = np.bincount(graph.dst[from_orn], weights=graph.count[from_orn], minlength=graph.n)
    frac = np.divide(got, tot, out=np.zeros(graph.n), where=tot > 0)
    return np.flatnonzero(pn & (frac >= float(cfg["pn_input_min"])))


def summarize(graph: Graph, steps: np.ndarray, neurons: np.ndarray, dt_ms: float,
              window_ms: float, idx: dict[str, np.ndarray], kc_pn: tuple[np.ndarray, ...]
              ) -> dict[str, Any]:
    m = steps * dt_ms < window_ms
    cnt = np.bincount(neurons[m], minlength=graph.n)
    kc, pn_s, apl, rest = idx["kc"], idx["stim_pn"], idx["apl"], idx["rest"]
    kc_c = cnt[kc]
    resp = kc_c[kc_c > 0]
    sec = window_ms / 1000.0
    pn_rate = cnt[pn_s] / sec if len(pn_s) else np.zeros(0)
    types = graph.columns["type"][pn_s]
    by_type = {t: round(float(pn_rate[types == t].mean()), 1) for t in sorted(set(types))}
    # 발톱: KC 마다 이 창에서 발화한 PN 짝(연결된 PN 중 스파이크 ≥ 1) 수
    k_kc, k_pn = kc_pn
    active_pn = cnt[k_pn] > 0
    claws = np.bincount(k_kc[active_pn], minlength=len(kc))
    claw_resp = {int(c): round(float((kc_c[claws == c] > 0).mean()), 4)
                 for c in sorted(set(claws.tolist())) if (claws == c).sum() >= 5}
    return {"window_ms": window_ms, "kc_active_fraction": round(float((kc_c > 0).mean()), 5),
            "kc_spikes_per_responder_median": float(np.median(resp)) if len(resp) else 0.0,
            "stim_pn_rate_hz_mean": round(float(pn_rate.mean()), 1) if len(pn_rate) else None,
            "stim_pn_rate_by_type": by_type,
            "kc_response_by_active_claws": claw_resp,
            "apl_rate_hz": round(float(cnt[apl].sum() / max(len(apl), 1) / sec), 2),
            "rest_rate_hz": round(float(cnt[rest].sum() / max(len(rest), 1) / sec), 5)}


def run(graph: Graph, lif, cfg: dict[str, Any], stim_types: list[str], seed: int,
        device: str, cell=None, windows_ms: list[float] | None = None) -> dict[str, Any]:
    from .sim import Engine

    cls, types = graph.columns["class"], graph.columns["type"]
    sizes = Counter(types[cls == "Kenyon_Cell"])
    kc_types = [t for t, n in sizes.items() if n >= int(cfg["kc_min_type_size"])]
    kc = np.flatnonzero((cls == "Kenyon_Cell") & np.isin(types, kc_types))
    orn = graph.select(type=stim_types) if stim_types else np.zeros(0, np.int64)
    pn_s = stim_pns(graph, orn, cfg) if len(orn) else np.zeros(0, np.int64)
    apl = graph.select(type=cfg["apl_type"])
    pn_all = np.flatnonzero(cls == cfg["pn_class"])
    excl = np.zeros(graph.n, bool)
    excl[np.concatenate([kc, orn, apl, pn_all, np.flatnonzero(cls == "Kenyon_Cell")])] = True
    rest = np.flatnonzero(~excl)
    src = graph.src()
    kc_pos = np.full(graph.n, -1)
    kc_pos[kc] = np.arange(len(kc))
    e = (kc_pos[graph.dst] >= 0) & (cls[src] == cfg["pn_class"])
    kc_pn = (kc_pos[graph.dst[e]], src[e])
    eng = Engine(graph, lif, device=device, seed=seed, cell=cell)
    windows = [float(w) for w in (windows_ms or cfg["windows_ms"])]
    if len(orn):
        eng.set_poisson(orn, np.full(len(orn), float(cfg["orn_rate_hz"])))
    t0 = time.time()
    eng.run(max(windows))
    steps, neurons = eng.spikes()
    idx = {"kc": kc, "stim_pn": pn_s, "apl": apl, "rest": rest}
    return {"stim_types": stim_types, "seed": seed, "kc_neurons": int(len(kc)),
            "stim_pns": int(len(pn_s)), "wall_s": round(time.time() - t0, 1),
            "windows": [summarize(graph, steps, neurons, lif.dt_ms, float(w), idx, kc_pn)
                        for w in windows]}


def measure_kc(graph: Graph, lif, cell, trials: int, seed: int, device: str,
               cfg: dict[str, Any] | None = None) -> dict[str, float]:
    """회로 학습(F-006) 평가용 — 양성 자극을 씨앗 `trials` 개로 · 음성 · 무자극 한 번씩.

    돌려주는 칸(physiology `mb.*` 측정 이름): kc_active_fraction · kc_spikes_per_responder ·
    kc_negative_fraction · kc_spontaneous_fraction · pop_rate_hz(KC · APL · 자극 · PN 밖).
    """
    cfg = cfg or config.load("fit")["kc_probe"]
    w = [float(cfg["loss_window_ms"])]
    pos = [run(graph, lif, cfg, cfg["stimuli"][cfg["target_stimulus"]], seed + k, device,
               cell, w)["windows"][0] for k in range(int(trials))]
    neg = run(graph, lif, cfg, cfg["stimuli"][cfg["negative_stimulus"]], seed, device, cell,
              w)["windows"][0]
    nil = run(graph, lif, cfg, [], seed, device, cell, w)["windows"][0]
    return {"kc_active_fraction": float(np.mean([x["kc_active_fraction"] for x in pos])),
            "kc_spikes_per_responder": float(np.mean([x["kc_spikes_per_responder_median"]
                                                      for x in pos])),
            "kc_negative_fraction": float(neg["kc_active_fraction"]),
            "kc_spontaneous_fraction": float(nil["kc_active_fraction"]),
            "pop_rate_hz": float(np.mean([x["rest_rate_hz"] for x in pos]))}


def main(args: argparse.Namespace) -> int:
    from . import graph as graph_mod
    from .guard import GuardRefused, heavy
    from .sim import LIFParams

    cfg = config.load("fit")["kc_probe"]
    rows = []
    try:
        with heavy("kc-probe") as guard:
            print(f"[guard] 메모리 여유 {guard.free_percent}%", flush=True)
            base = graph_mod.load()
            lif = LIFParams.from_config()
            sets = stimulus_sets(base, cfg)
            for brain in cfg["brains"]:
                g = (base if brain == "real"
                     else graph_mod.rewire(base, int(brain.rsplit("-s", 1)[1])))
                for name, st in sets.items():
                    for seed in cfg["seeds"]:
                        r = {"brain": brain, "stimulus": name,
                             **run(g, lif, cfg, st, int(seed), args.device)}
                        rows.append(r)
                        w = {x["window_ms"]: x for x in r["windows"]}
                        mid = w.get(300.0) or r["windows"][-1]
                        print(f"[kc-probe] {brain:10} {name:12} 씨앗 {seed} · KC "
                              f"{mid['kc_active_fraction']:.4f} · PN "
                              f"{mid['stim_pn_rate_hz_mean']} Hz"
                              f" · APL {mid['apl_rate_hz']} Hz · 나머지 {mid['rest_rate_hz']} Hz"
                              f" · {r['wall_s']} s", flush=True)
    except GuardRefused as err:
        print(f"[guard] 막음 — {err}")
        return 2
    out = paths.runs_dir() / f"kc-probe-{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.write_text(json.dumps({"cfg": cfg, "sim": lif.version, "sets": sets, "rows": rows},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[kc-probe] {out}")
    return 0


def register(sub) -> None:
    p = sub.add_parser("kc-probe", help="KC 희소성 M0 — 냄새 자극 KC · PN 반응(무거움 · 짧음)")
    p.add_argument("--device", default="cpu", choices=["cpu", "mps"])
    p.set_defaults(func=main)
