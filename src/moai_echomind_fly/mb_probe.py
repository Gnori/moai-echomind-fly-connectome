"""KC 희소성 M0 — 학습 없이 지금 매개변수로 냄새 자극의 KC · PN · APL 반응을 잰다(연구 노트 16 §6).

자극 묶음(식초 6 · DM1+VA2 · 무작위 10 사구체 · 없음) × 포아송 씨앗 × 뇌(진짜 · 재배선)마다 한 번
가장 긴 창만큼 돌리고, 앞 구간(100 · 300 · 500 ms)을 잘라 센다:
KC 주요 종류 중 한 번 이상 발화한 비율 · 반응 KC 스파이크 수 중앙 · 자극 PN 평균 발화율(종류별) ·
켜진 PN 짝 수별 KC 반응률(발톱 진단) · APL 발화율 · 나머지 뇌 발화율(KC · APL · 자극 제외).
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
    lobes = {}
    if "kc_lobe" in idx:                     # 엽마다 켜진 비율(physiology-v4)
        lb = idx["kc_lobe"]
        lobes = {str(name): round(float((kc_c[lb == name] > 0).mean()), 5)
                 for name in sorted(set(lb.tolist())) if name}
    mbon = {}
    if "mbon" in idx and len(idx["mbon"]):   # MBON 반응(physiology-v5 · 범위 kc-mbon-a)
        mc = cnt[idx["mbon"]]
        mbon = {"mbon_fire_fraction": round(float((mc > 0).mean()), 5),
                "mbon_rate_hz": round(float(mc.mean() / sec), 3)}
    return {"window_ms": window_ms, "kc_active_fraction": round(float((kc_c > 0).mean()), 5),
            "kc_lobe_fractions": lobes, **mbon,
            "kc_spikes_per_responder_median": float(np.median(resp)) if len(resp) else 0.0,
            "stim_pn_rate_hz_mean": round(float(pn_rate.mean()), 1) if len(pn_rate) else None,
            "stim_pn_rate_by_type": by_type,
            "kc_response_by_active_claws": claw_resp,
            "apl_rate_hz": round(float(cnt[apl].sum() / max(len(apl), 1) / sec), 2),
            "rest_rate_hz": round(float(cnt[rest].sum() / max(len(rest), 1) / sec), 5)}


def run(graph: Graph, lif, cfg: dict[str, Any], stim_types: list[str], seed: int,
        device: str, cell=None, windows_ms: list[float] | None = None,
        keep_kc: bool = False) -> dict[str, Any]:
    """keep_kc 이면 첫 창에서 발화한 KC 번호(`kc_on` · numpy · JSON 에 싣지 않는다)도 돌려준다."""
    from .sim import Engine

    cls, types = graph.columns["class"], graph.columns["type"]
    sizes = Counter(types[cls == "Kenyon_Cell"])
    kc_types = [t for t, n in sizes.items() if n >= int(cfg["kc_min_type_size"])]
    kc = np.flatnonzero((cls == "Kenyon_Cell") & np.isin(types, kc_types))
    orn = graph.select(type=stim_types) if stim_types else np.zeros(0, np.int64)
    pn_s = stim_pns(graph, orn, cfg) if len(orn) else np.zeros(0, np.int64)
    apl = graph.select(type=cfg["apl_type"])
    excl = np.zeros(graph.n, bool)
    # 나머지 뇌 = KC(전부) · APL · 자극 ORN 밖(연구 노트 16 §4 cns_quiet 정의 · PN 은 포함)
    excl[np.concatenate([orn, apl, np.flatnonzero(cls == "Kenyon_Cell")])] = True
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
    if cfg.get("mbon_class"):
        idx["mbon"] = np.flatnonzero(cls == cfg["mbon_class"])
    if cfg.get("lobes"):
        lobe_of = {t: name for name, ts in cfg["lobes"].items() for t in ts}
        idx["kc_lobe"] = np.array([lobe_of.get(str(t), "") for t in types[kc]], dtype=object)
    out = {"stim_types": stim_types, "seed": seed, "kc_neurons": int(len(kc)),
           "stim_pns": int(len(pn_s)), "wall_s": round(time.time() - t0, 1),
           "windows": [summarize(graph, steps, neurons, lif.dt_ms, float(w), idx, kc_pn)
                       for w in windows]}
    if keep_kc:
        m = steps * lif.dt_ms < windows[0]
        out["kc_on"] = np.intersect1d(kc, neurons[m])
    return out


def max_jaccard(sets: list[np.ndarray]) -> float:
    """KC 집합들의 짝마다 Jaccard 중 가장 큰 값(physiology-v7 · 냄새끼리 KC 겹침)."""
    best = 0.0
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            u = len(np.union1d(sets[i], sets[j]))
            if u:
                best = max(best, len(np.intersect1d(sets[i], sets[j])) / u)
    return best


def extra_odor_sets(graph: Graph, cfg: dict[str, Any]) -> list[list[str]]:
    """학습에 더 쓰는 냄새 묶음(`extra_odors`: n · size · seed) — 양성 자극 밖 ORN 에서 씨앗마다.

    plasticity.cs_minus_types 와 같은 뽑기(씨앗만 다름) — 대조 후보(씨앗 1~20)와 씨앗을 겹치지 않게.
    """
    ex = cfg["extra_odors"]
    target = set(cfg["stimuli"][cfg["target_stimulus"]])
    orn = sorted({t for t in graph.columns["type"].astype(str)
                  if t.startswith("ORN_") and t not in target})
    return [sorted(np.random.default_rng(int(ex["seed"]) + i).choice(
        orn, size=int(ex["size"]), replace=False).tolist()) for i in range(int(ex["n"]))]


def measure_kc(graph: Graph, lif, cell, trials: int, seed: int, device: str,
               cfg: dict[str, Any] | None = None) -> dict[str, float]:
    """회로 학습(F-006) 평가용 — 양성 자극을 씨앗 `trials` 개로 · 음성 · 무자극 한 번씩.

    돌려주는 칸(physiology `mb.*` 측정 이름): kc_active_fraction · kc_spikes_per_responder ·
    kc_negative_fraction · kc_spontaneous_fraction · pop_rate_hz(양성 중 나머지 뇌) ·
    pop_rate_spont_hz(무자극 중 나머지 뇌) · 설정에 `lobes` 가 있으면 kc_lobe_fraction_min · max
    (엽 γ · αβ · α′β′ 마다 켜진 비율을 씨앗 평균한 뒤의 최소 · 최대) · `mbon_class` 가 있으면
    mbon_fire_fraction(양성 중 한 번 이상 발화한 MBON 비율) · mbon_rate_hz(MBON 평균 발화율) ·
    `extra_odors` · `odor_overlap` 이 있으면 kc_overlap_max_odor(학습 냄새끼리 KC Jaccard 최대).
    """
    cfg = cfg or config.load("fit")["kc_probe"]
    w = [float(cfg["loss_window_ms"])]
    ov = bool(cfg.get("odor_overlap"))       # 냄새끼리 KC 겹침(physiology-v7 · 범위 kc-mbon-c)
    pos_runs = [run(graph, lif, cfg, cfg["stimuli"][cfg["target_stimulus"]], seed + k, device,
                    cell, w, keep_kc=ov) for k in range(int(trials))]
    pos = [r["windows"][0] for r in pos_runs]
    neg = run(graph, lif, cfg, cfg["stimuli"][cfg["negative_stimulus"]], seed, device, cell,
              w)["windows"][0]
    nil = run(graph, lif, cfg, [], seed, device, cell, w)["windows"][0]
    lobe = {}
    if cfg.get("lobes"):                     # 씨앗 평균한 엽별 비율의 최소 · 최대(physiology-v4)
        mean = {k: float(np.mean([x["kc_lobe_fractions"][k] for x in pos]))
                for k in pos[0]["kc_lobe_fractions"]}
        lobe = {"kc_lobe_fraction_min": min(mean.values()),
                "kc_lobe_fraction_max": max(mean.values())}
    if cfg.get("mbon_class"):                # 씨앗 평균(physiology-v5)
        for k in ("mbon_fire_fraction", "mbon_rate_hz"):
            lobe[k] = float(np.mean([x[k] for x in pos]))
    if cfg.get("extra_odors"):               # 학습 냄새 여럿(physiology-v6 · 범위 kc-mbon-b)
        ex_runs = [run(graph, lif, cfg, st, seed, device, cell, w, keep_kc=ov)
                   for st in extra_odor_sets(graph, cfg)]
        ex = [r["windows"][0] for r in ex_runs]
        if ov:                               # 같은 씨앗(seed)의 식초와 학습 냄새 묶음끼리
            lobe["kc_overlap_max_odor"] = max_jaccard(
                [pos_runs[0]["kc_on"]] + [r["kc_on"] for r in ex_runs])
        kcs = [float(np.mean([x["kc_active_fraction"] for x in pos]))]
        kcs += [x["kc_active_fraction"] for x in ex]
        lobe["kc_fraction_min_odor"] = min(kcs)
        lobe["kc_fraction_max_odor"] = max(kcs)
        if cfg.get("mbon_class"):
            mbs = [lobe["mbon_fire_fraction"]] + [x["mbon_fire_fraction"] for x in ex]
            lobe["mbon_fire_min_odor"] = min(mbs)
    return {**lobe,"kc_active_fraction": float(np.mean([x["kc_active_fraction"] for x in pos])),
            "kc_spikes_per_responder": float(np.mean([x["kc_spikes_per_responder_median"]
                                                      for x in pos])),
            "kc_negative_fraction": float(neg["kc_active_fraction"]),
            "kc_spontaneous_fraction": float(nil["kc_active_fraction"]),
            "pop_rate_hz": float(np.mean([x["rest_rate_hz"] for x in pos])),
            "pop_rate_spont_hz": float(nil["rest_rate_hz"])}


def active_kc(graph: Graph, lif, cfg: dict[str, Any], stim: list[str], seed: int, device: str,
              cell=None) -> tuple[np.ndarray, float]:
    """이 자극 · 창에서 발화한 KC 번호와 '켜진 PN 짝 ≥ 2' 인 반응 KC 비율(발톱 일치)."""
    from .sim import Engine

    cls, types = graph.columns["class"], graph.columns["type"]
    sizes = Counter(types[cls == "Kenyon_Cell"])
    kc = np.flatnonzero((cls == "Kenyon_Cell") & np.isin(
        types, [t for t, n in sizes.items() if n >= int(cfg["kc_min_type_size"])]))
    orn = graph.select(type=stim)
    eng = Engine(graph, lif, device=device, seed=seed, cell=cell)
    eng.set_poisson(orn, np.full(len(orn), float(cfg["orn_rate_hz"])))
    eng.run(float(cfg["loss_window_ms"]))
    _, neurons = eng.spikes()
    cnt = np.bincount(neurons, minlength=graph.n)
    on = kc[cnt[kc] > 0]
    src = graph.src()
    pn = cls == cfg["pn_class"]
    e = np.isin(graph.dst, on) & pn[src] & (cnt[src] > 0)
    claws = np.bincount(graph.dst[e], minlength=graph.n)[on]
    return on, (float((claws >= 2).mean()) if len(on) else float("nan"))


def evaluate_fit(run_dir, device: str = "cpu") -> dict[str, Any]:
    """M2 — 학습한 매개변수로 일반화(안 쓴 무작위 세트) · 손실 측정 · 도망 재채점 · 발톱 일치."""
    from pathlib import Path

    from . import cellparams as cp
    from . import graph as graph_mod
    from .circuit_fit import measure_escape
    from .sim import LIFParams

    run_dir = Path(run_dir)
    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    cell = cp.CellParams.load(run_dir / "params.npz")
    cfg = config.load("fit")["kc_probe"]
    gen = cfg["generalize"]
    base = graph_mod.load()
    brain = meta["brain"]
    g = base if brain == "real" else graph_mod.rewire(base, int(brain.rsplit("-s", 1)[1]))
    lif = LIFParams.from_config()
    orn = sorted({t for t in g.columns["type"].astype(str) if t.startswith("ORN_")})
    rng = np.random.default_rng(int(gen["seed"]))
    sets = [sorted(rng.choice(orn, size=int(cfg["random_size"]), replace=False).tolist())
            for _ in range(int(gen["sets"]))]
    out: dict[str, Any] = {"run": run_dir.name, "brain": brain, "sets": sets}
    for name, c in (("before", None), ("after", cell)):
        fr = [run(g, lif, cfg, st, 0, device, c, [float(cfg["loss_window_ms"])])["windows"][0]
              ["kc_active_fraction"] for st in sets]
        on, claw = active_kc(g, lif, cfg, cfg["stimuli"][cfg["target_stimulus"]], 0, device, c)
        esc = measure_escape(g, lif, c, int(gen["escape_trials"]), 0, device)
        out[name] = {"loss_measures": measure_kc(g, lif, c, 3, 0, device, cfg),
                     "generalize_fractions": [round(x, 5) for x in fr],
                     "generalize_in_range": bool(all(gen["range"][0] <= x <= gen["range"][1]
                                                     for x in fr)),
                     "target_active_kc": on.tolist(), "claw_consistency": claw,
                     "escape": esc}
    return out


def main_eval(args: argparse.Namespace) -> int:
    from .guard import GuardRefused, heavy

    try:
        with heavy("fit-eval") as guard:
            print(f"[guard] 메모리 여유 {guard.free_percent}%", flush=True)
            res = evaluate_fit(args.run, args.device)
    except GuardRefused as err:
        print(f"[guard] 막음 — {err}")
        return 2
    out = paths.runs_dir() / f"fit-eval-{res['run']}.json"
    out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    for k in ("before", "after"):
        r = res[k]
        print(f"[fit-eval] {res['brain']} {k:6} KC {r['loss_measures']['kc_active_fraction']:.4f}"
              f" · 일반화 {r['generalize_fractions']} 범위 안 {r['generalize_in_range']}"
              f" · 발톱 일치 {r['claw_consistency']:.3f} · GF {r['escape']['gf_hit']:.2f}",
              flush=True)
    print(f"[fit-eval] {out}")
    return 0


def overlap(a_json: str, b_json: str) -> dict[str, float]:
    """두 fit-eval 결과(진짜 · 재배선)의 식초 자극 켜진 KC 집합 겹침(Jaccard) — 학습 전 · 뒤."""
    a, b = (json.loads(open(p, encoding="utf-8").read()) for p in (a_json, b_json))
    out = {}
    for k in ("before", "after"):
        x, y = set(a[k]["target_active_kc"]), set(b[k]["target_active_kc"])
        out[k] = round(len(x & y) / max(len(x | y), 1), 4)
    return out


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
    e = sub.add_parser("fit-eval", help="회로 학습 뒤 M2 — 일반화 · 도망 재채점(무거움 · 짧음)")
    e.add_argument("--run", required=True, help="runs/fit-<범위>-<판>-<뇌>-<시각> 폴더")
    e.add_argument("--device", default="cpu", choices=["cpu", "mps"])
    e.set_defaults(func=main_eval)
