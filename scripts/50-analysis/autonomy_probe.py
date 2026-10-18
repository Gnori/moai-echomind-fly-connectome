"""스스로 켜지나 — 가능성 작은 시험(R-5402 2.1 · 요구자 10-18 "둘 다 — 시험 먼저" · 판정 아님).

입력 없이 바탕 활동만: 감각 뉴런(광수용체 뺌)이 쉬는 발화를 하게 두고(포아송 r Hz · 엔진 그대로)
window_ms 동안 → 몸 하강 뉴런 좌우 · 하강 뉴런 전체 · 발화가 많은 하강 뉴런 종류 · 뇌 전체 발화 몫 · 진짜 대 섞은 배선 셋.
다음으로 갈지는 config next_rule(결과 전 잠금). 값은 config/eyeworld.yaml autonomy_probe.
사용: python scripts/50-analysis/autonomy_probe.py → runs/autonomy-probe-<시각>.json
"""

from __future__ import annotations

import collections
import json
import sys
import time

import numpy as np

from moai_echomind_fly import config, paths
from moai_echomind_fly import graph as graph_mod
from moai_echomind_fly.guard import heavy
from moai_echomind_fly.loop.signals import side_of
from moai_echomind_fly.sim import Engine, LIFParams


def main() -> None:
    key = sys.argv[1] if len(sys.argv) > 1 else "autonomy_probe"   # autonomy_probe_b = 감각 종류마다 문헌 값
    cfg = config.load("eyeworld")[key]
    res: dict = {"rule": cfg, "runs": []}
    t0 = time.time()
    with heavy("autonomy-probe") as guard:
        print(f"[guard] 메모리 여유 {guard.free_percent}%", flush=True)
        base = graph_mod.load()
        lif = LIFParams.from_config()
        typ = base.columns["type"].astype(str)
        side = side_of(base).astype(str)
        sup = base.columns["superclass"].astype(str)
        cls = base.columns["class"].astype(str)
        arms = cfg.get("arms") or {f"{r:g}Hz": None for r in cfg["rates_hz"]}
        sens = np.isin(sup, cfg.get("sensory_superclass", [])) & ~np.isin(cls, cfg.get("exclude_class", []))
        read = {f"{t}_{s}": np.flatnonzero((typ == t) & (side == s)) for t in cfg["motor"] for s in "LR"}
        dn = np.flatnonzero(sup == "descending_neuron")
        sec = float(cfg["window_ms"]) / 1000.0
        print(f"[auto] 갈래 {list(arms)} · 하강 뉴런 {len(dn)}", flush=True)
        for brain in cfg["brains"]:
            g = graph_mod.brain_graph(base, brain)
            for arm, per in arms.items():
                if per is None:   # 판 a: 감각 전체 같은 세기
                    stim = np.flatnonzero(sens)
                    rates = np.full(len(stim), float(arm[:-2]))
                else:             # 판 b: 감각 종류(class)마다 문헌 값
                    stim = np.flatnonzero(np.isin(cls, list(per)))
                    rates = np.array([float(per[c]) for c in cls[stim]])
                eng = Engine(g, lif, device="cpu", seed=int(cfg["seed"]))
                eng.set_poisson(stim, rates)
                rate = arm
                eng.run(float(cfg["window_ms"]))
                _, neurons = eng.spikes()
                cnt = np.bincount(neurons, minlength=g.n)
                cnt[stim] = 0
                by = collections.Counter()
                for t, v in zip(typ[dn], cnt[dn]):
                    if v:
                        by[str(t)] += int(v)
                motor = sum(int(cnt[ix].sum()) for ix in read.values())
                nonstim = g.n - len(stim)
                r = {"brain": brain, "arm": rate, "n_stim": int(len(stim)),
                     "spiking_frac": round(float((cnt > 0).sum()) / nonstim, 4),
                     "dn_spiking": int((cnt[dn] > 0).sum()), "dn_spikes_per_s": round(cnt[dn].sum() / sec, 1),
                     "motor_per_s": round(motor / sec, 1),
                     **{k: int(cnt[ix].sum()) for k, ix in read.items()},
                     "top_dn": by.most_common(int(cfg["top_dn"]))}
                res["runs"].append(r)
                print(f"[auto] {brain:11} {rate} · 켬 {len(stim)} · 퍼짐 {r['spiking_frac']:.3f} · 하강 {r['dn_spiking']:4}"
                      f"/{r['dn_spikes_per_s']:7.1f}/s · 몸 {r['motor_per_s']:5.1f}/s"
                      f" · 켜진 대응 {[(k, v) for k, v in r.items() if k[-2:] in ('_L', '_R') and v][:8]}"
                      f" · 위 {r['top_dn'][:3]} · {time.time() - t0:.0f} s", flush=True)
    tag = cfg["version"]
    dst = paths.runs_dir() / f"{tag}-{time.strftime('%Y%m%d-%H%M%S')}.json"
    dst.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[auto] {dst}")


if __name__ == "__main__":
    main()
