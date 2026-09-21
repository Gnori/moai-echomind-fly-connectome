"""한계 측정 셋 (R-3340 F-005) — 통과가 아니라 **기록**이 인수 기준이다.

1. 광수용체(R1–R6) → 라미나·메둘라: 침묵 LIF 는 이중 반전(ON 경로)을 못 낸다는 예상
   (FLYCNS RESULTS §2)을 잰다.
2. 청각(존스턴 기관) → 거대섬유: 생체 기록의 정답은 문턱 아래 탈분극(발화 0).
   JO→GF 직접 연결을 끄면 사라지는지.
3. 미각 → MN9(주둥이 운동 뉴런): 억제를 푸는 경로라 침묵 LIF 로는 재현 실패가 예상된다
   (FLYCNS · Shiu 한계).
"""

from __future__ import annotations

import argparse
from typing import Any

import numpy as np

from .. import graph as graph_mod
from ..guard import GuardRefused, heavy
from ..runs import RunRecorder
from ..sim import Engine, LIFParams

RATE_HZ = 150.0      # Shiu 의 기본 자극 발화율
DURATION_MS = 300.0
# FLYCNS: 해부학 기준(모달리티 주석은 없다)
GUSTATORY_SUBCLASSES = ["labellar bristle", "taste peg", "pharyngeal sensillum"]


def _count(neuron: np.ndarray, idx: np.ndarray) -> tuple[int, int]:
    m = np.isin(neuron, idx)
    return int(m.sum()), int(len(np.unique(neuron[m])))


def photoreceptors(g: graph_mod.Graph, params: LIFParams, seed: int) -> dict[str, Any]:
    stim = g.select(type="R1-R6")
    eng = Engine(g, params, seed=seed)
    eng.set_poisson(stim, RATE_HZ)
    eng.run(DURATION_MS)
    _, neuron = eng.spikes()
    lamina = g.select(type=["L1", "L2", "L3", "L4", "L5"])
    ol = np.setdiff1d(g.select(superclass="ol_intrinsic"), stim)
    downstream = np.setdiff1d(np.arange(g.n), stim)
    return {
        "stimulated": {"type": "R1-R6", "neurons": int(len(stim)), "rate_hz": RATE_HZ},
        "lamina_L1_L5": dict(zip(("spikes", "neurons_fired"), _count(neuron, lamina))),
        "optic_lobe_intrinsic": dict(zip(("spikes", "neurons_fired"), _count(neuron, ol))),
        "all_downstream": dict(zip(("spikes", "neurons_fired"), _count(neuron, downstream))),
        "expected": "안 퍼짐 — 히스타민(−)이 쉬는(침묵) 라미나를 억제해도 발화가 안 생긴다",
    }


def auditory(g: graph_mod.Graph, params: LIFParams, seed: int) -> dict[str, Any]:
    stim = g.select(subclass="auditory")
    gf = g.select(type="DNp01")
    out: dict[str, Any] = {"stimulated": {"subclass": "auditory", "neurons": int(len(stim)),
                                          "rate_hz": RATE_HZ}}
    for arm in ("intact", "jo_gf_cut"):
        eng = Engine(g, params, seed=seed)
        cut = 0
        if arm == "jo_gf_cut":
            for i in stim:
                lo, hi = int(g.row_ptr[i]), int(g.row_ptr[i + 1])
                hit = np.flatnonzero(np.isin(g.dst[lo:hi], gf)) + lo
                eng.w[hit] = 0
                cut += len(hit)
        eng.set_poisson(stim, RATE_HZ)
        eng.record_voltage(gf)
        eng.run(DURATION_MS)
        _, neuron = eng.spikes()
        v = eng.voltage()
        out[arm] = {
            "gf_spikes": int(np.isin(neuron, gf).sum()),
            "gf_peak_depol_mV": float((v.max(axis=0) - params.v0_mV).mean()),
            "gf_mean_depol_mV": float((v.mean(axis=0) - params.v0_mV).mean()),
            "edges_cut": cut,
        }
    out["expected"] = ("GF 발화 0 · 문턱(7 mV) 아래 탈분극 — "
                       "생체 기록(Pezier & Blagburn 2013)과 FLYCNS 채점")
    return out


def gustatory(g: graph_mod.Graph, params: LIFParams, seed: int) -> dict[str, Any]:
    stim = g.select(klass="gustatory", subclass=GUSTATORY_SUBCLASSES)
    mn9 = g.select(type="MN9")
    eng = Engine(g, params, seed=seed)
    eng.set_poisson(stim, RATE_HZ)
    eng.record_voltage(mn9)
    eng.run(DURATION_MS)
    _, neuron = eng.spikes()
    v = eng.voltage()
    return {
        "stimulated": {"class": "gustatory", "subclass": GUSTATORY_SUBCLASSES,
                       "neurons": int(len(stim)), "rate_hz": RATE_HZ},
        "mn9_spikes": int(np.isin(neuron, mn9).sum()),
        "mn9_peak_depol_mV": float((v.max(axis=0) - params.v0_mV).mean()),
        "mn9_min_mV_rel_rest": float((v.min(axis=0) - params.v0_mV).mean()),
        "expected": "재현 실패 — 억제를 푸는 경로(FLYCNS: 2 단 경로는 음, 3~5 단은 양)",
    }


def main(args: argparse.Namespace) -> int:
    params = LIFParams.from_config()
    try:
        with heavy("bench-limits") as guard:
            g = graph_mod.load(args.graph)
            rec = RunRecorder("limits", {"graph": g.name, "seed": args.seed,
                                         "rate_hz": RATE_HZ, "duration_ms": DURATION_MS})
            rec.log(f"[guard] 메모리 여유 {guard.free_percent}%")
            result = {
                "photoreceptors": photoreceptors(g, params, args.seed),
                "auditory": auditory(g, params, args.seed),
                "gustatory": gustatory(g, params, args.seed),
                "params_version": params.version,
            }
            rec.write_json("summary.json", result)
            rec.honesty(wiring=["세 경로의 전파 · 막전위"],
                        added=[f"자극 = 포아송 {RATE_HZ} Hz · {DURATION_MS} ms",
                               f"w_scale={params.w_scale} ({params.version})",
                               "JO→GF 연결 끄기(청각 대조) — 코드가 배선을 바꾼 것"])
            import json

            rec.log(json.dumps(result, ensure_ascii=False, indent=2))
            rec.log(f"기록: {rec.dir}")
    except GuardRefused as err:
        print(f"[guard] 막음 — {err}")
        return 2
    return 0
