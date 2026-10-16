"""V0d — 뇌 쪽 필요 입력량(R-1177 · 요구자 10-17 "뇌 쪽 필요량 먼저 재기").

결과 전 잠금: config/eyeworld.yaml v0d(rule v0d-a).
V0 분석식 회전 흐름(cw · ccw · 90 °/s)의 T4/T5 발화율을 배율 k 로 줄여 가며
DNa02 좌우 차(R − L)의 부호가 cw · ccw 에서 뒤집히는지(V1 R1 과 같은 판정)를 잰다.
'새는' 꼴: 반대 방향 아형도 선호의 leak 배로 켬(F0-d 눈높이 띠의 선호 ÷ 반대 약 2.2 를 흉내).
입력량은 V1 F0-3 과 같은 단위(앞→뒤 흐름 눈의 T4a + T5a 평균 Hz)로 적는다.
사용: python scripts/60-eyeworld/v0_dose.py [--time-only]
     → runs/v0-dose-<시각>/{rows.jsonl, judge.json}
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from moai_echomind_fly import config, paths
from moai_echomind_fly import graph as graph_mod
from moai_echomind_fly.eye import flow
from moai_echomind_fly.eye.geometry import Positions, t4t5_positions
from moai_echomind_fly.guard import heavy
from moai_echomind_fly.loop.signals import side_of
from moai_echomind_fly.sim import Engine, LIFParams

READ = {"DNa02": ["DNa02"], "HS": ["HSE", "HSN", "HSS"]}
DIRS = ["cw", "ccw"]  # cw = 세계가 위에서 보아 시계(몸이 왼쪽으로 돎 · V1 rot-left 와 같은 흐름)


def stim_rates(pos: Positions, direction: str, cfg: dict, k: float) -> np.ndarray:
    """회전 흐름 → 뉴런마다 발화율 × k(새는 꼴이면 반대 방향 아형을 leak 배로 더함)."""
    r_max, v_ref = float(cfg["r_max_hz"]), float(cfg["v_ref_deg_s"])
    m_az, m_el = flow.rotation_field(pos, float(cfg["speed_deg_s"]), direction)
    pref = flow.rates(pos, m_az, m_el, r_max, v_ref)
    null = flow.rates(pos, -m_az, -m_el, r_max, v_ref)  # 반대 방향 흐름에서의 발화율
    return np.minimum(k * (pref + float(cfg["leak"]) * null), r_max)


def ftb_a_mean(pos: Positions, rates: np.ndarray, direction: str) -> float:
    """앞→뒤 흐름 눈(cw = 오른눈 · ccw = 왼눈)의 T4a + T5a 평균 Hz — V1 F0-3 과 같은 단위."""
    eye = "R" if direction == "cw" else "L"
    return float(rates[(pos.side == eye) & (pos.sub == "a")].mean())


def judge(rows: list[dict], cfg: dict, dose_hz: dict[str, float]) -> dict:
    ps = cfg["pass"]
    out: dict = {"rule": cfg["rule"], "doses": {}}
    ok_by_k = {}
    for k in map(str, cfg["doses"]):
        def diff(arm: str, d: str, k: str = k) -> np.ndarray:
            sel = [r for r in rows if r["arm"] == arm and r["dir"] == d and r["k"] == k]
            sel = sorted(sel, key=lambda r: r["seed"])
            return np.array([r["DNa02_R"] - r["DNa02_L"] for r in sel], float)

        cw, ccw = diff("real", "cw"), diff("real", "ccw")
        flips = int(np.sum(np.sign(cw) * np.sign(ccw) < 0))
        real_abs = float(np.mean(np.abs(np.concatenate([cw, ccw]))))
        rew = {a: float(np.mean(np.abs(np.concatenate([diff(a, "cw"), diff(a, "ccw")]))))
               for a in cfg["rewired"]}
        ok = flips >= int(ps["flip_min"]) and all(
            v == 0 or v < float(ps["rewired_max_ratio"]) * real_abs for v in rew.values()
        )
        ok_by_k[k] = ok
        out["doses"][k] = {
            "ftb_a_hz": dose_hz[k], "flips": flips, "n": len(cw), "real_mean_abs": real_abs,
            "rewired_mean_abs": rew, "cw_RminusL": cw.tolist(), "ccw_RminusL": ccw.tolist(),
            "ok": bool(ok),
        }
    ks = list(map(str, cfg["doses"]))
    th = None
    for i, k in enumerate(ks):  # 이 배율과 그 위 모두 ok 인 가장 작은 배율
        if all(ok_by_k[x] for x in ks[i:]):
            th = k
            break
    out["threshold_k"] = th
    out["threshold_hz"] = dose_hz[th] if th is not None else None
    v1 = float(cfg["v1_measured_hz"])
    if th is None:
        out["possibility"] = "낮음(가장 센 배율에서도 부호 없음)"
    elif v1 >= float(cfg["high_margin"]) * dose_hz[th]:
        out["possibility"] = "높음"
    elif v1 >= dose_hz[th]:
        out["possibility"] = "중간"
    else:
        out["possibility"] = "낮음"
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--time-only", action="store_true", help="한 번만 돌려 시간만(결과 안 읽음)")
    a = ap.parse_args()
    cfg = config.load("eyeworld")["v0d"]
    out = paths.runs_dir() / f"v0-dose-{time.strftime('%Y%m%d-%H%M%S')}"
    out.mkdir(parents=True)
    n_bins = int(round(float(cfg["window_ms"]) / float(cfg["bin_ms"])))
    rows: list[dict] = []
    t_start = time.time()
    with heavy("v0-dose") as guard:
        print(f"[guard] 메모리 여유 {guard.free_percent}%", flush=True)
        base = graph_mod.load()
        lif = LIFParams.from_config()
        typ = base.columns["type"].astype(str)
        side = side_of(base)
        groups = {f"{k}_{s}": np.flatnonzero(np.isin(typ, v) & (side == s))
                  for k, v in READ.items() for s in ("L", "R")}
        pos = t4t5_positions(base)
        sched = {(str(k), d): stim_rates(pos, d, cfg, float(k)) for k in cfg["doses"] for d in DIRS}
        dose_hz = {str(k): ftb_a_mean(pos, sched[(str(k), "cw")], "cw") for k in cfg["doses"]}
        print(f"[v0d] 배율 → 앞→뒤 눈 a 평균 Hz {dose_hz}", flush=True)
        if a.time_only:
            eng = Engine(base, lif, device="cpu", seed=0)
            eng.set_poisson(pos.idx, np.zeros(len(pos)))
            t0 = time.time()
            eng.reset()
            for _ in range(n_bins):
                eng.set_rates(sched[(str(cfg["doses"][0]), "cw")])
                eng.run(float(cfg["bin_ms"]))
            print(f"[v0d] 한 번 {time.time() - t0:.1f} s(결과 안 읽음)", flush=True)
            return
        f = (out / "rows.jsonl").open("w", encoding="utf-8")
        arms = [("real", "real", [int(s) for s in cfg["seeds"]])] + [
            (b, b, [int(s) for s in cfg["rewired_seeds"]]) for b in cfg["rewired"]
        ]
        for arm, brain, seeds in arms:
            g = graph_mod.brain_graph(base, brain)
            for seed in seeds:
                eng = Engine(g, lif, device="cpu", seed=seed)
                eng.set_poisson(pos.idx, np.zeros(len(pos)))
                for k in map(str, cfg["doses"]):
                    for d in DIRS:
                        eng.reset()
                        for _ in range(n_bins):
                            eng.set_rates(sched[(k, d)])
                            eng.run(float(cfg["bin_ms"]))
                        _, neurons = eng.spikes()
                        cnt = np.bincount(neurons, minlength=g.n)
                        row = {"arm": arm, "k": k, "dir": d, "seed": seed,
                               **{n: int(cnt[ix].sum()) for n, ix in groups.items()}}
                        rows.append(row)
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")
                        f.flush()
                print(f"[v0d] {arm} 씨앗 {seed} 끝 · {time.time() - t_start:.0f} s", flush=True)
        f.close()
    res = judge(rows, cfg, dose_hz)
    res["cost_s"] = round(time.time() - t_start, 1)
    (out / "judge.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    for k, v in res["doses"].items():
        print(f"[v0d] k {k} · 앞→뒤 눈 a {v['ftb_a_hz']:.1f} Hz · 뒤집힘 {v['flips']}/{v['n']}"
              f" · 진짜 |R−L| {v['real_mean_abs']:.1f} · 재배선 {v['rewired_mean_abs']}"
              f" · ok {v['ok']}", flush=True)
    print(f"[v0d] 문턱 k {res['threshold_k']} · {res['threshold_hz']} Hz"
          f" · V1 {cfg['v1_measured_hz']} Hz → 가능성 {res['possibility']}"
          f" · {res['cost_s']} s · {out}")


if __name__ == "__main__":
    main()
