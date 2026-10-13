"""가치 읽기 A1 판정(config/plasticity.yaml valence_readout vr-a · 결과 전 잠금 · 10-14).

뇌마다 plasticity-controls --detail 결과(mbon_counts · mbon_types · mbon_nt)를 읽어 씨앗마다
ΔV 바뀜 = [V(CS+) − V(CS−)]_뒤 − [ … ]_전 을 낸다. V = Σ s_i r_i(s 는 신경전달물질로 고정).
사용: python scripts/50-analysis/valence_judge.py → runs/valence-judge-<시각>.json
"""

from __future__ import annotations

import glob
import itertools
import json
import time

import numpy as np

from moai_echomind_fly import config, paths


def latest(brain: str) -> dict:
    f = sorted(glob.glob(str(paths.runs_dir() / f"plasticity-controls-{brain}-*.json")))[-1]
    d = json.loads(open(f, encoding="utf-8").read())
    d["_file"] = f
    return d


def signs(d: dict, vr: dict, type_signs: dict | None = None) -> np.ndarray:
    keep = set(vr["mbon_types"])
    s = np.zeros(len(d["mbon_types"]))
    for i, (t, nt) in enumerate(zip(d["mbon_types"], d["mbon_nt"])):
        if t in keep:
            s[i] = (type_signs or {}).get(t, vr["sign_by_nt"].get(nt, 0))
    return s


def changes(d: dict, s: np.ndarray, cond: str, seeds: list[int]) -> np.ndarray:
    rows = {r["seed"]: r for r in d["rows"] if r["cond"] == cond}
    out = []
    for sd in seeds:
        mc = rows[sd]["mbon_counts"]
        dv = [float(s @ np.array(mc["cs_plus"][k]) - s @ np.array(mc["cs_minus"][k])) for k in (0, 1)]
        out.append(dv[1] - dv[0])
    return np.array(out)


def p_exact(x: np.ndarray) -> float:
    obs = abs(x.mean())
    hit = sum(abs((x * np.array(sg)).mean()) >= obs - 1e-12
              for sg in itertools.product([1, -1], repeat=len(x)))
    return hit / 2 ** len(x)


def main() -> None:
    vr = config.load("plasticity")["valence_readout"]
    seeds = [int(x) for x in vr["seeds"]]
    real = latest("real")
    s = signs(real, vr)
    res = {"rule": vr, "real_file": real["_file"], "conditions": {}}
    for cond in ("paired", "punish", "unpaired", "shuffled", "off"):
        c = changes(real, s, cond, seeds)
        res["conditions"][cond] = {"change": c.tolist(), "mean": float(c.mean()), "p": p_exact(c),
                                   "up": int((c > 0).sum())}
        print(f"[valence] 진짜 {cond:8} 바뀜 평균 {c.mean():+.1f} · 올라감 {(c > 0).sum()}/10 · "
              f"p {p_exact(c):.4f}", flush=True)
    pm = res["conditions"]["paired"]["mean"]
    types = [t for t in vr["mbon_types"]]
    base = {t: vr["sign_by_nt"].get(next(nt for tt, nt in zip(real["mbon_types"], real["mbon_nt"])
                                         if tt == t), 0) for t in types}
    vals = list(base.values())
    rng = np.random.default_rng(int(vr["random_signs"]["seed"]))
    rand = []
    for _ in range(int(vr["random_signs"]["n"])):
        perm = dict(zip(types, rng.permutation(vals).tolist()))
        rand.append(float(changes(real, signs(real, vr, perm), "paired", seeds).mean()))
    rank = float(np.mean(np.array(rand) < pm))
    ps = vr["pass"]
    verdict = {
        "reward_up": bool(pm > 0 and res["conditions"]["paired"]["p"] < ps["reward_up"]["sign_flip_p"]),
        "punish_down": bool(res["conditions"]["punish"]["mean"] < 0
                            and res["conditions"]["punish"]["p"] < ps["punish_down"]["sign_flip_p"]),
        "controls_small": {c: bool(abs(res["conditions"][c]["mean"]) <= abs(pm) * ps["controls_small"])
                           for c in ("unpaired", "shuffled", "off")},
        "random_rank": rank, "random_pass": bool(rank >= ps["random_rank"])}
    res.update({"random_sign_means": rand, "verdict": verdict, "arms": {}})
    print(f"[valence] 무작위 부호 20 벌 짝짓기 바뀜 {np.round(sorted(rand), 1).tolist()} · 진짜 {pm:+.1f} · "
          f"순위 {rank:.2f}", flush=True)
    rc = np.array(res["conditions"]["paired"]["change"])
    for arm, brains in vr["arms"].items():
        ctl = {b: changes(latest(b), signs(latest(b), vr), "paired", seeds) for b in brains}
        ens = np.mean(np.array(list(ctl.values())), axis=0)
        above = all(pm > c.mean() for c in ctl.values())
        p = p_exact(rc - ens)
        res["arms"][arm] = {"means": {b: float(c.mean()) for b, c in ctl.items()}, "above_all": above,
                            "p": p, "pass": bool(above and p < 0.05)}
        print(f"[valence] {arm}: 섞은 셋 짝짓기 바뀜 {[round(c.mean(), 1) for c in ctl.values()]} · "
              f"진짜가 모두 위 {above} · p {p:.4f}", flush=True)
    print(f"[valence] 판정 {verdict}", flush=True)
    dst = paths.runs_dir() / f"valence-judge-{time.strftime('%Y%m%d-%H%M%S')}.json"
    dst.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[valence] {dst}")


if __name__ == "__main__":
    main()
