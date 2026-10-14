"""가치 읽기 A1 — 정규화 읽기 탐색(10-14 · 결과 뒤라 판정 아님 · 판정은 vr-a 그대로).

왜: A1 보조 비교에서 KC→MBON 섞은 뇌의 짝짓기 바뀜(122~162)이 진짜(114)보다 컸다. 섞은 뇌는 MBON 반응 자체가
커서(짝짓기 전 CS+ 4,700~5,000 대 3,000) 날 발화 수 차이가 클 수 있다 — 크기 탓인지 가른다.
읽기(계산 전에 정함 · 셋 모두 보고):
  N0 날 값 — vr-a 그대로(재현 확인)
  N1 뇌 크기로 나눔 — 바뀜 / 읽기 뉴런(55) 짝짓기 전 CS+ 발화 합 × 1000(씨앗마다)
  N2 뉴런마다 자기 크기로 나눔 — r̃_i = r_i / (½(r_i^{CS+,전} + r_i^{CS−,전}) + 1) 로 V 를 다시 냄
  N3 가치 비율 — V̂ = Σ s_i r_i ÷ Σ |s_i| r_i(냄새 · 전후마다 · Sakagiannis 2026 MB_out 형) · ⚠️ N0~N2 결과를 본 뒤 더함(A2 읽기 후보라서)
덧붙여 날 바뀜을 부호 무리(− 글루탐산 · + ACh · GABA)로 나눠 어디서 커지는지 본다.
입력: 판정 파일(runs/valence-judge-*.json)이 읽은 파일 — 진짜는 기록된 경로 · 섞은 뇌는 판정 시각 이전 마지막 파일.
사용: python scripts/50-analysis/valence_norm.py [runs/valence-judge-….json] → runs/valence-norm-<시각>.json
"""

from __future__ import annotations

import glob
import json
import sys
import time
from pathlib import Path

import numpy as np
from valence_judge import p_exact, signs

from moai_echomind_fly import config, paths


def judge_file(arg: str | None) -> Path:
    if arg:
        return Path(arg)
    return Path(sorted(glob.glob(str(paths.runs_dir() / "valence-judge-*.json")))[-1])


def at_or_before(brain: str, stamp: str) -> dict:
    """판정 시각(파일 이름의 YYYYMMDD-HHMMSS) 이전의 마지막 결과 — 판정이 읽은 것과 같은 파일."""
    fs = [f for f in sorted(glob.glob(str(paths.runs_dir() / f"plasticity-controls-{brain}-*.json")))
          if f.rsplit("-", 2)[-2] + "-" + f.rsplit("-", 1)[-1][:6] <= stamp]
    d = json.loads(open(fs[-1], encoding="utf-8").read())
    d["_file"] = fs[-1]
    return d


def counts(d: dict, cond: str, seeds: list[int]) -> dict[int, dict]:
    rows = {r["seed"]: r for r in d["rows"] if r["cond"] == cond}
    return {sd: {k: np.array(v, dtype=float) for k, v in rows[sd]["mbon_counts"].items()} for sd in seeds}


def change(mc: dict, s: np.ndarray, how: str) -> float:
    """씨앗 하나의 바뀜 = [V(CS+) − V(CS−)]뒤 − 전 · how = N0 | N1 | N2."""
    cp, cm = mc["cs_plus"], mc["cs_minus"]  # [전, 뒤] × 뉴런
    if how == "N2":
        scale = 0.5 * (cp[0] + cm[0]) + 1.0
        cp, cm = cp / scale, cm / scale
    if how == "N3":
        a = np.abs(s)
        dv = [float(s @ cp[k] / max(a @ cp[k], 1.0) - s @ cm[k] / max(a @ cm[k], 1.0)) for k in (0, 1)]
        return dv[1] - dv[0]
    dv = [float(s @ cp[k] - s @ cm[k]) for k in (0, 1)]
    out = dv[1] - dv[0]
    if how == "N1":
        out = out / float((s != 0) @ mc["cs_plus"][0]) * 1000.0
    return out


def by_group(mc: dict, s: np.ndarray) -> dict[str, float]:
    """날 바뀜을 부호 무리로 — 피함(−) 쪽 · 다가감(+) 쪽 몫."""
    cp, cm = mc["cs_plus"], mc["cs_minus"]
    out = {}
    for name, mask in (("avoid_glu", s < 0), ("approach_ach_gaba", s > 0)):
        ss = s * mask
        out[name] = float((ss @ cp[1] - ss @ cm[1]) - (ss @ cp[0] - ss @ cm[0]))
    return out


def main() -> None:
    vr = config.load("plasticity")["valence_readout"]
    seeds = [int(x) for x in vr["seeds"]]
    jf = judge_file(sys.argv[1] if len(sys.argv) > 1 else None)
    judge = json.loads(jf.read_text(encoding="utf-8"))
    stamp = jf.stem.removeprefix("valence-judge-")
    real = json.loads(open(judge["real_file"], encoding="utf-8").read())
    real["_file"] = judge["real_file"]
    brains = {"real": real} | {b: at_or_before(b, stamp) for v in vr["arms"].values() for b in v}
    res = {"judge_file": str(jf), "rule": vr["version"], "files": {b: d["_file"] for b, d in brains.items()},
           "pre_cs_plus_total": {}, "groups": {}, "reads": {}}
    ch = {}
    for b, d in brains.items():
        s = signs(d, vr)
        mc = counts(d, "paired", seeds)
        res["pre_cs_plus_total"][b] = float(np.mean([(s != 0) @ mc[sd]["cs_plus"][0] for sd in seeds]))
        g = [by_group(mc[sd], s) for sd in seeds]
        res["groups"][b] = {k: float(np.mean([x[k] for x in g])) for k in g[0]}
        ch[b] = {h: np.array([change(mc[sd], s, h) for sd in seeds]) for h in ("N0", "N1", "N2", "N3")}
    print("[norm] 짝짓기 전 CS+ 읽기 뉴런 발화 합(씨앗 평균): "
          + " · ".join(f"{b} {v:,.0f}" for b, v in res["pre_cs_plus_total"].items()), flush=True)
    print("[norm] 날 바뀜 부호 무리(피함 글루탐산 / 다가감 ACh·GABA): "
          + " · ".join(f"{b} {g['avoid_glu']:+.1f}/{g['approach_ach_gaba']:+.1f}"
                       for b, g in res["groups"].items()), flush=True)
    rs = brains["real"]
    s_real = signs(rs, vr)
    for h in ("N0", "N1", "N2", "N3"):
        rc = ch["real"][h]
        real_conds = {}
        for cond in ("paired", "punish", "unpaired", "shuffled", "off"):
            mc = counts(rs, cond, seeds)
            c = np.array([change(mc[sd], s_real, h) for sd in seeds])
            real_conds[cond] = {"mean": float(c.mean()), "p": p_exact(c), "up": int((c > 0).sum())}
        arms = {}
        for arm, bs in vr["arms"].items():
            ens = np.mean([ch[b][h] for b in bs], axis=0)
            arms[arm] = {"means": {b: float(ch[b][h].mean()) for b in bs},
                         "above_all": bool(all(rc.mean() > ch[b][h].mean() for b in bs)),
                         "p": p_exact(rc - ens)}
        res["reads"][h] = {"real": real_conds, "arms": arms}
        pm = real_conds["paired"]["mean"]
        ctl = max(abs(real_conds[c]["mean"]) for c in ("unpaired", "shuffled", "off")) / abs(pm)
        print(f"[norm] {h}: 진짜 짝짓기 {pm:+.3f}(p {real_conds['paired']['p']:.4f}) · 처벌 "
              f"{real_conds['punish']['mean']:+.2f} · 대조 최대 / 짝짓기 {ctl:.2f}", flush=True)
        for arm, a in arms.items():
            print(f"[norm] {h}: {arm} 섞은 셋 {[round(v, 3) for v in a['means'].values()]} · "
                  f"진짜가 모두 위 {a['above_all']} · p {a['p']:.4f}", flush=True)
    dst = paths.runs_dir() / f"valence-norm-{time.strftime('%Y%m%d-%H%M%S')}.json"
    dst.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[norm] {dst}")


if __name__ == "__main__":
    main()
