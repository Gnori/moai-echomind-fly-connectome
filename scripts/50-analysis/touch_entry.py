"""촉각 입구 재기(R-1177 F-011 ① · 요구자 10-18 "시작" · 연결 지도만 · 뇌 안 돌림 · 판정 아님).

물음: 몸 부위별 촉각 털 뉴런을 연결 지도 주석으로 고를 수 있나, 그리고 그 무리에서 보기 창 몸을 움직이는
하강 뉴런(DNp09 앞 · MDN 뒤 · DNa01 · DNa02 돌기 · DNp01 GF · DNp02)까지 길이 얼마나 되나 — 섞은 배선과 나란히.

잰 것(무리 S · 하강 뉴런 T 마다):
  한 단계 몫 = S → T 시냅스 / T 가 받는 모든 시냅스
  두 단계 몫 = Σ_m (S → m 시냅스 / m 이 받는 시냅스) × (m → T 시냅스 / T 가 받는 시냅스)
  부호 붙인 두 단계 = 위에 출발 뉴런 부호(흥분 +1 · 억제 −1)를 곱함(S 는 감각이라 +)
  · 부위 무리마다 두 단계 몫이 큰 하강 뉴런 종류(어디로 가나)
값은 config/eyeworld.yaml touch_entry. 사용: python scripts/50-analysis/touch_entry.py → runs/touch-entry-<시각>.json
"""

from __future__ import annotations

import collections
import json
import time

import numpy as np

from moai_echomind_fly import config, paths
from moai_echomind_fly import graph as graph_mod
from moai_echomind_fly.loop.signals import side_of


def groups(g, cfg: dict) -> dict[str, np.ndarray]:
    """부위 × 좌우 → 뉴런 인덱스."""
    cls = g.columns["class"].astype(str)
    sub = g.columns["subclass"].astype(str)
    nerve = g.columns["entryNerve"].astype(str)
    side = side_of(g).astype(str)
    tact = np.isin(cls, cfg["tactile_class"])
    out = {}
    for part, nerves in cfg["parts"].items():
        for s in "LR":
            out[f"{part}_{s}"] = np.flatnonzero(tact & np.isin(nerve, nerves) & (side == s))
    typ = g.columns["type"].astype(str)
    hc = cfg["head"]
    head = (np.isin(cls, hc["class"]) & np.char.startswith(typ, hc["type_prefix"])
            & ~np.isin(typ, hc["exclude"]))
    ant = np.isin(cls, cfg["antenna"]["class"]) & np.isin(sub, cfg["antenna"]["subclass"])
    for s in "LR":
        out[f"머리털_{s}"] = np.flatnonzero(head & (side == s))
        out[f"더듬이털_{s}"] = np.flatnonzero(ant & (side == s))
    for name, types in cfg["reference"].items():   # 비교 기준(감각 아님)
        for s in "LR":
            out[f"[기준]{name}_{s}"] = np.flatnonzero(np.isin(typ, types) & (side == s))
    return out


class Flow:
    """그래프 하나의 입력 정규화 길(한 단계 · 두 단계)."""

    def __init__(self, g) -> None:
        self.g = g
        self.src = g.src()
        self.w = g.count.astype(np.float64)
        self.inn = np.bincount(g.dst, weights=self.w, minlength=g.n)
        self.sgn = g.sign.astype(np.float64)
        self._cache: dict = {}

    def first(self, s_idx: np.ndarray) -> np.ndarray:
        """S → 모든 뉴런: (S 에서 온 시냅스 / 그 뉴런이 받는 시냅스)."""
        m = np.isin(self.src, s_idx)
        a = np.bincount(self.g.dst[m], weights=self.w[m], minlength=self.g.n).astype(np.float64)
        return np.divide(a, self.inn, out=np.zeros_like(a), where=self.inn > 0)

    def to_targets(self, a: np.ndarray, t_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """한 단계 몫 a(뉴런마다)를 T 로 한 번 더: 부호 없음 · 부호 붙임(T 마다) · T 쪽 연결은 한 번만 고름."""
        key = (len(t_idx), int(t_idx[0]) if len(t_idx) else -1, int(t_idx.sum()))
        if key not in self._cache:
            m = np.isin(self.g.dst, t_idx)
            pre, post = self.src[m], self.g.dst[m]
            pos = np.full(self.g.n, -1, np.int64)
            pos[t_idx] = np.arange(len(t_idx))
            self._cache[key] = (pre, self.w[m] / self.inn[post], pos[post])
        pre, frac, k = self._cache[key]
        two = np.bincount(k, weights=a[pre] * frac, minlength=len(t_idx)).astype(np.float64)
        two_s = np.bincount(k, weights=a[pre] * frac * self.sgn[pre],
                            minlength=len(t_idx)).astype(np.float64)
        return two, two_s


def measure(g, grp: dict, targets: dict, dn_idx: np.ndarray, dn_type: np.ndarray,
            top: int) -> dict:
    f = Flow(g)
    out = {}
    for name, s_idx in grp.items():
        a = f.first(s_idx)
        row = {"n": int(len(s_idx))}
        for tname, t_idx in targets.items():
            two, two_s = f.to_targets(a, t_idx)
            row[tname] = {"one": float(a[t_idx].sum()), "two": float(two.sum()),
                          "two_signed": float(two_s.sum())}
        two_dn, _ = f.to_targets(a, dn_idx)
        by = collections.Counter()
        for t, v in zip(dn_type, two_dn + a[dn_idx]):
            by[str(t)] += float(v)
        row["dn_all"] = float((two_dn + a[dn_idx]).sum())
        row["top_dn"] = [(k, round(v, 5)) for k, v in by.most_common(top)]
        out[name] = row
    return out


def main() -> None:
    cfg = config.load("eyeworld")["touch_entry"]
    t0 = time.time()
    base = graph_mod.load()
    typ = base.columns["type"].astype(str)
    side = side_of(base).astype(str)
    grp = groups(base, cfg)
    targets = {f"{t}_{s}": np.flatnonzero((typ == t) & (side == s))
               for t in cfg["targets"] for s in "LR"}
    dn_idx = np.flatnonzero(base.columns["superclass"].astype(str) == "descending_neuron")
    res = {"brains": {}, "groups": {k: int(len(v)) for k, v in grp.items()},
           "targets": {k: int(len(v)) for k, v in targets.items()}, "config": cfg}
    for brain in ["real", *cfg["controls"]]:
        g = graph_mod.brain_graph(base, brain)
        res["brains"][brain] = measure(g, grp, targets, dn_idx, typ[dn_idx], int(cfg["top_dn"]))
        print(f"[{brain}] {time.time() - t0:.0f} s", flush=True)
    real = res["brains"]["real"]
    print("\n부위_쪽     n    하강 전체(두 단계)  진짜 / 섞은 최대   가장 큰 하강 종류(진짜)")
    for name in grp:
        r = real[name]["dn_all"]
        c = max(res["brains"][b][name]["dn_all"] for b in cfg["controls"])
        tops = " · ".join(f"{k} {v:.4f}" for k, v in real[name]["top_dn"][:4])
        print(f"{name:10s} {real[name]['n']:5d}  {r:9.4f}  {c:9.4f}   {tops}")
    print("\n몸 하강 뉴런(두 단계 몫 · 부호 붙임) — 진짜 | 섞은 s0")
    tn = list(targets)
    print("부위_쪽     " + " ".join(f"{t:>9s}" for t in tn))
    for name in grp:
        print(f"{name:10s} " + " ".join(f"{real[name][t]['two_signed']:9.5f}" for t in tn))
        c0 = res["brains"][cfg["controls"][0]][name]
        print(f"{'  섞은':10s} " + " ".join(f"{c0[t]['two_signed']:9.5f}" for t in tn))
    out = paths.runs_dir() / f"touch-entry-{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n→ {out}  ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
