"""나침반 봉우리 검사 (R-7469 F-008 3단계) — 자료 v2 의 공간 채널(EPG)이 몸 방향을 따라가나.

EPG 는 인스턴스 이름에 PB 구획이 있다(예: `EPG(PB08)_L3`). PB 반쪽마다 구획 1~8 이
360° 를 한 번 덮는다(Wolff et al. 2015 · Hulse et al. 2021 — 연구 노트 12 [3D-2])
→ 구획 i 에 (i − 1) × 45° 를 준다.
결정마다 발화율로 가중한 벡터 합의 **위상**과 **길이**(봉우리가 한쪽에 몰렸나)를 구하고,
위상과 실제 몸 방향(참값 `heading_deg`)의 **원형 상관**(Jammalamadaka)을 잰다.
방향 · 기준각은 알 수 없으므로 상관의 크기(|r|)를 본다. 귀무 = 에피소드 단위로 방향을 섞는다.

⚠️ 이 모델은 쉬는 상태가 침묵 · 신경조절 · 전기 시냅스가 없어 봉우리가 서지 않을 수 있다
— 실패도 기록한다.
"단서를 끈 뒤 유지"는 자료로는 못 잰다(몸 없이 뇌만 돌리는 검사가 따로 필요).
"""

from __future__ import annotations

import argparse
import json
import re
import time
from typing import Any

import numpy as np

from .. import datasets, paths

_GLOM = re.compile(r"_([LR])(\d)")


def epg_units(units: list[str]) -> list[tuple[int, float]]:
    """공간 채널 단위 중 EPG → (단위 번호, 구획 각도 라디안). 구획 9 는 뺀다(8 구획이 360°)."""
    out = []
    for i, u in enumerate(units):
        name = u.rsplit(":", 1)[0]
        if not name.startswith("EPG("):
            continue
        m = _GLOM.search(name)
        if m and 1 <= int(m.group(2)) <= 8:
            out.append((i, np.deg2rad((int(m.group(2)) - 1) * 45.0)))
    return out


def circ_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Jammalamadaka 원형 상관(라디안)."""
    sa = np.sin(a - np.angle(np.mean(np.exp(1j * a))))
    sb = np.sin(b - np.angle(np.mean(np.exp(1j * b))))
    den = np.sqrt((sa ** 2).sum() * (sb ** 2).sum())
    return float((sa * sb).sum() / den) if den > 0 else 0.0


def phase_of(feats: np.ndarray, units: list[str], bins: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """결정마다 EPG 봉우리 위상 · 길이(0 = 고르게 · 1 = 한 구획). 발화가 없으면 길이 0."""
    n = len(units)
    rate = feats[:, : bins * n].reshape(-1, bins, n).mean(1)
    ep = epg_units(units)
    idx = np.array([i for i, _ in ep])
    ang = np.array([a for _, a in ep])
    w = np.clip(rate[:, idx], 0, None)
    z = (w * np.exp(1j * ang)).sum(1)
    tot = w.sum(1)
    length = np.where(tot > 0, np.abs(z) / np.maximum(tot, 1e-12), 0.0)
    return np.angle(z), length


def check(dataset: str, n_perm: int = 2000, seed: int = 0) -> dict[str, Any]:
    meta = json.loads((datasets.store_dir() / dataset / "meta.json").read_text())
    units = meta["channel_units"]["space"]
    brains = ["real", *datasets.rewired_brains(dataset)]
    rng = np.random.default_rng(seed)
    out: dict[str, Any] = {"dataset": dataset, "epg_units": len(epg_units(units)), "brains": {},
                           "when": time.strftime("%Y-%m-%d %H:%M:%S")}
    for brain in brains:
        feats, rows = [], []
        for split in ("train", "eval"):
            f, r = datasets.load(dataset, split, brain)
            feats.append(f["space"])
            rows += r
        x = np.concatenate(feats)
        heading = np.deg2rad(np.array([r["truth"]["heading_deg"] for r in rows]))
        eps = np.array([r["seed"] for r in rows])
        phase, length = phase_of(x, units)
        live = length > 0
        r_obs = abs(circ_corr(phase[live], heading[live])) if live.sum() > 2 else 0.0
        # 귀무: 에피소드마다 방향 줄을 다른 에피소드 것과 바꾼다(길이가 달라 원형 이동으로 맞춘다)
        ids = np.unique(eps[live])
        null = []
        for _ in range(n_perm):
            perm = rng.permutation(ids)
            h = heading[live].copy()
            for a, b in zip(ids, perm):
                src = heading[live][eps[live] == b]
                dst = eps[live] == a
                h[dst] = np.resize(np.roll(src, rng.integers(len(src))), dst.sum())
            null.append(abs(circ_corr(phase[live], h)))
        out["brains"][brain] = {
            "decisions": int(len(rows)), "live": int(live.sum()),
            "bump_length_median": float(np.median(length)),
            "circ_corr_abs": round(r_obs, 4),
            "p_perm": float((np.sum(np.array(null) >= r_obs) + 1) / (n_perm + 1)),
        }
    rew = [v["circ_corr_abs"] for k, v in out["brains"].items() if k != "real"]
    real = out["brains"]["real"]["circ_corr_abs"]
    out["real_rank_pct"] = float(100.0 * np.mean(np.array(rew) < real)) if rew else None
    return out


def main(args: argparse.Namespace) -> int:
    res = check(args.dataset, args.permutations)
    path = paths.runs_dir() / f"bump-check-{time.strftime('%Y%m%d-%H%M%S')}.json"
    path.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"EPG 단위 {res['epg_units']} · 진짜 순위 {res['real_rank_pct']} %")
    for b, v in res["brains"].items():
        print(f"  {b:<12} 결정 {v['decisions']} · 발화 있음 {v['live']} · 봉우리 길이 중앙 "
              f"{v['bump_length_median']:.3f} · |원형 상관| {v['circ_corr_abs']:.3f} · "
              f"p {v['p_perm']:.4f}")
    print(f"기록 {path}")
    return 0


def register(isub) -> None:
    b = isub.add_parser("bump", help="나침반 봉우리 — EPG 위상이 몸 방향을 따라가나(R-7469 F-008)")
    b.add_argument("--dataset", default="fly-loop-v2")
    b.add_argument("--permutations", type=int, default=2000)
    b.set_defaults(func=main)
