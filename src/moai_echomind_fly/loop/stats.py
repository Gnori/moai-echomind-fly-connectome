"""판정 통계 stats-v1 (R-7469 F-001) — 진짜 배선이 재배선 앙상블을 "앞선다"고 말할 기준.

평가 결정들은 몇 에피소드에 몰려 있어 서로 독립이 아니다. 그래서 **에피소드를 한 덩어리**로 다룬다:

- 쌍 비교: 같은 평가 결정에서 진짜 배선이 맞혔나(0/1) − 재배선들이 맞힌 비율(평균)
  → 에피소드마다 합. 귀무가설(차이 없음)에서는 에피소드마다 부호가 뒤집혀도 같다
  → **에피소드 부호 뒤집기 순열**(한쪽 · 큰 쪽).
  에피소드가 적으면(≤ `exact_max_episodes`) 모든 경우를 센다(정확한 p).
- 여러 칸을 한꺼번에 보므로 **Holm** 으로 보정한다.
- 정확도 구간은 **에피소드 부트스트랩**(에피소드를 뽑아 다시 잰다).
- 판정 = 앙상블 순위 ≥ `rank_pct` **그리고** 보정 p < `alpha` **그리고** 찍기(최빈값)보다 높다.

참고: Combrisson & Jerbi 2015 · Varoquaux 2018(표본이 적을 때의 우연 · 오차) — 연구 노트 12 §5.
"""

from __future__ import annotations

import itertools
from typing import Any

import numpy as np

STATS_RULE: dict[str, Any] = {
    "version": "stats-v1",
    "alpha": 0.05,
    "rank_pct": 95.0,               # 진짜가 재배선 앙상블의 몇 % 위에 있어야 하나
    "permutations": 10000,          # 에피소드가 많을 때 몬테카를로 횟수
    "exact_max_episodes": 16,       # 이 이하면 2^n 경우를 모두 센다
    "bootstrap": 2000,
    "seed": 0,
    # 판정에서 빼는 채널 — 하강 뉴런은 교사 명령(DNp09 · DNa02)을 받는 회로 안에 있어
    # 정답이 샐 수 있다
    # (운동 → 교사 돌기 0.610 · 찍기 0.590 — 약하지만 근거로 쓰지 않는다 · R-7469 F-003)
    "exclude_from_judgement": ["motor"],
}


def episode_sums(values: np.ndarray, episodes: np.ndarray) -> np.ndarray:
    """결정마다 값 → 에피소드마다 합(에피소드 순서는 정렬된 고유값 순)."""
    _, inv = np.unique(episodes, return_inverse=True)
    return np.bincount(inv, weights=values.astype(np.float64))


def sign_flip_p(block_sums: np.ndarray, rule: dict[str, Any] = STATS_RULE) -> float:
    """에피소드 부호 뒤집기 순열의 한쪽 p — "합이 0 보다 크다"의 증거.

    관측 합 이상이 나오는 경우의 비율(관측 자체 포함).
    """
    d = np.asarray(block_sums, dtype=np.float64)
    obs = d.sum()
    n = len(d)
    if n == 0:
        return 1.0
    if n <= rule["exact_max_episodes"]:
        signs = np.array(list(itertools.product((1.0, -1.0), repeat=n)))
        null = signs @ d
        return float(np.mean(null >= obs - 1e-12))
    rng = np.random.default_rng(rule["seed"])
    signs = rng.choice((1.0, -1.0), size=(int(rule["permutations"]), n))
    null = signs @ d
    return float((np.sum(null >= obs - 1e-12) + 1) / (len(null) + 1))


def holm(pvalues: dict[str, float]) -> dict[str, float]:
    """Holm 보정 — 작은 p 부터 (m − i) 를 곱하고, 앞보다 작아지지 않게 누적 최대."""
    keys = sorted(pvalues, key=lambda k: pvalues[k])
    m = len(keys)
    out, running = {}, 0.0
    for i, k in enumerate(keys):
        running = max(running, min(1.0, (m - i) * pvalues[k]))
        out[k] = running
    return out


def bootstrap_ci(correct: np.ndarray, episodes: np.ndarray,
                 rule: dict[str, Any] = STATS_RULE) -> tuple[float, float]:
    """정확도의 95 % 구간 — 에피소드를 복원 추출해 다시 잰다."""
    ids, inv = np.unique(episodes, return_inverse=True)
    hits = np.bincount(inv, weights=correct.astype(np.float64))
    counts = np.bincount(inv).astype(np.float64)
    rng = np.random.default_rng(rule["seed"])
    pick = rng.integers(0, len(ids), size=(int(rule["bootstrap"]), len(ids)))
    acc = hits[pick].sum(1) / counts[pick].sum(1)
    lo, hi = np.percentile(acc, [2.5, 97.5])
    return float(lo), float(hi)


def judge_cell(real_correct: np.ndarray, rewired_correct: list[np.ndarray],
               baseline_correct: np.ndarray, episodes: np.ndarray,
               rule: dict[str, Any] = STATS_RULE) -> dict[str, Any]:
    """한 칸(채널 | 칸)의 판정 재료 — Holm 보정 전 p 까지. `pass` 는 `finalize` 가 정한다."""
    real_acc = float(real_correct.mean())
    rew = np.array([float(r.mean()) for r in rewired_correct])
    ens = np.mean(np.stack(rewired_correct).astype(np.float64), axis=0)
    diff = real_correct.astype(np.float64) - ens
    return {
        "real": real_acc,
        "baseline": float(baseline_correct.mean()),
        "rewired": {"values": [round(float(v), 4) for v in rew], "mean": float(rew.mean()),
                    "p05": float(np.percentile(rew, 5)), "p95": float(np.percentile(rew, 95))},
        "rank_pct": float(100.0 * np.mean(rew < real_acc)),
        "p_block_perm": sign_flip_p(episode_sums(diff, episodes), rule),
        "p_vs_baseline": sign_flip_p(
            episode_sums(real_correct.astype(np.float64) - baseline_correct, episodes), rule),
        "ci95_real": list(bootstrap_ci(real_correct, episodes, rule)),
        "episodes": int(len(np.unique(episodes))),
        "decisions": int(len(real_correct)),
    }


def finalize(cells: dict[str, dict[str, Any]], rule: dict[str, Any] = STATS_RULE) -> None:
    """판정 대상 칸끼리 Holm 보정 · `pass` 를 채운다(제자리)."""
    judged = {k: c["p_block_perm"] for k, c in cells.items() if c.get("judge", True)}
    adj = holm(judged)
    for k, c in cells.items():
        if k in adj:
            c["p_holm"] = adj[k]
            c["pass"] = bool(c["rank_pct"] >= rule["rank_pct"] and adj[k] < rule["alpha"]
                             and c["real"] > c["baseline"])
        else:
            c["p_holm"] = None
            c["pass"] = None
