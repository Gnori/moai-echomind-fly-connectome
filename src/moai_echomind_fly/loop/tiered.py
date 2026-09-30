"""두 층 제어 · 사건 기반 LLM 호출 (R-1027 F-001 · F-003 · F-005 · P loop §1).

깊은 판단이 필요할 때만 Qwen 을 부른다. 결정마다:

1. **반사**(1층) — 빠른 정책의 위험 점수가 문턱 위면 LLM 없이 회피
   (도망 반사는 회로가 뇌 구간 안에서 그대로).
2. **사건**이 있으면 Qwen(3층) — 목표 설정 · 시간 상한 · 충돌 · 애매함 · 놀람.
3. 그 밖에는 **빠른 정책**(2층 · 초파리 신호 선형 읽기)이 몬다
   — 또는 `hold` 면 마지막 LLM 행동 유지.

control-v2(2026-10-01 · 연구 노트 15 §5): 문턱은 **보정 나눔 분위수**로 잠근다(뇌마다 같은 절차) ·
놀람 = 전방 모형 오차² · 충돌 = 위험 × 먹이 쪽 확신도 · 호출 뒤 최소 간격 · 냄새 맡기 동안은 안 부름
(행동이 STOP 으로 덮이므로). 대조 모드 `random` · `periodic` 은 gated 가 실제로 낸 호출률에 맞춘다.

`Loop.episode` 의 policy 자리에 끼우는 감싸개라 몸 · 뇌 · 냄새 맡기 구간은 그대로다(P §1.1).
⚠️ 정직: 게이트 · 빠른 정책 · 반사는 **초파리 신호(압축 특징)만** 본다
— 참값(`truth`)은 LLM 에 넘길 뿐 쓰지 않는다.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from .. import config

MODES = ("always", "never", "gated", "random", "periodic")
CONDITIONS = ("gap", "conflict", "ambiguous", "surprise")      # 목표(goal)는 늘 켜짐
NEEDS = {"gap": "max_gap_decisions", "conflict": "conflict_q", "ambiguous": "confidence_q",
         "surprise": "surprise_q"}
STOP = {"walk": 0.0, "turn": "none", "turn_strength": 0.0, "stop": True, "groom": False}


class Gate:
    """언제 Qwen 을 부르나 — 조건 계산과 기억(마지막 호출).

    `fast` 는 문턱 분위수를 절댓값으로 바꾸는 데만 쓴다(`FastPolicy.quantile`).
    `rate` 는 random · periodic 모드의 결정당 호출 확률(gated 실측에서 맞춘 값).
    """

    def __init__(self, cfg: dict[str, Any], fast=None, sniff_decisions: int = 0,
                 rate: float | None = None, seed: int = 0) -> None:
        self.mode = cfg["mode"]
        if self.mode not in MODES:
            raise ValueError(f"mode 는 {MODES} 중 하나 — {self.mode}")
        self.th = dict(cfg.get("thresholds") or {})
        self.conds = [c for c in (cfg.get("conditions") or CONDITIONS)]
        bad = [c for c in self.conds if c not in CONDITIONS]
        if bad:
            raise ValueError(f"조건 이름 {bad} — {CONDITIONS} 중에서")
        self.abs: dict[str, float] = {}
        if self.mode == "gated":
            missing = [NEEDS[c] for c in self.conds if self.th.get(NEEDS[c]) is None]
            if self.th.get("min_interval_decisions") is None:
                missing.append("min_interval_decisions")
            if missing:
                raise ValueError(f"gated 모드인데 문턱 미정 {missing} — config/control.yaml 에서 "
                                 "결과를 보기 전에 잠근다(R-1027 F-003)")
            if fast is None:
                raise ValueError("gated 는 빠른 정책(분위수 → 문턱)이 필요하다")
            for c, sig in (("surprise", "surprise"), ("ambiguous", "confidence"),
                           ("conflict", "conflict")):
                if c in self.conds:
                    self.abs[c] = fast.quantile(sig, float(self.th[NEEDS[c]]))
        self.rate = None
        if self.mode in ("random", "periodic"):
            if rate is None or not 0.0 < float(rate) <= 1.0:
                raise ValueError(f"{self.mode} 모드는 0 < rate ≤ 1 이 필요하다 — {rate}")
            self.rate = float(rate)
            self.period = max(1, int(round(1.0 / self.rate)))
        self.sniff = int(sniff_decisions)
        self.seed = int(seed)
        self.reset()

    def reset(self, episode_seed: int = 0) -> None:
        self.last_call: int | None = None
        self.rng = np.random.default_rng([self.seed, int(episode_seed)])

    def trigger(self, fast_out, d: int, surprise: float = 0.0) -> str | None:
        """부를 까닭 또는 None. 순서: 목표 → (최소 간격) → 시간 → 충돌 → 애매함 → 놀람."""
        if self.mode == "always":
            return "always"
        if self.mode == "never" or d < self.sniff:
            return None                 # 냄새 맡기 동안은 행동이 STOP 으로 덮인다
        if d == self.sniff or self.last_call is None:
            return "goal"
        since = d - self.last_call
        if self.mode == "random":
            return "random" if self.rng.random() < self.rate else None
        if self.mode == "periodic":
            return "periodic" if since >= self.period else None
        th = self.th
        if since < int(th["min_interval_decisions"]):
            return None
        if "gap" in self.conds and since >= int(th["max_gap_decisions"]):
            return "gap"
        if "conflict" in self.conds and fast_out.conflict > self.abs["conflict"]:
            return "conflict"
        if "ambiguous" in self.conds and fast_out.turn_conf < self.abs["ambiguous"]:
            return "ambiguous"
        if "surprise" in self.conds and surprise > self.abs["surprise"]:
            return "surprise"
        return None

    def called(self, d: int) -> None:
        self.last_call = d


class TieredPolicy:
    """policy 자리에 끼우는 층 구조 — (action, info) · info 에 tier · trigger · 신호 · 특징."""

    def __init__(self, fast, llm: Callable | None, gate: Gate,
                 featurize: Callable[[dict], np.ndarray], *, between_calls: str = "fast",
                 reflex_threat_min: float | None = None, record_z: bool = True) -> None:
        if between_calls not in ("fast", "hold"):
            raise ValueError("between_calls 는 fast | hold")
        if llm is None and gate.mode != "never":
            raise ValueError("LLM 없이 쓰려면 mode=never(fast-only)")
        self.fast, self.llm, self.gate, self.featurize = fast, llm, gate, featurize
        self.between_calls = between_calls
        self.reflex_threat_min = reflex_threat_min
        self.record_z = record_z
        self.reset()

    def reset(self, seed: int = 0) -> None:
        self.gate.reset(seed)
        self.last_llm: dict | None = None
        self.llm_calls = 0
        self.prev_z: np.ndarray | None = None
        self.prev_a: dict | None = None

    @property
    def violations(self) -> int:
        """성적표가 세는 형식 위반 — LLM 쪽 값(부르지 않으면 0)."""
        return int(getattr(self.llm, "violations", 0)) if self.llm is not None else 0

    def _remember(self, z: np.ndarray, d: int, action: dict) -> None:
        self.prev_z = np.asarray(z, dtype=np.float64)
        self.prev_a = dict(STOP) if d < self.gate.sniff else dict(action)   # 실행된 행동

    def __call__(self, vectors, truth, obs, decision):
        z = self.featurize(vectors)
        fo = self.fast.predict(z)
        sur = self.fast.surprise(self.prev_z, self.prev_a, z)
        base = {"fast_conf": round(float(fo.turn_conf), 4), "threat": round(float(fo.threat), 4),
                "conflict": round(float(fo.conflict), 4), "surprise": round(float(sur), 4),
                "fast_action": fo.action}
        if self.record_z:               # 연구 노트 15 T3b — 성적표 뒤 "쓸모 있는 호출"을 재려고
            base["z"] = [round(float(v), 4) for v in z]
        if self.reflex_threat_min is not None and fo.threat >= self.reflex_threat_min:
            self._remember(z, decision, fo.threat_action)
            return dict(fo.threat_action), {**base, "source": "reflex", "tier": "reflex",
                                            "trigger": None, "llm_calls": self.llm_calls}
        why = self.gate.trigger(fo, decision, sur)
        if why is not None:
            assert self.llm is not None
            action, info = self.llm(vectors, truth, obs, decision)
            self.llm_calls += 1
            self.last_llm = action
            self.gate.called(decision)
            self._remember(z, decision, action)
            return action, {**info, **base, "tier": "llm", "trigger": why,
                            "llm_calls": self.llm_calls}
        action = (self.last_llm if self.between_calls == "hold" and self.last_llm is not None
                  else fo.action)
        self._remember(z, decision, action)
        return dict(action), {**base, "source": "fast", "tier": "fast", "trigger": None,
                              "llm_calls": self.llm_calls}


def load_config() -> dict[str, Any]:
    return config.load("control")
