"""두 층 제어 · 사건 기반 LLM 호출 (R-1027 F-001 · F-003 · F-005 · P loop §1).

깊은 판단이 필요할 때만 Qwen 을 부른다. 결정마다:

1. **반사**(1층) — 빠른 정책의 위험 점수가 문턱 위면 LLM 없이 회피
   (도망 반사는 회로가 뇌 구간 안에서 그대로).
2. **사건**이 있으면 Qwen(3층) — 목표 설정 · 시간 상한 · 충돌 · 애매함 · 놀람.
3. 그 밖에는 **빠른 정책**(2층 · 초파리 신호 선형 읽기)이 몬다
   — 또는 `hold` 면 마지막 LLM 행동 유지.

`Loop.episode` 의 policy 자리에 끼우는 감싸개라 몸 · 뇌 · 냄새 맡기 구간은 그대로다(P §1.1).
⚠️ 정직: 게이트 · 빠른 정책 · 반사는 **초파리 신호(압축 특징)만** 본다
— 참값(`truth`)은 LLM 에 넘길 뿐 쓰지 않는다.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from .. import config

MODES = ("always", "never", "gated")
TRIGGERS = ("goal", "gap", "conflict", "ambiguous", "surprise")


class Gate:
    """언제 Qwen 을 부르나 — 조건 계산과 기억(마지막 호출 · 직전 특징)."""

    def __init__(self, cfg: dict[str, Any], delta_quantiles: dict[str, float] | None = None,
                 sniff_decisions: int = 0) -> None:
        self.mode = cfg["mode"]
        if self.mode not in MODES:
            raise ValueError(f"mode 는 {MODES} 중 하나 — {self.mode}")
        self.th = dict(cfg.get("thresholds") or {})
        if self.mode == "gated":
            missing = [k for k in ("max_gap_decisions", "surprise_q", "confidence_min",
                                   "conflict_min") if self.th.get(k) is None]
            if missing:
                raise ValueError(f"gated 모드인데 문턱 미정 {missing} — config/control.yaml 에서 "
                                 "결과를 보기 전에 잠근다(R-1027 F-003)")
        self.surprise_abs = None
        if self.mode == "gated":
            q = f"{float(self.th['surprise_q']):.2f}"
            if not delta_quantiles or q not in delta_quantiles:
                raise ValueError(f"놀람 문턱 분위수 {q} 가 빠른 정책 meta 에 없다")
            self.surprise_abs = float(delta_quantiles[q])
        self.sniff = int(sniff_decisions)
        self.reset()

    def reset(self) -> None:
        self.prev_z: np.ndarray | None = None
        self.last_call: int | None = None

    def trigger(self, z: np.ndarray, fast_out, d: int) -> str | None:
        """부를 까닭(TRIGGERS 중 하나) 또는 None. 순서: 목표 → 시간 → 충돌 → 애매함 → 놀람."""
        if self.mode == "always":
            return "always"
        if self.mode == "never":
            return None
        th = self.th
        if d == self.sniff or self.last_call is None:
            return "goal"
        if d - self.last_call >= int(th["max_gap_decisions"]):
            return "gap"
        if fast_out.threat >= th["conflict_min"] and fast_out.turn_conf >= th["conflict_min"]:
            return "conflict"
        if fast_out.turn_conf < th["confidence_min"]:
            return "ambiguous"
        if self.prev_z is not None and float(np.linalg.norm(z - self.prev_z)) > self.surprise_abs:
            return "surprise"
        return None

    def remember(self, z: np.ndarray, d: int, called: bool) -> None:
        self.prev_z = np.asarray(z, dtype=np.float64)
        if called:
            self.last_call = d


class TieredPolicy:
    """policy 자리에 끼우는 층 구조 — (action, info) · info 에 tier · trigger · 확신도."""

    def __init__(self, fast, llm: Callable | None, gate: Gate,
                 featurize: Callable[[dict], np.ndarray], *, between_calls: str = "fast",
                 reflex_threat_min: float | None = None) -> None:
        if between_calls not in ("fast", "hold"):
            raise ValueError("between_calls 는 fast | hold")
        if llm is None and gate.mode != "never":
            raise ValueError("LLM 없이 쓰려면 mode=never(fast-only)")
        self.fast, self.llm, self.gate, self.featurize = fast, llm, gate, featurize
        self.between_calls = between_calls
        self.reflex_threat_min = reflex_threat_min
        self.reset()

    def reset(self) -> None:
        self.gate.reset()
        self.last_llm: dict | None = None
        self.llm_calls = 0

    @property
    def violations(self) -> int:
        """성적표가 세는 형식 위반 — LLM 쪽 값(부르지 않으면 0)."""
        return int(getattr(self.llm, "violations", 0)) if self.llm is not None else 0

    def __call__(self, vectors, truth, obs, decision):
        z = self.featurize(vectors)
        fo = self.fast.predict(z)
        base = {"fast_conf": round(float(fo.turn_conf), 4), "threat": round(float(fo.threat), 4),
                "fast_action": fo.action}
        if self.reflex_threat_min is not None and fo.threat >= self.reflex_threat_min:
            self.gate.remember(z, decision, called=False)
            return dict(fo.threat_action), {**base, "source": "reflex", "tier": "reflex",
                                            "trigger": None, "llm_calls": self.llm_calls}
        why = self.gate.trigger(z, fo, decision)
        if why is not None:
            assert self.llm is not None
            action, info = self.llm(vectors, truth, obs, decision)
            self.llm_calls += 1
            self.last_llm = action
            self.gate.remember(z, decision, called=True)
            return action, {**info, **base, "tier": "llm", "trigger": why,
                            "llm_calls": self.llm_calls}
        self.gate.remember(z, decision, called=False)
        action = (self.last_llm if self.between_calls == "hold" and self.last_llm is not None
                  else fo.action)
        return dict(action), {**base, "source": "fast", "tier": "fast", "trigger": None,
                              "llm_calls": self.llm_calls}


def load_config() -> dict[str, Any]:
    return config.load("control")
