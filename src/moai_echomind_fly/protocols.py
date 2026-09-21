"""자극 규약 (R-3340 F-004 · F-005). 규약마다 출처는 config/*.yaml 에 값과 함께 있다."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import config
from .graph import Graph
from .sim import Engine


@dataclass(frozen=True)
class LoomSpec:
    version: str
    gap_ms: float
    burst_ms: float
    peak_hz: float
    gain_sigma: float
    amplitude: dict[str, float]

    @classmethod
    def from_config(cls, **override: float) -> LoomSpec:
        cfg = config.load("loom")
        values = {k: cfg[k] for k in ("version", "gap_ms", "burst_ms", "peak_hz", "gain_sigma")}
        values["amplitude"] = {str(k): float(v) for k, v in cfg["amplitude"].items()}
        values.update({k: v for k, v in override.items() if v is not None})
        return cls(**values)


@dataclass
class LoomTrials:
    onsets_ms: list[float]
    gains: list[float]
    stim_idx: np.ndarray
    missing_types: list[str]


def loom_targets(graph: Graph, spec: LoomSpec) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """다가옴에 반응하는 종류의 뉴런 인덱스와 상대 세기(좌우 모두)."""
    idx, amp, missing = [], [], []
    for typ, a in spec.amplitude.items():
        sel = graph.select(type=typ)
        if not len(sel):
            missing.append(typ)
            continue
        idx.append(sel)
        amp.append(np.full(len(sel), a))
    return np.concatenate(idx), np.concatenate(amp), missing


def run_loom(engine: Engine, spec: LoomSpec, n_trials: int, seed: int) -> LoomTrials:
    """쉼 → 폭발(포아송) → … → 쉼. 시행 게인은 로그정규(σ = gain_sigma)."""
    idx, amp, missing = loom_targets(engine.graph, spec)
    rng = np.random.default_rng(seed)
    engine.set_poisson(idx, 0.0)
    onsets, gains = [], []
    for _ in range(n_trials):
        engine.set_rates(0.0)
        engine.run(spec.gap_ms)
        gain = float(rng.lognormal(0.0, spec.gain_sigma))
        onsets.append(engine.t_ms)
        gains.append(gain)
        engine.set_rates(amp * spec.peak_hz * gain)
        engine.run(spec.burst_ms)
    engine.set_rates(0.0)
    engine.run(spec.gap_ms)
    return LoomTrials(onsets, gains, idx, missing)
