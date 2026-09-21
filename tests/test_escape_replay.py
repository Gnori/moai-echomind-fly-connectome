"""F-007: 기록만 보고 다시 돌리면 같은 결과(CPU) · F-004 채점 계산."""

from __future__ import annotations

import argparse

import numpy as np
import pytest

from moai_echomind_fly import graph
from moai_echomind_fly.bench import escape
from moai_echomind_fly.protocols import LoomSpec
from moai_echomind_fly.sim import LIFParams


@pytest.mark.data
@pytest.mark.heavy
def test_replay_from_record_gives_same_spikes(tmp_path, monkeypatch):
    monkeypatch.setattr("moai_echomind_fly.paths.runs_dir", lambda: tmp_path)
    args = argparse.Namespace(trials=2, seed=3, device="cpu")
    r = escape.run_arm("replay-test", graph.load(), LIFParams.from_config(),
                       LoomSpec.from_config(), args)
    with np.load(tmp_path / r.run_id / "spikes.npz") as z:
        step, neuron = z["step"], z["neuron"]
    s2, n2 = escape.replay(tmp_path / r.run_id)
    assert np.array_equal(step, s2) and np.array_equal(neuron, n2)


def test_score_trials_counts_hits_latency_and_per_cell():
    dt = 0.1
    gf = np.array([5, 6])
    # 시행 두 번(0 ms · 400 ms). 첫 시행: GF 5 가 30 ms · 32 ms 에 두 번 → 반응당 2, 잠복 30
    step = np.array([300, 320, 4500])
    neuron = np.array([5, 5, 9])
    s = escape.score_trials(step, neuron, dt, gf, [0.0, 400.0], burst_ms=60)
    assert s["hits"] == 1 and s["gf_hit"] == 0.5
    assert s["gf_per_hit"] == 2.0 and s["gf_latency_ms"] == pytest.approx(30.0)


def test_wilson_interval_bounds():
    lo, hi = escape.wilson(0, 40)
    assert lo == 0.0 and 0 < hi < 0.1
