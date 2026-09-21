"""F-003: 우리 엔진 = Shiu 방정식의 Brian2 판 (작은 그래프 · 같은 결정적 입력).

Brian2 쪽은 Shiu model.py 의 방정식·리셋·불응·지연을 그대로 적고, 포아송 입력 대신 같은 사건표를
SpikeGeneratorGroup → v += kick(지연 0, 'synapses' 칸)로 넣는다 — PoissonInput 과 같은 칸이다.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from moai_echomind_fly.graph import Graph
from moai_echomind_fly.sim import Engine, LIFParams


def small_graph(n: int = 300, e: int = 4000, seed: int = 1) -> Graph:
    rng = np.random.default_rng(seed)
    src = rng.integers(0, n, e)
    dst = rng.integers(0, n, e)
    keep = src != dst
    pairs = np.unique(np.stack([src[keep], dst[keep]], 1), axis=0)
    src, dst = pairs[:, 0], pairs[:, 1]
    count = rng.integers(1, 40, len(src)).astype(np.int32)
    sign = rng.choice(np.array([1, 1, 1, -1, 0], dtype=np.int8), n)
    row_ptr = np.concatenate([[0], np.cumsum(np.bincount(src, minlength=n))]).astype(np.int64)
    cols = {"type": np.array([f"T{i % 7}" for i in range(n)], dtype=object)}
    return Graph("small", cols, np.arange(n, dtype=np.int64), sign, row_ptr,
                 dst.astype(np.int32), count, {})


def brian2_spikes(graph: Graph, p: LIFParams, stim: np.ndarray, kicks: np.ndarray):
    b2 = pytest.importorskip("brian2")
    from brian2 import (
        Network,
        NeuronGroup,
        SpikeGeneratorGroup,
        SpikeMonitor,
        Synapses,
        defaultclock,
        ms,
        mV,
        prefs,
    )

    prefs.codegen.target = "numpy"
    defaultclock.dt = p.dt_ms * ms
    ns = dict(v_0=p.v0_mV * mV, v_rst=p.v_reset_mV * mV, v_th=p.v_th_mV * mV,
              t_mbr=p.t_mbr_ms * ms, tau=p.tau_syn_ms * ms)
    eqs = """
    dv/dt = (v_0 - v + g) / t_mbr : volt (unless refractory)
    dg/dt = -g / tau               : volt (unless refractory)
    rfc                            : second
    """
    neu = NeuronGroup(graph.n, eqs, method="linear", threshold="v > v_th",
                      reset="v = v_rst; g = 0 * mV", refractory="rfc", namespace=ns)
    neu.v = p.v0_mV * mV
    neu.g = 0 * mV
    neu.rfc = p.t_rfc_ms * ms
    neu.rfc[stim] = 0 * ms
    syn = Synapses(neu, neu, "w : volt", on_pre="g += w", delay=p.t_dly_ms * ms)
    syn.connect(i=graph.src(), j=graph.dst)
    syn.w = graph.sign[graph.src()] * graph.count * p.w_syn_mV * p.w_scale * mV
    steps, k = kicks.shape
    t_idx, s_idx = np.nonzero(kicks)
    gen = SpikeGeneratorGroup(k, s_idx, t_idx * p.dt_ms * ms)
    kick = Synapses(gen, neu, on_pre=f"v_post += {p.poisson_kick_mV}*mV")
    kick.connect(i=np.arange(k), j=stim)
    mon = SpikeMonitor(neu)
    Network(neu, syn, gen, kick, mon).run(steps * p.dt_ms * ms)
    step = np.round(np.asarray(mon.t / ms) / p.dt_ms).astype(np.int64)
    return step, np.asarray(mon.i, dtype=np.int64), b2


def test_engine_matches_brian2_shiu_equations():
    graph = small_graph()
    p = LIFParams(w_scale=1.0)
    stim = np.arange(0, 20)
    rng = np.random.default_rng(7)
    steps = 2000  # 200 ms
    kicks = rng.random((steps, len(stim))) < 150 * p.dt_ms / 1000

    eng = Engine(graph, p, device="cpu", dtype=torch.float64)
    eng.set_poisson(stim, 150.0)
    eng.run(steps * p.dt_ms, kicks=kicks)
    ours_step, ours_i = eng.spikes()

    b_step, b_i, _ = brian2_spikes(graph, p, stim, kicks)

    ours = sorted(zip(ours_step.tolist(), ours_i.tolist()))
    theirs = sorted(zip(b_step.tolist(), b_i.tolist()))
    assert len(ours) > 200, "대조가 의미 있으려면 자극 뉴런 밖에서도 발화가 퍼져야 한다"
    assert len({i for _, i in ours} - set(stim.tolist())) > 20
    assert len(ours) == len(theirs)
    assert ours == theirs


def test_silent_without_stimulus():
    graph = small_graph()
    eng = Engine(graph, LIFParams(), device="cpu", dtype=torch.float64)
    eng.run(1000)
    assert eng.spikes()[0].size == 0


def test_silence_removes_influence():
    graph = small_graph()
    p = LIFParams()
    stim = np.arange(0, 20)
    kicks = np.random.default_rng(3).random((1000, 20)) < 0.05
    eng = Engine(graph, p, dtype=torch.float64)
    eng.set_poisson(stim, 150.0)
    eng.silence(stim)
    eng.run(100, kicks=kicks)
    _, i = eng.spikes()
    assert set(i.tolist()) <= set(stim.tolist()), "침묵시킨 뉴런은 아무에게도 영향을 주지 않는다"


def test_same_seed_same_spikes_cpu():
    graph = small_graph()
    runs = []
    for _ in range(2):
        eng = Engine(graph, LIFParams(), dtype=torch.float32, seed=11)
        eng.set_poisson(np.arange(20), 150.0)
        eng.run(100)
        runs.append(eng.spikes())
    assert np.array_equal(runs[0][0], runs[1][0]) and np.array_equal(runs[0][1], runs[1][1])
