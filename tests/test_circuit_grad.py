"""F-006 2 단계 준비 — 함수형 LIF = 엔진 발화 · 기울기는 범위 안에서만(작은 그래프 · CPU)."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from test_sim_parity import small_graph

from moai_echomind_fly import cellparams as cp
from moai_echomind_fly.circuit_grad import FuncLIF
from moai_echomind_fly.sim import Engine, LIFParams

P = LIFParams(w_scale=1.0)
STIM = np.arange(20)
READ = np.arange(100, 140)


def _kicks(steps=1500, seed=0):
    return np.random.default_rng(seed).random((steps, len(STIM))) < 0.02


def _cell(g):
    c = cp.init(g, ["T0", "T1", "T2"], P.t_mbr_ms, P.v_th_mV, pair_min_synapses=30)
    c.tau_ms[:] = [15.0, 25.0, 5.0]        # 5 ms = tau_syn → 극한 경로도 지난다
    c.v_th_mV[:] = [-46.0, -44.0, -45.5]
    c.alpha[:] = np.linspace(0.5, 1.5, len(c.alpha))
    return c


@pytest.mark.parametrize("scope", [None, "all"])
def test_functional_forward_matches_engine_spikes(scope):
    g = small_graph()
    cell = _cell(g)
    k = _kicks()
    eng = Engine(g, P, dtype=torch.float64, cell=cell)
    eng.set_poisson(STIM, np.ones(len(STIM)))
    eng.run(len(k) * P.dt_ms, kicks=k)
    ref = eng.spikes()
    f = FuncLIF(g, P, cell, stim_idx=STIM, readout_idx=READ,
                grad_scope=np.arange(g.n) if scope else None)
    with torch.no_grad():
        _, _, got = f.run(f.initial_state(), k, record=True)
    assert len(ref[0]) > 100
    assert np.array_equal(ref[0], got[0]) and np.array_equal(ref[1], got[1])


def test_gradients_flow_only_through_scope():
    g = small_graph()
    cell = _cell(g)
    k = _kicks()
    f = FuncLIF(g, P, cell, stim_idx=STIM, readout_idx=READ, grad_scope=np.arange(g.n))
    _, counts, _ = f.run(f.initial_state(), k)
    counts.sum().backward()
    grads = [t.grad for t in f.params()]
    assert all(gr is not None and torch.isfinite(gr).all() for gr in grads)
    assert any(gr.abs().sum() > 0 for gr in grads)
    none = FuncLIF(g, P, cell, stim_idx=STIM, readout_idx=READ, grad_scope=None)
    _, c2, _ = none.run(none.initial_state(), k)
    assert not c2.requires_grad                                  # 범위가 비면 기울기 길이 없다
    half = FuncLIF(g, P, cell, stim_idx=STIM, readout_idx=READ, grad_scope=READ)
    _, c3, _ = half.run(half.initial_state(), k)
    assert torch.equal(c3.detach(), counts.detach()) and c3.requires_grad
    assert torch.equal(c2, counts.detach())                      # 정방향은 같다


def test_checkpoint_gives_same_gradient_and_truncation_detaches():
    g = small_graph()
    cell = _cell(g)
    k = _kicks(600)
    out = []
    for ck in (0, 50):
        f = FuncLIF(g, P, cell, stim_idx=STIM, readout_idx=READ, grad_scope=np.arange(g.n))
        _, counts, _ = f.run(f.initial_state(), k, checkpoint_steps=ck)
        counts.sum().backward()
        out.append([t.grad.clone() for t in f.params()])
    for a, b in zip(*out):
        assert torch.allclose(a, b, rtol=1e-9, atol=1e-12)
    f = FuncLIF(g, P, cell, stim_idx=STIM, readout_idx=READ, grad_scope=np.arange(g.n))
    st, _, _ = f.run(f.initial_state(), k[:300])
    st = f.detach(st)
    assert not any(t.requires_grad for t in st)


def test_to_cell_round_trip_keeps_wiring():
    g = small_graph()
    cell = _cell(g)
    f = FuncLIF(g, P, cell, stim_idx=STIM, readout_idx=READ)
    back = f.to_cell(cell)
    assert np.allclose(back.tau_ms, cell.tau_ms) and np.array_equal(back.pair_id, cell.pair_id)
    back.check(g)


def test_probe_sgbptt_runs_on_a_small_graph_with_loom_types(monkeypatch):
    """명령 경로가 작은 그래프에서 돈다(뇌 전체 아님) — 다가옴 종류 이름을 붙여 흉내."""
    from moai_echomind_fly import circuit_fit as cf
    from moai_echomind_fly import config
    from moai_echomind_fly.protocols import LoomSpec

    g = small_graph()
    names = list(LoomSpec.from_config().amplitude)
    g.columns["type"] = np.array([names[i % len(names)] if i < 200 else
                                  ("DNp01" if i < 210 else "T9") for i in range(g.n)],
                                 dtype=object)
    cell = cp.init(g, names + ["DNp01"], P.t_mbr_ms, P.v_th_mV, 1, pair_dst=["DNp01"])
    sg = {**config.load("fit")["sgbptt"], "dtype": "float64", "checkpoint_steps": 25}
    monkeypatch.setattr(cf, "_footprint_gb", lambda: 0.0)
    monkeypatch.setattr(cf, "_free_percent", lambda: 100)
    r = cf.probe_sgbptt(g, P, cell, 100, sg, "cpu")
    assert r["window_steps"] == 100 and r["scope_neurons"] == 210
    assert set(r["grad_abs_sum"]) == {"tau", "vth", "alpha"}
