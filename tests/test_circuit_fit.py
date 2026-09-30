"""F-006 1 단계 준비 — CMA-ES · 손실 · θ 매핑 · 옮기기 (가벼움 · 뇌 전체 안 돌림)."""

from __future__ import annotations

import numpy as np
import pytest
from test_sim_parity import small_graph

from moai_echomind_fly import cellparams as cp
from moai_echomind_fly import circuit_fit as cf
from moai_echomind_fly import config, graph
from moai_echomind_fly.sim import LIFParams

LIF = LIFParams(w_scale=1.0)
BOUNDS = {"tau_ms": [5.0, 200.0], "v_th_above_rest_mV": [2.0, 15.0], "alpha": [0.1, 10.0]}


def test_cmaes_minimises_a_shifted_quadratic():
    target = np.array([1.0, -2.0, 0.5, 3.0])
    es = cf.CMAES(np.zeros(4), 1.0, seed=0)
    for _ in range(120):
        xs = es.ask()
        es.tell(xs, np.sum((xs - target) ** 2, axis=1))
    assert np.allclose(es.mean, target, atol=1e-3) and es.lam == 4 + int(3 * np.log(4))


def test_theta_round_trip_and_bounds():
    g = small_graph()
    cell = cp.init(g, ["T0", "T1"], LIF.t_mbr_ms, LIF.v_th_mV, 50)
    th = cf.to_theta(cell, LIF.v0_mV)
    lo, hi = cf.bounds_log(cell, BOUNDS)
    back = cf.from_theta(th, cell, LIF.v0_mV, lo, hi)
    assert np.allclose(back.tau_ms, cell.tau_ms) and np.allclose(back.v_th_mV, cell.v_th_mV)
    assert np.allclose(back.alpha, 1.0)
    wild = cf.from_theta(th + 50.0, cell, LIF.v0_mV, lo, hi)
    assert wild.tau_ms.max() == pytest.approx(200.0) and wild.alpha.max() == pytest.approx(10.0)
    assert wild.v_th_mV.max() == pytest.approx(LIF.v0_mV + 15.0)
    assert np.array_equal(wild.pair_id, cell.pair_id)       # 연결 쪽은 그대로


def test_item_and_physiology_loss_use_ranges_and_scope():
    assert cf.item_loss(0.7, [0.5, 1.0], 0.1, 3.0) == 0.0
    assert cf.item_loss(0.3, [0.5, 1.0], 0.1, 3.0) == pytest.approx(4.0)
    assert cf.item_loss(float("nan"), [10, 60], 10, 3.0) == 9.0
    phys = config.load("physiology")
    good = {"gf_hit": 0.65, "gf_per_hit": 1.4, "gf_latency_ms": 34.0, "pop_rate_hz": 0.003,
            "null_gf_hit": 0.0}
    loss, parts = cf.physiology_loss(good, "escape-small", phys, 3.0)
    assert loss == 0.0 and len(parts) == 5          # sim-v2 확인 값(sim.yaml 주석)은 전부 안
    bad, _ = cf.physiology_loss({**good, "pop_rate_hz": 0.09}, "escape-small", phys, 3.0)
    assert bad == pytest.approx(2.0 * (0.04 / 0.02) ** 2)   # 가중 2 · 허용폭 0.02


def test_transfer_keeps_type_params_and_matches_pairs_by_name():
    g = small_graph()
    cell = cp.init(g, ["T0", "T1"], LIF.t_mbr_ms, LIF.v_th_mV, 1)
    cell.alpha[:] = 3.0
    cell.tau_ms[:] = 30.0
    rw = graph.rewire(g, 5)
    moved = cf.transfer(cell, rw, None)
    assert np.all(moved.tau_ms == 30.0)
    names = {(int(a), int(b)) for a, b in cell.pairs}
    for (a, b), x in zip(moved.pairs, moved.alpha):
        assert x == (3.0 if (int(a), int(b)) in names else 1.0)
    moved.check(rw)


def test_objective_and_run_cmaes_with_a_stub_measure():
    """평가를 흉내(문턱이 낮을수록 GF 반응↑)해 손실 · 기록 · 최선 추적이 도는지만 본다."""
    g = small_graph()
    cell = cp.init(g, ["T0"], LIF.t_mbr_ms, LIF.v_th_mV, 1)
    seen = []

    def stub(graph_, lif, c, trials, seed, device):
        seen.append(graph_.name)
        hit = float(np.clip((LIF.v_th_mV - 1.0 - c.v_th_mV[0]) / 6.0, 0, 1))
        return {"gf_hit": hit if graph_ is g else 0.0, "gf_per_hit": 1.5,
                "gf_latency_ms": 30.0, "pop_rate_hz": 0.0}

    rule = {"trials": 2, "seed": 0, "device": "cpu", "nan_tolerances": 3.0, "reg_lambda": 0.0,
            "sigma0": 0.3, "popsize": 6, "generations": 15}
    obj = cf.make_objective(g, graph.rewire(g, 9), LIF, cell, "escape-small", rule, None,
                            measure=stub)
    f0, info0 = obj(cf.to_theta(cell, LIF.v0_mV))
    assert f0 > 0 and info0["measures"]["null_gf_hit"] == 0.0
    best, best_f, log = cf.run_cmaes(obj, cf.to_theta(cell, LIF.v0_mV), rule)
    assert best_f < f0 and len(log) == 15 and log[-1]["best"] == best_f
    assert len(set(seen)) == 2                               # 학습 그래프 + 섞은 배선


def test_graph_hash_changes_only_with_wiring():
    g = small_graph()
    assert cf.graph_hash(g) == cf.graph_hash(small_graph())
    assert cf.graph_hash(g) != cf.graph_hash(graph.rewire(g, 1))
