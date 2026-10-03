"""F-006 준비 — sim-v3 매개변수 파일(종류 단위 tau · 문턱 · 쌍 배율)과 엔진(작은 그래프 · CPU)."""

from __future__ import annotations

import numpy as np
import pytest
from test_sim_parity import small_graph

from moai_echomind_fly import cellparams as cp
from moai_echomind_fly.sim import Engine, LIFParams, _decay_mix

P = LIFParams(w_scale=1.0)
TYPES = ["T0", "T1", "T2"]


def _spikes(g, cell=None, steps=3000, seed=0):
    rng = np.random.default_rng(seed)
    stim = np.arange(20)
    kicks = rng.random((steps, len(stim))) < 0.02
    e = Engine(g, P, cell=cell)
    e.set_poisson(stim, np.full(len(stim), 1.0))
    e.run(steps * P.dt_ms, kicks=kicks)
    return e.spikes()


def test_init_params_give_identical_spikes():
    g = small_graph()
    base = _spikes(g)
    cell = cp.init(g, TYPES, P.t_mbr_ms, P.v_th_mV, pair_min_synapses=50)
    same = _spikes(g, cell)
    assert len(base[0]) > 50
    assert np.array_equal(base[0], same[0]) and np.array_equal(base[1], same[1])


def test_pairs_cover_only_scoped_edges_and_alpha_scales_them():
    g = small_graph()
    cell = cp.init(g, TYPES, P.t_mbr_ms, P.v_th_mV, pair_min_synapses=50)
    k = cell.neuron_type(g)
    inside = cell.pair_id >= 0
    assert (k[g.src()][inside] >= 0).all() and (k[g.dst][inside] >= 0).all()
    assert len(cell.alpha) == len(cell.pairs) > 0
    cell.alpha[:] = 0.0                                     # 범위 안 쌍을 끊으면
    changed = _spikes(g, cell)
    base = _spikes(g)
    assert not (np.array_equal(base[0], changed[0]) and np.array_equal(base[1], changed[1]))


def test_tau_and_threshold_change_only_scoped_neurons():
    g = small_graph()
    cell = cp.init(g, ["T0"], P.t_mbr_ms, P.v_th_mV, pair_min_synapses=10 ** 9)
    cell.tau_ms[:] = 40.0
    cell.v_th_mV[:] = -40.0
    tau, vth = cell.per_neuron(g, P.t_mbr_ms, P.v_th_mV)
    t0 = g.columns["type"] == "T0"
    assert (tau[t0] == 40.0).all() and (tau[~t0] == P.t_mbr_ms).all()
    assert (vth[t0] == -40.0).all() and (vth[~t0] == P.v_th_mV).all()
    assert len(cell.pairs) == 0 and (cell.edge_scale() == 1.0).all()


def test_decay_mix_is_continuous_when_time_constants_meet():
    dt = 0.1
    at = float(_decay_mix(dt, 5.0, 5.0))
    near = float(_decay_mix(dt, 5.0, 5.0 + 1e-4))
    assert at == pytest.approx(near, rel=1e-4)
    ref = 5.0 / (5.0 - 20.0) * (np.exp(-dt / 5.0) - np.exp(-dt / 20.0))
    assert float(_decay_mix(dt, 5.0, 20.0)) == pytest.approx(ref)


def test_identity_check_refuses_bad_params_and_round_trips(tmp_path):
    g = small_graph()
    cell = cp.init(g, TYPES, P.t_mbr_ms, P.v_th_mV, pair_min_synapses=50, meta={"scope": "t"})
    cell.save(tmp_path / "p.npz")
    back = cp.CellParams.load(tmp_path / "p.npz")
    assert list(back.types) == TYPES and np.array_equal(back.pair_id, cell.pair_id)
    assert back.meta["scope"] == "t"
    back.alpha[0] = -1.0
    with pytest.raises(ValueError, match="alpha"):
        back.check(g)
    bad = cp.init(g, ["없는종류"], P.t_mbr_ms, P.v_th_mV, 1)
    with pytest.raises(ValueError, match="없는 종류"):
        bad.check(g)


def test_scope_config_names_escape_small_and_refuses_unlocked():
    types = cp.scope_types("escape-small")
    assert "LPLC2" in types and "DNp01" in types and len(types) == 17
    with pytest.raises(ValueError, match="잠기지"):
        cp.scope_types("space")


def test_pair_dst_keeps_only_pairs_into_those_types():
    g = small_graph()
    cell = cp.init(g, TYPES, P.t_mbr_ms, P.v_th_mV, pair_min_synapses=1, pair_dst=["T2"])
    assert len(cell.pairs) > 0 and (cell.pairs[:, 1] == 2).all()
    assert (g.columns["type"][g.dst][cell.pair_id >= 0] == "T2").all()


def test_groups_share_one_alpha_and_keep_type_pairs_first():
    g = small_graph()
    groups = [{"name": "T0-any", "src": {"type": ["T0"]}, "dst": {"type": ["T1", "T2"]}},
              {"name": "T0-T1", "src": {"type": ["T0"]}, "dst": {"type": ["T1"]}}]
    cell = cp.init(g, TYPES, P.t_mbr_ms, P.v_th_mV, None, groups=groups)
    assert cell.meta["groups"] == ["T0-any", "T0-T1"] and len(cell.alpha) == 2
    src_t, dst_t = g.columns["type"][g.src()], g.columns["type"][g.dst]
    m = (src_t == "T0") & np.isin(dst_t, ["T1", "T2"])
    assert (cell.pair_id[m] == 0).all() and (cell.pair_id[~m] == -1).all()   # 앞 묶음이 먼저
    cell.check(g)
    both = cp.init(g, TYPES, P.t_mbr_ms, P.v_th_mV, 50, groups=groups[:1])
    n_pairs = len(both.alpha) - 1
    assert (both.pair_id[m] >= 0).all() and (both.pair_id[m] < n_pairs).any()   # 종류 쌍이 먼저
    assert tuple(both.pairs[-1]) == (-1, -1)


def test_kc_scopes_count_as_proposed():
    from moai_echomind_fly import config

    cfg = config.load("fit")
    assert len(cp.scope_types("kc-sparse-a", cfg)) == 10
    assert len(cfg["scopes"]["kc-sparse-b"]["groups"]) == 4
