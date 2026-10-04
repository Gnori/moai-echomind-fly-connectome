"""MBON 침묵 진단 — 출처별 평균 탈분극 · 까닭 판정 · 작은 그래프 한 번 달리기."""

from __future__ import annotations

import numpy as np
import torch
from test_plasticity import mb_graph

from moai_echomind_fly import config
from moai_echomind_fly import mbon_probe as mp
from moai_echomind_fly.sim import Engine, LIFParams

CFG = config.load("plasticity")


def test_inputs_and_drive_split_kc_from_other_sources():
    g = mb_graph()
    w = Engine(g, LIFParams(w_scale=1.0), dtype=torch.float64).w.numpy()
    mbon = np.array([4, 5])
    s = mp.mbon_inputs(g, w, mbon, CFG)
    assert s["kc_partners"].tolist() == [3, 3]          # MBON01 ← KC 0·1·2 · MBON11 ← KC 0·1·3
    assert s["kc_syn"].tolist() == [15, 15]
    assert s["other_exc_syn"].tolist() == [30, 30]      # DAN(부호 +1 · 이 작은 그래프에서)
    cnt = np.zeros(g.n)
    cnt[[0, 2]] = 2                                      # KC 0 · 2 가 두 번씩
    d = mp.drive(g, w, cnt, mbon, CFG, window_ms=100.0, tau_syn_ms=5.0)
    one = 5 * 0.275                                      # 시냅스 5 × w_syn
    assert np.allclose(d["kc_mV"], [4 * one * 5 / 100, 2 * one * 5 / 100])
    assert np.allclose(d["kc_partners_on"], [2 / 3, 1 / 3])
    assert (d["inh_mV"] == 0).all() and (d["other_exc_mV"] == 0).all()


def test_verdict_names_the_reason():
    v = CFG["mbon_probe"]
    assert mp.verdict(7.0, 1.0, 0.0, 0.0, 3, v) == "fires"
    assert mp.verdict(7.0, 6.0, 2.0, -3.0, 0, v) == "inhibition"
    assert mp.verdict(7.0, 0.0, 1.0, 0.0, 0, v) == "no_kc_input"
    assert mp.verdict(7.0, 1.0, 0.0, 0.0, 0, v) == "weak_kc"


def test_run_case_on_small_graph(monkeypatch):
    g = mb_graph()
    cfg = {**CFG, "mbon_probe": {**CFG["mbon_probe"], "windows_ms": [10, 20]}}
    r = mp.run_case(g, LIFParams(w_scale=1.0), None, cfg, ["KCg"], 150.0, 0, "cpu")
    assert r["mbon_neurons"] == 2 and len(r["windows"]) == 2
    row = r["windows"][1]["mbon"][0]
    assert row["gap_mV"] == 7.0 and row["why"] in {"fires", "weak_kc", "no_kc_input", "inhibition"}
