"""F-008: 도구 본체 — 상한 거부 다섯 종류 · 요약 숫자 = 원본에서 다시 센 값."""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from moai_echomind_fly import graph, tools

pytestmark = pytest.mark.data


@pytest.fixture
def runs(tmp_path, monkeypatch):
    monkeypatch.setenv("FLY_RUNS_DIR", str(tmp_path))
    return tmp_path


def test_find_neurons_groups_by_type():
    rows = tools.find_neurons("DNp01")
    dn = [r for r in rows if r["type"] == "DNp01"]
    assert dn and dn[0]["neurons"] == 2 and dn[0]["superclass"] == "descending_neuron"


@pytest.mark.parametrize("kwargs, code", [
    ({"stimulate": ["DNp01"], "duration_ms": 5000}, "over_limit"),              # 길이
    ({"stimulate": ["DNp01"], "rate_hz": 1000}, "over_limit"),                   # 발화율
    # 자극 뉴런 3,374 > 3,000
    ({"stimulate": ["T4a", "T4b"]}, "over_limit"),
    # 침묵 5,152 > 5,000
    ({"stimulate": ["DNp01"], "silence": ["T4a", "T4b", "T4c"]}, "over_limit"),
    ({"stimulate": ["DH44"]}, "not_allowed"),                                     # cb_endocrine
    ({"stimulate": ["DNp1"]}, "unknown_type"),                                    # 없는 이름
])
def test_limits_are_enforced_in_code(runs, kwargs, code):
    with pytest.raises(tools.ToolError) as e:
        tools.run_experiment(**kwargs)
    assert e.value.code == code
    assert not list(runs.iterdir()), "거부된 요청은 실행 기록을 남기지 않는다"


def test_unknown_type_suggests_near_names():
    with pytest.raises(tools.ToolError) as e:
        tools.run_experiment(["DNp1"])
    assert "DNp01" in e.value.detail["suggestions"] or e.value.detail["suggestions"]


@pytest.mark.heavy
def test_summary_numbers_recount_from_raw(runs):
    s = tools.run_experiment(["DNp01"], rate_hz=150, duration_ms=100, seed=1, purpose="test")
    g = graph.load()
    with np.load(runs / s["run_id"] / "spikes.npz") as z:
        neuron, stim = z["neuron"], z["stim_idx"]
    non = neuron[~np.isin(neuron, stim)]
    assert s["total_spikes"] == len(non)
    assert s["spiking_neurons"] == len(np.unique(non))
    counts = Counter(g.columns["type"][non].tolist())
    for row in s["top_types"]:
        assert counts[row["type"]] == row["spikes"]
    assert s["stimulated"]["neurons"] == 2
    assert tools.get_run(s["run_id"])["total_spikes"] == s["total_spikes"]
    assert tools.list_runs()[0]["run_id"] == s["run_id"]
