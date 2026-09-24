"""compress — 압축기 저장 · 불러오기 왕복 · 채널 건너뛰기 · 재배선 뇌 이름 순서 · v3 크기 맞춤.

가벼운 시험 · 실제 자료 없음(임시 폴더에 가짜 묶음).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from moai_echomind_fly import datasets
from moai_echomind_fly.loop import compress


def _fake_dataset(root: Path, name: str, brains: list[str], rng: np.random.Generator) -> None:
    for split in ("train", "eval"):
        d = root / name / split
        d.mkdir(parents=True)
        for b in brains:
            np.savez_compressed(d / f"features-{b}.npz",
                                a=rng.standard_normal((60, 24)).astype(np.float32),
                                b=rng.standard_normal((60, 18)).astype(np.float32))
        (d / "samples.jsonl").write_text("", encoding="utf-8")


def test_fit_then_load_round_trips_the_same_model(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    store = tmp_path / "datasets"
    _fake_dataset(store, "toy", ["real", "rewired-s3"], rng)
    monkeypatch.setattr(datasets, "store_dir", lambda: store)
    monkeypatch.setenv("FLY_RUNS_DIR", str(tmp_path / "runs"))

    meta = compress.fit("toy")
    assert meta["brains"] == ["real", "rewired-s3"]
    out = compress.out_dir("toy")
    assert (out / "real.npz").exists() and (out / "rewired-s3.npz").exists()
    assert (out / "meta.json").exists()

    ftr, _ = datasets.load("toy", "train", "real")
    expect = {ch: compress.fit_channel(x, meta["rule"]) for ch, x in ftr.items()}
    loaded = compress.load("toy", "real")
    assert set(loaded) == {"a", "b"}
    for ch in ("a", "b"):
        np.testing.assert_array_equal(loaded[ch]["mean"], expect[ch]["mean"])
        np.testing.assert_array_equal(loaded[ch]["components"], expect[ch]["components"])
        assert float(loaded[ch]["explained"]) == float(expect[ch]["explained"])
        assert meta["channels"]["real"][ch]["k"] == expect[ch]["components"].shape[0]
    z1 = compress.apply(loaded, ftr)
    z2 = compress.apply(expect, ftr)
    np.testing.assert_allclose(z1["a"], z2["a"], atol=1e-5)


def test_apply_skips_channels_missing_from_features():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((30, 12))
    m = compress.fit_channel(x, {"var_target": 0.9, "k_min": 2, "k_max": 6, "center": True})
    z = compress.apply({"a": m, "b": m}, {"a": x.astype(np.float32)})
    assert list(z) == ["a"] and z["a"].shape == (30, m["components"].shape[0])
    assert z["a"].dtype == np.float32


def test_rewired_brains_orders_v1_name_first_then_seeds_numerically(tmp_path, monkeypatch):
    store = tmp_path / "datasets"
    d = store / "toy" / "eval"
    d.mkdir(parents=True)
    for b in ("rewired-s10", "rewired", "rewired-s2", "real"):
        np.savez(d / f"features-{b}.npz", a=np.zeros((1, 1), np.float32))
    monkeypatch.setattr(datasets, "store_dir", lambda: store)
    assert datasets.rewired_brains("toy") == ["rewired", "rewired-s2", "rewired-s10"]


def test_unit_scale_makes_mean_norm_one_for_every_channel_and_brain():
    """v3 — 크기가 다른 두 '뇌'를 같은 규칙으로 압축하면 학습 나눔의 평균 크기가 둘 다 1 이 된다."""
    rng = np.random.default_rng(2)
    base = rng.standard_normal((80, 20))
    cfg = {"k_fixed": 8, "unit_scale": True, "scale_floor": 1e-6, "center": True}
    for scale in (0.05, 7.0):                      # 진짜 0.9 대 재배선 1.8 · 몸 8.3 같은 상황
        x = base * scale
        m = compress.fit_channel(x, cfg)
        z = compress.apply({"c": m}, {"c": x.astype(np.float32)})["c"]
        assert z.shape == (80, 8)
        np.testing.assert_allclose(np.linalg.norm(z, axis=1).mean(), 1.0, rtol=1e-4)
    # 축 사이 비율은 그대로(백색화가 아니다): 첫 축이 여전히 가장 크다
    assert z[:, 0].std() > z[:, -1].std()
    # 크기를 맞추지 않으면(v2) 값이 그대로 — scale 이 없을 때 apply 는 나누지 않는다
    m2 = compress.fit_channel(base, {"k_fixed": 8, "center": True})
    assert "scale" not in m2
    z2 = compress.apply({"c": m2}, {"c": base.astype(np.float32)})["c"]
    assert abs(np.linalg.norm(z2, axis=1).mean() - 1.0) > 0.1


def test_unit_scale_silent_channel_does_not_blow_up():
    x = np.zeros((40, 10))
    m = compress.fit_channel(x, {"k_fixed": 4, "unit_scale": True, "scale_floor": 1e-6,
                                 "center": True})
    z = compress.apply({"c": m}, {"c": x.astype(np.float32)})["c"]
    assert np.isfinite(z).all() and float(np.abs(z).max()) == 0.0
