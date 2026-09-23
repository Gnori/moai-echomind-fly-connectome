"""compress-v1 — 압축기 저장 · 불러오기 왕복 · 채널 건너뛰기 · 재배선 뇌 이름 순서.

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
