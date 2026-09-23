"""채널별 압축 · 선형 관문 compress-v1 (R-7469 F-004 · I §3.4).

규칙은 `config/compress.yaml`(결과 보기 전에 적음). 뇌마다 · 채널마다 PCA 를
**학습 나눔으로만** 맞추고, 압축한 특징으로 선형 검사 판정(stats) →
한 칸이라도 통과한 채널만 다음 단계(투영기 train-v4)로 넘긴다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from .. import config, datasets, paths


def fit_channel(x: np.ndarray, cfg: dict[str, Any]) -> dict[str, np.ndarray]:
    """한 채널 PCA — 설명 분산이 `var_target` 이상인 가장 작은 k([k_min, k_max] 안)."""
    x = x.astype(np.float64)
    mean = x.mean(0) if cfg.get("center", True) else np.zeros(x.shape[1])
    _, sv, vt = np.linalg.svd(x - mean, full_matrices=False)
    var = sv ** 2
    frac = np.cumsum(var) / max(var.sum(), 1e-12)
    k = int(np.searchsorted(frac, cfg["var_target"]) + 1)
    k = int(np.clip(k, cfg["k_min"], min(cfg["k_max"], vt.shape[0])))
    return {"mean": mean.astype(np.float32), "components": vt[:k].astype(np.float32),
            "explained": np.float32(frac[k - 1])}


def apply(model: dict[str, dict[str, np.ndarray]], feats: dict[str, np.ndarray]
          ) -> dict[str, np.ndarray]:
    return {ch: ((feats[ch] - m["mean"]) @ m["components"].T).astype(np.float32)
            for ch, m in model.items() if ch in feats}


def out_dir(dataset: str, cfg: dict[str, Any] | None = None) -> Path:
    cfg = cfg or config.load("compress")
    return paths.runs_dir() / f"compress-{dataset}-{cfg['version']}"


def fit(dataset: str) -> dict[str, Any]:
    cfg = config.load("compress")
    out = out_dir(dataset, cfg)
    out.mkdir(parents=True, exist_ok=True)
    brains = ["real", *datasets.rewired_brains(dataset)]
    info: dict[str, Any] = {}
    for brain in brains:
        ftr, _ = datasets.load(dataset, "train", brain)
        model = {ch: fit_channel(x, cfg) for ch, x in ftr.items()}
        np.savez_compressed(out / f"{brain}.npz", **{f"{ch}.{k}": v for ch, m in model.items()
                                                    for k, v in m.items()})
        info[brain] = {ch: {"k": int(m["components"].shape[0]), "dims": int(ftr[ch].shape[1]),
                            "explained": round(float(m["explained"]), 4)}
                       for ch, m in model.items()}
    sums = datasets.store_dir() / dataset / "SHA256SUMS"
    meta = {"rule": cfg, "dataset": dataset, "brains": brains, "channels": info,
            "dataset_sha256sums": hashlib.sha256(sums.read_bytes()).hexdigest()
            if sums.exists() else None, "when": time.strftime("%Y-%m-%d %H:%M:%S")}
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def load(dataset: str, brain: str) -> dict[str, dict[str, np.ndarray]]:
    z = np.load(out_dir(dataset) / f"{brain}.npz")
    model: dict[str, dict[str, np.ndarray]] = {}
    for key in z.files:
        ch, part = key.rsplit(".", 1)
        model.setdefault(ch, {})[part] = z[key]
    return model


def gate(dataset: str) -> dict[str, Any]:
    from .linear_probe import run

    cache: dict[str, dict] = {}

    def transform(brain: str, feats: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        if brain not in cache:
            cache[brain] = load(dataset, brain)
        return apply(cache[brain], feats)

    res = run(dataset, transform=transform)
    passed = sorted({k.split("|", 1)[0] for k, c in res["cells"].items() if c["pass"]})
    out = {"dataset": dataset, "rule": config.load("compress")["version"],
           "stats": res["stats"]["version"], "passed_channels": passed,
           "passed_cells": [k for k, c in res["cells"].items() if c["pass"]],
           "cells": res["cells"], "when": time.strftime("%Y-%m-%d %H:%M:%S")}
    (out_dir(dataset) / "gate.json").write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                                encoding="utf-8")
    return out


def main_fit(args: argparse.Namespace) -> int:
    meta = fit(args.dataset)
    for brain, chs in meta["channels"].items():
        print(brain, " · ".join(f"{ch} {v['dims']}→{v['k']}({v['explained']:.2f})"
                                for ch, v in chs.items()))
    print(f"[compress] 저장 {out_dir(args.dataset)}")
    return 0


def main_gate(args: argparse.Namespace) -> int:
    g = gate(args.dataset)
    print(f"[gate] 통과 채널 {g['passed_channels']} · 칸 {g['passed_cells']}")
    print(f"[gate] 기록 {out_dir(args.dataset) / 'gate.json'}")
    return 0


def register(sub) -> None:
    p = sub.add_parser("compress", help="채널별 압축 · 선형 관문(R-7469 F-004 · CPU)")
    cs = p.add_subparsers(dest="compress_cmd", required=True)
    f = cs.add_parser("fit", help="뇌마다 · 채널마다 PCA(학습 나눔으로만)")
    f.add_argument("--dataset", default="fly-loop-v2")
    f.set_defaults(func=main_fit)
    g = cs.add_parser("gate", help="압축 특징으로 판정 → 통과 채널")
    g.add_argument("--dataset", default="fly-loop-v2")
    g.set_defaults(func=main_gate)
