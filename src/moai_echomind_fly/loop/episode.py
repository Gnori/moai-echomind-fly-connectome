"""조율기 (R-3340 F-015 · 06 §10) — 비실시간 닫힌 고리.

한 결정 창(300 ms)을 3 구간(100 ms)으로: 몸 관측 → 인코더 → 뇌 100 ms → 하강 뉴런 발화 → 몸 100 ms.
창이 끝나면 정책(교사 · 벡터 Qwen · 글 Qwen · …)이 다음 행동을 정한다.
자료 모으기에서는 **뇌 여럿(진짜 · 섞은 배선)** 에 같은 감각 입력을 주고,
몸은 `primary` 뇌를 따른다.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .. import config
from ..graph import Graph
from ..protocols import LoomSpec
from ..sim import Engine, LIFParams
from . import teacher
from .decoder import IDLE, Decoder
from .encoder import Encoder, SensoryMap, build_map
from .signals import ChannelTable, ChannelVector, SignalReader
from .world import BodyClient, Obs, make_task, truth_of


class Brain:
    """엔진 하나 + 채널 읽기 + 자극 배치(감각 뉴런 뒤에 하강 뉴런)."""

    def __init__(self, graph: Graph, params: LIFParams, table: ChannelTable, smap: SensoryMap,
                 decoder: Decoder, seed: int) -> None:
        self.graph = graph
        self.eng = Engine(graph, params, seed=seed)
        self.reader = SignalReader(self.eng, table)
        self.n_sens = len(smap.stim_idx)
        self.eng.set_poisson(np.concatenate([smap.stim_idx, decoder.dn_idx]), 0.0)

    def bin(self, sensory: np.ndarray, dn: np.ndarray):
        self.eng.set_rates(np.concatenate([sensory, dn]))
        return self.reader.run_bin()


@dataclass
class Step:
    """결정 하나의 기록."""
    decision: int
    t_ms: float
    truth: teacher.Truth
    caption: str
    teacher_action: dict
    action: dict
    action_source: str
    reflex_bins: int
    drive: tuple[float, float]
    info: dict


Policy = Callable[[dict[str, dict[str, ChannelVector]], teacher.Truth, Obs, int], tuple[dict, dict]]


def teacher_policy(vectors, truth, obs, decision):
    return teacher.action(truth), {"source": "teacher"}


class Loop:
    def __init__(self, graphs: dict[str, Graph], primary: str, *, seed: int = 0) -> None:
        self.cfg = config.load("task")
        self.loom = LoomSpec.from_config()
        self.params = LIFParams.from_config()
        g0 = graphs[primary]
        self.smap = build_map(g0, self.cfg, self.loom)
        self.decoder = Decoder(g0, self.cfg)
        # 직접 자극하는 뉴런은 벡터에서 뺀다(ChannelTable.build 설명)
        self.table = ChannelTable.build(
            g0, exclude=np.concatenate([self.smap.stim_idx, self.decoder.dn_idx]))
        self.encoder = Encoder(self.smap, self.cfg, self.loom)
        self.brains = {name: Brain(g, self.params, self.table, self.smap, self.decoder, seed)
                       for name, g in graphs.items()}
        self.primary = primary
        self.body = BodyClient()
        self.bin_ms = self.table.window_ms / self.table.bins

    def close(self) -> None:
        self.body.close()

    def episode(self, seed: int, policy: Policy, *, on_step=None) -> dict[str, Any]:
        ep = self.cfg["episode"]
        task = make_task(seed, self.cfg)
        for b in self.brains.values():
            b.eng.reset()
            b.reader = SignalReader(b.eng, self.table)
        obs = self.body.reset(seed, task)
        action = dict(IDLE)
        dn_prev = np.zeros(len(self.decoder.dn_idx))
        steps: list[Step] = []
        min_loom = float("inf")
        reached_at = None
        t0 = time.time()
        timing = {"body_s": 0.0, "brain_s": 0.0, "policy_s": 0.0}
        for d in range(int(ep["max_decisions"])):
            dn_new = self.decoder.target_rates(action)
            bins = {name: [] for name in self.brains}
            reflex_bins = 0
            drive = (0.0, 0.0)
            for b in range(self.table.bins):
                dn = 0.5 * (dn_prev + dn_new) if b == 0 else dn_new
                sens = self.encoder.rates(obs)
                tb = time.time()
                spikes = {}
                for name, br in self.brains.items():
                    rates, depol, spk = br.bin(sens, dn)
                    bins[name].append((rates, depol))
                    spikes[name] = spk
                timing["brain_s"] += time.time() - tb
                drive, reflex = self.decoder.drive(spikes[self.primary], self.bin_ms)
                reflex_bins += int(reflex)
                tb = time.time()
                obs = self.body.step(drive, self.bin_ms)
                timing["body_s"] += time.time() - tb
                lo = obs.looming
                if lo["present"] and lo["distance_mm"] is not None:
                    min_loom = min(min_loom, lo["distance_mm"])
            dn_prev = dn_new
            vectors = {name: self.brains[name].reader.assemble(bins[name]) for name in self.brains}
            truth = truth_of(obs, self.cfg["encoder"]["smell"]["sensors"])
            tp = time.time()
            action, info = policy(vectors, truth, obs, d)
            timing["policy_s"] += time.time() - tp
            step = Step(d, obs.t_ms, truth, teacher.caption(truth), teacher.action(truth), action,
                        info.get("source", "?"), reflex_bins, drive, info)
            steps.append(step)
            if on_step is not None:
                on_step(step, vectors)
            if obs.food_distance_mm <= ep["reach_mm"] and reached_at is None:
                reached_at = obs.t_ms
                break
            if obs.dead:
                break
        radius = task.get("looming", {}).get("radius_mm", 0.0)
        return {
            "seed": seed, "task": task, "decisions": len(steps),
            "reached": reached_at is not None, "reach_ms": reached_at,
            "final_food_mm": round(obs.food_distance_mm, 2),
            "looming": "looming" in task,
            "looming_min_mm": None if min_loom == float("inf") else round(min_loom, 2),
            "looming_hit": bool(min_loom <= radius + ep["hit_mm"]) if "looming" in task else None,
            "dead": obs.dead, "wall_s": round(time.time() - t0, 1),
            "timing": {k: round(v, 1) for k, v in timing.items()},
            "reflex_bins": sum(s.reflex_bins for s in steps),
        }


class DatasetWriter:
    """결정마다 채널 특징(뇌별) · 참값 · 캡션 · 교사 행동을 모은다(F-017 학습 자료)."""

    def __init__(self, out: Path, table: ChannelTable) -> None:
        self.out = out
        self.table = table
        out.mkdir(parents=True, exist_ok=True)
        self.feats: dict[str, dict[str, list[np.ndarray]]] = {}
        self.rows: list[dict[str, Any]] = []

    def add(self, seed: int, step: Step, vectors: dict[str, dict[str, ChannelVector]]) -> None:
        t = self.table
        for brain, vec in vectors.items():
            per = self.feats.setdefault(brain, {n: [] for n in vec})
            for n, v in vec.items():
                per[n].append(v.features(t.rate_scale_hz, t.depol_scale_mV))
        self.rows.append({"seed": seed, "decision": step.decision, "t_ms": step.t_ms,
                          "truth": step.truth.__dict__, "caption": step.caption,
                          "teacher_action": step.teacher_action, "reflex_bins": step.reflex_bins})

    def save(self, meta: dict[str, Any]) -> None:
        for brain, per in self.feats.items():
            np.savez_compressed(self.out / f"features-{brain}.npz",
                                **{n: np.stack(v) for n, v in per.items()})
        (self.out / "samples.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in self.rows), encoding="utf-8")
        (self.out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                            encoding="utf-8")
