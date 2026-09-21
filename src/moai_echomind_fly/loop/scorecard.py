"""비교군·대조군 성적표 (R-3340 F-016 · I §3.5).

| 구성 | 앞(인코더) | 뒤 |
| vector     | 진짜 배선 → 채널 벡터 → 투영기(학습) | Qwen |
| text       | 진짜 배선 → 글 요약                   | Qwen(학습 없음) |
| no_circuit | 세계의 원시 값(숫자 글)               | Qwen(학습 없음) |
| rewired    | 섞은 배선 → 채널 벡터 → 투영기(똑같이 학습) | Qwen |
| no_llm     | 진짜 배선 → 하강 뉴런 그대로(명령 없음) | 없음 |

⚠️ no_circuit 은 R 이 적은 "영상 한 장"을 주지 않는다 —
mlx-lm 은 Qwen 의 글 부분만 올린다(정직 목록).
`rewired` 를 이기지 못하면 보고서에 "초파리 뇌"라는 말을 쓰지 않는다(R 2.0-6).
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Any

import numpy as np

from .. import config, paths
from .. import graph as graph_mod
from ..guard import GuardRefused, heavy
from . import fly_prompt as fp
from .collect import parse_seeds
from .decoder import IDLE, parse_action
from .episode import Loop
from .signals import text_summary

CONFIGS = ["vector", "text", "no_circuit", "rewired", "no_llm"]

SYSTEM_RAW = (
    "너는 초파리의 뇌다. 아래는 세계의 원시 값이다(각도는 + 가 왼쪽). "
    "목표: 먹이 냄새 쪽으로 가되, 다가오는 물체는 피한다. "
    'JSON 한 줄로만 답한다: {"walk":0~1,"turn":"left|right|none","turn_strength":0~1,'
    '"stop":true|false,"groom":true|false}'
)


def raw_text(obs) -> str:
    lo = obs.looming
    loom = (f"물체 방위 {lo['azimuth_deg']:+.0f}° · 각 크기 {lo['size_deg']:.1f}° · "
            f"확대 {lo['expansion_deg_s']:+.0f}°/s") if lo["present"] else "물체 없음"
    legs = " ".join(f"{k} {v:.1f}" for k, v in obs.leg_contact.items())
    return (f"시각: {loom}\n냄새 농도: 왼 더듬이 {obs.odor_sensors[0]:.2f} · "
            f"오른 더듬이 {obs.odor_sensors[1]:.2f}\n다리 접촉 힘: {legs}")


class QwenPolicy:
    """vector · text · no_circuit — 결정마다 Qwen 이 행동 JSON 을 낸다. 위반이면 직전 행동 유지."""

    def __init__(self, kind: str, qwen, names: list[str], k: int, primary: str,
                 projector=None, table=None) -> None:
        self.kind, self.q, self.names, self.k = kind, qwen, names, k
        self.primary, self.proj, self.table = primary, projector, table
        self.slot = qwen.tok.convert_tokens_to_ids(fp.SLOT)
        self.act_prompt = qwen.render(
            fp.messages(fp.SYSTEM_ACTION, fp.INSTRUCTION_ACTION, names, k))
        self.last = dict(IDLE)
        self.violations = 0
        self.decisions = 0

    def reset(self) -> None:
        self.last = dict(IDLE)

    def __call__(self, vectors, truth, obs, decision):
        import mlx.core as mx

        t0 = time.time()
        if self.kind in ("vector", "rewired"):
            v = vectors[self.primary]
            feats = {n: mx.array(v[n].features(self.table.rate_scale_hz, self.table.depol_scale_mV))
                     for n in self.names}
            emb = self.q.splice(self.act_prompt, self.slot, self.proj(feats))
            gen = self.q.generate(self.act_prompt, max_tokens=60, input_embeddings=emb)
        else:
            body = text_summary(vectors[self.primary]) if self.kind == "text" else raw_text(obs)
            system = fp.SYSTEM_ACTION if self.kind == "text" else SYSTEM_RAW
            prompt = self.q.render(fp.messages(system, fp.INSTRUCTION_ACTION, self.names, self.k,
                                               text_summary=body))
            gen = self.q.generate(prompt, max_tokens=60)
        self.decisions += 1
        action, why = parse_action(gen.text)
        if action is None:
            self.violations += 1
            action = self.last
        self.last = action
        return action, {"source": self.kind, "raw": gen.text, "valid": not why, "why": why,
                        "prompt_tokens": gen.prompt_tokens, "latency_s": round(time.time() - t0, 2)}


def no_llm_policy(vectors, truth, obs, decision):
    return dict(IDLE), {"source": "no_llm"}


def _load_projector(qwen, brain: str):
    import mlx.core as mx

    from .projector import ChannelProjector
    from .train_cli import TRAIN_RULE

    pdir = paths.runs_dir() / f"projector-{brain}-{TRAIN_RULE['version']}"
    meta = json.loads((pdir / "meta.json").read_text())
    pm = meta["projector"]
    proj = ChannelProjector(meta["dims"], hidden=qwen.hidden, k=pm["k"], kind=pm["kind"],
                            width=meta["rule"]["width"], target_rms=pm["target_rms"])
    proj.load_weights(str(pdir / "final.safetensors"))
    mx.eval(proj.parameters())
    return proj, list(meta["dims"]), pm["k"]


def summarize(eps: list[dict[str, Any]]) -> dict[str, Any]:
    reached = [e for e in eps if e["reached"]]
    loom = [e for e in eps if e["looming"]]
    return {
        "episodes": len(eps),
        "reach_rate": round(len(reached) / len(eps), 3) if eps else None,
        "reach_ms_mean": (round(float(np.mean([e["reach_ms"] for e in reached])), 1)
                          if reached else None),
        "reach_ms_std": (round(float(np.std([e["reach_ms"] for e in reached])), 1)
                         if reached else None),
        "final_food_mm_mean": round(float(np.mean([e["final_food_mm"] for e in eps])), 2),
        "looming_episodes": len(loom),
        "avoid_rate": round(1 - np.mean([e["looming_hit"] for e in loom]), 3) if loom else None,
        "schema_violations": int(sum(e.get("violations", 0) for e in eps)),
        "decisions": int(sum(e["decisions"] for e in eps)),
        "latency_s_mean": round(float(np.mean([e["latency_s"] for e in eps if e.get("latency_s")]
                                              or [0.0])), 2),
    }


def main(args: argparse.Namespace) -> int:
    from .qwen import QwenLM

    seeds = parse_seeds(args.seeds)
    configs = args.configs.split(",")
    out = paths.runs_dir() / f"scorecard-{time.strftime('%Y%m%d-%H%M%S')}"
    out.mkdir(parents=True)
    try:
        with heavy("scorecard", extra_gb=float(config.load("llm")["weights_gb"])) as guard:
            print(f"[guard] 메모리 여유 {guard.free_percent}%", flush=True)
            g = graph_mod.load(args.graph)
            q = QwenLM()
            results: dict[str, list[dict]] = {}
            for cfg_name in configs:
                brain = "rewired" if cfg_name == "rewired" else "real"
                graphs = {brain: (graph_mod.rewire(g, args.rewired_seed)
                                  if brain == "rewired" else g)}
                loop = Loop(graphs, brain)
                if cfg_name == "no_llm":
                    policy = None
                else:
                    proj, names, k = (_load_projector(q, brain) if cfg_name in ("vector", "rewired")
                                      else (None, list(loop.table.channels), 2))
                    policy = QwenPolicy(cfg_name, q, names, k, brain, proj, loop.table)
                eps = []
                try:
                    for seed in seeds:
                        lat: list[float] = []

                        def on_step(step, vectors, lat=lat):
                            if "latency_s" in step.info:
                                lat.append(step.info["latency_s"])

                        if policy is not None:
                            policy.reset()
                            v0 = policy.violations
                        res = loop.episode(seed, policy or no_llm_policy, on_step=on_step)
                        res["violations"] = (policy.violations - v0) if policy is not None else 0
                        res["latency_s"] = float(np.mean(lat)) if lat else None
                        res.pop("task", None)
                        eps.append(res)
                        print(f"[{cfg_name}] seed {seed} · 도착 {res['reached']} "
                              f"({res['reach_ms']}) · 부딪힘 {res['looming_hit']} · 위반 "
                              f"{res['violations']} · {res['wall_s']} s", flush=True)
                finally:
                    loop.close()
                results[cfg_name] = eps
                (out / f"{cfg_name}.jsonl").write_text(
                    "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in eps),
                    encoding="utf-8")
            table = {c: summarize(e) for c, e in results.items()}
            v, r = table.get("vector"), table.get("rewired")
            beats = None
            if v and r:
                beats = bool((v["reach_rate"] or 0) > (r["reach_rate"] or 0)
                             or ((v["avoid_rate"] or 0) > (r["avoid_rate"] or 0)))
            wording = ("초파리 배선 벡터가 섞은 배선보다 나았다" if beats
                       else "섞은 배선 대조를 이기지 못했다 — '초파리 뇌'라는 말을 쓰지 않는다")
            card = {"seeds": seeds, "llm": q.label, "configs": table,
                    "vector_beats_rewired": beats, "wording": wording,
                    "notes": ["no_circuit 에는 영상 이미지를 주지 않았다(mlx-lm 은 글 부분만)",
                              "몸의 걷기 리듬은 FlyGym 1.x 제어기가 만든다(06 (나) 갈래)"]}
            (out / "scorecard.json").write_text(json.dumps(card, ensure_ascii=False, indent=2),
                                                encoding="utf-8")
            print(json.dumps(card, ensure_ascii=False, indent=2))
    except GuardRefused as err:
        print(f"[guard] 막음 — {err}")
        return 2
    return 0


def register(sub) -> None:
    p = sub.add_parser("scorecard", help="F-016 비교군·대조군 성적표(무거움 · Qwen)")
    p.add_argument("--seeds", default="2000-2009")
    p.add_argument("--configs", default=",".join(CONFIGS))
    p.add_argument("--graph", default="traced-v1")
    p.add_argument("--rewired-seed", type=int, default=0)
    p.set_defaults(func=main)
