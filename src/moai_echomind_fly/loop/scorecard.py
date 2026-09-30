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
from ..guard import GuardRefused, Watchdog, heavy
from . import fly_prompt as fp
from .collect import parse_seeds
from .decoder import IDLE, parse_action
from .episode import Loop
from .signals import text_summary
from .train_cli import TRAIN_RULE

CONFIGS = ["vector", "text", "no_circuit", "rewired", "no_llm"]
# scorecard-v3(2026-09-26 · R-5777 F-002): 판정은 v2 그대로 ·
# 구성마다 결정 기록 `<구성>.decisions.jsonl` 을 더 남긴다
# v4(2026-10-01 · R-1027): 층 구조 구성(-gated · fast-only) · 결정 기록 tier 칸
RECORD_VERSION = "scorecard-v4"
TIER_SUFFIX = {"-gated": "gated", "-fast-only": "never"}
RAW_CHARS = 200            # Qwen 원문은 앞 이만큼만(파일 크기)


def decision_row(seed: int, step) -> dict[str, Any]:
    """결정 하나 → 대시보드가 읽는 한 줄(R-5777 3.3). 냄새 · 먹이 값은 참값(회로 밖)."""
    t = step.truth
    info = step.info or {}
    return {"seed": seed, "d": step.decision, "t_ms": step.t_ms, "truth": dict(t.__dict__),
            "caption": step.caption, "teacher": step.teacher_action, "action": step.action,
            "source": step.action_source, "raw": (info.get("raw") or "")[:RAW_CHARS],
            "valid": info.get("valid"), "why": info.get("why") or None,
            "odor_L": t.odor_left, "odor_R": t.odor_right,
            "food_bearing_deg": t.odor_azimuth_deg, "food_mm": t.food_distance_mm,
            "drive": list(step.drive), "reflex_bins": step.reflex_bins,
            **{k: info[k] for k in ("tier", "trigger", "fast_conf", "threat") if k in info}}


def parse_config(name: str) -> dict[str, Any]:
    """구성 이름 → 바탕 구성 · 뇌 · 층 모드(R-1027). 기존 이름은 층 모드 None(지금과 같은 길).

    `vector-gated` · `rewired-s0-gated` → gated ·
    `fast-only`(= 진짜) · `rewired-s0-fast-only` → never.
    """
    mode = None
    base = name
    if name == "fast-only":
        base, mode = "vector", "never"
    for suf, m in TIER_SUFFIX.items():
        if name.endswith(suf):
            base, mode = name[: -len(suf)], m
    brain = base if base.startswith("rewired") else "real"
    return {"base": base, "brain": brain, "tier_mode": mode}


def build_tiered(llm, brain: str, loop, mode: str):
    """층 구조 조립 — 빠른 정책 · 게이트 · 문턱은 config/control.yaml(문턱 미정이면 gated 거부)."""
    from . import fast_policy as fp
    from .tiered import Gate, TieredPolicy, load_config

    cfg = load_config()
    fast = fp.FastPolicy.load(fp.out_dir(config.current_dataset(), brain, cfg["fast_policy"]))
    gate = Gate({"mode": mode, "thresholds": cfg["thresholds"]}, fast.meta["delta_quantiles"],
                sniff_decisions=int(loop.cfg["episode"].get("sniff_decisions", 0)))
    return TieredPolicy(fast, llm, gate, fp.make_featurizer(fast, loop.table, brain),
                        between_calls=cfg["between_calls"],
                        reflex_threat_min=(cfg["thresholds"] or {}).get("reflex_threat_min"))


def paired(results: dict[str, list[dict]], suffix: str) -> dict[str, list[dict]]:
    """층 구성끼리 판정하려고 이름을 바탕 이름으로(vector-gated → vector)."""
    out = {}
    for c, e in results.items():
        if c == "fast-only" and suffix == "-fast-only":
            out["vector"] = e
        elif c.endswith(suffix):
            out[c[: -len(suffix)]] = e
    return out

SYSTEM_RAW = (
    "너는 초파리의 뇌다. 아래는 세계의 원시 값이다(각도는 + 가 왼쪽). "
    "목표: 먹이 냄새 쪽으로 가되, 다가오는 물체는 피한다. "
    'JSON 한 줄로만 답한다: {"walk":0~1,"turn":"left|right|none","turn_strength":0~1,'
    '"stop":true|false,"groom":true|false}'
)


class _Abort(Exception):
    """감시가 멈추라고 했다 — 끝난 에피소드까지 저장하고 나간다."""


def raw_text(obs, sensors: dict[str, int]) -> str:
    """no_circuit 의 글 — 냄새는 인코더와 같은 센서(task.yaml `encoder.smell.sensors`)."""
    lo = obs.looming
    loom = (f"물체 방위 {lo['azimuth_deg']:+.0f}° · 각 크기 {lo['size_deg']:.1f}° · "
            f"확대 {lo['expansion_deg_s']:+.0f}°/s") if lo["present"] else "물체 없음"
    legs = " ".join(f"{k} {v:.1f}" for k, v in obs.leg_contact.items())
    return (f"시각: {loom}\n냄새 농도: 왼 더듬이 {obs.odor_sensors[sensors['L']]:.2f} · "
            f"오른 더듬이 {obs.odor_sensors[sensors['R']]:.2f}\n다리 접촉 힘: {legs}")


class QwenPolicy:
    """vector · text · no_circuit — 결정마다 Qwen 이 행동 JSON 을 낸다. 위반이면 직전 행동 유지."""

    def __init__(self, kind: str, qwen, names: list[str], k: int, primary: str,
                 projector=None, table=None, squeeze=None,
                 sensors: dict[str, int] | None = None) -> None:
        self.squeeze = squeeze
        self.sensors = sensors or config.load("task")["encoder"]["smell"]["sensors"]
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
        if self.kind == "vector" or self.kind.startswith("rewired"):
            v = vectors[self.primary]
            raw = {n: v[n].features(self.table.rate_scale_hz, self.table.depol_scale_mV)[None]
                   for n in v}
            if self.squeeze is not None:          # train-v4 — 학습 때와 같은 압축
                raw = self.squeeze(raw)
            feats = {n: mx.array(raw[n][0]) for n in self.names}
            emb = self.q.splice(self.act_prompt, self.slot, self.proj(feats))
            gen = self.q.generate(self.act_prompt, max_tokens=60, input_embeddings=emb)
        else:
            body = (text_summary(vectors[self.primary]) if self.kind == "text"
                    else raw_text(obs, self.sensors))
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


def _load_projector(qwen, brain: str, version: str | None = None):
    import mlx.core as mx

    from .projector import ChannelProjector
    from .train_cli import projector_dir

    pdir = projector_dir(brain, version)
    meta = json.loads((pdir / "meta.json").read_text())
    pm = meta["projector"]
    proj = ChannelProjector(meta["dims"], hidden=qwen.hidden, k=pm["k"], kind=pm["kind"],
                            width=meta["rule"]["width"], target_rms=pm["target_rms"])
    proj.load_weights(str(pdir / "final.safetensors"))
    mx.eval(proj.parameters())
    from .train_cli import compress_for

    squeeze, _ = compress_for(meta["rule"], meta.get("data") or {}, brain)
    return proj, list(meta["dims"]), pm["k"], squeeze


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
        **({"llm_calls": int(sum(e["llm_calls"] for e in eps)),
            "call_rate": round(sum(e["llm_calls"] for e in eps)
                               / max(sum(e["decisions"] for e in eps), 1), 3),
            "est_llm_wall_s": round(float(sum(e["llm_calls"] * (e.get("latency_s") or 0.0)
                                              for e in eps)), 1)}
           if eps and all("llm_calls" in e for e in eps) else {}),
    }


def judge(results: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """`vector` 가 재배선(들)을 앞섰나 — scorecard-v2(R-7469 F-001 · F-005).

    재배선이 하나면(v2 결과와 같은 구성) 옛 규칙 그대로: 도착률 또는 회피율이 높으면.
    여럿이면: 먹이 도착률이 **재배선 전부보다 높고**, 씨앗별 쌍 차이(vector − 재배선 평균)의
    부호 뒤집기 순열 p < alpha. 회피율은 가르는 힘이 없어(움직이면 전부 1.0) 쓰지 않는다.
    """
    from .stats import STATS_RULE, bootstrap_ci, sign_flip_p

    rew = sorted(c for c in results if c.startswith("rewired"))
    if "vector" not in results or not rew:
        return {"rule": "scorecard-v2", "vector_beats_rewired": None}
    seeds = [e["seed"] for e in results["vector"]]
    reach = {c: np.array([float(next(e for e in results[c] if e["seed"] == s)["reached"])
                          for s in seeds]) for c in ["vector", *rew]}
    ci = {c: list(bootstrap_ci(v.astype(bool), np.array(seeds))) for c, v in reach.items()}
    if len(rew) == 1:
        v, r = summarize(results["vector"]), summarize(results[rew[0]])
        beats = bool((v["reach_rate"] or 0) > (r["reach_rate"] or 0)
                     or ((v["avoid_rate"] or 0) > (r["avoid_rate"] or 0)))
        return {"rule": "scorecard-v1(재배선 하나)", "rewired": rew, "reach_ci95": ci,
                "vector_beats_rewired": beats}
    ens = np.mean([reach[c] for c in rew], axis=0)
    p = sign_flip_p(reach["vector"] - ens)
    rank = float(100.0 * np.mean([reach[c].mean() < reach["vector"].mean() for c in rew]))
    beats = bool(rank == 100.0 and p < STATS_RULE["alpha"])
    return {"rule": "scorecard-v2", "rewired": rew, "reach_ci95": ci, "rank_pct": rank,
            "p_sign_flip": p, "vector_beats_rewired": beats}


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
            watchdog = Watchdog("scorecard")   # v5 성적표는 감시 없이 13 시간 돌았다
            done = 0
            results: dict[str, list[dict]] = {}
            for cfg_name in configs:
                pc = parse_config(cfg_name)
                base, brain = pc["base"], pc["brain"]
                seed_k = (int(base.rsplit("-s", 1)[1]) if base.startswith("rewired-s")
                          else args.rewired_seed)
                graphs = {brain: (graph_mod.rewire(g, seed_k) if brain != "real" else g)}
                loop = Loop(graphs, brain)
                if base == "no_llm":
                    policy = None
                else:
                    llm = None
                    if pc["tier_mode"] != "never":
                        uses_proj = base == "vector" or base.startswith("rewired")
                        proj, names, k, squeeze = (_load_projector(q, brain, args.version)
                                                   if uses_proj
                                                   else (None, list(loop.table.channels), 2,
                                                         None))
                        llm = QwenPolicy(base, q, names, k, brain, proj, loop.table, squeeze)
                    policy = (build_tiered(llm, brain, loop, pc["tier_mode"])
                              if pc["tier_mode"] else llm)
                eps = []
                dec_file = (out / f"{cfg_name}.decisions.jsonl").open("w", encoding="utf-8")
                try:
                    for seed in seeds:
                        lat: list[float] = []

                        def on_step(step, vectors, lat=lat, seed=seed):
                            if "latency_s" in step.info:
                                lat.append(step.info["latency_s"])
                            dec_file.write(json.dumps(decision_row(seed, step),
                                                      ensure_ascii=False) + "\n")

                        if policy is not None:
                            policy.reset()
                            v0 = policy.violations
                        res = loop.episode(seed, policy or no_llm_policy, on_step=on_step)
                        res["violations"] = (policy.violations - v0) if policy is not None else 0
                        res["llm_calls"] = (int(getattr(policy, "llm_calls", res["decisions"]))
                                            if policy is not None else 0)
                        res["latency_s"] = float(np.mean(lat)) if lat else None
                        res.pop("task", None)
                        eps.append(res)
                        print(f"[{cfg_name}] seed {seed} · 도착 {res['reached']} "
                              f"({res['reach_ms']}) · 부딪힘 {res['looming_hit']} · 위반 "
                              f"{res['violations']} · {res['wall_s']} s", flush=True)
                        done += 1
                        why = watchdog.check(done)
                        if why:
                            raise _Abort(why)
                finally:
                    loop.close()
                    dec_file.close()
                    (out / f"{cfg_name}.jsonl").write_text(
                        "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in eps),
                        encoding="utf-8")
                results[cfg_name] = eps
            table = {c: summarize(e) for c, e in results.items()}
            verdict = judge(results)
            for suf, key in (("-gated", "judgement_gated"), ("-fast-only", "judgement_fast_only")):
                pr = paired(results, suf)
                if "vector" in pr and any(c.startswith("rewired") for c in pr):
                    verdict[key] = judge(pr)
            beats = verdict["vector_beats_rewired"]
            wording = ("초파리 배선 벡터가 섞은 배선보다 나았다" if beats
                       else "섞은 배선 대조를 이기지 못했다 — '초파리 뇌'라는 말을 쓰지 않는다")
            card = {"seeds": seeds, "llm": q.label, "configs": table,
                    "projector_version": args.version or TRAIN_RULE["version"],
                    "judgement": verdict, "record": RECORD_VERSION,
                    "vector_beats_rewired": beats, "wording": wording,
                    "notes": ["no_circuit 에는 영상 이미지를 주지 않았다(mlx-lm 은 글 부분만)",
                              "몸의 걷기 리듬은 FlyGym 1.x 제어기가 만든다(06 (나) 갈래)"]}
            (out / "scorecard.json").write_text(json.dumps(card, ensure_ascii=False, indent=2),
                                                encoding="utf-8")
            print(json.dumps(card, ensure_ascii=False, indent=2))
    except GuardRefused as err:
        print(f"[guard] 막음 — {err}")
        return 2
    except _Abort as err:
        # 성적표는 다시 돌려야 한다(구성 사이 비교가 같은 씨앗이어야 함) — 끝난 것은 기록으로만
        (out / "aborted.json").write_text(json.dumps(
            {"reason": str(err), "watchdog": watchdog.last, "episodes_done": done,
             "configs_done": list(results)}, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[watchdog] 멈춤 — {err} · {watchdog.last} · 끝난 에피소드 {done} · {out}",
              flush=True)
        return 3
    return 0


def register(sub) -> None:
    p = sub.add_parser("scorecard", help="F-016 비교군·대조군 성적표(무거움 · Qwen)")
    p.add_argument("--seeds", default="2000-2009")
    p.add_argument("--configs", default=",".join(CONFIGS))
    p.add_argument("--graph", default="traced-v1")
    p.add_argument("--rewired-seed", type=int, default=0)
    p.add_argument("--version", default=None,
                   help="투영기 규칙 판(기본 = 지금 TRAIN_RULE · 예전 결과는 train-v2)")
    p.set_defaults(func=main)
