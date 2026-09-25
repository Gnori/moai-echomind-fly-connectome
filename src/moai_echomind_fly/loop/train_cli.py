"""투영기 학습 · 캡션 평가 명령 (R-3340 F-017).

학습(`train`): ① 정렬 — 신호 → 캡션 ② 행동 — 신호 → 행동 JSON
(캡션을 일부 섞어 두 쓰임을 함께 지킨다). Qwen 은 얼리고 투영기만 배운다 — 파인튜닝이 아니다.
진짜 배선과 섞은 배선 벡터로 **똑같이** 학습한다(대조군).
평가(`eval-caption`): 학습에 안 쓴 씨앗에서 캡션을 만들게 하고 칸별 정확도를 잰다
(최빈값 기준선과 나란히).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import shutil
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from .. import config, paths
from ..guard import GuardRefused, Watchdog, heavy
from . import fly_prompt as fp
from .collect import parse_seeds
from .status import StatusFile
from .status import now as status_now

TRAIN_RULE = {
    # v2(2026-09-22): 행동 단계는 새 옵티마이저로 시작한다 — 캡션 단계 저장본에서 이어 가도
    # 진짜 · 섞은 배선이 같은 절차가 되게(저장본에는 옵티마이저 상태가 없다).
    # 캡션 단계는 v1 과 같다.
    # v3(2026-09-23): 투영기 초기값에 씨앗(init_seed)을 준다 — v2 까지는 초기값이 실행마다 달라
    # 같은 설정으로도 같은 저장본이 나오지 않았다(W loop 잔여 이슈 5). 나머지는 v2 와 같다.
    # v4(2026-09-23 · R-7469 F-005): 입력 = 관문을 통과한 채널의 **압축 벡터**(compress-v1) ·
    # 채널당 **선형 토큰 1 개** · 손실 가중 = 칸 값 토큰 1 · 틀 토큰 `template_weight`.
    # 틀을 0 으로 두지 않는 까닭: 시스템 글에 캡션 서식이 없어 서식을 배울 곳이 정답뿐이다
    # (연구 노트 12 §4).
    # v5(2026-09-24): 입력 압축을 compress-v3(채널마다 평균 크기 1)으로 — v4 는 뇌 · 채널마다
    # 입력 크기가 달라 입력이 2 배 큰 재배선이 먼저 배웠다(config/compress.yaml v3 주석 · 실측).
    # 나머지는 v4 와 같다.
    "version": "train-v5",
    "input": "compress", "compress": "compress-v3",   # v1 차원 다름 → v2 같은 64 → v3 같은 크기
    "k": 1, "width": 0, "kind": "linear", "lr": 2e-4,
    "caption_epochs": 2, "action_epochs": 1, "caption_rehearsal": 0.3,
    "val_samples": 24, "seed": 0, "init_seed": 0, "template_weight": 0.1,
}

_CAP_VALUE = re.compile(r": ([^/]+?)(?= / |$)")
_ACT_VALUE = re.compile(r':("?[^",}]*"?)')


def value_spans(text: str, task: str) -> list[tuple[int, int]]:
    """정답 글에서 **칸 값**의 글자 구간 — 캡션은 `: 값 /`, 행동 JSON 은 `:값,`."""
    pat = _CAP_VALUE if task == "cap" else _ACT_VALUE
    return [m.span(1) for m in pat.finditer(text)]


def token_weights(offsets: list[tuple[int, int]], spans: list[tuple[int, int]],
                  template_weight: float) -> list[float]:
    """토큰마다 가중 — 칸 값 구간과 겹치면 1, 아니면 `template_weight`."""
    return [1.0 if any(a < e and b > s for s, e in spans) else float(template_weight)
            for a, b in offsets]


def projector_dir(brain: str, version: str | None = None) -> Path:
    """투영기 결과 폴더 — 규칙 판마다 따로(`runs/projector-<뇌>-<판>`)."""
    return paths.runs_dir() / f"projector-{brain}-{version or TRAIN_RULE['version']}"


def new_projector(dims: dict[str, int], hidden: int, rule: dict, target_rms: float):
    """규칙대로 투영기를 만든다. `init_seed` 가 있으면 초기값이 씨앗으로 정해진다."""
    import mlx.core as mx

    from .projector import ChannelProjector

    if rule.get("init_seed") is not None:
        mx.random.seed(int(rule["init_seed"]))
    proj = ChannelProjector(dims, hidden=hidden, k=rule["k"], kind=rule["kind"],
                            width=rule["width"], target_rms=target_rms)
    mx.eval(proj.parameters())
    return proj


def data_source(data: str, val: str = "eval") -> dict[str, Any]:
    """학습 자료가 어디 있나 — 묶음(`data/datasets/<이름>`)이 있으면 그것, 없으면 `runs/` 조각."""
    from .. import datasets

    pack = datasets.store_dir() / data
    if pack.is_dir():
        sums = pack / "SHA256SUMS"
        digest = hashlib.sha256(sums.read_bytes()).hexdigest() if sums.exists() else None
        return {"pack": data, "sha256sums": digest}
    return {"runs": {"train": expand(data), "eval": expand(val)}}


def load_split(src: dict[str, Any], split: str, brain: str
               ) -> tuple[dict[str, np.ndarray], list[dict]]:
    """`data_source` 가 가리키는 자료에서 train · eval 한 쪽을 불러온다."""
    from .. import datasets

    if "pack" in src:
        return datasets.load(src["pack"], split, brain)
    return load_datasets(src["runs"][split], brain)


def load_datasets(names: list[str], brain: str) -> tuple[dict[str, np.ndarray], list[dict]]:
    feats: dict[str, list[np.ndarray]] = {}
    rows: list[dict] = []
    for name in names:
        d = paths.runs_dir() / f"dataset-{name}"
        with np.load(d / f"features-{brain}.npz") as z:
            for ch in z.files:
                feats.setdefault(ch, []).append(z[ch])
        lines = (d / "samples.jsonl").read_text().splitlines()
        rows += [json.loads(line) for line in lines if line]
    return {ch: np.concatenate(v) for ch, v in feats.items()}, rows


def expand(names: str) -> list[str]:
    out = []
    for n in names.split(","):
        workers = sorted(paths.runs_dir().glob(f"dataset-{n}-w*"))
        out += [w.name.removeprefix("dataset-") for w in workers] or [n]
    return out


class _Setup:
    def __init__(self, qwen, names: list[str], k: int,
                 template_weight: float | None = None) -> None:
        self.q = qwen
        self.names = names
        self.template_weight = template_weight
        self.im_end = qwen.tok.convert_tokens_to_ids("<|im_end|>")
        self.cap_prompt = qwen.render(
            fp.messages(fp.SYSTEM_CAPTION, fp.INSTRUCTION_CAPTION, names, k))
        self.act_prompt = qwen.render(
            fp.messages(fp.SYSTEM_ACTION, fp.INSTRUCTION_ACTION, names, k))

    def target(self, text: str) -> list[int]:
        return list(self.q.tok.encode(text, add_special_tokens=False)) + [self.im_end]

    def weighted(self, text: str, task: str) -> tuple[list[int], list[float] | None]:
        """정답 토큰 + 토큰마다 손실 가중(train-v4). 가중 규칙이 없으면 (토큰, None)."""
        if self.template_weight is None:
            return self.target(text), None
        enc = self.q.tok._tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        w = token_weights(enc["offset_mapping"], value_spans(text, task), self.template_weight)
        return list(enc["input_ids"]) + [self.im_end], w + [float(self.template_weight)]


def _sample(setup: _Setup, feats, rows, i: int, task: str):
    from .train import Sample

    f = {n: feats[n][i] for n in setup.names}
    if task == "cap":
        tgt, w = setup.weighted(rows[i]["caption"], "cap")
        return Sample(setup.cap_prompt, tgt, f, w)
    tgt, w = setup.weighted(fp.action_text(rows[i]["teacher_action"]), "act")
    return Sample(setup.act_prompt, tgt, f, w)


def compress_for(rule: dict, src: dict, brain: str):
    """train-v4 입력 — (압축기 적용 함수, 통과 채널). 규칙에 압축이 없으면 (None, None)."""
    if rule.get("input") != "compress":
        return None, None
    from . import compress

    dataset = src["pack"]
    ver = rule["compress"]                          # 학습 규칙이 정한 압축 판(설정 판이 아니라)
    gate = json.loads((compress.out_dir(dataset, version=ver) / "gate.json").read_text())
    model = compress.load(dataset, brain, version=ver)
    passed = gate["passed_channels"]
    if not passed:
        raise GuardRefused("gate", "관문을 통과한 채널이 없다 — train-v4 를 하지 않는다(F-004)")
    return (lambda feats: compress.apply({c: model[c] for c in passed}, feats)), passed


class TrainingAborted(RuntimeError):
    """감시가 멈춘 학습(메모리) — 그때까지의 투영기는 저장된다."""


def _mem() -> dict[str, float]:
    import mlx.core as mx

    return {"active_gb": round(mx.get_active_memory() / 1e9, 2),
            "cache_gb": round(mx.get_cache_memory() / 1e9, 2),
            "peak_gb": round(mx.get_peak_memory() / 1e9, 2)}


def qwen_fingerprint(qwen) -> str:
    """Qwen 전체 가중치의 지문 — 학습 전후가 같아야 한다(파인튜닝이 아니다 · R F-013)."""
    import hashlib

    import mlx.core as mx
    from mlx.utils import tree_flatten

    h = hashlib.sha256()
    for name, v in sorted(tree_flatten(qwen.model.parameters()), key=lambda kv: kv[0]):
        h.update(name.encode())
        h.update(float(mx.sum(v.astype(mx.float32))).hex().encode())
        h.update(float(mx.sum(mx.abs(v.astype(mx.float32)))).hex().encode())
    return h.hexdigest()


def train(args: argparse.Namespace) -> int:
    import mlx.core as mx
    from mlx.utils import tree_flatten

    from .qwen import QwenLM
    from .train import Trainer

    rule = dict(TRAIN_RULE)
    gcfg = config.load("guard")
    out = projector_dir(args.brain, rule["version"])
    if (out / "final.safetensors").exists():
        # 끝난 학습을 덮어쓰지 않는다 — 규칙을 바꿨으면 판을 올린다(결과 폴더가 새로 생긴다)
        print(f"[train] 막음 — {out}/final.safetensors 가 이미 있다. 같은 판으로 다시 돌리려면 "
              "폴더를 옮긴 뒤 돌린다.")
        return 2
    out.mkdir(parents=True, exist_ok=True)
    log = (out / "train.log").open("a", encoding="utf-8")
    # 상태 파일(R-7347) — 대시보드 · 다른 세션이 로그 대신 읽는다. 학습 규칙에는 영향 없음(기록만).
    status = StatusFile(out / "status.json", brain=args.brain, version=rule["version"],
                        rule=rule, state="running", stage="loading")

    def say(msg: str) -> None:
        print(msg, flush=True)
        log.write(msg + "\n")
        log.flush()

    try:
        with heavy("train", min_free_percent=int(gcfg["training_min_free_percent"])) as guard:
            say(f"[guard] 메모리 여유 {guard.free_percent}% ≥ 학습 문턱 {guard.threshold}%")
            src = data_source(args.dataset, args.val)
            feats, rows = load_split(src, "train", args.brain)
            vfeats, vrows = load_split(src, "eval", args.brain)
            if args.train_seeds:               # 학습 표본을 에피소드(씨앗)로 줄인다 — 시간(B 안)
                keep = set(parse_seeds(args.train_seeds))
                idx = np.array([i for i, r in enumerate(rows) if int(r["seed"]) in keep])
                feats = {n: v[idx] for n, v in feats.items()}
                rows = [rows[i] for i in idx]
                src = {**src, "train_seeds": args.train_seeds, "train_samples": int(len(idx))}
            squeeze, passed = compress_for(rule, src, args.brain)
            if squeeze is not None:
                feats, vfeats = squeeze(feats), squeeze(vfeats)
                src = {**src, "compress": rule["compress"], "channels": passed}
            say(f"[data] {src}")
            status.update(data=src)
            names = list(feats)
            q = QwenLM()
            fp_before = qwen_fingerprint(q)
            say(f"[qwen] 가중치 지문(학습 전) {fp_before[:16]}")
            emb = q.embed(mx.arange(1000, 3000)).astype(mx.float32)
            rms = float(mx.sqrt(mx.mean(emb * emb)))
            proj = new_projector({n: feats[n].shape[1] for n in names}, q.hidden, rule, rms)
            trainer = Trainer(q, proj, lr=rule["lr"],
                              cache_limit_gb=float(gcfg["training_cache_limit_gb"]))
            watchdog = Watchdog()
            total = [0]
            setup = _Setup(q, names, rule["k"], rule.get("template_weight"))
            rng = random.Random(rule["seed"])
            n, nv = len(rows), min(rule["val_samples"], len(vrows))
            say(f"[train] {args.brain} · 학습 {n} · 검증 {nv} · "
                f"투영기 {proj.n_params():,} 매개변수 · "
                f"입력 {len(setup.cap_prompt)}/{len(setup.act_prompt)} 토큰 · 규칙 {rule}")
            cap_steps = n * int(rule["caption_epochs"])
            status.update(train_samples=n, val_samples=nv, params=proj.n_params(),
                          fingerprint_before=fp_before[:16], caption_steps=cap_steps,
                          total_steps=cap_steps + n * int(rule["action_epochs"])
                          + int(n * rule["caption_rehearsal"]), stage="before")

            def val(task: str) -> float:
                return float(np.mean([trainer.loss_only(_sample(setup, vfeats, vrows, i, task))
                                      for i in range(nv)]))

            def run_epochs(plan: list[tuple[int, str]], stage: str, epochs: int,
                           replay_only: bool = False) -> None:
                step = 0
                for ep in range(epochs):
                    order = plan[:]
                    rng.shuffle(order)
                    if replay_only:   # 이어 가기 — 이 단계가 쓴 난수만 소비해 다음 단계 순서를 같게
                        continue
                    losses, t0 = [], time.time()
                    for i, task in order:
                        losses.append(trainer.step(_sample(setup, feats, rows, i, task)))
                        step += 1
                        total[0] += 1
                        why = watchdog.check(total[0])
                        if why:
                            mx.save_safetensors(str(out / f"aborted-{stage}.safetensors"),
                                                dict(tree_flatten(proj.parameters())))
                            say(f"[watchdog] 멈춤 — {why} · {watchdog.last} · {_mem()} · "
                                f"저장 aborted-{stage}.safetensors")
                            status.update(state="aborted", reason=why, finished=status_now(),
                                          saved=f"aborted-{stage}.safetensors", exit_code=3,
                                          mem={**_mem(), **watchdog.last})
                            raise TrainingAborted(why)
                        if step % 25 == 0:
                            say(f"[mem] {_mem()} · {watchdog.last}")
                            say(f"[{stage}] ep {ep} step {step} "
                                f"loss {np.mean(losses[-25:]):.3f} · "
                                f"{(time.time() - t0) / len(losses):.1f} s/걸음")
                            status.update(stage=stage, epoch=ep, stage_step=step, step=total[0],
                                          loss=float(np.mean(losses[-25:])),
                                          sec_per_step=(time.time() - t0) / len(losses),
                                          mem={**_mem(), **watchdog.last})
                    vc, va = val("cap"), val("act")
                    say(f"[{stage}] ep {ep} 끝 · 학습 {np.mean(losses):.3f} · 검증 캡션 "
                        f"{vc:.3f} · 검증 행동 {va:.3f}")
                    status.update(stage=stage, epoch=ep, stage_step=step, step=total[0],
                                  val={"stage": stage, "epoch": ep,
                                       "train": float(np.mean(losses)), "caption": vc,
                                       "action": va})

            caption_from = None
            if args.resume_caption:
                src = Path(args.resume_caption)
                caption_from = {"path": str(src),
                                "sha256": hashlib.sha256(src.read_bytes()).hexdigest()}
                proj.load_weights(str(src))
                if src.resolve() != (out / "stage1-caption.safetensors").resolve():
                    shutil.copyfile(src, out / "stage1-caption.safetensors")
                run_epochs([(i, "cap") for i in range(n)], "caption", rule["caption_epochs"],
                           replay_only=True)
                say(f"[resume] 캡션 단계 = {src} · sha256 {caption_from['sha256'][:16]} · "
                    f"검증 캡션 {val('cap'):.3f} · 검증 행동 {val('act'):.3f}")
            else:
                vc, va = val("cap"), val("act")
                say(f"[before] 검증 캡션 {vc:.3f} · 검증 행동 {va:.3f}")
                status.update(before={"caption": vc, "action": va})
                run_epochs([(i, "cap") for i in range(n)], "caption", rule["caption_epochs"])
                mx.save_safetensors(str(out / "stage1-caption.safetensors"),
                                    dict(tree_flatten(proj.parameters())))
                status.update(saved="stage1-caption.safetensors")
            trainer.reset_optimizer()   # 행동 단계는 새 옵티마이저(TRAIN_RULE v2 부터)
            reh = rng.sample(range(n), int(n * rule["caption_rehearsal"]))
            run_epochs([(i, "act") for i in range(n)] + [(i, "cap") for i in reh], "action",
                       rule["action_epochs"])
            mx.save_safetensors(str(out / "final.safetensors"),
                                dict(tree_flatten(proj.parameters())))
            fp_after = qwen_fingerprint(q)
            say(f"[qwen] 가중치 지문(학습 뒤) {fp_after[:16]} · 같음 {fp_after == fp_before}")
            status.update(state="done", finished=status_now(), saved="final.safetensors",
                          fingerprint_same=fp_after == fp_before, exit_code=0)
            (out / "meta.json").write_text(json.dumps({
                "brain": args.brain, "data": src,
                "rule": rule, "caption_stage_from": caption_from, "projector": proj.meta(),
                "dims": {k: int(feats[k].shape[1]) for k in names},
                "qwen_fingerprint": {"before": fp_before, "after": fp_after,
                                     "unchanged": fp_after == fp_before},
                "llm": q.label, "versions": {c: config.load(c)["version"]
                                             for c in ("sim", "loom", "task", "channels", "llm")},
            }, ensure_ascii=False, indent=2), encoding="utf-8")
            say(f"[train] 저장 {out}")
    except GuardRefused as err:
        say(f"[guard] 막음 — {err}")
        status.update(state="refused", reason=str(err), finished=status_now(), exit_code=2)
        return 2
    except TrainingAborted:
        return 3
    finally:
        log.close()
    return 0


def probe(args: argparse.Namespace) -> int:
    """짧게 돌리며 걸음마다 메모리를 잰다 — 긴 학습 전에 평탄한지 본다(저장 없음)."""
    import mlx.core as mx

    from .qwen import QwenLM
    from .train import Trainer

    rule = dict(TRAIN_RULE)
    gcfg = config.load("guard")
    out = paths.runs_dir() / f"train-probe-{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
    rec = out.open("w", encoding="utf-8")
    try:
        with heavy("train-probe",
                   min_free_percent=int(gcfg["training_min_free_percent"])) as guard:
            print(f"[guard] 메모리 여유 {guard.free_percent}%", flush=True)
            src = data_source(args.dataset)
            feats, rows = load_split(src, "train", args.brain)
            squeeze, _ = compress_for(rule, src, args.brain)      # v4: 학습과 같은 입력
            if squeeze is not None:
                feats = squeeze(feats)
            names = list(feats)
            q = QwenLM()
            emb = q.embed(mx.arange(1000, 3000)).astype(mx.float32)
            proj = new_projector({n: feats[n].shape[1] for n in names}, q.hidden, rule,
                                 float(mx.sqrt(mx.mean(emb * emb))))
            cache = float(gcfg["training_cache_limit_gb"])
            trainer = Trainer(q, proj, lr=rule["lr"], cache_limit_gb=cache)
            setup = _Setup(q, names, rule["k"], rule.get("template_weight"))
            watchdog = Watchdog()
            rng = random.Random(1)
            print(f"[probe] {args.steps} 걸음 · 캐시 상한 {cache} GB · 기록 {out}", flush=True)
            for step in range(1, args.steps + 1):
                # 입력 길이가 번갈아 바뀐다(지난번 가장 나빴던 조건)
                task = "act" if step % 2 else "cap"
                i = rng.randrange(len(rows))
                mx.reset_peak_memory()
                t0 = time.time()
                loss = trainer.step(_sample(setup, feats, rows, i, task))
                row = {"step": step, "task": task, "sec": round(time.time() - t0, 2),
                       "loss": round(loss, 3), **_mem()}
                why = watchdog.check(step)
                if step % watchdog.every == 0:
                    row.update(watchdog.last)
                rec.write(json.dumps(row) + "\n")
                rec.flush()
                if step % watchdog.every == 0 or step <= 2:
                    print(f"[probe] {row}", flush=True)
                if why:
                    print(f"[watchdog] 멈춤 — {why}", flush=True)
                    return 3
    except GuardRefused as err:
        print(f"[guard] 막음 — {err}")
        return 2
    finally:
        rec.close()
    return 0


# ---------------------------------------------------------------------- 캡션 평가
_FIELDS = ["loom", "odor", "legs", "sound", "female"]


def caption_fields(text: str) -> dict[str, str]:
    """캡션을 칸으로.

    형식: "다가오는 물체: … / 먹이 냄새: … / 다리 접촉: N개 / 소리: … / 암컷 대용 물체: …"
    """
    parts = [p.strip() for p in text.split("/")]
    out = {f: "" for f in _FIELDS}
    for p in parts:
        key, _, val = p.partition(":")
        key, val = key.strip(), val.strip()
        if key.startswith("다가오는"):
            out["loom"] = val
        elif key.startswith("먹이"):
            out["odor"] = val
        elif key.startswith("다리"):
            out["legs"] = re.sub(r"\s", "", val)
        elif key.startswith("소리"):
            out["sound"] = val
        elif key.startswith("암컷"):
            out["female"] = val
    return out


def _sub(fields: dict[str, str]) -> dict[str, str]:
    """칸을 더 잘게: 물체 있음·쪽·거리·움직임 / 냄새 쪽·세기."""
    loom = [x.strip() for x in fields["loom"].split("·")]
    odor = [x.strip() for x in fields["odor"].split("·")]
    return {
        "loom_present": "없음" if loom[0] in ("없음", "") else "있음",
        "loom_side": loom[0] if len(loom) > 1 else "-",
        "loom_distance": loom[1] if len(loom) > 2 else "-",
        "loom_motion": loom[2] if len(loom) > 2 else "-",
        "odor_side": odor[0] if len(odor) > 1 else odor[0],
        "odor_level": odor[1] if len(odor) > 1 else "-",
        "legs": fields["legs"],
    }


def eval_caption(args: argparse.Namespace) -> int:
    import mlx.core as mx

    from .projector import ChannelProjector
    from .qwen import QwenLM
    from .train import Trainer

    pdir = projector_dir(args.brain, args.version)
    meta = json.loads((pdir / "meta.json").read_text())
    try:
        with heavy("eval-caption", extra_gb=float(config.load("llm")["weights_gb"])):
            src = meta.get("data") or data_source(args.dataset, args.val)
            vfeats, vrows = load_split(src, "eval", args.brain)
            squeeze, _ = compress_for(meta["rule"], src, args.brain)
            if squeeze is not None:                # train-v4 — 학습 때와 같은 압축
                vfeats = squeeze(vfeats)
            names = list(meta["dims"])
            q = QwenLM()
            pm = meta["projector"]
            proj = ChannelProjector(meta["dims"], hidden=q.hidden, k=pm["k"], kind=pm["kind"],
                                    width=meta["rule"]["width"], target_rms=pm["target_rms"])
            proj.load_weights(str(pdir / f"{args.checkpoint}.safetensors"))
            mx.eval(proj.parameters())
            trainer = Trainer(q, proj)
            setup = _Setup(q, names, pm["k"])
            n = len(vrows) if not args.limit else min(args.limit, len(vrows))
            truth_sub, pred_sub, examples = [], [], []
            for i in range(n):
                text = trainer.generate(_sample(setup, vfeats, vrows, i, "cap"), max_tokens=60)
                t = _sub(caption_fields(vrows[i]["caption"]))
                p = _sub(caption_fields(text))
                truth_sub.append(t)
                pred_sub.append(p)
                if i < 8:
                    examples.append({"truth": vrows[i]["caption"], "pred": text})
            keys = list(truth_sub[0])
            acc = {k: float(np.mean([p[k] == t[k] for p, t in zip(pred_sub, truth_sub)]))
                   for k in keys}
            base = {}
            for k in keys:
                major = Counter(t[k] for t in truth_sub).most_common(1)[0][0]
                base[k] = float(np.mean([t[k] == major for t in truth_sub]))
            result = {"brain": args.brain, "checkpoint": args.checkpoint, "samples": n,
                      "accuracy": acc, "majority_baseline": base, "examples": examples}
            (pdir / f"eval-caption-{args.checkpoint}.json").write_text(
                json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            print(json.dumps(result, ensure_ascii=False, indent=2))
    except GuardRefused as err:
        print(f"[guard] 막음 — {err}")
        return 2
    return 0


def eval_scored(args: argparse.Namespace) -> int:
    """채점 평가 eval-v2(빈칸 채우기 · 정답 접두 · 후보 첫 토큰 비교) — 서식과 무관하게 값을 잰다.

    한 표본 = 앞으로 한 번(정답 캡션을 그대로 넣음). 결과 `eval-scored-<저장본>.json`.
    """
    import mlx.core as mx

    from . import caption_score as cs
    from .projector import ChannelProjector
    from .qwen import QwenLM
    from .train import Trainer

    pdir = projector_dir(args.brain, args.version)
    meta = json.loads((pdir / "meta.json").read_text())
    try:
        with heavy("eval-scored", extra_gb=float(config.load("llm")["weights_gb"])):
            src = meta.get("data") or data_source(args.dataset, args.val)
            vfeats, vrows = load_split(src, "eval", args.brain)
            squeeze, _ = compress_for(meta["rule"], src, args.brain)
            if squeeze is not None:                # 학습 때와 같은 압축
                vfeats = squeeze(vfeats)
            names = list(meta["dims"])
            q = QwenLM()
            pm = meta["projector"]
            proj = ChannelProjector(meta["dims"], hidden=q.hidden, k=pm["k"], kind=pm["kind"],
                                    width=meta["rule"]["width"], target_rms=pm["target_rms"])
            proj.load_weights(str(pdir / f"{args.checkpoint}.safetensors"))
            mx.eval(proj.parameters())
            trainer = Trainer(q, proj)
            setup = _Setup(q, names, pm["k"])

            def encode(text: str) -> list[int]:
                return list(q.tok.encode(text, add_special_tokens=False))

            n = len(vrows) if not args.limit else min(args.limit, len(vrows))
            truths, preds, examples, per_sample, t0 = [], [], [], [], time.time()
            for i in range(n):
                slots = cs.slots_for(vrows[i]["caption"])
                logits = trainer.target_logits(_sample(setup, vfeats, vrows, i, "cap"))
                pred = cs.predict(slots, encode, lambda j: np.asarray(logits[j]))
                truth = {s.name: s.truth for s in slots}
                truths.append(cs.to_fields(slots, truth))
                preds.append(cs.to_fields(slots, pred))
                per_sample.append({"i": i, "seed": int(vrows[i]["seed"]),
                                   "truth": truths[-1], "pred": preds[-1]})
                if i < 8:
                    examples.append({"truth": vrows[i]["caption"], "pred": pred})
                if (i + 1) % 50 == 0:
                    print(f"[eval-scored] {i + 1}/{n} · {(time.time() - t0) / (i + 1):.2f} s/표본",
                          flush=True)
            cells = cs.score(truths, preds)
            result = {"brain": args.brain, "checkpoint": args.checkpoint, "samples": n,
                      "rule": cs.RULE, "cells": cells,
                      "accuracy": {k: v["accuracy"] for k, v in cells.items()},
                      "majority_baseline": {k: v["majority_baseline"] for k, v in cells.items()},
                      "examples": examples, "elapsed_s": round(time.time() - t0, 1),
                      "per_sample": per_sample}   # 판정(judge-captions)이 쓴다
            (pdir / f"eval-scored-{args.checkpoint}.json").write_text(
                json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            print(json.dumps({k: v for k, v in result.items()
                              if k not in ("examples", "per_sample")},
                             ensure_ascii=False, indent=2))
    except GuardRefused as err:
        print(f"[guard] 막음 — {err}")
        return 2
    return 0


def judge_captions(args: argparse.Namespace) -> int:
    """채점 평가 판정 — 진짜 대 재배선(표본별 기록 · stats · Holm). 가벼움(CPU · 파일만)."""
    from . import caption_score as cs

    brains = ["real", *[f"rewired-s{k}" for k in parse_seeds(args.rewired_seeds)]]
    per = {}
    for b in brains:
        f = projector_dir(b, args.version) / f"eval-scored-{args.checkpoint}.json"
        rec = json.loads(f.read_text(encoding="utf-8"))
        if "per_sample" not in rec:
            print(f"[judge] {f} 에 표본별 기록이 없다 — eval-scored 를 다시 돌린다")
            return 2
        per[b] = rec["per_sample"]
    out = cs.judge(per)
    out.update(version=args.version, checkpoint=args.checkpoint,
               when=time.strftime("%Y-%m-%d %H:%M:%S"))
    path = paths.runs_dir() / f"caption-judge-{args.version}-{args.checkpoint}.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    for k, c in out["cells"].items():
        print(f"[judge] {k:13s} 진짜 {c['real']:.3f} · 재배선 {c['rewired']['values']} · "
              f"찍기 {c['baseline']:.3f} · Holm {c['p_holm']:.3f} · 통과 {c['pass']}")
    print(f"[judge] 통과 {out['passed']} · 기록 {path}")
    return 0


def _brain(text: str) -> str:
    if not re.fullmatch(r"real|rewired|rewired-s\d+", text):
        raise argparse.ArgumentTypeError("real · rewired · rewired-s<K>")
    return text


def register(sub) -> None:
    p = sub.add_parser("train",
                       help="F-017 투영기 학습(Qwen 얼림 · 최대 약 32 GB · 학습 전용 문턱)")
    p.add_argument("--brain", type=_brain, required=True, help="real · rewired · rewired-s<K>")
    p.add_argument("--dataset", default="fly-loop-v2",
                   help="묶음 이름(data/datasets/) — 없으면 runs/dataset-<이름> 조각으로 읽는다")
    p.add_argument("--val", default="eval", help="조각으로 읽을 때만: 검증 조각 이름")
    p.add_argument("--resume-caption", default="",
                   help="캡션 단계 저장본(.safetensors)에서 이어 행동 단계만 돈다")
    p.add_argument("--train-seeds", default="",
                   help="학습에 쓸 에피소드 씨앗 범위(예: 0-119) — 비우면 전부")
    p.set_defaults(func=train)
    pr = sub.add_parser("train-probe", help="짧게 돌리며 걸음마다 메모리를 잰다(저장 없음)")
    pr.add_argument("--brain", type=_brain, default="real")
    pr.add_argument("--dataset", default="fly-loop-v2")
    pr.add_argument("--steps", type=int, default=50)
    pr.set_defaults(func=probe)
    e = sub.add_parser("eval-caption", help="F-017 캡션 칸별 정확도(학습에 안 쓴 씨앗)")
    e.add_argument("--brain", type=_brain, required=True)
    e.add_argument("--checkpoint", default="stage1-caption", choices=["stage1-caption", "final"])
    e.add_argument("--dataset", default="fly-loop-v2", help="묶음 이름(eval 나눔을 쓴다)")
    e.add_argument("--val", default="eval", help="조각으로 읽을 때만: 평가 조각 이름")
    e.add_argument("--version", default=None,
                   help=f"투영기 규칙 판(기본 {TRAIN_RULE['version']} · 예전 결과는 train-v2)")
    e.add_argument("--limit", type=int, default=0)
    e.set_defaults(func=eval_caption)
    es = sub.add_parser("eval-scored", help="캡션 채점 평가 eval-v2 — 빈칸 채우기(서식 무관)")
    es.add_argument("--brain", type=_brain, required=True)
    es.add_argument("--checkpoint", default="final", choices=["stage1-caption", "final"])
    es.add_argument("--dataset", default="fly-loop-v2", help="묶음 이름(eval 나눔을 쓴다)")
    es.add_argument("--val", default="eval", help="조각으로 읽을 때만: 평가 조각 이름")
    es.add_argument("--version", default=None, help="투영기 판(기본 = 지금 규칙의 판)")
    es.add_argument("--limit", type=int, default=0, help="앞에서 N 표본만(0 = 전부)")
    es.set_defaults(func=eval_scored)
    jc = sub.add_parser("judge-captions", help="채점 평가 판정 — 진짜 대 재배선(stats · Holm)")
    jc.add_argument("--version", default=None)
    jc.add_argument("--checkpoint", default="final", choices=["stage1-caption", "final"])
    jc.add_argument("--rewired-seeds", default="0-2")
    jc.set_defaults(func=judge_captions)
