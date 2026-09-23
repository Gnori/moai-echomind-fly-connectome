"""학습 자료 묶음 (R-3340 F-017) — 다음에도 쓸 수 있게 오래 둘 자리에 묶는다.

`runs/` 는 실행 기록을 쌓는 임시 자리다. 모은 자료는 `data/datasets/<이름>/` 에 학습·평가로 합쳐 묶고
자료 카드(README.md) · 메타데이터 · 체크섬을 붙인다. 형식 설명은 저장소 `docs/datasets.md`(git).

    data/datasets/<이름>/
      README.md            자료 카드 — 무엇 · 어떻게 만들었나 · 불러오는 법 · 다시 만드는 법
      meta.json            판 번호 · 코드 커밋 · 채널 단위 · 감각 대응표 · 교사 규칙 · 조각 목록
      SHA256SUMS
      train/  eval/        features-real.npz · features-rewired.npz · samples.jsonl · episodes.jsonl
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np

from . import paths

BRAINS = ("real", "rewired")


def store_dir() -> Path:
    return paths.root() / "data" / "datasets"


def shards(name: str) -> list[Path]:
    found = sorted(paths.runs_dir().glob(f"dataset-{name}-w*"))
    return found or [paths.runs_dir() / f"dataset-{name}"]


def _merge(parts: list[Path], out: Path) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    for brain in BRAINS:
        per: dict[str, list[np.ndarray]] = {}
        for p in parts:
            with np.load(p / f"features-{brain}.npz") as z:
                for ch in z.files:
                    per.setdefault(ch, []).append(z[ch])
        np.savez_compressed(out / f"features-{brain}.npz",
                            **{ch: np.concatenate(v) for ch, v in per.items()})
    rows, eps = [], []
    for p in parts:
        rows += [line for line in (p / "samples.jsonl").read_text().splitlines() if line]
        eps += [line for line in (p / "episodes.jsonl").read_text().splitlines() if line]
    (out / "samples.jsonl").write_text("\n".join(rows) + "\n", encoding="utf-8")
    (out / "episodes.jsonl").write_text("\n".join(eps) + "\n", encoding="utf-8")
    ep = [json.loads(e) for e in eps]
    seeds = sorted({json.loads(r)["seed"] for r in rows})
    return {"samples": len(rows), "episodes": len(ep), "seeds": seeds,
            "reached": sum(e["reached"] for e in ep), "looming": sum(e["looming"] for e in ep),
            "looming_hit": sum(bool(e["looming_hit"]) for e in ep),
            "shards": [p.name for p in parts]}


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _git_head() -> str:
    try:
        return subprocess.run(["git", "-C", str(paths.root()), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def pack(name: str, train: str, eval_: str, generated_with: str = "") -> Path:
    out = store_dir() / name
    if out.exists():
        raise FileExistsError(f"{out} 가 이미 있다 — 판을 올린 새 이름으로 묶는다")
    splits = {"train": shards(train), "eval": shards(eval_)}
    stats = {s: _merge(parts, out / s) for s, parts in splits.items()}
    src_meta = json.loads((splits["train"][0] / "meta.json").read_text())
    meta = {
        "name": name, "packed": time.strftime("%Y-%m-%d %H:%M:%S"), "code_commit": _git_head(),
        "generated_with_commit": generated_with or _git_head(),
        "requirement": "R-3340 F-017", "splits": stats,
        "from_runs": {s: [p.name for p in parts] for s, parts in splits.items()},
        **{k: src_meta[k] for k in ("graph", "rewired_seed", "versions", "channels",
                                     "channel_units", "sensory_map", "teacher_rules")},
    }
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "README.md").write_text(card(meta), encoding="utf-8")
    files = sorted(p for p in out.rglob("*") if p.is_file() and p.name != "SHA256SUMS")
    (out / "SHA256SUMS").write_text(
        "".join(f"{_sha(p)}  {p.relative_to(out)}\n" for p in files), encoding="utf-8")
    return out


def _features_file(d: Path, brain: str) -> Path:
    """뇌 이름 → 특징 파일. v1 의 `features-rewired.npz` 는 재배선 씨앗 0 이다(I-7469 §3.1)."""
    f = d / f"features-{brain}.npz"
    if f.exists():
        return f
    if brain == "rewired-s0" and (d / "features-rewired.npz").exists():
        return d / "features-rewired.npz"
    if brain == "rewired" and (d / "features-rewired-s0.npz").exists():
        return d / "features-rewired-s0.npz"
    return f


def rewired_brains(name: str, split: str = "eval") -> list[str]:
    """묶음에 들어 있는 재배선 뇌 이름들 — `rewired`(v1) 또는 `rewired-s<K>`."""
    d = store_dir() / name / split
    out = [p.name.removeprefix("features-").removesuffix(".npz")
           for p in sorted(d.glob("features-rewired*.npz"))]
    return sorted(out, key=lambda b: (b != "rewired", int(b.rsplit("-s", 1)[1]) if "-s" in b else 0))


def load(name: str, split: str, brain: str) -> tuple[dict[str, np.ndarray], list[dict]]:
    """묶음에서 불러온다: (채널 → [결정 수, 차원] 특징, 결정마다 참값·캡션·교사 행동).

    `brain` = `real` · `rewired`(= 씨앗 0) · `rewired-s<K>`.
    """
    d = store_dir() / name / split
    with np.load(_features_file(d, brain)) as z:
        feats = {ch: z[ch] for ch in z.files}
    rows = [json.loads(line) for line in (d / "samples.jsonl").read_text().splitlines() if line]
    return feats, rows


def verify(name: str) -> list[str]:
    """체크섬이 다른 파일 목록(비면 온전)."""
    out = store_dir() / name
    bad = []
    for line in (out / "SHA256SUMS").read_text().splitlines():
        sha, rel = line.split("  ", 1)
        if _sha(out / rel) != sha:
            bad.append(rel)
    return bad


def card(meta: dict[str, Any]) -> str:
    tr, ev = meta["splits"]["train"], meta["splits"]["eval"]
    ch = meta["channels"]
    rows = "\n".join(f"| {n} | {c['neurons']:,} | {c['units']:,} | {c['dim']:,} |" for n, c in ch.items())
    total = sum(c["dim"] for c in ch.values())
    return f"""# {meta['name']} — 초파리 닫힌 고리 학습 자료

> R-3340 F-017 · 묶은 때 {meta['packed']} · 모은 코드 커밋 `{meta['generated_with_commit']}` ·
> 묶은 코드 커밋 `{meta['code_commit']}` · 그래프 `{meta['graph']}` ·
> 섞은 배선 씨앗 {meta['rewired_seed']} · 설정 판 {meta['versions']}

## 무엇인가

교사 규칙이 FlyGym 1.x 몸을 모는 동안, **진짜 배선 뇌와 섞은 배선 뇌 둘에 같은 감각 입력**을 넣고
결정 창(300 ms)마다 남긴 것이다. 투영기 학습(초파리 신호 → Qwen 입력)의 재료다.

| 나눔 | 결정 수 | 에피소드 | 씨앗 | 교사 도착 | 다가오는 물체 | 부딪힘 |
|---|---|---|---|---|---|---|
| train | {tr['samples']} | {tr['episodes']} | {tr['seeds'][0]}~{tr['seeds'][-1]} | {tr['reached']} | {tr['looming']} | {tr['looming_hit']} |
| eval | {ev['samples']} | {ev['episodes']} | {ev['seeds'][0]}~{ev['seeds'][-1]} | {ev['reached']} | {ev['looming']} | {ev['looming_hit']} |

eval 씨앗은 학습에 쓰지 않았다.

## 파일

| 파일 | 내용 |
|---|---|
| `train/`·`eval/` `features-real.npz` | 진짜 배선 뇌의 채널 벡터 — 채널 이름 → float32 [결정 수, 차원] |
| `features-rewired.npz` | 섞은 배선 뇌(같은 감각 입력) — 같은 형식 |
| `samples.jsonl` | 결정마다 `seed` · `decision` · `t_ms` · `truth`(세계 참값) · `caption` · `teacher_action` · `reflex_bins` |
| `episodes.jsonl` | 에피소드마다 도착 여부 · 시간 · 다가오는 물체 · 부딪힘 · 벽시계 |
| `meta.json` | 판 번호 · 채널 단위 이름(`channel_units`) · 감각 대응표 · 교사 규칙 · 원래 조각 |
| `SHA256SUMS` | 체크섬 — `fly-connectome dataset verify {meta['name']}` |

## 채널 (직접 자극한 뉴런은 뺐다 — 회로를 거친 활동만)

| 채널 | 뉴런 | 단위 | 차원 |
|---|---|---|---|
{rows}

합계 {total:,} 차원. 특징 = 창 300 ms 를 3 구간으로 나눠 단위마다 `log1p(발화율)/log1p(20)` 와 `탈분극/7 mV` 를 이어 붙인 것
(`config/channels.yaml` · `loop/signals.py`). 순서는 `[발화율 구간×단위, 탈분극 구간×단위]`.

## 불러오기

```python
from moai_echomind_fly import datasets
feats, rows = datasets.load("{meta['name']}", "train", "real")   # 또는 "rewired" · "eval"
feats["smell"].shape, rows[0]["caption"], rows[0]["teacher_action"]
```

## 다시 만들기

```bash
uv run fly-connectome collect --name main --seeds 0-59 --workers 3 --threads 4
uv run fly-connectome collect --name eval --seeds 1000-1011 --workers 3 --threads 4
uv run fly-connectome dataset pack --name <새 이름> --train main --eval eval
```

같은 코드 커밋 · 같은 설정 판이면 같은 씨앗은 같은 자료를 낸다(포아송 · 과제 · 몸 모두 씨앗).

## 주의

- 정답(캡션 · 행동)은 **세계의 참값에서 코드가 만든 것**이다 — 교사 규칙은 `meta.json` 의 `teacher_rules`.
- 몸은 FlyGym 1.x(보행 제어기 · 냄새 원) · 다가오는 물체는 해석적(부딪히지 않음).
- 시각 입구는 LC/LPLC(광수용체 경로는 침묵 LIF 에서 안 퍼진다) · 청각 → 거대섬유는 이 모델에서 발화한다(생체는 문턱 아래).
- 설정 판이나 채널 표가 바뀌면 자료도 새로 모아 **새 이름**으로 묶는다(차원이 달라진다).
"""


def register(sub) -> None:
    p = sub.add_parser("dataset", help="학습 자료 묶음 — pack · verify · list")
    dsub = p.add_subparsers(dest="dataset_cmd", required=True)
    k = dsub.add_parser("pack", help="runs/ 의 조각을 data/datasets/<이름>/ 에 합쳐 묶는다(복사)")
    k.add_argument("--name", required=True)
    k.add_argument("--train", default="main")
    k.add_argument("--eval", default="eval")
    k.add_argument("--generated-with", default="", help="자료를 모은 코드 커밋(비우면 지금 HEAD)")
    k.set_defaults(func=lambda a: print(pack(a.name, a.train, a.eval, a.generated_with)) or 0)
    v = dsub.add_parser("verify", help="체크섬 대조")
    v.add_argument("name")
    v.set_defaults(func=lambda a: (print(verify(a.name) or "온전") or 0))
    ls = dsub.add_parser("list", help="묶음 목록")
    ls.set_defaults(func=lambda a: print("\n".join(
        p.name for p in sorted(store_dir().glob("*")) if p.is_dir())) or 0)

