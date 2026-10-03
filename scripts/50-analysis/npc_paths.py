"""NPC 실험 이동 그림(R-5138) — 위에서 내려다본 NPC 길 · 위험물 길 · 부딪힘 판정 원.

결정 기록(`<구성>.decisions.jsonl` 의 truth.pos_xy · 0.3 s 마다)으로 NPC 길을 그리고, 위험물 길은 같은 씨앗의
과제(make_task)로 다시 계산한다(S1: NPC 가 서 있다가 생겨날 때 자리로 곧게 옴 · 가짜 세계 기준).
사용: uv run --no-sync --with matplotlib python scripts/50-analysis/npc_paths.py <npc 결과 폴더> [씨앗 수]
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams["font.family"] = ["AppleGothic", "Apple SD Gothic Neo", "DejaVu Sans"]  # 한글
plt.rcParams["axes.unicode_minus"] = False

from moai_echomind_fly import config  # noqa: E402
from moai_echomind_fly.loop.world import make_task  # noqa: E402

COLORS = {"real": "#1f77b4", "rewired-s0": "#d62728", "rewired-s1": "#ff7f0e",
          "rewired-s2": "#9467bd", "no_brain": "#7f7f7f"}


def load(folder: Path, cfg: str) -> dict[int, list[tuple[float, float]]]:
    f = folder / f"{cfg}.decisions.jsonl"
    out: dict[int, list] = {}
    if not f.exists():
        return out
    for line in f.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        out.setdefault(r["seed"], []).append(tuple(r["truth"]["pos_xy"]))
    return out


def hits(folder: Path, cfg: str) -> dict[int, dict]:
    f = folder / f"{cfg}.jsonl"
    return ({json.loads(x)["seed"]: json.loads(x) for x in f.read_text().splitlines()}
            if f.exists() else {})


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("folder")
    ap.add_argument("n", nargs="?", type=int, default=6)
    a = ap.parse_args()
    folder = Path(a.folder)
    task_cfg = config.load("task")
    cfgs = [c for c in COLORS if (folder / f"{c}.decisions.jsonl").exists()]
    paths = {c: load(folder, c) for c in cfgs}
    res = {c: hits(folder, c) for c in cfgs}
    seeds = sorted(paths["real"])[: a.n]
    cols = 3
    rows = math.ceil(len(seeds) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows), squeeze=False)
    for ax, s in zip(axes.flat, seeds):
        lo = make_task(s, task_cfg)["looming"]
        b = math.radians(lo["bearing_deg"])
        sx, sy = lo["start_mm"] * math.cos(b), lo["start_mm"] * math.sin(b)
        ax.plot([sx, 0], [sy, 0], color="black", lw=1, ls="--", label="위험물 길(곧게)")
        ax.plot([sx], [sy], marker="o", color="black", ms=6)
        thr = lo["radius_mm"] + float(task_cfg["episode"]["hit_mm"])
        ax.add_patch(plt.Circle((0, 0), thr, fill=False, color="gray", ls=":"))
        for c in cfgs:
            p = paths[c].get(s, [])
            if not p:
                continue
            xs, ys = [0.0] + [q[0] for q in p], [0.0] + [q[1] for q in p]
            r = res[c].get(s)
            tag = "진행 중" if r is None else ("부딪힘" if r.get("looming_hit") else "피함")
            ax.plot(xs, ys, color=COLORS[c], lw=2, marker=".", label=f"{c} · {tag}")
        ax.set_aspect("equal")
        ax.set_xlim(-8, 8)
        ax.set_ylim(-8, 8)
        ax.set_title(f"씨앗 {s} · 위험 방위 {lo['bearing_deg']:+.0f}°", fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="lower left")
    for ax in list(axes.flat)[len(seeds):]:
        ax.axis("off")
    fig.suptitle(f"NPC S1 반사 — 위에서 본 길(과제 mm · 1 mm = 0.15 m) · {folder.name}\n"
                 "점선 원 = 부딪힘 판정선(반지름 1.5 + 1.0 mm) · NPC 는 서 있다가 반사 때만 옆으로",
                 fontsize=11)
    out = folder / "paths.png"
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
