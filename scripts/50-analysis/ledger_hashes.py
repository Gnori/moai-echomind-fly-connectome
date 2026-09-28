"""학습 안내서 10장 대장용 표 — 투영기 저장본 · 판정 · 성적표의 sha256(앞 16자)과 손실(가벼움).

사용: uv run python scripts/50-analysis/ledger_hashes.py train-v6
출력: 화면 + runs/scripts-ledger-<판>.md(대장 1.1 · 1.2 · 1.3 · 2.x 에 옮길 줄)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from moai_echomind_fly import paths


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("version")
    a = ap.parse_args()
    runs = paths.runs_dir()
    out = [
        f"# 대장 줄 — {a.version}",
        "",
        "| 뇌 | 학습 전 | 캡션 단계 끝 | 최종(캡션 / 행동) | stage1 · final sha256 | 지문 같음 |",
        "|---|---|---|---|---|---|",
    ]
    for d in sorted(runs.glob(f"projector-*-{a.version}")):
        log = (d / "train.log").read_text(encoding="utf-8") if (d / "train.log").exists() else ""
        before = re.findall(r"\[before\] 검증 캡션 ([\d.]+) · 검증 행동 ([\d.]+)", log)
        ends = re.findall(r"끝 · 학습 [\d.]+ · 검증 캡션 ([\d.]+) · 검증 행동 ([\d.]+)", log)
        cap_end = [
            e
            for e, line in zip(ends, re.findall(r"\[(caption|action)\] ep \d+ 끝", log))
            if line == "caption"
        ]
        same = "같음 True" in log
        sh = " · ".join(
            _sha(d / f) if (d / f).exists() else "—"
            for f in ("stage1-caption.safetensors", "final.safetensors")
        )
        brain = d.name.removeprefix("projector-").removesuffix(f"-{a.version}")
        b = "/".join(before[-1]) if before else "—"
        c = cap_end[-1][0] if cap_end else "—"
        f = "/".join(ends[-1]) if ends else "—"
        out.append(f"| {brain} | {b} | {c} | {f} | `{sh}` | {same} |")
    out += ["", "| 파일 | sha256(앞 16자) |", "|---|---|"]
    for p in sorted(runs.glob(f"caption-judge-{a.version}-*.json")):
        out.append(f"| `runs/{p.name}` | `{_sha(p)}` |")
    for s in sorted(runs.glob("scorecard-*/scorecard.json")):
        card = json.loads(s.read_text(encoding="utf-8"))
        if card.get("projector_version") == a.version:
            out.append(f"| `runs/{s.parent.name}/scorecard.json` | `{_sha(s)}` |")
    text = "\n".join(out) + "\n"
    print(text)
    (runs / f"scripts-ledger-{a.version}.md").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
