"""로컬 대시보드 dashboard-v1 (R-7928) — `runs/` 를 읽기만 해서 HTML 한 장으로.

무엇을 읽나(전부 없어도 된다 → "아직 없음"):
- `runs/.heavy.lock` · 최신 `runs/full-v4-*.log`(단계 · 끝 코드 · 기다림 줄) · `memory_pressure`
- `runs/projector-*-train-v4/train.log`(단계 · 걸음 · 손실 · s/걸음 · [mem] · 검증 손실 ·
  남은 시간 추정) · `meta.json`(Qwen 지문)
- 최신 `linear-probe-*.json`(통과 칸) · `compress-*/gate.json`(통과 채널) ·
  최신 `bump-check-*.json` · `projector-*/eval-caption-*.json`(칸별 정확도 · 찍기) ·
  최신 `scorecard-*/scorecard.json`(도착률 · `wording` 그대로)
- `data/datasets/*/meta.json` · 백업 `BACKUP.md` 의 마지막 복사 줄

읽기만 한다 — 어떤 실행도 시작하지 않고 잠금도 만지지 않는다. 표준 라이브러리만.
남은 시간 = 실측 s/걸음 × 남은 걸음.
총 걸음 = n × (caption_epochs + action_epochs) + int(n × caption_rehearsal) —
n 은 `[train] <뇌> · 학습 n` 줄, 규칙은 그 줄의 `규칙 {...}`(없으면 `train_cli.py` 소스의
`TRAIN_RULE` 을 **import 없이** 읽는다 — torch · numpy 를 끌어오지 않으려고).

가벼움의 근거(2026-09-24): 이 모듈만 올린 프로세스는 30 MB 인데, `fly-connectome` 명령으로 띄우면
cli.py 가 다른 명령의 무거운 모듈(torch · numpy · pyarrow)까지 올려 237~249 MB 가 됐다
(새지 않고 일정).
그래서 `dashboard` 는 cli.py 에서 **빠른 길**로 이 모듈의 `cli()` 만 부른다.
"""

from __future__ import annotations

import argparse
import ast
import html
import json
import re
import resource
import subprocess
import threading
import time
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable

from . import config, paths

NONE = "아직 없음"

CHANNEL_KO = {"vision": "시각", "touch": "촉각", "hearing": "청각", "smell": "후각",
              "taste": "미각", "courtship": "구애", "value": "가치", "body": "몸",
              "motor": "운동", "space": "공간", "all": "전 채널"}
FIELD_KO = {"loom_present": "물체 있음", "loom_side": "물체 쪽", "loom_distance": "물체 거리",
            "loom_motion": "물체 움직임", "odor_present": "냄새 있음", "odor_side": "냄새 쪽",
            "odor_level": "냄새 세기", "legs": "다리 접촉", "sound": "소리", "female": "암컷 대용"}

_FREE_RE = re.compile(r"free percentage:\s*(\d+)%")
_FULL_RE = re.compile(r"^=== (\S+ \S+) (.*)$")
_TRAIN_RE = re.compile(r"^\[train\] (\S+) · 학습 (\d+) · 검증 (\d+)(?: · .*?규칙 (\{.*\}))?")
_STEP_RE = re.compile(r"^\[(caption|action)\] ep (\d+) step (\d+) loss ([\d.]+) · ([\d.]+) s/걸음")
_END_RE = re.compile(r"^\[(caption|action)\] ep (\d+) 끝 · 학습 ([\d.]+) · 검증 캡션 ([\d.]+)"
                     r" · 검증 행동 ([\d.]+)")
_MEM_RE = re.compile(r"^\[mem\] (\{.*?\}) · (\{.*\})$")
_QWEN_RE = re.compile(r"^\[qwen\] 가중치 지문\(학습 뒤\) \S+ · 같음 (\w+)")


# ---------------------------------------------------------------- 읽기

def _latest(paths_: list[Path]) -> Path | None:
    return max(paths_, key=lambda p: p.stat().st_mtime) if paths_ else None


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8", errors="replace").splitlines()


def memory_free_percent() -> int | None:
    """`memory_pressure` 의 시스템 여유 % — 못 읽으면 None(맥이 아니거나 실패)."""
    try:
        out = subprocess.run(["memory_pressure"], capture_output=True, text=True,
                             timeout=20).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    m = _FREE_RE.search(out)
    return int(m.group(1)) if m else None


def read_lock(runs: Path) -> dict[str, Any]:
    path = runs / ".heavy.lock"
    if not path.exists():
        return {"held": False, "text": "비어 있음"}
    try:
        d = _load_json(path)
        started = time.strftime("%m-%d %H:%M", time.localtime(float(d.get("started", 0))))
        return {"held": True,
                "text": f"{d.get('kind', '?')} · pid {d.get('pid', '?')} · {started} 시작"}
    except (ValueError, OSError):
        return {"held": True, "text": "있음(내용을 못 읽음)"}


def read_full_jsonl(path: Path) -> dict[str, Any]:
    """전체 순서 사건 파일(`full-v*.jsonl` · R-7347) — 한 줄 = 한 사건.

    사건: begin · start · end · abort · done.
    """
    steps: list[dict[str, Any]] = []
    started = done = aborted = None
    for ln in _lines(path):
        try:
            ev = json.loads(ln)
        except ValueError:
            continue
        kind, ts = ev.get("event"), ev.get("ts")
        if kind == "begin":
            started = ts
        elif kind == "done":
            done = ts
        elif kind == "abort":
            aborted = f"{ts} 중단 · {ev.get('cmd', '')} · 끝 코드 {ev.get('code')}"
        elif kind == "start":
            steps.append({"cmd": ev.get("cmd"), "start": ts, "free": ev.get("free"),
                          "end": None, "code": None})
        elif kind == "end":
            for st in reversed(steps):
                if st["cmd"] == ev.get("cmd") and st["end"] is None:
                    st["end"], st["code"] = ts, ev.get("code")
                    break
    current = next((s for s in reversed(steps) if s["end"] is None), None)
    return {"path": path.name, "started": started, "done": done, "aborted": aborted,
            "steps": steps, "current": current, "notes": [], "source": "jsonl",
            "mtime": time.strftime("%m-%d %H:%M", time.localtime(path.stat().st_mtime))}


def read_full(runs: Path, jsonl_pattern: str, log_pattern: str) -> dict[str, Any]:
    """전체 순서 — 사건 파일(jsonl)이 로그보다 새로우면 그것, 아니면 로그 파싱(v4 까지 · 대체)."""
    j = _latest(list(runs.glob(jsonl_pattern)))
    lg = _latest(list(runs.glob(log_pattern)))
    if j is not None and (lg is None or j.stat().st_mtime >= lg.stat().st_mtime):
        return read_full_jsonl(j)
    return read_full_log(runs, log_pattern)


def read_full_log(runs: Path, pattern: str) -> dict[str, Any]:
    """최신 전체 순서 로그(대체 읽기) — 단계마다 시작 · 끝 코드, 지금 도는 단계, 기다림 줄."""
    path = _latest(list(runs.glob(pattern)))
    if path is None:
        return {"path": None}
    steps: list[dict[str, Any]] = []
    started = done = aborted = None
    last_sep = -1
    lines = _lines(path)
    for i, line in enumerate(lines):
        m = _FULL_RE.match(line)
        if not m:
            continue
        last_sep = i
        ts, rest = m.group(1), m.group(2)
        if rest.startswith("시작"):
            started = ts
        elif rest.startswith("전체 끝"):
            done = ts
        elif rest.startswith("중단"):
            aborted = f"{ts} {rest}"
        elif (s := re.match(r"여유 (\d+)% · (.*)", rest)):
            steps.append({"cmd": s.group(2), "start": ts, "free": int(s.group(1)),
                          "end": None, "code": None})
        elif (e := re.match(r"끝 코드 (\d+) · (.*)", rest)):
            for st in reversed(steps):
                if st["cmd"] == e.group(2) and st["end"] is None:
                    st["end"], st["code"] = ts, int(e.group(1))
                    break
    notes = [ln for ln in lines[last_sep + 1:] if ln.startswith("--- ")]
    current = next((s for s in reversed(steps) if s["end"] is None), None)
    return {"path": path.name, "started": started, "done": done, "aborted": aborted,
            "steps": steps, "current": current, "notes": notes[-3:], "source": "log",
            "mtime": time.strftime("%m-%d %H:%M", time.localtime(path.stat().st_mtime))}


def _train_rule(text: str | None) -> dict[str, Any] | None:
    if text:
        try:
            return ast.literal_eval(text)
        except (ValueError, SyntaxError):
            pass
    return train_rule_from_source()


def train_rule_from_source(path: Path | None = None) -> dict[str, Any] | None:
    """`loop/train_cli.py` 의 `TRAIN_RULE = {...}` 를 **import 없이**(ast) 읽는다."""
    path = path or Path(__file__).with_name("loop") / "train_cli.py"
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:
            if (isinstance(node, ast.Assign) and len(node.targets) == 1
                    and getattr(node.targets[0], "id", None) == "TRAIN_RULE"):
                rule = ast.literal_eval(node.value)
                return rule if isinstance(rule, dict) else None
    except (OSError, ValueError, SyntaxError):
        return None
    return None


def parse_train_log(lines: list[str]) -> dict[str, Any]:
    """train.log → 뇌 · n · 단계 · 걸음 · 손실 · s/걸음 · 메모리 · 검증 · 남은 시간(초)."""
    out: dict[str, Any] = {"brain": None, "n": None, "stage": None, "step": None, "loss": None,
                           "sec_per_step": None, "epoch": None, "mem": None, "val": None,
                           "before": None, "fingerprint_same": None, "saved": False,
                           "total_steps": None, "done_steps": None, "eta_s": None,
                           "state": None, "reason": None, "source": "log"}
    rule_text = None
    for ln in lines:
        if (m := _TRAIN_RE.match(ln)):
            out["brain"], out["n"] = m.group(1), int(m.group(2))
            rule_text = m.group(4)
        elif (m := _STEP_RE.match(ln)):
            out.update(stage=m.group(1), epoch=int(m.group(2)), step=int(m.group(3)),
                       loss=float(m.group(4)), sec_per_step=float(m.group(5)))
        elif (m := _END_RE.match(ln)):
            out["val"] = {"stage": m.group(1), "epoch": int(m.group(2)),
                          "train": float(m.group(3)), "caption": float(m.group(4)),
                          "action": float(m.group(5))}
        elif (m := _MEM_RE.match(ln)):
            try:
                a, b = ast.literal_eval(m.group(1)), ast.literal_eval(m.group(2))
                out["mem"] = {**a, **b}
            except (ValueError, SyntaxError):
                pass
        elif ln.startswith("[before]"):
            out["before"] = ln.removeprefix("[before] ")
        elif (m := _QWEN_RE.match(ln)):
            out["fingerprint_same"] = m.group(1) == "True"
        elif ln.startswith("[train] 저장"):
            out["saved"] = True
    rule = _train_rule(rule_text) if out["n"] else None
    if rule and out["n"]:
        n = out["n"]
        cap = n * int(rule.get("caption_epochs", 0))
        act = n * int(rule.get("action_epochs", 0))
        act += int(n * float(rule.get("caption_rehearsal", 0)))
        out["caption_steps"], out["total_steps"] = cap, cap + act
        if out["step"] is not None:
            done = out["step"] + (cap if out["stage"] == "action" else 0)
            out["done_steps"] = done
            if out["sec_per_step"]:
                out["eta_s"] = max(out["total_steps"] - done, 0) * out["sec_per_step"]
    return out


def val_history(lines: list[str]) -> list[dict[str, Any]]:
    """검증 손실 내력 — 학습 전(before) · 회전 끝마다(대조 비교 칸이 쓴다)."""
    hist: list[dict[str, Any]] = []
    for ln in lines:
        if (m := re.match(r"^\[before\] 검증 캡션 ([\d.]+) · 검증 행동 ([\d.]+)", ln)):
            hist.append({"at": "시작", "caption": float(m.group(1)), "action": float(m.group(2))})
        elif (m := _END_RE.match(ln)):
            hist.append({"at": f"{m.group(1)} {int(m.group(2)) + 1}회전 끝",
                         "caption": float(m.group(4)), "action": float(m.group(5))})
    return hist


def from_status(st: dict[str, Any]) -> dict[str, Any]:
    """status.json(학습이 직접 씀 · R-7347) → 로그 파싱과 같은 모양(그리는 쪽이 하나만 알게)."""
    step, total = st.get("step"), st.get("total_steps")
    sps = st.get("sec_per_step")
    eta = (max(total - step, 0) * sps) if (step is not None and total and sps) else None
    return {"brain": st.get("brain"), "n": st.get("train_samples"), "stage": st.get("stage"),
            "step": st.get("stage_step"), "loss": st.get("loss"), "sec_per_step": sps,
            "epoch": st.get("epoch"), "mem": st.get("mem"), "val": st.get("val"),
            "before": st.get("before"), "fingerprint_same": st.get("fingerprint_same"),
            "saved": st.get("state") == "done", "total_steps": total, "done_steps": step,
            "caption_steps": st.get("caption_steps"), "eta_s": eta,
            "state": st.get("state"), "reason": st.get("reason"), "source": "status"}


def read_projectors(runs: Path, pattern: str, eval_pattern: str,
                    stale_s: float = 3600.0) -> list[dict[str, Any]]:
    out = []
    for d in sorted(p for p in runs.glob(pattern) if p.is_dir()):
        rec: dict[str, Any] = {"dir": d.name, "log": None, "meta": None,
                               "final": (d / "final.safetensors").exists(), "evals": [],
                               "_mtime": d.stat().st_mtime}
        log, st_path = d / "train.log", d / "status.json"
        st = _load_json(st_path) if st_path.exists() else None
        if isinstance(st, dict) and st.get("state"):
            rec["log"] = from_status(st)                     # 만드는 쪽이 쓴 상태(v5 부터)
            rec["_mtime"] = st_path.stat().st_mtime
            rec["log_mtime"] = time.strftime("%m-%d %H:%M", time.localtime(rec["_mtime"]))
        elif log.exists():                                   # 옛 폴더(v1~v4) — 로그 파싱으로 대체
            rec["log"] = parse_train_log(_lines(log))
            rec["_mtime"] = log.stat().st_mtime
            rec["log_mtime"] = time.strftime("%m-%d %H:%M", time.localtime(rec["_mtime"]))
        rec["history"] = val_history(_lines(log)) if log.exists() else []
        meta = d / "meta.json"
        if meta.exists():
            try:
                m = _load_json(meta)
                rec["meta"] = {"unchanged": (m.get("qwen_fingerprint") or {}).get("unchanged"),
                               "version": (m.get("rule") or {}).get("version")}
            except (ValueError, OSError):
                rec["meta"] = {"unchanged": None, "version": None}
        for ev in sorted(d.glob(eval_pattern)):
            try:
                e = _load_json(ev)
                rec["evals"].append({"file": ev.name, "checkpoint": e.get("checkpoint"),
                                     "accuracy": e.get("accuracy", {}),
                                     "baseline": e.get("majority_baseline", {})})
            except (ValueError, OSError):
                rec["evals"].append({"file": ev.name, "checkpoint": None, "accuracy": {},
                                     "baseline": {}})
        out.append(rec)
    # 끝나지 않았는데 오래 갱신이 없으면 = 멈춘 기록(도는 중 아님)
    for rec in out:
        rec["stale"] = (not rec["final"] and rec["log"] is not None
                        and not rec["log"].get("saved")
                        and rec["log"].get("state") not in ("aborted", "refused")
                        and time.time() - rec["_mtime"] > stale_s)
    # 최근에 움직인 투영기가 위(요구자 2026-09-24 "투영기 날짜로 정렬") — 판이 섞여도 도는 것이 먼저
    out.sort(key=lambda r: r["_mtime"], reverse=True)
    return out


def read_linear_probe(runs: Path, pattern: str) -> dict[str, Any]:
    path = _latest(list(runs.glob(pattern)))
    if path is None:
        return {"path": None}
    try:
        r = _load_json(path)
    except (ValueError, OSError):
        return {"path": path.name, "error": "읽기 실패"}
    cells = r.get("cells") or {}
    passed = [(k, c) for k, c in cells.items() if c.get("pass")]
    judged = sum(1 for c in cells.values() if c.get("judge"))
    return {"path": path.name, "dataset": r.get("dataset"),
            "stats": (r.get("stats") or {}).get("version"),
            "rewired": len(r.get("rewired_brains") or []), "judged": judged, "passed": passed,
            "when": r.get("when")}


def read_gates(runs: Path, pattern: str) -> list[dict[str, Any]]:
    out = []
    for d in sorted(p for p in runs.glob(pattern) if p.is_dir()):
        g = d / "gate.json"
        if not g.exists():
            out.append({"dir": d.name, "rule": None, "passed": None, "cells": None})
            continue
        try:
            j = _load_json(g)
            out.append({"dir": d.name, "rule": j.get("rule"),
                        "passed": j.get("passed_channels") or [],
                        "cells": j.get("passed_cells") or [], "when": j.get("when")})
        except (ValueError, OSError):
            out.append({"dir": d.name, "rule": None, "passed": None, "cells": None})
    return out


def read_bump(runs: Path, pattern: str) -> dict[str, Any]:
    path = _latest(list(runs.glob(pattern)))
    if path is None:
        return {"path": None}
    try:
        b = _load_json(path)
    except (ValueError, OSError):
        return {"path": path.name, "error": "읽기 실패"}
    return {"path": path.name, "dataset": b.get("dataset"), "epg_units": b.get("epg_units"),
            "brains": b.get("brains") or {}, "real_rank_pct": b.get("real_rank_pct"),
            "when": b.get("when")}


def read_scorecard(runs: Path, pattern: str) -> dict[str, Any]:
    files = [p / "scorecard.json" for p in runs.glob(pattern) if (p / "scorecard.json").exists()]
    path = _latest(files)
    if path is None:
        return {"path": None}
    try:
        s = _load_json(path)
    except (ValueError, OSError):
        return {"path": str(path.parent.name), "error": "읽기 실패"}
    return {"path": path.parent.name, "configs": s.get("configs") or {},
            "seeds": len(s.get("seeds") or []), "beats": s.get("vector_beats_rewired"),
            "wording": s.get("wording"), "judge": s.get("judge")}


def read_datasets(datasets_dir: Path, pattern: str) -> list[dict[str, Any]]:
    out = []
    for m in sorted(datasets_dir.glob(f"{pattern}/meta.json")):
        try:
            j = _load_json(m)
            splits = {k: (v.get("samples"), v.get("episodes"))
                      for k, v in (j.get("splits") or {}).items()}
            out.append({"name": j.get("name", m.parent.name), "splits": splits,
                        "versions": j.get("versions") or {}, "packed": j.get("packed")})
        except (ValueError, OSError):
            out.append({"name": m.parent.name, "splits": {}, "versions": {}, "packed": "읽기 실패"})
    return out


def read_backup(path: Path) -> str | None:
    if not path.exists():
        return None
    for ln in _lines(path):
        if ln.startswith("- 마지막 복사"):
            return ln.removeprefix("- ").strip()
    return "복사 줄 없음"


def collect(runs: Path | None = None, datasets_dir: Path | None = None,
            backup_md: Path | None = None, cfg: dict[str, Any] | None = None,
            memory_free: Callable[[], int | None] = memory_free_percent) -> dict[str, Any]:
    """읽을 것을 전부 읽어 한 덩어리로(그리기와 분리 — 시험은 이 덩어리를 본다)."""
    cfg = cfg or config.load("dashboard")
    runs = runs or paths.runs_dir()
    datasets_dir = datasets_dir or paths.root() / "data" / "datasets"
    backup_md = backup_md or Path(cfg["backup_md"]).expanduser()
    g = cfg["globs"]
    full = read_full(runs, g["full_jsonl"], g["full_log"])
    return {
        "version": cfg["version"], "now": time.strftime("%Y-%m-%d %H:%M:%S"), "runs": str(runs),
        "lock": read_lock(runs), "full": full,
        "free": memory_free(),
        "projectors": read_projectors(runs, g["projector"], g["eval_caption"],
                                      60.0 * float(cfg.get("stale_minutes", 60))),
        "probe": read_linear_probe(runs, g["linear_probe"]),
        "gates": read_gates(runs, g["compress"]),
        "bump": read_bump(runs, g["bump"]), "scorecard": read_scorecard(runs, g["scorecard"]),
        "datasets": read_datasets(datasets_dir, cfg["datasets_glob"]),
        "backup": read_backup(backup_md),
        "links": cfg.get("links") or [], "refresh_s": int(cfg["refresh_s"]),
        "train_rule": cfg.get("train_rule"),
        "plan": plan_status(cfg.get("plan") or [], _same_run_steps(runs, full)),
        "caption_judge": _load_json(runs / f"caption-judge-{cfg.get('train_rule')}-final.json")
        if (runs / f"caption-judge-{cfg.get('train_rule')}-final.json").exists() else None,
    }


# ---------------------------------------------------------------- 그리기

CSS = """
:root{--ground:#F1F4F5;--panel:#FAFBFB;--plate:#E7ECEE;--ink:#12171A;--ink-2:#414A4F;
 --ink-3:#6B757A;--rule:#D3DADD;--a:#0E7C86;--a-soft:#DCEFF1;--b:#AB2A79;--b-soft:#F7E3EF;
 --ok:#2E7D4F;--ok-soft:#DFF2E6;--warn:#9A5B00;--warn-soft:#FBEED6;--code-bg:#E9EEF0;
 --f-display:"Gowun Batang","AppleMyungjo","Nanum Myeongjo",serif;
 --f-body:"Gothic A1","Apple SD Gothic Neo","Malgun Gothic",system-ui,sans-serif;
 --f-mono:"JetBrains Mono","Gothic A1",ui-monospace,Menlo,monospace}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--ground:#0E1214;
 --panel:#161B1E;--plate:#1C2327;--ink:#E6EBEC;--ink-2:#B2BCC0;--ink-3:#859096;--rule:#263036;
 --a:#4ECDD6;--a-soft:#102E33;--b:#F07AC0;--b-soft:#33122A;--ok:#7BD39A;--ok-soft:#12321F;
 --warn:#F0B860;--warn-soft:#3A2A0C;--code-bg:#171E21}}
:root[data-theme="dark"]{--ground:#0E1214;--panel:#161B1E;--plate:#1C2327;--ink:#E6EBEC;
 --ink-2:#B2BCC0;--ink-3:#859096;--rule:#263036;--a:#4ECDD6;--a-soft:#102E33;--b:#F07AC0;
 --b-soft:#33122A;--ok:#7BD39A;--ok-soft:#12321F;--warn:#F0B860;--warn-soft:#3A2A0C;
 --code-bg:#171E21}
*,*::before,*::after{box-sizing:border-box}
body{margin:0;background:var(--ground);color:var(--ink);font-family:var(--f-body);
 font-size:.97rem;line-height:1.65;word-break:keep-all;padding:28px clamp(16px,4vw,56px) 72px}
h1{font-family:var(--f-display);font-size:1.7rem;margin:0 0 4px}
h2{font-family:var(--f-display);font-size:1.25rem;margin:36px 0 10px;padding-top:14px;
 border-top:1px solid var(--rule)}
h3{font-size:1rem;margin:20px 0 6px}
.sub{font-family:var(--f-mono);font-size:.74rem;color:var(--ink-3);margin:0 0 18px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:12px}
.card{background:var(--panel);border:1px solid var(--rule);border-radius:4px;padding:12px 16px}
.card .k{font-family:var(--f-mono);font-size:.7rem;letter-spacing:.08em;color:var(--ink-3)}
.card .v{font-size:1.15rem;font-weight:700;margin-top:2px}
.tbl{overflow-x:auto;margin:10px 0 16px;border:1px solid var(--rule);border-radius:3px;
 background:var(--panel)}
table{border-collapse:collapse;width:100%;font-size:.88rem}
th,td{text-align:left;vertical-align:top;padding:7px 12px;border-bottom:1px solid var(--rule)}
tr:last-child td{border-bottom:0}
th{font-family:var(--f-mono);font-weight:400;font-size:.7rem;letter-spacing:.08em;
 color:var(--ink-3);white-space:nowrap;background:var(--plate)}
code{font-family:var(--f-mono);font-size:.84em;background:var(--code-bg);padding:.05em .3em;
 border-radius:2px}
.ok{color:var(--ok);background:var(--ok-soft);padding:.05em .45em;border-radius:3px;font-weight:700}
.warn{color:var(--warn);background:var(--warn-soft);padding:.05em .45em;border-radius:3px;
 font-weight:700}
.bad{color:var(--b);background:var(--b-soft);padding:.05em .45em;border-radius:3px;font-weight:700}
.muted{color:var(--ink-3)}
.wording{border-left:3px solid var(--b);background:var(--b-soft);padding:10px 14px;margin:8px 0;
 font-weight:700}
a{color:var(--a)}
footer{margin-top:36px;font-size:.78rem;color:var(--ink-3)}
"""


def _e(x: Any) -> str:
    return html.escape("" if x is None else str(x))


def _f(x: Any, nd: int = 3) -> str:
    return "—" if x is None else f"{float(x):.{nd}f}"


def _cell_name(key: str) -> str:
    ch, _, field = key.partition("|")
    return f"{CHANNEL_KO.get(ch, ch)} → {FIELD_KO.get(field, field)}"


def _dur(seconds: float | None) -> str:
    if seconds is None:
        return "—"
    s = int(seconds)
    h, m = s // 3600, (s % 3600) // 60
    return f"{h} 시간 {m} 분" if h else f"{m} 분"


def _tag(cls: str, text: str) -> str:
    return f'<span class="{cls}">{text}</span>'


def _table(head: list[str], rows: list[list[str]]) -> str:
    if not rows:
        return f'<p class="muted">{NONE}</p>'
    th = "".join(f"<th>{_e(h)}</th>" for h in head)
    tr = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<div class="tbl"><table><thead><tr>{th}</tr></thead><tbody>{tr}</tbody></table></div>'


def _guard_text() -> str:
    """가드 문턱을 설정(`config/guard.yaml`)에서 읽어 적는다 — 코드에 박지 않는다."""
    try:
        from . import config

        g = config.load("guard")
        return (f"학습 문턱 {g['training_min_free_percent']} % · 평소 {g['min_free_percent']} % · "
                f"감시: 여유 < {g['training_abort_free_percent']} % 또는 "
                f"프로세스 > {g['training_abort_footprint_gb']} GB 면 저장하고 멈춤")
    except Exception:                  # noqa: BLE001 — 대시보드는 읽기 실패로 죽지 않는다
        return "가드 설정을 읽지 못함"


def _card(key: str, value: str, sub: str) -> str:
    return (f'<div class="card"><div class="k">{_e(key)}</div><div class="v">{value}</div>'
            f'<div class="muted">{sub}</div></div>')


def _full_summary(full: dict[str, Any]) -> tuple[str, str]:
    if full.get("path") is None:
        return NONE, ""
    sub = _e(full["path"])
    if full.get("done"):
        return f'{_tag("ok", "전체 끝")} {_e(full["done"])}', sub
    if full.get("aborted"):
        return _tag("bad", _e(full["aborted"])), sub
    if full.get("current"):
        c = full["current"]
        txt = f'{_tag("warn", "도는 중")} <code>{_e(c["cmd"])}</code> · {_e(c["start"])} 시작'
        return txt, sub
    return f"단계 사이 · 마지막 갱신 {_e(full.get('mtime'))}", sub


def _section_status(d: dict[str, Any]) -> str:
    lock, free, full = d["lock"], d["free"], d["full"]
    free_txt = "알 수 없음" if free is None else f"{free} %"
    if free is None:
        free_cls = "muted"
    else:
        free_cls = "ok" if free >= 40 else "warn" if free >= 25 else "bad"
    full_txt, full_sub = _full_summary(full)
    out = '<div class="grid">' + "".join([
        _card("무거운 실행 잠금", _tag("warn" if lock["held"] else "ok", _e(lock["text"])), ""),
        _card("전체 순서(run_full)", full_txt, full_sub),
        _card("시스템 메모리 여유", _tag(free_cls, free_txt), _guard_text()),
        _card("마지막 갱신", _e(d["now"]), f'{d["refresh_s"]} 초마다 새로 읽음'),
    ]) + "</div>"
    if full.get("path"):
        rows = []
        for st in full["steps"]:
            code = st["code"]
            state = (_tag("warn", "도는 중") if st["end"] is None
                     else _tag("ok" if code == 0 else "bad", f"끝 코드 {code}"))
            rows.append([f"<code>{_e(st['cmd'])}</code>", _e(st["start"]),
                         _e(st["end"] or "—"), state])
        out += "<h3>단계</h3>" + _table(["명령", "시작", "끝", "상태"], rows)
        if full.get("notes"):
            out += "<p>" + "<br>".join(f"<code>{_e(n)}</code>" for n in full["notes"]) + "</p>"
    return out


def _train_row(p: dict[str, Any]) -> list[str]:
    lg = p["log"]
    name = f"<code>{_e(p['dir'])}</code><br><span class=\"muted\">{_e(p.get('log_mtime'))}</span>"
    if not lg:
        return [name, "로그 없음"] + ["—"] * 7
    finished = p["final"] or lg["saved"]
    if lg.get("state") in ("aborted", "refused"):
        what = "멈춤" if lg["state"] == "aborted" else "막음"
        state = _tag("bad", f'{what} · {_e(lg.get("reason"))}')
    elif p["final"]:
        state = _tag("ok", "끝(final)")
    elif lg["saved"]:
        state = _tag("ok", "저장됨")
    elif p.get("stale"):
        state = _tag("bad", f'멈춘 기록 · {_e(lg["stage"])} ep {lg["epoch"]}에서 끊김')
    elif lg["step"] is not None:
        state = _tag("warn", f'{_e(lg["stage"])} ep {lg["epoch"]}')
    else:
        state = "시작 중"
    if lg["done_steps"] is None:
        prog = "—"
    else:
        pct = 100 * lg["done_steps"] / max(lg["total_steps"], 1)
        prog = f'{lg["done_steps"]:,} / {lg["total_steps"]:,} ({pct:.0f} %)'
    mem = lg["mem"] or {}
    mem_txt = ("—" if not mem else
               f'최대 {_f(mem.get("peak_gb"), 1)} GB · 캐시 {_f(mem.get("cache_gb"), 2)} GB'
               f' · 여유 {_e(mem.get("sys_free_percent"))} %')
    val = lg["val"]
    val_txt = ("—" if not val else
               f'{_e(val["stage"])} ep {val["epoch"]} 끝 · 캡션 {_f(val["caption"])}'
               f' · 행동 {_f(val["action"])}')
    fp = (p["meta"] or {}).get("unchanged")
    if fp is None:
        fp = lg["fingerprint_same"]
    fp_txt = "—" if fp is None else (_tag("ok", "같음") if fp else _tag("bad", "다름!"))
    return [name, state, prog, _f(lg["loss"]), _f(lg["sec_per_step"], 1),
            "끝" if finished else "—" if p.get("stale") else _dur(lg["eta_s"]),
            mem_txt, val_txt, fp_txt]


BRAIN_KO = {"real": "진짜 배선 — 실제 수컷 초파리 연결 지도 그대로",
            "rewired": "재배선 — 같은 뉴런 · 같은 연결 수, 연결 상대만 무작위로 섞음"}


def _brain_of(dirname: str) -> str:
    m = re.match(r"projector-(.+)-train-v\d+", dirname)
    return m.group(1) if m else dirname


def _section_compare(d: dict[str, Any]) -> str:
    """진짜 대 재배선 — 지금 판(train_rule)의 투영기끼리 같은 칸을 나란히."""
    ver = d.get("train_rule")
    ps = [p for p in d["projectors"] if ver and p["dir"].endswith(f"-{ver}")]
    if not ps:
        return f'<p class="muted">{NONE}</p>'
    ps.sort(key=lambda p: (0 if _brain_of(p["dir"]) == "real" else 1, p["dir"]))
    intro = (f'<p>판 <code>{_e(ver)}</code> — <b>네 뇌 모두 같은 자료 · 같은 정답 · 같은 규칙 · '
             '같은 초기값 · 같은 걸음 순서</b>로 학습하고, '
             '<b>다른 것은 신호를 만든 뇌의 배선 하나</b>다. '
             '진짜 배선이 재배선 셋보다 확실히 앞서야 "초파리 뇌 덕"이라 말한다(R 2.0-6). '
             '손실은 참고 · 판정은 채점 평가 · 성적표.</p>')
    ats = []
    for p in ps:
        for h in p.get("history") or []:
            if h["at"] not in ats:
                ats.append(h["at"])
    rows, rew = [], {a: [] for a in ats}
    for p in ps:
        b = _brain_of(p["dir"])
        kind = "real" if b == "real" else "rewired"
        hist = {h["at"]: h for h in p.get("history") or []}
        cells = []
        for a in ats:
            h = hist.get(a)
            cells.append("—" if not h else f'{_f(h["caption"])} · {_f(h["action"])}')
            if h and kind == "rewired":
                rew[a].append(h)
        lg = p.get("log") or {}
        state = ("끝" if p["final"] else "멈춘 기록" if p.get("stale")
                 else f'도는 중 {lg.get("done_steps") or 0:,}/{lg.get("total_steps") or 0:,}')
        rows.append([f"<b>{_e(b)}</b><br><span class=\"muted\">{_e(BRAIN_KO[kind])}</span>",
                     state] + cells)
    real = next((p for p in ps if _brain_of(p["dir"]) == "real"), None)
    rhist = {h["at"]: h for h in (real or {}).get("history") or []}
    mean_cells, diff_cells = [], []
    for a in ats:
        hs = rew[a]
        if not hs:
            mean_cells.append("—")
            diff_cells.append("—")
            continue
        mc = sum(h["caption"] for h in hs) / len(hs)
        ma = sum(h["action"] for h in hs) / len(hs)
        mean_cells.append(f"{_f(mc)} · {_f(ma)} <span class=\"muted\">({len(hs)} 뇌)</span>")
        r = rhist.get(a)
        if r:
            dc = r["caption"] - mc
            cls = "ok" if dc < -0.05 else "bad" if dc > 0.05 else "warn"
            diff_cells.append(_tag(cls, f"{dc:+.3f}") + f' · {r["action"] - ma:+.3f}')
        else:
            diff_cells.append("—")
    rows.append(["재배선 평균", ""] + mean_cells)
    rows.append(["진짜 − 재배선 평균<br><span class=\"muted\">음수 = 진짜가 낮음(좋음)</span>", ""]
                + diff_cells)
    head = ["뇌", "상태"] + [f"{a} (캡션 · 행동)" for a in ats]
    return (intro + "<h3>검증 손실(같은 24 표본)</h3>" + _table(head, rows) + _compare_scored(ps)
            + _compare_judge(d.get("caption_judge")))


def _compare_judge(j: dict[str, Any] | None) -> str:
    """판정(judge-captions · stats-v2 · Holm) — 판정 칸 넷."""
    if not j:
        return ('<h3>판정(에피소드 단위 검정 · Holm)</h3>'
                '<p class="muted">아직 없음 — 채점 평가 8 뒤 '
                '<code>fly-connectome judge-captions --version train-v5</code></p>')
    rows = []
    for k, c in j.get("cells", {}).items():
        ok = c.get("pass")
        rows.append([FIELD_KO.get(k, k), _f(c["real"]),
                     " · ".join(_f(v) for v in c["rewired"]["values"]), _f(c["baseline"]),
                     f'{c["episodes"]} / {c["decisions"]}', _f(c.get("p_holm")),
                     _tag("ok" if ok else "bad", "통과" if ok else "불통")])
    return ("<h3>판정(에피소드 단위 검정 · Holm · stats-v2)</h3>"
            + _table(["칸", "진짜", "재배선 s0 · s1 · s2", "찍기", "에피소드 / 표본", "Holm p",
                      "판정"], rows))


def _compare_scored(ps: list[dict[str, Any]]) -> str:
    """채점 평가(eval-scored) — 칸마다 진짜 · 재배선 평균 · 찍기(저장본 final)."""
    per: dict[str, dict[str, Any]] = {}
    base: dict[str, float] = {}
    for p in ps:
        for e in p["evals"]:
            if e["file"] == "eval-scored-final.json":
                per[_brain_of(p["dir"])] = e["accuracy"]
                base.update({k: v for k, v in e["baseline"].items() if v is not None})
    if "real" not in per:
        return '<h3>채점 평가(최종본)</h3><p class="muted">아직 없음 — 학습 4 개가 끝난 뒤</p>'
    rews = [a for b, a in per.items() if b != "real"]
    rows = []
    for f in FIELD_KO:
        if f not in per["real"]:
            continue
        r = per["real"][f]
        vals = [a[f] for a in rews if f in a]
        m = sum(vals) / len(vals) if vals else None
        win = m is not None and r > max(vals) and r > (base.get(f) or 0) + 0.05
        rows.append([FIELD_KO[f], _tag("ok" if win else "", _f(r)),
                     "—" if m is None else f"{_f(m)} ({' · '.join(_f(v) for v in vals)})",
                     _f(base.get(f))])
    return ("<h3>채점 평가(최종본) — 진짜가 재배선 셋 모두 · 찍기보다 높으면 초록</h3>"
            + _table(["칸", "진짜", "재배선 평균(각각)", "찍기"], rows))


def _same_run_steps(runs: Path, full: dict[str, Any]) -> dict[str, Any]:
    """같은 판(full-v5-*) 사건 파일을 모두 합친 단계 — 중단 뒤 다시 띄워도 계획이 이어진다."""
    name = full.get("path") or ""
    m = re.match(r"(full-v\d+)-", name)
    if not m or not list(runs.glob(f"{m.group(1)}-*.jsonl")):   # 사건 파일이 없는 옛 판(v4)
        return full
    steps: list[dict[str, Any]] = []
    for f in sorted(runs.glob(f"{m.group(1)}-*.jsonl")):
        steps += read_full_jsonl(f)["steps"]
    return {**full, "steps": steps}


def plan_status(plan: list[dict[str, Any]], full: dict[str, Any]) -> list[dict[str, Any]]:
    """계획(설정 파일) × 전체 순서 사건 → 단계마다 완료 · 진행 중 · 남음 · 실패."""
    steps = full.get("steps") or []
    out = []
    for item in plan:
        hit = [s for s in steps if (s.get("cmd") or "").startswith(item["match"])]
        n = int(item.get("count", 1))
        done = [s for s in hit if s["end"] is not None and s["code"] == 0]
        bad = [s for s in hit if s["end"] is not None and s["code"] != 0]
        running = [s for s in hit if s["end"] is None]
        if bad and len(done) < n and not running:
            state = "실패"
        elif len(done) >= n:
            state = "완료"
        elif running or done:
            state = "진행 중"
        else:
            state = "남음"
        out.append({**item, "state": state, "done": len(done), "count": n,
                    "start": hit[0]["start"] if hit else None,
                    "end": done[-1]["end"] if state == "완료" else None})
    return out


def _section_plan(d: dict[str, Any]) -> str:
    items = d.get("plan") or []
    if not items:
        return f'<p class="muted">{NONE}</p>'
    tag = {"완료": "ok", "진행 중": "warn", "실패": "bad", "남음": ""}
    rows = []
    for it in items:
        cnt = f' {it["done"]}/{it["count"]}' if it["count"] > 1 else ""
        rows.append([f'<b>{_e(it["name"])}</b>', _tag(tag[it["state"]], it["state"] + cnt),
                     _e(it["start"] or "—"), _e(it["end"] or "—"), _e(it["content"])])
    n = {k: sum(1 for it in items if it["state"] == k) for k in ("완료", "진행 중", "남음", "실패")}
    head = (f'<p>완료 <b>{n["완료"]}</b> · 진행 중 <b>{n["진행 중"]}</b> · 남음 <b>{n["남음"]}</b>'
            + (f' · 실패 <b>{n["실패"]}</b>' if n["실패"] else "") + "</p>")
    return head + _table(["단계", "상태", "시작", "끝", "무엇을 하나(학습 내용)"], rows)


def _section_train(d: dict[str, Any]) -> str:
    head = ["투영기", "단계", "걸음", "손실", "s/걸음", "남은 시간(추정)", "메모리", "검증 손실",
            "Qwen 지문"]
    out = _table(head, [_train_row(p) for p in d["projectors"]])
    evals = [(p["dir"], e) for p in d["projectors"] for e in p["evals"]]
    if not evals:
        return out + f'<h3>캡션 평가</h3><p class="muted">{NONE}</p>'
    # 채점(eval-scored · v2 · 서식 무관)을 먼저, 자유 생성(eval-caption · v1 · 서식을 잰다)은 뒤에
    evals.sort(key=lambda de: 0 if de[1]["file"].startswith("eval-scored") else 1)
    fields = [f for f in FIELD_KO if any(f in e["accuracy"] for _, e in evals)]
    erows = []
    for dname, e in evals:
        how = "채점(v2)" if e["file"].startswith("eval-scored") else "생성(v1)"
        cells = []
        for f in fields:
            a, b = e["accuracy"].get(f), e["baseline"].get(f)
            if a is None:
                cells.append("—")
                continue
            cls = "ok" if (b is not None and a > b + 0.05) else ""
            cells.append(f'{_tag(cls, _f(a))} <span class="muted">/ {_f(b)}</span>')
        erows.append([f"<code>{_e(dname)}</code>", _e(e["checkpoint"]), how] + cells)
    title = "캡션 평가 (정확도 / 찍기 — 채점 v2 는 해당 표본만 · 생성 v1 은 서식을 잰다)"
    return (out + f"<h3>{title}</h3>"
            + _table(["투영기", "저장본", "방식"] + [FIELD_KO[f] for f in fields], erows))


def _section_probe(pr: dict[str, Any]) -> str:
    if pr.get("path") is None:
        return f'<h3>선형 검사 판정</h3><p class="muted">{NONE}</p>'
    rows = [[_e(_cell_name(k)), f"<b>{_f(c.get('real'))}</b>",
             _f((c.get("rewired") or {}).get("mean")), _f(c.get("baseline")),
             _f(c.get("p_holm"))] for k, c in pr["passed"]]
    return (f'<h3>선형 검사 판정 — <code>{_e(pr["path"])}</code></h3>'
            f'<p class="muted">{_e(pr.get("dataset"))} · {_e(pr.get("stats"))} · '
            f'재배선 {pr.get("rewired")} · 판정 칸 {pr.get("judged")} · '
            f'<b>통과 {len(pr["passed"])}</b></p>'
            + _table(["채널 → 칸", "진짜", "재배선 평균", "찍기", "Holm p"], rows))


def _section_gates(gates: list[dict[str, Any]]) -> str:
    if not gates:
        return f'<h3>압축 관문</h3><p class="muted">{NONE}</p>'
    rows = []
    for g in gates:
        if g["passed"] is None:
            passed = "gate.json 없음"
        else:
            passed = ", ".join(CHANNEL_KO.get(c, c) for c in g["passed"]) or _tag("bad", "없음")
        rows.append([f"<code>{_e(g['dir'])}</code>", _e(g.get("rule") or "—"), passed,
                     "—" if g["cells"] is None else str(len(g["cells"]))])
    return "<h3>압축 관문</h3>" + _table(["폴더", "판", "통과 채널", "통과 칸"], rows)


def _section_bump(b: dict[str, Any]) -> str:
    if b.get("path") is None:
        return f'<h3>나침반 봉우리</h3><p class="muted">{NONE}</p>'
    rows = []
    for name, v in (b.get("brains") or {}).items():
        live = v.get("live")
        rows.append([_e(name), _e(v.get("decisions")), _tag("bad" if not live else "", _e(live)),
                     _f(v.get("bump_length_median")), _f(v.get("circ_corr_abs")),
                     _f(v.get("p_perm"), 2)])
    return (f'<h3>나침반 봉우리 — <code>{_e(b["path"])}</code></h3>'
            f'<p class="muted">{_e(b.get("dataset"))} · EPG 단위 {_e(b.get("epg_units"))} · '
            f'진짜 순위 {_e(b.get("real_rank_pct"))} %</p>'
            + _table(["뇌", "결정", "EPG 발화 있는 결정", "봉우리 길이 중앙", "|원형 상관|", "p"],
                     rows))


def _section_scorecard(s: dict[str, Any]) -> str:
    if s.get("path") is None:
        return f'<h3>성적표</h3><p class="muted">{NONE}</p>'
    rows = [[_e(name), _f(c.get("reach_rate"), 2), _f(c.get("final_food_mm_mean"), 1),
             _f(c.get("avoid_rate"), 2), _e(c.get("schema_violations"))]
            for name, c in (s.get("configs") or {}).items()]
    beats = s.get("beats")
    if beats is None:
        beats_txt = "—"
    elif beats:
        beats_txt = _tag("ok", "vector 가 재배선을 앞섰다")
    else:
        beats_txt = _tag("bad", "vector 가 재배선을 못 이겼다")
    judge = s.get("judge") or {}
    judge_txt = (f' · 규칙 {_e(judge.get("rule"))} · 순위 {_e(judge.get("rank_pct"))} %'
                 f' · p {_f(judge.get("p_sign_flip"))}' if judge else "")
    wording = f'<div class="wording">{_e(s["wording"])}</div>' if s.get("wording") else ""
    return (f'<h3>성적표 — <code>{_e(s["path"])}</code></h3>'
            f'<p class="muted">씨앗 {s.get("seeds")} · {beats_txt}{judge_txt}</p>'
            + _table(["구성", "먹이 도착률", "끝 거리 mm", "회피율", "형식 위반"], rows) + wording)


def _section_data(d: dict[str, Any]) -> str:
    rows = []
    for ds in d["datasets"]:
        sp = " · ".join(f"{k} {n:,} 결정 / {ep} 에피소드" if isinstance(n, int) else f"{k} —"
                        for k, (n, ep) in ds["splits"].items())
        rows.append([f"<code>{_e(ds['name'])}</code>", sp,
                     " · ".join(_e(v) for v in ds["versions"].values()), _e(ds.get("packed"))])
    bk = d["backup"]
    backup = f'<p>백업: {_e(bk) if bk else NONE}</p>'
    return _table(["묶음", "나눔", "판", "묶은 때"], rows) + backup


def render(d: dict[str, Any]) -> str:
    links = " · ".join(f'<a href="{_e(u)}">{_e(t)}</a>' for t, u in d["links"])
    results = (_section_probe(d["probe"]) + _section_gates(d["gates"])
               + _section_bump(d["bump"]) + _section_scorecard(d["scorecard"]))
    return f"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="refresh" content="{d['refresh_s']}">
<title>초파리 실행 대시보드</title>
<style>{CSS}</style>
</head>
<body>
<h1>초파리 실행 대시보드</h1>
<p class="sub">{_e(d['version'])} · {_e(d['runs'])} · 읽기 전용 · 자료 <code>state.json</code>
 · {links}</p>
<h2>지금 상태</h2>
{_section_status(d)}
<h2>순서 — 완료 · 진행 중 · 남음</h2>
{_section_plan(d)}
<h2>진짜 대 재배선 (지금 판)</h2>
{_section_compare(d)}
<h2>학습 진행 (투영기 · 최근 순)</h2>
{_section_train(d)}
<h2>결과</h2>
{results}
<h2>자료 · 백업</h2>
{_section_data(d)}
<footer>runs/ 를 읽기만 한다 — 실행 · 잠금을 만지지 않는다. R-7928 · 만든 때 {_e(d['now'])}</footer>
</body>
</html>
"""


def build(runs: Path | None = None, datasets_dir: Path | None = None,
          backup_md: Path | None = None, cfg: dict[str, Any] | None = None,
          memory_free: Callable[[], int | None] = memory_free_percent) -> Path:
    """페이지를 만들어 `runs/<out_dir>/index.html` 에 쓴다(폴더가 없으면 만든다)."""
    cfg = cfg or config.load("dashboard")
    runs = runs or paths.runs_dir()
    out = runs / cfg["out_dir"] / "index.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    state = collect(runs, datasets_dir, backup_md, cfg, memory_free)
    # 자료 파일(R-7347 · 요구자 "데이터 파일로 가지고 있고 주기적으로 갱신") — 화면과 같은 덩어리를
    # runs/<out_dir>/state.json 에 둔다. 세션 · 스크립트는 HTML 이 아니라 이것을 읽는다.
    state_path = out.parent / cfg.get("state_file", "state.json")
    tmp = state_path.with_name(state_path.name + ".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(state_path)
    page = render(state)
    tmp = out.with_suffix(".html.tmp")
    tmp.write_text(page, encoding="utf-8")
    tmp.replace(out)                    # 브라우저가 반쯤 쓰인 파일을 읽지 않게
    return out


class _Handler(SimpleHTTPRequestHandler):
    """저장소 뿌리를 읽기 전용으로 내준다 · `/` 는 대시보드로 · 접속 로그 없음."""

    index_path = "/runs/dashboard/index.html"

    def do_GET(self) -> None:        # noqa: N802 — http.server 규약
        if self.path in ("/", "/index.html"):
            self.send_response(302)
            self.send_header("Location", self.index_path)
            self.end_headers()
            return
        super().do_GET()

    def log_message(self, format: str, *args: Any) -> None:   # noqa: A002 — 부모 서명
        return


def make_server(port: int, root: Path, index_path: str) -> ThreadingHTTPServer:
    handler = partial(_Handler, directory=str(root))
    _Handler.index_path = index_path
    return ThreadingHTTPServer(("127.0.0.1", port), handler)


def serve(port: int, every_s: int, cfg: dict[str, Any] | None = None) -> int:
    cfg = cfg or config.load("dashboard")
    root = paths.root()
    out = build(cfg=cfg)
    rel = "/" + out.relative_to(root).as_posix() if out.is_relative_to(root) else "/" + out.name
    srv = make_server(port, root, rel)
    stop = threading.Event()

    def loop() -> None:
        while not stop.wait(every_s):
            try:
                build(cfg=cfg)
            except Exception as err:          # noqa: BLE001 — 대시보드는 죽지 않는다
                print(f"[dashboard] 다시 만들기 실패: {err}", flush=True)

    threading.Thread(target=loop, daemon=True).start()
    print(f"[dashboard] http://127.0.0.1:{port}/ → {rel} · {every_s} 초마다 다시 만듦 · Ctrl-C"
          f" · RSS {process_rss_mb():.0f} MB", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        srv.server_close()
    return 0


def process_rss_mb() -> float:
    """이 프로세스의 최대 RSS(MB) — 가벼움 확인용(맥은 바이트 · 리눅스는 KB)."""
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return raw / 1e6 if raw > 1e7 else raw / 1e3


def main(args: argparse.Namespace) -> int:
    cfg = config.load("dashboard")
    if args.serve is not None:
        port = int(cfg["port"]) if args.serve == -1 else int(args.serve)
        return serve(port, int(args.every or cfg["every_s"]), cfg)
    out = build(cfg=cfg)
    print(f"[dashboard] {out}")
    return 0


HELP = "R-7928 진행 · 메모리 · 결과 한 장(runs/ 읽기만 · 가벼움)"


def _add_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--serve", nargs="?", const=-1, default=None, type=int,
                   help="로컬 HTTP 로 띄운다(127.0.0.1 · 포트 생략하면 설정 값)")
    p.add_argument("--every", type=int, default=None,
                   help="--serve 일 때 다시 만드는 간격 초(기본 설정 값)")
    p.set_defaults(func=main)


def register(sub) -> None:
    _add_args(sub.add_parser("dashboard", help=HELP))


def cli(argv: list[str]) -> int:
    """`fly-connectome dashboard …` 의 빠른 길 — 다른 명령의 무거운 모듈을 올리지 않는다."""
    p = argparse.ArgumentParser(prog="fly-connectome dashboard", description=HELP)
    _add_args(p)
    return main(p.parse_args(argv))
