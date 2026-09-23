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
n 은 `[train] <뇌> · 학습 n` 줄, 규칙은 그 줄의 `규칙 {...}`(없으면 `train_cli.TRAIN_RULE`).
"""

from __future__ import annotations

import argparse
import ast
import html
import json
import re
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
            "loom_motion": "물체 움직임", "odor_side": "냄새 쪽", "odor_level": "냄새 세기",
            "legs": "다리 접촉"}

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


def read_full_log(runs: Path, pattern: str) -> dict[str, Any]:
    """최신 전체 순서 로그 — 단계마다 시작 · 끝 코드, 지금 도는 단계, 기다림 줄."""
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
            "steps": steps, "current": current, "notes": notes[-3:],
            "mtime": time.strftime("%m-%d %H:%M", time.localtime(path.stat().st_mtime))}


def _train_rule(text: str | None) -> dict[str, Any] | None:
    if text:
        try:
            return ast.literal_eval(text)
        except (ValueError, SyntaxError):
            pass
    try:
        # 무거운 의존 — 로그에 규칙이 없을 때만 불러온다
        from .loop.train_cli import TRAIN_RULE
        return dict(TRAIN_RULE)
    except Exception:                                # noqa: BLE001 — 없으면 추정을 안 할 뿐
        return None


def parse_train_log(lines: list[str]) -> dict[str, Any]:
    """train.log → 뇌 · n · 단계 · 걸음 · 손실 · s/걸음 · 메모리 · 검증 · 남은 시간(초)."""
    out: dict[str, Any] = {"brain": None, "n": None, "stage": None, "step": None, "loss": None,
                           "sec_per_step": None, "epoch": None, "mem": None, "val": None,
                           "before": None, "fingerprint_same": None, "saved": False,
                           "total_steps": None, "done_steps": None, "eta_s": None}
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


def read_projectors(runs: Path, pattern: str, eval_pattern: str) -> list[dict[str, Any]]:
    out = []
    for d in sorted(p for p in runs.glob(pattern) if p.is_dir()):
        rec: dict[str, Any] = {"dir": d.name, "log": None, "meta": None,
                               "final": (d / "final.safetensors").exists(), "evals": []}
        log = d / "train.log"
        if log.exists():
            rec["log"] = parse_train_log(_lines(log))
            rec["log_mtime"] = time.strftime("%m-%d %H:%M", time.localtime(log.stat().st_mtime))
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
    return {
        "version": cfg["version"], "now": time.strftime("%Y-%m-%d %H:%M:%S"), "runs": str(runs),
        "lock": read_lock(runs), "full": read_full_log(runs, g["full_log"]),
        "free": memory_free(),
        "projectors": read_projectors(runs, g["projector"], g["eval_caption"]),
        "probe": read_linear_probe(runs, g["linear_probe"]),
        "gates": read_gates(runs, g["compress"]),
        "bump": read_bump(runs, g["bump"]), "scorecard": read_scorecard(runs, g["scorecard"]),
        "datasets": read_datasets(datasets_dir, cfg["datasets_glob"]),
        "backup": read_backup(backup_md),
        "links": cfg.get("links") or [], "refresh_s": int(cfg["refresh_s"]),
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
        _card("전체 순서(run_full_v4)", full_txt, full_sub),
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
    if p["final"]:
        state = _tag("ok", "끝(final)")
    elif lg["saved"]:
        state = _tag("ok", "저장됨")
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
            "끝" if finished else _dur(lg["eta_s"]), mem_txt, val_txt, fp_txt]


def _section_train(d: dict[str, Any]) -> str:
    head = ["투영기", "단계", "걸음", "손실", "s/걸음", "남은 시간(추정)", "메모리", "검증 손실",
            "Qwen 지문"]
    out = _table(head, [_train_row(p) for p in d["projectors"]])
    evals = [(p["dir"], e) for p in d["projectors"] for e in p["evals"]]
    if not evals:
        return out + f'<h3>캡션 평가</h3><p class="muted">{NONE}</p>'
    fields = [f for f in FIELD_KO if any(f in e["accuracy"] for _, e in evals)]
    erows = []
    for dname, e in evals:
        cells = []
        for f in fields:
            a, b = e["accuracy"].get(f), e["baseline"].get(f)
            if a is None:
                cells.append("—")
                continue
            cls = "ok" if (b is not None and a > b + 0.05) else ""
            cells.append(f'{_tag(cls, _f(a))} <span class="muted">/ {_f(b)}</span>')
        erows.append([f"<code>{_e(dname)}</code>", _e(e["checkpoint"])] + cells)
    return (out + "<h3>캡션 평가 (정확도 / 찍기)</h3>"
            + _table(["투영기", "저장본"] + [FIELD_KO[f] for f in fields], erows))


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
<p class="sub">{_e(d['version'])} · {_e(d['runs'])} · 읽기 전용 · {links}</p>
<h2>지금 상태</h2>
{_section_status(d)}
<h2>학습 v4 진행</h2>
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
    page = render(collect(runs, datasets_dir, backup_md, cfg, memory_free))
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
    print(f"[dashboard] http://127.0.0.1:{port}/ → {rel} · {every_s} 초마다 다시 만듦 · Ctrl-C",
          flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        srv.server_close()
    return 0


def main(args: argparse.Namespace) -> int:
    cfg = config.load("dashboard")
    if args.serve is not None:
        port = int(cfg["port"]) if args.serve == -1 else int(args.serve)
        return serve(port, int(args.every or cfg["every_s"]), cfg)
    out = build(cfg=cfg)
    print(f"[dashboard] {out}")
    return 0


def register(sub) -> None:
    p = sub.add_parser("dashboard",
                       help="R-7928 진행 · 메모리 · 결과 한 장(runs/ 읽기만 · 가벼움)")
    p.add_argument("--serve", nargs="?", const=-1, default=None, type=int,
                   help="로컬 HTTP 로 띄운다(127.0.0.1 · 포트 생략하면 설정 값)")
    p.add_argument("--every", type=int, default=None,
                   help="--serve 일 때 다시 만드는 간격 초(기본 설정 값)")
    p.set_defaults(func=main)
