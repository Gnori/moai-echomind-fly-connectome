"""대시보드 dashboard-v1 (R-7928) · v2 자료 파일(R-7347) — 가짜 runs/ 로 기대 문자열이 들어가나 ·
파일이 없어도 만들어지나 · status.json · jsonl 이 있으면 로그 대신 그것을 읽고 state.json 을 쓰나.

runs 자리는 `FLY_RUNS_DIR` 로 바꾼다(paths.runs_dir) · 자료 · 백업 자리는 인자로 준다.
memory_pressure 는 부르지 않는다(가짜 함수).
"""

from __future__ import annotations

import json
import threading
import urllib.request
from pathlib import Path

from moai_echomind_fly import config, dashboard

TRAIN_LOG = "\n".join([
    "[guard] 메모리 여유 88% ≥ 학습 문턱 80%",
    "[qwen] 가중치 지문(학습 전) abc",
    "[train] rewired-s0 · 학습 100 · 검증 24 · 투영기 1,331,200 매개변수 · 입력 90/120 토큰"
    " · 규칙 {'version': 'train-v4', 'caption_epochs': 2, 'action_epochs': 1,"
    " 'caption_rehearsal': 0.3}",
    "[before] 검증 캡션 3.300 · 검증 행동 0.400",
    "[mem] {'active_gb': 29.0, 'cache_gb': 0.22, 'peak_gb': 31.5}"
    " · {'step': 25, 'sys_free_percent': 83, 'footprint_gb': 28.0}",
    "[caption] ep 0 step 25 loss 2.500 · 6.0 s/걸음",
    "[caption] ep 0 step 50 loss 1.234 · 6.0 s/걸음",
    "",
])

FULL_LOG = """=== 2026-09-24 00:26:30 시작 · 자료 fly-loop-v2 · 판 train-v4 · 학습 건너뜀 0 · 로그 x
=== 2026-09-24 00:26:30 여유 70% · compress fit --dataset fly-loop-v2
=== 2026-09-24 00:28:33 끝 코드 0 · compress fit --dataset fly-loop-v2
=== 2026-09-24 00:28:33 여유 74% · train --brain real --dataset fly-loop-v2
--- 메모리 여유 62% < 80% — 기다림
"""


def _cfg() -> dict:
    return config.load("dashboard")


def _fake_runs(runs: Path) -> None:
    (runs / "full-v4-20260924-002630.log").write_text(FULL_LOG, encoding="utf-8")
    (runs / ".heavy.lock").write_text(json.dumps({"pid": 1, "kind": "train", "started": 0}))
    pj = runs / "projector-rewired-s0-train-v4"
    pj.mkdir()
    (pj / "train.log").write_text(TRAIN_LOG, encoding="utf-8")
    (pj / "eval-caption-final.json").write_text(json.dumps(
        {"brain": "rewired-s0", "checkpoint": "final",
         "accuracy": {"loom_side": 0.81, "legs": 0.5},
         "majority_baseline": {"loom_side": 0.73, "legs": 0.36}}))
    (pj / "eval-scored-final.json").write_text(json.dumps(
        {"brain": "rewired-s0", "checkpoint": "final", "rule": {"version": "eval-v2-scored"},
         "accuracy": {"loom_side": 0.66, "legs": 0.55, "sound": 0.9},
         "majority_baseline": {"loom_side": 0.5, "legs": 0.36, "sound": 0.9}}))
    (runs / "linear-probe-20260923-201422.json").write_text(json.dumps(
        {"dataset": "fly-loop-v2", "stats": {"version": "stats-v2"}, "rewired_brains": ["a", "b"],
         "cells": {"vision|loom_side": {"real": 0.801, "rewired": {"mean": 0.619},
                                        "baseline": 0.726, "p_holm": 0.004, "judge": True,
                                        "pass": True},
                   "touch|loom_side": {"real": 0.4, "rewired": {"mean": 0.6}, "baseline": 0.726,
                                       "p_holm": 1.0, "judge": True, "pass": False}}}))
    g = runs / "compress-fly-loop-v2-compress-v2"
    g.mkdir()
    (g / "gate.json").write_text(json.dumps(
        {"rule": "compress-v2", "passed_channels": ["vision", "body"],
         "passed_cells": ["vision|loom_side", "body|legs"]}))
    (runs / "bump-check-20260923-195413.json").write_text(json.dumps(
        {"dataset": "fly-loop-v2", "epg_units": 16, "real_rank_pct": 0.0,
         "brains": {"real": {"decisions": 3297, "live": 0, "bump_length_median": 0.0,
                             "circ_corr_abs": 0.0, "p_perm": 1.0}}}))
    sc = runs / "scorecard-20260923-031557"
    sc.mkdir()
    (sc / "scorecard.json").write_text(json.dumps(
        {"seeds": [1, 2], "configs": {"vector": {"reach_rate": 0.2, "final_food_mm_mean": 41.2,
                                                 "avoid_rate": 1.0, "schema_violations": 0}},
         "vector_beats_rewired": False,
         "wording": "섞은 배선 대조를 이기지 못했다 — '초파리 뇌'라는 말을 쓰지 않는다"}))


def test_build_without_any_file_says_not_yet(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    runs.mkdir()
    monkeypatch.setenv("FLY_RUNS_DIR", str(runs))
    out = dashboard.build(datasets_dir=tmp_path / "no-datasets", backup_md=tmp_path / "no.md",
                          memory_free=lambda: None)
    assert out == runs / "dashboard" / "index.html" and out.exists()
    page = out.read_text(encoding="utf-8")
    assert dashboard.NONE in page and "알 수 없음" in page and "비어 있음" in page
    assert '<meta http-equiv="refresh" content="60">' in page


def test_build_with_fake_runs_shows_progress_and_results(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    runs.mkdir()
    _fake_runs(runs)
    ds = tmp_path / "datasets" / "fly-loop-v2"
    ds.mkdir(parents=True)
    (ds / "meta.json").write_text(json.dumps(
        {"name": "fly-loop-v2", "splits": {"train": {"samples": 2921, "episodes": 300}},
         "versions": {"sim": "sim-v2"}, "packed": "2026-09-23 19:43:32"}))
    bk = tmp_path / "BACKUP.md"
    bk.write_text("# 백업\n\n- 마지막 복사: 2026-09-23 20:22:50 · 저장소 커밋 `5765b43`\n",
                  encoding="utf-8")
    monkeypatch.setenv("FLY_RUNS_DIR", str(runs))
    d = dashboard.collect(datasets_dir=tmp_path / "datasets", backup_md=bk, memory_free=lambda: 77)
    lg = d["projectors"][0]["log"]
    assert lg["brain"] == "rewired-s0" and lg["n"] == 100
    assert lg["total_steps"] == 330 and lg["done_steps"] == 50            # 100×(2+1) + 30
    assert lg["eta_s"] == (330 - 50) * 6.0 and lg["loss"] == 1.234
    assert d["full"]["current"]["cmd"].startswith("train --brain real")
    assert d["full"]["steps"][0]["code"] == 0 and "기다림" in d["full"]["notes"][0]
    page = dashboard.render(d)
    for text in ("rewired-s0", "50 / 330", "28 분", "1.234", "77 %", "train · pid 1",
                 "시각 → 물체 쪽", "0.801", "0.619", "시각, 몸", "3297",
                 "vector", "섞은 배선 대조를 이기지 못했다", "2,921 결정 / 300 에피소드", "sim-v2",
                 "마지막 복사: 2026-09-23 20:22:50", "0.810", "/ 0.730"):
        assert text in page, text
    eval_part = page.split("캡션 평가")[1].split("<h2>")[0]
    assert eval_part.index("채점(v2)") < eval_part.index("생성(v1)")   # 채점이 먼저
    assert "0.660" in eval_part and "소리" in eval_part
    probe_part = page.split("선형 검사 판정")[1].split("압축 관문")[0]
    assert "touch" not in probe_part                      # 불통 칸은 표에 없다
    out = dashboard.build(datasets_dir=tmp_path / "datasets", backup_md=bk, memory_free=lambda: 77)
    assert "28 분" in out.read_text(encoding="utf-8")


def test_parse_train_log_action_stage_counts_caption_steps_as_done():
    lines = TRAIN_LOG.splitlines() + [
        "[caption] ep 1 끝 · 학습 0.2 · 검증 캡션 0.15 · 검증 행동 0.30",
        "[action] ep 0 step 10 loss 0.5 · 8.0 s/걸음"]
    lg = dashboard.parse_train_log(lines)
    assert lg["stage"] == "action" and lg["done_steps"] == 200 + 10 and lg["eta_s"] == 120 * 8.0
    assert lg["val"]["caption"] == 0.15
    saved = dashboard.parse_train_log(lines + ["[qwen] 가중치 지문(학습 뒤) abc · 같음 True",
                                               "[train] 저장 /x/projector-rewired-s0-train-v4"])
    assert saved["saved"] and saved["fingerprint_same"] is True


def test_server_redirects_root_and_serves_page(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    runs.mkdir()
    monkeypatch.setenv("FLY_RUNS_DIR", str(runs))
    out = dashboard.build(datasets_dir=tmp_path / "x", backup_md=tmp_path / "y",
                          memory_free=lambda: 50)
    srv = dashboard.make_server(0, tmp_path, "/runs/dashboard/index.html")
    port = srv.server_address[1]
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=5) as r:  # 302 따라감
            body = r.read().decode("utf-8")
            assert r.geturl().endswith("/runs/dashboard/index.html")
            assert "초파리 실행 대시보드" in body
    finally:
        srv.shutdown()
        srv.server_close()
    assert out.exists()


def test_status_json_and_jsonl_win_over_logs_and_state_json_is_written(tmp_path, monkeypatch):
    """v2(R-7347): status.json · 사건 jsonl 을 먼저 읽고, 같은 덩어리를 state.json 에 둔다."""
    runs = tmp_path / "runs"
    runs.mkdir()
    _fake_runs(runs)                                   # 옛 로그(v4)도 같이 있다
    pj = runs / "projector-real-train-v5"
    pj.mkdir()
    (pj / "train.log").write_text("[caption] ep 0 step 25 loss 9.999 · 1.0 s/걸음\n",
                                  encoding="utf-8")
    (pj / "status.json").write_text(json.dumps(
        {"brain": "real", "version": "train-v5", "state": "running", "stage": "action", "epoch": 0,
         "stage_step": 100, "step": 2478, "total_steps": 3923, "caption_steps": 2378,
         "loss": 0.777, "sec_per_step": 5.0, "train_samples": 1189,
         "mem": {"peak_gb": 31.0, "cache_gb": 0.01, "sys_free_percent": 66},
         "val": {"stage": "caption", "epoch": 1, "train": 2.9, "caption": 2.774, "action": 0.705},
         "started": "2026-09-24 13:00:00", "updated": "2026-09-24 16:00:00"}), encoding="utf-8")
    ab = runs / "projector-rewired-s0-train-v5"
    ab.mkdir()
    (ab / "status.json").write_text(json.dumps(
        {"brain": "rewired-s0", "version": "train-v5", "state": "aborted",
         "reason": "시스템 메모리 여유 21% < 25%", "stage": "caption", "step": 270,
         "total_steps": 3923}), encoding="utf-8")
    lines = [
        {"ts": "2026-09-24 12:00:00", "event": "begin", "cmd": "run_full_v5", "free": 60,
         "code": None},
        {"ts": "2026-09-24 12:00:01", "event": "start", "cmd": "train --brain real", "free": 60,
         "code": None},
        {"ts": "2026-09-24 18:00:00", "event": "end", "cmd": "train --brain real", "free": 70,
         "code": 0},
        {"ts": "2026-09-24 18:00:01", "event": "start", "cmd": "train --brain rewired-s0",
         "free": 70, "code": None},
    ]
    (runs / "full-v5-20260924-120000.jsonl").write_text(
        "\n".join(json.dumps(x, ensure_ascii=False) for x in lines) + "\n", encoding="utf-8")
    monkeypatch.setenv("FLY_RUNS_DIR", str(runs))
    d = dashboard.collect(datasets_dir=tmp_path / "x", backup_md=tmp_path / "y",
                          memory_free=lambda: 66)
    assert d["full"]["source"] == "jsonl" and d["full"]["path"].endswith(".jsonl")
    assert d["full"]["steps"][0]["code"] == 0
    assert d["full"]["current"]["cmd"] == "train --brain rewired-s0"
    by = {p["dir"]: p["log"] for p in d["projectors"]}
    real = by["projector-real-train-v5"]
    assert real["source"] == "status" and real["loss"] == 0.777          # 로그의 9.999 가 아니라
    assert real["done_steps"] == 2478 and real["eta_s"] == (3923 - 2478) * 5.0
    assert by["projector-rewired-s0-train-v5"]["state"] == "aborted"
    assert by["projector-rewired-s0-train-v4"]["source"] != "status"     # 옛 폴더는 로그 파싱
    page = dashboard.render(d)
    for text in ("0.777", "2,478 / 3,923", "멈춤 · 시스템 메모리 여유 21%",
                 "train --brain rewired-s0", "2.774"):
        assert text in page, text
    out = dashboard.build(datasets_dir=tmp_path / "x", backup_md=tmp_path / "y",
                          memory_free=lambda: 66)
    state = json.loads((out.parent / "state.json").read_text(encoding="utf-8"))
    assert state["version"] == "dashboard-v2" and state["free"] == 66
    assert state["full"]["current"]["cmd"] == "train --brain rewired-s0"
    assert any(p["dir"] == "projector-real-train-v5" and p["log"]["loss"] == 0.777
               for p in state["projectors"])


def test_dashboard_command_stays_light_no_torch_and_rule_from_source(tmp_path):
    """`fly-connectome dashboard` 빠른 길 — torch · numpy · train_cli 를 올리지 않고
    TRAIN_RULE 은 소스에서 읽는다. 새 프로세스에서 확인(이 프로세스는 이미 올렸을 수 있다)."""
    import os
    import subprocess
    import sys

    code = (
        "import sys, json\n"
        "from moai_echomind_fly import cli, dashboard\n"
        "rc = cli.main(['dashboard'])\n"
        "rule = dashboard.train_rule_from_source()\n"
        "print(json.dumps({'rc': rc, 'heavy': [m for m in ('torch', 'numpy', 'pyarrow',"
        " 'moai_echomind_fly.loop.train_cli') if m in sys.modules],"
        " 'rule_ok': isinstance(rule, dict) and 'caption_epochs' in rule,"
        " 'rss_mb': dashboard.process_rss_mb()}))\n")
    env = {**os.environ, "FLY_RUNS_DIR": str(tmp_path / "runs")}
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env,
                         timeout=60)
    assert out.returncode == 0, out.stderr
    got = json.loads(out.stdout.strip().splitlines()[-1])
    assert got["rc"] == 0 and got["heavy"] == [] and got["rule_ok"], got
    assert got["rss_mb"] < 120, got                       # 인수 기준 200 MB · 실측 약 30 MB
    assert (tmp_path / "runs" / "dashboard" / "index.html").exists()
