"""대시보드 2단계(R-5777) — 캡션 표본(F-001) · 성적표 에피소드(F-003) 쪽.

가짜 runs/ 로: 칸별 맞음 수를 다시 세면 파일의 accuracy 와 같나 · 진짜만 맞음/틀림 · 재배선별 차이 ·
도착 수 = scorecard.json · 결정 기록이 있으면 결정 표 · 도는 쪽 불일치 · 없으면 "결정 기록 전".
실제 v5 파일이 있으면 그것으로도 맞춰 본다(없으면 건너뜀).
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from moai_echomind_fly import config, dashboard, dashboard_debug, paths

BRAINS = dashboard_debug.BRAIN_ORDER
FIELDS = ["loom_present", "loom_side", "legs"]


def _fake_eval(runs: Path, version: str = "train-v5", n: int = 60, seed: int = 0) -> dict:
    """뇌 넷 × 저장본 final 의 eval-scored — 물체가 있을 때만 물체 쪽 칸이 있다."""
    rng = random.Random(seed)
    truths = []
    for i in range(n):
        present = rng.random() < 0.4
        t = {"loom_present": "있음" if present else "없음", "legs": f"{rng.randint(4, 6)}개"}
        if present:
            t["loom_side"] = rng.choice(["왼쪽", "오른쪽"])
        truths.append({"i": i, "seed": 1000 + i // 10, "truth": t})
    acc = {}
    for k, b in enumerate(BRAINS):
        hit = 0.8 - 0.1 * k
        ps = []
        for s in truths:
            pred = {f: (v if rng.random() < hit else f"틀림{k}") for f, v in s["truth"].items()}
            ps.append({**s, "pred": pred})
        acc[b] = {f: sum(1 for s in ps if f in s["truth"] and s["pred"][f] == s["truth"][f])
                  / sum(1 for s in ps if f in s["truth"]) for f in FIELDS}
        d = runs / f"projector-{b}-{version}"
        d.mkdir(parents=True)
        (d / "eval-scored-final.json").write_text(json.dumps(
            {"brain": b, "checkpoint": "final", "accuracy": acc[b],
             "majority_baseline": {f: 0.5 for f in FIELDS}, "per_sample": ps},
            ensure_ascii=False), encoding="utf-8")
    return acc


def _fake_scorecard(runs: Path, with_decisions: bool) -> Path:
    sc = runs / "scorecard-20260930-000000"
    sc.mkdir()
    seeds = [2000, 2001, 2002]
    reached = {"vector": [True, False, False], "rewired-s0": [False, False, False]}
    (sc / "scorecard.json").write_text(json.dumps(
        {"seeds": seeds, "judgement": {"rule": "scorecard-v3", "p_sign_flip": 0.5},
         "configs": {c: {"episodes": 3, "reach_rate": sum(r) / 3, "final_food_mm_mean": 30.0}
                     for c, r in reached.items()}}), encoding="utf-8")
    for c, r in reached.items():
        (sc / f"{c}.jsonl").write_text("\n".join(json.dumps(
            {"seed": s, "decisions": 2, "reached": ok, "reach_ms": 2400 if ok else None,
             "final_food_mm": 1.5 if ok else 40.0}) for s, ok in zip(seeds, r)) + "\n")
    if with_decisions:        # loop/scorecard.py decision_row 형식(R-5777 3.3)
        rows = []
        for s in seeds:
            for d, (tt, at) in enumerate([("left", "left"), ("left", "right")]):
                rows.append({"seed": s, "d": d, "t_ms": 300 * (d + 1),
                             "truth": {"looming_present": d == 1, "looming_azimuth_deg": -20.0,
                                       "odor_left": 1.2, "odor_right": 1.1},
                             "caption": "다가오는 물체: 없음 / 먹이 냄새: 왼쪽 · 강함",
                             "teacher": {"walk": 0.6, "turn": tt, "turn_strength": 0.5},
                             "action": {"walk": 0.5, "turn": at, "turn_strength": 1.0},
                             "source": "qwen", "raw": '{"turn":"x"}</script><b>',
                             "valid": True, "why": None, "odor_L": 1.2, "odor_R": 1.1,
                             "food_bearing_deg": 30.0, "food_mm": 20.0, "drive": [0.5, 0.5],
                             "reflex_bins": 0})
        (sc / "vector.decisions.jsonl").write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    return sc


def test_samples_recount_equals_file_accuracy_and_pair_diffs(tmp_path):
    acc = _fake_eval(tmp_path)
    s = dashboard_debug.read_samples(tmp_path, "train-v5", ["final", "stage1-caption"],
                                     list(dashboard.FIELD_KO), 5000)
    assert list(s["checkpoints"]) == ["final"]                    # 없는 저장본은 건너뜀
    c = s["checkpoints"]["final"]
    assert c["brains"] == list(BRAINS) and c["missing_pairs"] == 0
    assert c["fields"] == ["loom_present", "loom_side", "legs"]    # FIELD_KO 순서 · 있는 칸만
    for f, sm in c["summary"].items():
        for b in BRAINS:
            assert sm["accuracy"][b] == pytest.approx(acc[b][f]), (f, b)
        for b, p in sm["pair"].items():                           # 정의상 맞음 수의 차이
            assert p["diff"] == sm["correct"]["real"] - sm["correct"][b]
        # 진짜만 맞음 · 틀림을 직접 다시 센다
        j = c["fields"].index(f)
        right = wrong = 0
        for r in c["rows"]:
            if r["t"][j] is None:
                continue
            ok = [p[j] == r["t"][j] for p in r["p"]]
            right += ok[0] and not any(ok[1:])
            wrong += (not ok[0]) and all(ok[1:])
        assert (sm["only_real_right"], sm["only_real_wrong"]) == (right, wrong)
    assert [r["d"] for r in c["rows"][:11]] == list(range(10)) + [0]   # 에피소드 안 결정 번호


def test_build_writes_both_pages_with_links_and_no_external_scripts(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    runs.mkdir()
    _fake_eval(runs)
    _fake_scorecard(runs, with_decisions=False)
    monkeypatch.setenv("FLY_RUNS_DIR", str(runs))
    out = dashboard.build(datasets_dir=tmp_path / "x", backup_md=tmp_path / "y",
                          memory_free=lambda: 70)
    idx = out.read_text(encoding="utf-8")
    assert 'href="samples.html"' in idx and 'href="episodes.html"' in idx
    sp = (out.parent / "samples.html").read_text(encoding="utf-8")
    ep = (out.parent / "episodes.html").read_text(encoding="utf-8")
    for page in (sp, ep):
        assert "<script src" not in page and "http" not in page.split("<body>")[1]
    assert "진짜만 맞음" in sp and "물체 있음" in sp
    assert "결정 기록 전(scorecard-v2)" in ep                     # 결정 파일이 없는 성적표
    st = json.loads((out.parent / "state.json").read_text(encoding="utf-8"))["debug"]
    assert st["episodes"]["check"]["vector"] == {"episodes": 3, "reached": 1,
                                                 "scorecard_reached": 1, "match": True}
    assert "final" in st["samples"]


def test_episodes_with_decisions_show_table_and_turn_mismatch(tmp_path):
    _fake_scorecard(tmp_path, with_decisions=True)
    e = dashboard_debug.read_episodes(tmp_path, "scorecard-*", 20000, 200)
    assert e["has_decisions"] == {"vector": True, "rewired-s0": False}
    assert e["check"]["vector"]["decision_rows"] == 6 and e["check"]["vector"]["turn_mismatch"] == 3
    assert all(k["match"] for k in e["check"].values())
    rows = e["decisions"]["vector"]["2000"]
    assert [r[-1] for r in rows] == [False, True]                 # 두 번째 결정만 도는 쪽 다름
    assert rows[0][3] is None and rows[1][3] == -20.0              # 물체 방위(있을 때만)
    assert rows[1][4] == "왼쪽 0.50 · 걷기 0.60" and rows[1][5] == "오른쪽 1.00 · 걷기 0.50"
    assert e["captions"] == ["다가오는 물체: 없음 / 먹이 냄새: 왼쪽 · 강함"] and rows[0][2] == 0
    page = "".join(dashboard_debug.render_episodes(e, dashboard.FIELD_KO))
    blob = page.split('id="data">')[1].split("</script>")[0]      # 원문의 </script> 가 새지 않는다
    assert json.loads(blob)["dec"]["vector"]["2001"][1][-1] is True
    assert "결정 기록 전(scorecard-v2)" not in page.split("<script")[0]
    assert "다름" in page and "3 / 6" in page and "<b>1</b> / 3" in page   # 다름 · 도착
    cut = dashboard_debug.read_episodes(tmp_path, "scorecard-*", 4, 200)
    assert cut["truncated"] and cut["check"]["vector"]["decision_rows"] == 4


def test_pages_are_not_rewritten_when_sources_are_unchanged(tmp_path):
    runs = tmp_path / "runs"
    _fake_eval(runs)
    out = runs / "dashboard"
    out.mkdir()
    cfg = config.load("dashboard")
    dashboard_debug._SIGS.clear()
    dashboard_debug.build_pages(runs, out, cfg, "", dashboard.FIELD_KO, [])
    sp = out / "samples.html"
    first = sp.stat().st_mtime_ns
    sp.write_text("표시", encoding="utf-8")                       # 원본이 같으면 손대지 않는다
    dashboard_debug.build_pages(runs, out, cfg, "", dashboard.FIELD_KO, [])
    assert sp.read_text(encoding="utf-8") == "표시"
    f = runs / "projector-real-train-v5" / "eval-scored-final.json"
    f.write_text(f.read_text(encoding="utf-8") + " ", encoding="utf-8")
    dashboard_debug.build_pages(runs, out, cfg, "", dashboard.FIELD_KO, [])
    assert sp.read_text(encoding="utf-8") != "표시" and sp.stat().st_mtime_ns >= first


def test_real_v5_files_recount_matches_accuracy_and_scorecard():
    """실제 v5 파일(git 밖) — 있으면 맞춰 보고 없으면 건너뜀."""
    runs = paths.runs_dir()
    if not (runs / "projector-real-train-v5" / "eval-scored-final.json").exists():
        pytest.skip("v5 채점 파일 없음")
    s = dashboard_debug.read_samples(runs, "train-v5", ["final", "stage1-caption"],
                                     list(dashboard.FIELD_KO), 5000)
    for cp, c in s["checkpoints"].items():
        assert len(c["rows"]) == 376 and c["brains"] == list(BRAINS) and not c["missing_pairs"]
        for f, sm in c["summary"].items():
            for b in c["brains"]:
                assert sm["accuracy"][b] == pytest.approx(c["file_accuracy"][b][f]), (cp, f, b)
    sc = runs / "scorecard-20260925-175025"
    if (sc / "scorecard.json").exists():
        e = dashboard_debug.read_episodes(runs, sc.name, 20000, 200)
        assert all(k["match"] and k["episodes"] == 40 for k in e["check"].values())
