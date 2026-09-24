"""상태 파일(R-7347) — 쓰기 · 합치기 · 원자적 교체 · 깨진 파일은 None."""

from __future__ import annotations

import json

import numpy as np

from moai_echomind_fly.loop import status


def test_update_merges_fields_and_rewrites_the_file(tmp_path):
    p = tmp_path / "status.json"
    st = status.StatusFile(p, brain="real", version="train-v5", state="running")
    first = json.loads(p.read_text(encoding="utf-8"))
    assert first["brain"] == "real" and first["state"] == "running"
    assert first["started"] and first["updated"] and first["pid"] > 0

    st.update(stage="caption", step=25, loss=np.float32(3.25), mem={"peak_gb": 31.0})
    st.update(state="done", exit_code=0)
    d = status.read(p)
    assert d is not None
    assert d["brain"] == "real" and d["stage"] == "caption" and d["step"] == 25
    assert d["loss"] == 3.25 and isinstance(d["loss"], float)   # numpy 숫자도 JSON 으로
    assert d["state"] == "done" and d["exit_code"] == 0
    assert not (tmp_path / "status.json.tmp").exists()          # 임시 파일은 이름이 바뀌어 사라진다


def test_read_missing_or_broken_returns_none(tmp_path):
    assert status.read(tmp_path / "없음.json") is None
    bad = tmp_path / "bad.json"
    bad.write_text("{반쯤", encoding="utf-8")
    assert status.read(bad) is None
