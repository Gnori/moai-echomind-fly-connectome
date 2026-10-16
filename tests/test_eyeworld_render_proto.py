"""렌더러 통로 · 이웃 감지(R-1177 · 설계 문서 I 2.6 · 2.9) — 크롬 없이."""

import subprocess

from moai_echomind_fly.eyeworld import render, safety


def test_pack_unpack_roundtrip():
    head = {"type": "frames", "tick": 3, "faces": ["front", "back"], "글": "한글"}
    body = bytes(range(256)) * 3
    h, b = render.unpack(render.pack(head, body))
    assert h == head and b == body
    h2, b2 = render.unpack(render.pack({"type": "bye"}))
    assert h2 == {"type": "bye"} and b2 == b""


def test_other_headless_chromes_excludes_ours_and_helpers(monkeypatch):
    ps = "\n".join([
        "100 /Applications/Google Chrome.app/Contents/MacOS/Google Chrome --headless=new "
        "--user-data-dir=shots/loop_B/_chrome about:blank",
        "101 /Applications/Google Chrome.app/.../Google Chrome Helper (GPU) "
        "--type=gpu-process --headless=new",
        "102 /Applications/Google Chrome.app/Contents/MacOS/Google Chrome --headless=new "
        "--user-data-dir=/x/runs/eyeworld-chrome/r1 http://127.0.0.1:8790/web/render.html",
        "103 /usr/bin/python something --headless",
    ])

    class R:
        stdout = ps

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: R())
    assert safety.other_headless_chromes("runs/eyeworld-chrome") == [(100, "shots/loop_B/_chrome")]
