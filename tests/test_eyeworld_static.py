"""정적 서버(R-1177 · 설계 문서 I 2.9) — GET 만 · 경로 탈출 거부 · 금지 포트 거부."""

import socket
import urllib.error
import urllib.request

import pytest

from moai_echomind_fly.eyeworld import static
from moai_echomind_fly.eyeworld.safety import SafetyError


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_get_only_and_no_escape(tmp_path):
    root = tmp_path / "web"
    root.mkdir()
    (root / "a.js").write_text("ok", encoding="utf-8")
    (tmp_path / "secret.txt").write_text("no", encoding="utf-8")
    srv = static.StaticServer(port=_free_port(), table={"/web/": root})
    srv.start()
    base = f"http://127.0.0.1:{srv.port}"
    try:
        r = urllib.request.urlopen(f"{base}/web/a.js")
        assert r.read() == b"ok" and r.headers["Content-Type"].startswith("text/javascript")
        for bad in ("/web/../secret.txt", "/web/%2e%2e/secret.txt", "/other/a.js", "/web/"):
            with pytest.raises(urllib.error.HTTPError) as e:
                urllib.request.urlopen(f"{base}{bad}")
            assert e.value.code == 404
        req = urllib.request.Request(f"{base}/web/a.js", data=b"x", method="PUT")
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(req)
        assert e.value.code == 405
    finally:
        srv.stop()


def test_forbidden_port_refused(tmp_path):
    with pytest.raises(SafetyError):
        static.StaticServer(port=9212, table={"/web/": tmp_path}).start()


def test_default_mounts_point_at_echoworld_readonly_dirs():
    m = static.mounts()
    assert m["/babylon/"].name == "babylon" and m["/assets/"].name == "assets"
    assert (m["/babylon/"] / "babylon.js").is_file()
