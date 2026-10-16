"""정적 파일 서버 — 127.0.0.1 · GET/HEAD 만(설계 문서 I 2.3 · 2.9).

마운트: `/web/` → 이 저장소 `web/eyeworld/` · `/babylon/` → 에코월드 `vendor/babylon`(읽기 전용) ·
`/assets/` → 에코월드 `assets`(읽기 전용). 경로 탈출(`..` · 링크로 밖)은 거부한다. 쓰는 요청은 405.
"""

from __future__ import annotations

import mimetypes
import threading
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

from .. import config, paths
from .safety import check_port, repo_path

TYPES = {
    ".js": "text/javascript",
    ".mjs": "text/javascript",
    ".wasm": "application/wasm",
    ".glb": "model/gltf-binary",
    ".gltf": "model/gltf+json",
    ".bin": "application/octet-stream",
    ".ktx2": "image/ktx2",
    ".json": "application/json",
    ".html": "text/html; charset=utf-8",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
}


def mounts() -> dict[str, Path]:
    w3d = repo_path(config.load("eyeworld")["render"]["world3d_root"])
    return {
        "/web/": paths.root() / "web" / "eyeworld",
        "/babylon/": w3d / "vendor" / "babylon",
        "/assets/": w3d / "assets",
    }


def resolve(url_path: str, table: dict[str, Path]) -> Path | None:
    """URL 경로 → 파일(마운트 안 · 실제 경로가 마운트 밖이면 None)."""
    p = unquote(urlsplit(url_path).path)
    for prefix, root in table.items():
        if p.startswith(prefix):
            rel = p[len(prefix) :]
            if not rel or "\x00" in rel:
                return None
            cand = (root / rel).resolve()
            base = root.resolve()
            if cand != base and base not in cand.parents:
                return None
            return cand if cand.is_file() else None
    return None


class Handler(BaseHTTPRequestHandler):
    server_version = "fly-eyeworld"

    def __init__(self, *a, table: dict[str, Path], **kw) -> None:
        self.table = table
        super().__init__(*a, **kw)

    def log_message(self, format: str, *args) -> None:  # noqa: A002 · 표준 출력을 더럽히지 않음
        return

    def _send(self, head_only: bool) -> None:
        f = resolve(self.path, self.table)
        if f is None:
            self.send_error(404)
            return
        data = f.read_bytes()
        ctype = (
            TYPES.get(f.suffix.lower())
            or mimetypes.guess_type(f.name)[0]
            or "application/octet-stream"
        )
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if not head_only:
            self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802
        self._send(False)

    def do_HEAD(self) -> None:  # noqa: N802
        self._send(True)

    def _refuse(self) -> None:
        self.send_error(405, "read-only")  # 응답 머리는 라틴 문자만

    do_POST = do_PUT = do_DELETE = do_PATCH = _refuse  # noqa: N815


class StaticServer:
    """백그라운드 실에서 도는 정적 서버."""

    def __init__(self, port: int | None = None, table: dict[str, Path] | None = None) -> None:
        self.port = int(config.load("eyeworld")["render"]["http_port"]) if port is None else port
        self.table = mounts() if table is None else table
        self.httpd: ThreadingHTTPServer | None = None
        self.thread: threading.Thread | None = None

    def start(self) -> None:
        check_port(self.port)
        self.httpd = ThreadingHTTPServer(
            ("127.0.0.1", self.port), partial(Handler, table=self.table)
        )
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        if self.httpd:
            self.httpd.shutdown()
            self.httpd.server_close()
            self.httpd = None
