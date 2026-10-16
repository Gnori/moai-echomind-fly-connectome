"""렌더러 몰기 — 헤드리스 크롬(우리 프로필 · 우리 그룹) + WebSocket 통로(설계 문서 I 2.1 · 2.6).

메시지 = `[u32 머리 길이(little)][머리 JSON(UTF-8)][본문 바이트]`.
크롬 원격 조종(CDP)은 쓰지 않는다 — 페이지(web/eyeworld/render.html)가 우리 통로에 붙는다.
시작 전: 금지 포트 · 포트 비었나 · 에코월드 헤드리스 잠금 · 다른 헤드리스 크롬 ·
에코월드 저장소 표본 수정 시각(끝날 때 비교 — 바뀌면 오류).
"""

from __future__ import annotations

import asyncio
import json
import struct
import threading
import time
from pathlib import Path

import numpy as np

from .. import config, paths
from . import safety
from .static import StaticServer


def pack(head: dict, body: bytes = b"") -> bytes:
    h = json.dumps(head, ensure_ascii=False).encode("utf-8")
    return struct.pack("<I", len(h)) + h + body


def unpack(buf: bytes) -> tuple[dict, bytes]:
    (n,) = struct.unpack_from("<I", buf, 0)
    return json.loads(buf[4 : 4 + n].decode("utf-8")), buf[4 + n :]


def texture_index() -> dict | None:
    """에코월드 재질 색인(설정 render.texture_index) → 페이지 주소 — 없으면 None(묶지 않음)."""
    ti = config.load("eyeworld")["render"].get("texture_index")
    if not ti:
        return None
    return {
        "url": f"/assets/{ti['index']}",
        "base": f"/assets/{ti['base']}",
        "alpha_cut": float(ti["alpha_cut"]),
    }


class RenderError(RuntimeError):
    pass


class Renderer:
    """`with Renderer() as r: r.load_scene(...); frames = r.render(...)`."""

    def __init__(self, gpu: bool = False, run_dir: Path | None = None) -> None:
        self.cfg = config.load("eyeworld")["render"]
        self.gpu = gpu
        self.run_dir = (
            run_dir or paths.runs_dir() / f"eyeworld-render-{time.strftime('%Y%m%d-%H%M%S')}"
        )
        self.static = StaticServer()
        self.chrome: safety.OwnedProcess | None = None
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.loop.run_forever, daemon=True)
        self.inbox: asyncio.Queue | None = None
        self.conn = None
        self.server = None
        self.ready: dict = {}
        self.before: dict[str, float] = {}

    # --- 통로(이벤트 루프 실) ---
    async def _handler(self, ws) -> None:
        from websockets.exceptions import ConnectionClosed

        self.conn = ws
        assert self.inbox is not None
        try:
            async for msg in ws:
                await self.inbox.put(msg if isinstance(msg, bytes) else msg.encode())
        except ConnectionClosed:  # 끝낼 때 크롬이 먼저 닫히면 — 정상
            return

    async def _serve(self, port: int) -> None:
        from websockets.asyncio.server import serve

        self.inbox = asyncio.Queue()
        self.server = await serve(self._handler, "127.0.0.1", port, max_size=None)

    async def _send_recv(self, data: bytes | None, timeout: float) -> bytes:
        if data is not None:
            await self.conn.send(data)
        return await asyncio.wait_for(self.inbox.get(), timeout)

    def _call(self, head: dict | None, timeout_s: float) -> tuple[dict, bytes]:
        data = pack(head) if head is not None else None
        fut = asyncio.run_coroutine_threadsafe(self._send_recv(data, timeout_s), self.loop)
        try:
            msg = fut.result(timeout_s + 1.0)
        except Exception as e:
            raise RenderError(f"렌더러 응답 없음({head and head.get('type')}): {e!r}") from e
        h, body = unpack(msg)
        if h.get("type") == "error":
            raise RenderError(f"렌더러 오류 {h.get('code')}: {h.get('msg')}")
        return h, body

    # --- 시작 · 끝 ---
    def start(self, timeout_s: float = 90.0) -> dict:
        busy = safety.echomind_headless_busy()
        if busy:
            raise safety.SafetyError(f"에코월드 헤드리스 잠금이 살아 있다({busy}) — 기다렸다 다시")
        others = safety.other_headless_chromes()
        if others:
            raise safety.SafetyError(
                f"다른 헤드리스 크롬이 떠 있다 {others[:3]} — 겹치지 않게 기다렸다 다시"
            )
        ws_port = int(self.cfg["ws_port"])
        safety.check_port(ws_port)
        self.before = safety.readonly_snapshot()
        self.static.start()
        self.thread.start()
        asyncio.run_coroutine_threadsafe(self._serve(ws_port), self.loop).result(10)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        profile = paths.root() / self.cfg["profile_root"] / self.run_dir.name
        profile.mkdir(parents=True, exist_ok=True)
        url = (
            f"http://127.0.0.1:{self.static.port}/web/render.html?ws={ws_port}"
            f"&gpu={1 if self.gpu else 0}"
        )
        cmd = [
            self.cfg["chrome"],
            *self.cfg["chrome_flags"],
            f"--user-data-dir={profile}",
            "--window-size=64,64",
            url,
        ]
        self.chrome = safety.OwnedProcess(cmd, log=self.run_dir / "chrome.log")
        pid = self.chrome.start()
        h, _ = self._call(None, timeout_s)
        if h.get("type") != "ready":
            raise RenderError(f"ready 대신 {h}")
        self.ready = {**h, "chrome_pid": pid}
        return self.ready

    def stop(self) -> list[str]:
        """끝 — bye · 우리 크롬 그룹만 종료 · 통로 · 정적 서버 닫기.

        읽기 전용 확인 결과(바뀐 파일 목록)를 돌려준다.
        """
        try:
            if self.conn is not None:
                asyncio.run_coroutine_threadsafe(
                    self.conn.send(pack({"type": "bye"})), self.loop
                ).result(5)
        except Exception:  # noqa: BLE001 — 끝내는 길은 계속
            pass
        if self.chrome:
            self.chrome.stop()
        if self.server is not None:
            self.server.close()
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.static.stop()
        changed = safety.readonly_changed(self.before) if self.before else []
        if changed:
            raise safety.SafetyError(f"에코월드 저장소 파일이 바뀌었다: {changed[:5]}")
        return changed

    def __enter__(self) -> Renderer:
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

    # --- 일 ---
    def load_scene(
        self,
        scene: dict,
        poses: list[dict],
        markers: list[dict] | None = None,
        timeout_s: float = 180.0,
    ) -> dict:
        all_assets = scene.get("assets", {})
        assets = {
            k: f"/assets/{v['path']}"
            for k, v in all_assets.items()
            if v.get("kind", "model") == "model"
        }
        ground = dict(scene.get("ground") or {})
        tex = ground.pop("texture", None)
        if tex:  # 땅 텍스처 = 에셋 키(허가 확인을 같이 거침)
            ground["texture_url"] = f"/assets/{all_assets[tex]['path']}"
        head = {
            "type": "scene.load",
            "assets": assets,
            "poses": poses,
            "ground": ground,
            "sky": scene.get("sky"),
            "light": scene.get("light"),
            "markers": markers or [],
            "texture_index": texture_index(),
        }
        h, _ = self._call(head, timeout_s)
        if h.get("type") != "scene.ready":
            raise RenderError(f"scene.ready 대신 {h}")
        return h

    def info(self, eid: str, timeout_s: float = 10.0) -> dict:
        """진단 — 개체 노드 아래 메시 상태."""
        h, _ = self._call({"type": "info", "id": eid}, timeout_s)
        return h

    def render(
        self,
        tick: int,
        poses: list[dict],
        rig: list[dict],
        face_px: int,
        timeout_s: float | None = None,
    ) -> tuple[np.ndarray, dict]:
        """→ (밝기 u8 [눈, 6 면, N, N] · 줄 0 = 아래, 머리)."""
        timeout_s = float(self.cfg["timeout_ms"]) / 1000.0 if timeout_s is None else timeout_s
        head = {"type": "render", "tick": tick, "poses": poses, "eyes": rig, "face_px": face_px}
        h, body = self._call(head, timeout_s)
        if h.get("type") != "frames":
            raise RenderError(f"frames 대신 {h}")
        arr = np.frombuffer(body, dtype=np.uint8).reshape(h["n_eyes"], 6, face_px, face_px)
        return arr, h
