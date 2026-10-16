"""보기 창 뇌 연결(R-1177 · 요구자 10-17 "초파리 움직이게" · 탐색 — 판정 아님).

브라우저(보기 창)가 초파리 뇌 캐릭터 두 눈 큐브맵을 걸음(10 ms)마다 보내면:
  그림 → 앞단 v3b(FrontendV3) → 발화율 × 다이얼(손으로 · 1 = 잰 그대로 · r_max 로 자름)
  → 뇌 둘(진짜 배선 · 섞은 배선 rewired-s0)을 **따로 프로세스에서 나란히** 10 ms 달림
  → 읽기 뉴런 발화를 창(window_ms)으로 셈
  → 진짜 배선 뇌의 **하강 뉴런 그대로** 몸 명령(지어낸 행동 규칙 없음):
     앞뒤 = DNp09(앞) − MDN(뒤) · 돌기 = DNa02 · DNa01 오른 − 왼(같은 쪽으로) · GF(DNp01) = 뛰어오름
     · 발화 → 속도 환산 숫자만 선언 · 상한은 몸 규약.
값은 config/eyeworld.yaml view_brain(선언). 섞은 배선 뇌는 같은 입력 · 값만 보여 줌(몸은 안 움직임).
통로 = 렌더러와 같은 꼴([u32 머리 길이][머리 JSON][본문]) · 127.0.0.1 만.
"""

from __future__ import annotations

import asyncio
import multiprocessing as mp
import os
import time
from collections import deque

import numpy as np

from .. import config
from .. import graph as graph_mod
from ..eye.frontend import FrontendV3
from ..loop.signals import side_of
from .render import pack, unpack
from .safety import check_port

READ = {  # 몸을 움직이는 하강 뉴런(설정 view_brain.motor) + 보기만 하는 것
    "DNp09": ["DNp09"],
    "MDN": ["MDN"],
    "DNa02": ["DNa02"],
    "DNa01": ["DNa01"],
    "GF": ["DNp01"],
    "DNp02": ["DNp02"],
    "LC4": ["LC4"],
    "LPLC2": ["LPLC2"],
    "HS": ["HSE", "HSN", "HSS"],
}
KEY = {t: k for k, v in READ.items() for t in v}  # 종류 이름 → 읽기 묶음


def labels(g) -> tuple[np.ndarray, list[str]]:
    """뉴런마다 읽기 묶음 번호(없으면 −1) · 묶음 이름(종류_눈)."""
    typ = g.columns["type"].astype(str)
    side = side_of(g)
    lab = np.full(g.n, -1, np.int64)
    names = []
    for k, v in READ.items():
        for s in ("L", "R"):
            lab[np.isin(typ, v) & (side == s)] = len(names)
            names.append(f"{k}_{s}")
    return lab, names


def _worker(brain: str, seed: int, entry: np.ndarray, dt_ms: float, threads: int, conn) -> None:
    """뇌 하나(프로세스) — 발화율을 받으면 dt 만큼 달리고 묶음별 발화 수를 돌려준다."""
    import torch

    from ..sim import Engine, LIFParams

    torch.set_num_threads(threads)
    base = graph_mod.load()
    g = graph_mod.brain_graph(base, brain)
    lab, names = labels(base)
    eng = Engine(g, LIFParams.from_config(), device="cpu", seed=seed)
    eng.set_poisson(entry, np.zeros(len(entry)))
    conn.send(("ready", names))
    while True:
        msg = conn.recv()
        if msg is None:
            break
        if isinstance(msg, str):  # "reset"
            eng.reset()
            continue
        t0 = time.perf_counter()
        s0 = eng.step
        eng.set_rates(msg)
        eng.run(dt_ms)
        _, nn = eng.spikes_since(s0)
        ln = lab[nn]
        cnt = np.bincount(ln[ln >= 0], minlength=len(names))
        conn.send((cnt, time.perf_counter() - t0, len(nn)))


class BrainLink:
    """앞단 + 뇌 둘(프로세스) + 해독. `serve()` 는 웹소켓(127.0.0.1:port)."""

    def __init__(self) -> None:
        ecfg = config.load("eyeworld")
        self.cfg = ecfg["view_brain"]
        self.dt_ms = float(self.cfg["dt_ms"])
        self.r_max = float(ecfg["frontend"]["r_max_hz"])
        base = graph_mod.load()
        self.fe = FrontendV3(base, int(ecfg["v1"]["face_px"]), self.dt_ms)
        self.n_px = 2 * 6 * int(ecfg["v1"]["face_px"]) ** 2
        t = self.fe.t45
        self.ftb = {s: (t.side == s) & (t.sub == "a") for s in "LR"}
        per = max(1, (os.cpu_count() or 2) // len(self.cfg["brains"]))
        threads = min(int(self.cfg["threads"]), per)
        ctx = mp.get_context("spawn")
        self.brains = {}
        for b in self.cfg["brains"]:
            parent, child = ctx.Pipe()
            p = ctx.Process(
                target=_worker,
                args=(b, int(self.cfg["seed"]), self.fe.entry, self.dt_ms, threads, child),
                daemon=True,
            )
            p.start()
            self.brains[b] = (p, parent)
        self.names: list[str] = []
        for _, conn in self.brains.values():
            _, self.names = conn.recv()
        n_win = int(round(float(self.cfg["window_ms"]) / self.dt_ms))
        self.win = {b: deque(maxlen=n_win) for b in self.brains}
        self.body = ecfg["world"]["body"]
        self.steps = 0
        self.t_wall: float | None = None

    def reset(self) -> None:
        self.t_wall = None
        self.steps = 0
        self.fe.reset()
        for _, conn in self.brains.values():
            conn.send("reset")
        for w in self.win.values():
            w.clear()

    def step(self, img: np.ndarray, dial: float) -> dict:
        """그림 [2, 6, N, N] u8 → 뇌 둘 한 걸음 → 창 합 · 몸 명령."""
        t0 = time.time()
        if self.t_wall is None:  # 벽시계는 첫 걸음부터(서버 띄운 때가 아님)
            self.t_wall = t0
        rates = np.minimum(self.fe.step(img) * float(dial), self.r_max)
        t1 = time.time()
        for _, conn in self.brains.values():
            conn.send(rates)
        sums = {}
        own = {}
        for b, (_, conn) in self.brains.items():
            cnt, sec, n_spk = conn.recv()
            self.win[b].append(cnt)
            own[b] = {"ms": round(1000 * sec, 1), "spikes": int(n_spk)}
            sums[b] = dict(zip(self.names, map(int, np.sum(self.win[b], axis=0))))
        t2 = time.time()
        m, r, b = self.cfg["motor"], sums["real"], self.body

        def both(part: str) -> int:
            return sum(r[f"{KEY[t]}_{s}"] for t in m[part]["types"] for s in "LR")

        def diff(part: str) -> int:
            return sum(r[f"{KEY[t]}_R"] - r[f"{KEY[t]}_L"] for t in m[part]["types"])

        speed = (float(m["forward"]["m_s_per_spike"]) * both("forward")
                 - float(m["backward"]["m_s_per_spike"]) * both("backward"))
        speed = float(np.clip(speed, -float(b["walk_max_m_s"]), float(b["run_max_m_s"])))
        turn = -float(m["turn"]["deg_s_per_spike"]) * diff("turn")  # 오른쪽 많으면 오른쪽으로
        turn = float(np.clip(turn, -float(b["turn_max_deg_s"]), float(b["turn_max_deg_s"])))
        jump = both("jump") >= int(m["jump"]["spikes"])
        n45 = len(self.fe.t45)
        self.steps += 1
        return {
            "type": "out",
            "sums": sums,
            "cmd": {"turn_deg_s": turn, "speed_m_s": speed, "jump": bool(jump)},
            "input_hz": {s: float(rates[:n45][self.ftb[s]].mean()) for s in "LR"},
            "dial": float(dial),
            "sim_ms": self.steps * self.dt_ms,
            "ms": {"frontend": round(1000 * (t1 - t0), 1), "brains": round(1000 * (t2 - t1), 1),
                   "own": own},
            "wall_s": time.time() - self.t_wall,
        }

    async def _handler(self, ws) -> None:
        from websockets.exceptions import ConnectionClosed

        try:
            async for msg in ws:
                head, body = unpack(msg if isinstance(msg, bytes) else msg.encode())
                if head.get("type") == "reset":
                    self.reset()
                    await ws.send(pack({"type": "reset.ok"}))
                    continue
                img = np.frombuffer(body, np.uint8)
                if img.size != self.n_px:
                    await ws.send(pack({"type": "error", "msg": f"그림 크기 {img.size}"}))
                    continue
                n = int(round((self.n_px / 12) ** 0.5))
                out = await asyncio.to_thread(
                    self.step, img.reshape(2, 6, n, n), float(head.get("dial", 1.0))
                )
                out["tick"] = head.get("tick")
                await ws.send(pack(out))
        except ConnectionClosed:
            return

    async def serve(self, port: int, stop: asyncio.Event) -> None:
        import websockets

        check_port(port)
        async with websockets.serve(self._handler, "127.0.0.1", port, max_size=2**22):
            await stop.wait()

    def close(self) -> None:
        for p, conn in self.brains.values():
            try:
                conn.send(None)
            except (BrokenPipeError, OSError):
                pass
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()
