"""보기 창 뇌 연결(R-1177 · 요구자 10-17 "초파리 움직이게" · 탐색 — 판정 아님).

브라우저(보기 창)가 초파리 뇌 캐릭터 두 눈 큐브맵을 걸음(10 ms)마다 보내면:
  그림 → 앞단 v3b(FrontendV3) → 발화율 × 다이얼(손으로 · 1 = 잰 그대로 · r_max 로 자름)
  → 뇌 둘(진짜 배선 · 섞은 배선 rewired-s0)을 **따로 프로세스에서 나란히** 10 ms 달림
  → 읽기 뉴런 발화를 창(window_ms)으로 셈
  → 진짜 배선 뇌의 **하강 뉴런 그대로** 몸 명령(지어낸 행동 규칙 없음):
     앞뒤 = DNp09(앞) − MDN(뒤) · 돌기 = DNa02 · DNa01 오른 − 왼(같은 쪽으로) · GF(DNp01) = 뛰어오름
     · 발화 → 속도 환산 숫자만 선언 · 상한은 몸 규약.
닿음(10-18 요구자 "머리 털 반응을 보기 창에"): 브라우저가 보낸 닿음(몸 기준 방위 · 틈)
  → 부위 × 좌우 촉각 털 무리(senses.touch · 연결 지도 주석) → 닿아 있는 동안 rate_hz 로 같은 뇌 둘에
  · 머리 털이 켜는 하강 뉴런(몸 규약 밖)은 보기만
  · 몸은 그대로 하강 뉴런 명령(닿음으로 몸을 움직이는 규칙 없음).
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
from ..neuron_names import Names
from ..senses.touch import contact_groups, groups
from .render import pack, unpack
from .safety import check_port


def read_types(motor: dict) -> list[str]:
    """표에 줄로 보일 종류(config/neurons.yaml groups) + 몸을 움직이는 종류(view_brain.motor).

    겹침 없이 · 순서 유지.
    """
    out: list[str] = []
    motor_types = [m["types"] for m in motor.values() if isinstance(m, dict)]
    for v in [*Names().groups().values(), *motor_types]:
        out += [t for t in v if t not in out]
    return out


def labels(g, types: list[str]) -> tuple[np.ndarray, list[str]]:
    """뉴런마다 읽기 줄 번호(없으면 −1) · 줄 이름(종류_눈).

    종류 하나 = 줄 하나(묶어 더하지 않음).
    """
    typ = g.columns["type"].astype(str)
    side = side_of(g)
    lab = np.full(g.n, -1, np.int64)
    names = []
    for t in types:
        for s in ("L", "R"):
            lab[(typ == t) & (side == s)] = len(names)
            names.append(f"{t}_{s}")
    return lab, names


def type_side_ids(g) -> tuple[np.ndarray, np.ndarray]:
    """뉴런마다 (종류 번호 × 2 + 오른쪽이면 1) · 종류 이름 목록.

    프로세스마다 같은 순서(np.unique).
    """
    names, tid = np.unique(g.columns["type"].astype(str), return_inverse=True)
    return tid.astype(np.int64) * 2 + (side_of(g) == "R"), names


def _worker(brain: str, seed: int, entry: np.ndarray, dt_ms: float, threads: int,
            types: list[str], conn) -> None:
    """뇌 하나(프로세스) — 발화율을 받으면 dt 만큼 달린다.

    돌려주는 것: 줄별 발화 수 · 걸린 시간 · 전체 발화 · 하강 뉴런 전체 발화 ·
    입력으로 켠 뉴런을 뺀 (종류 × 쪽) 발화(지금 많이 켜진 종류용).
    """
    import torch

    from ..sim import Engine, LIFParams

    torch.set_num_threads(threads)
    base = graph_mod.load()
    g = graph_mod.brain_graph(base, brain)
    lab, names = labels(base, types)
    dn = base.columns["superclass"].astype(str) == "descending_neuron"
    ts, _ = type_side_ids(base)
    own = np.zeros(base.n, bool)
    own[entry] = True
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
        keys, kc = np.unique(ts[nn[~own[nn]]], return_counts=True)
        conn.send((cnt, time.perf_counter() - t0, len(nn), int(dn[nn].sum()), keys, kc))


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
        self.tcfg = self.cfg["touch"]
        tg = groups(base, ecfg["touch_entry"])
        want = {p for sec in self.tcfg["sectors"] for p in sec["parts"]}
        self.tgroups = {k: v for k, v in tg.items() if k.rsplit("_", 1)[0] in want}
        self.t_entry = np.unique(np.concatenate(list(self.tgroups.values())))
        self.t_pos = {k: np.searchsorted(self.t_entry, v) for k, v in self.tgroups.items()}
        self.n_vis = len(self.fe.entry)
        entry = np.concatenate([self.fe.entry, self.t_entry])
        self.types = read_types(self.cfg["motor"])
        self.names_cfg = Names()
        self.top_n = int(self.names_cfg.cfg["top_n"])
        self.ts, self.type_names = type_side_ids(base)
        sup = base.columns["superclass"].astype(str)
        cls = base.columns["class"].astype(str)
        first = np.unique(self.ts // 2, return_index=True)[1]   # 종류마다 첫 뉴런의 주석
        self.type_meta = {int(t): (sup[i], cls[i]) for t, i in zip(self.ts[first] // 2, first)}
        per = max(1, (os.cpu_count() or 2) // len(self.cfg["brains"]))
        threads = min(int(self.cfg["threads"]), per)
        ctx = mp.get_context("spawn")
        self.brains = {}
        for b in self.cfg["brains"]:
            parent, child = ctx.Pipe()
            p = ctx.Process(
                target=_worker,
                args=(b, int(self.cfg["seed"]), entry, self.dt_ms, threads, self.types, child),
                daemon=True,
            )
            p.start()
            self.brains[b] = (p, parent)
        self.names: list[str] = []
        for _, conn in self.brains.values():
            _, self.names = conn.recv()
        n_win = int(round(float(self.cfg["window_ms"]) / self.dt_ms))
        self.win = {b: deque(maxlen=n_win) for b in self.brains}
        self.win_dn = {b: deque(maxlen=n_win) for b in self.brains}
        self.win_top = {b: deque(maxlen=n_win) for b in self.brains}
        self.body = self.cfg[self.cfg["motor"]["limits"]]   # 10-18: 초파리 크기(fly_body)
        self.steps = 0
        self.t_wall: float | None = None

    def reset(self) -> None:
        self.t_wall = None
        self.steps = 0
        self.fe.reset()
        for _, conn in self.brains.values():
            conn.send("reset")
        for w in [*self.win.values(), *self.win_dn.values(), *self.win_top.values()]:
            w.clear()

    def step(self, img: np.ndarray, dial: float, contacts: list[dict] | None = None) -> dict:
        """그림 [2, 6, N, N] u8 + 닿음 → 뇌 둘 한 걸음 → 창 합 · 몸 명령."""
        t0 = time.time()
        if self.t_wall is None:  # 벽시계는 첫 걸음부터(서버 띄운 때가 아님)
            self.t_wall = t0
        vis = np.minimum(self.fe.step(img) * float(dial), self.r_max)   # 다이얼은 눈 입력만
        touch = np.zeros(len(self.t_entry))
        on = contact_groups(contacts or [], self.tcfg)
        for k in on:
            touch[self.t_pos[k]] = float(self.tcfg["rate_hz"])
        rates = np.concatenate([vis, touch])
        t1 = time.time()
        for _, conn in self.brains.values():
            conn.send(rates)
        sums = {}
        own = {}
        for b, (_, conn) in self.brains.items():
            cnt, sec, n_spk, n_dn, keys, kc = conn.recv()
            self.win[b].append(cnt)
            self.win_dn[b].append(n_dn)
            self.win_top[b].append((keys, kc))
            own[b] = {"ms": round(1000 * sec, 1), "spikes": int(n_spk)}
            sums[b] = dict(zip(self.names, map(int, np.sum(self.win[b], axis=0))))
        t2 = time.time()
        m, r, b = self.cfg["motor"], sums["real"], self.body

        def both(part: str) -> int:
            return sum(r[f"{t}_{s}"] for t in m[part]["types"] for s in "LR")

        def diff(part: str) -> int:
            return sum(r[f"{t}_R"] - r[f"{t}_L"] for t in m[part]["types"])

        speed = (float(m["forward"]["m_s_per_spike"]) * both("forward")
                 - float(m["backward"]["m_s_per_spike"]) * both("backward"))
        speed = float(np.clip(speed, -float(b["back_max_m_s"]), float(b["fwd_max_m_s"])))
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
            "touch": {"on": on, "neurons": int((touch > 0).sum()),
                      "rate_hz": float(self.tcfg["rate_hz"])},
            "dn_all": {b: int(sum(w)) for b, w in self.win_dn.items()},
            "top": self.top(),
            "dial": float(dial),
            "sim_ms": self.steps * self.dt_ms,
            "ms": {"frontend": round(1000 * (t1 - t0), 1), "brains": round(1000 * (t2 - t1), 1),
                   "own": own},
            "wall_s": time.time() - self.t_wall,
        }

    def _window_counts(self, b: str) -> dict[int, int]:
        out: dict[int, int] = {}
        for keys, kc in self.win_top[b]:
            for k, c in zip(keys.tolist(), kc.tolist()):
                out[k] = out.get(k, 0) + c
        return out

    def top(self) -> list[dict]:
        """진짜 배선에서 최근 창에 가장 많이 발화한 종류(입력으로 켠 뉴런 뺌).

        섞은 배선의 같은 종류를 나란히.
        """
        real = self._window_counts("real")
        others = [b for b in self.brains if b != "real"]
        ctl = self._window_counts(others[0]) if others else {}
        per: dict[int, list[int]] = {}
        for k, c in real.items():
            per.setdefault(k // 2, [0, 0, 0, 0])[k % 2] += c
        for k, c in ctl.items():
            if k // 2 in per:
                per[k // 2][2 + k % 2] += c
        rows = sorted(per.items(), key=lambda kv: -(kv[1][0] + kv[1][1]))[: self.top_n]
        out = []
        for t, (lt, rt, cl, cr) in rows:
            sup, cls = self.type_meta.get(t, ("", ""))
            d = self.names_cfg.describe(str(self.type_names[t]), sup, cls)
            out.append({"type": str(self.type_names[t]), "L": lt, "R": rt, "cL": cl, "cR": cr, **d})
        return out

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
                    self.step, img.reshape(2, 6, n, n), float(head.get("dial", 1.0)),
                    list(head.get("contacts") or []),
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
