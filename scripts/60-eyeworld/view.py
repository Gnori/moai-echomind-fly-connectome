"""보기 창(R-1177 F-008 최소판 · 요구자 10-17 "띄워서 보여줘") — 브라우저 탭으로 장면을 둘러본다.

판정 실행이 아니다(뇌 · 앞단 판정 없음). 하는 일:
① 장면(+ V1 의 사람 · 다가옴 대본) · 초파리 뇌 캐릭터 두 눈 자리 ·
   낱눈 기둥 표본 무게(앞단과 같음) · 충돌 원기둥 · 내 캐릭터 시작 자리
   → runs/eyeworld-view/<장면>.json
② 정적 서버(127.0.0.1 · 읽기 전용 · 보기 포트 config render.view_port)를 띄우고 기본 브라우저로 엶
③ 페이지: 내 캐릭터(마음대로 이동) · 초파리 뇌 캐릭터(따로 · 서 있음) ·
   카메라 1인칭 · 3인칭 · 자유 · 초파리 눈으로 · 그 캐릭터 두 눈이 지금 받는 그림(실시간)
④ config render.view_serve_min 분 뒤(또는 Ctrl-C) 서버를 닫음.
사용: python scripts/60-eyeworld/view.py [--scene arena-b]
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import subprocess
import threading
import time

import numpy as np

from moai_echomind_fly import config, paths
from moai_echomind_fly import graph as graph_mod
from moai_echomind_fly.eye.frontend import FrontendV2
from moai_echomind_fly.eyeworld import scene as scene_mod
from moai_echomind_fly.eyeworld.render import texture_index
from moai_echomind_fly.eyeworld.static import StaticServer, mounts
from moai_echomind_fly.eyeworld.world import Entity, body_dir


def with_person(world, scene: dict, side: str | None) -> list[dict]:
    """V1 과 같은 사람 · 다가옴 대본(v1.approach) — side 가 None 이면 서 있는 자리만."""
    ap = config.load("eyeworld")["v1"]["approach"]
    scene["assets"][ap["asset"]] = {"path": ap["asset_path"], "license": ap["license"],
                                    "source": ap["source"]}
    scene_mod.verify_assets({"assets": {ap["asset"]: scene["assets"][ap["asset"]]}})
    fly = world.ents["fly"]
    rig = world.eye_rig("fly")
    mid = (np.asarray(rig[0]["pos"]) + np.asarray(rig[1]["pos"])) / 2
    b = float(ap["bearings_deg"][side or "left"])
    start = mid + float(ap["start_m"]) * body_dir(fly.yaw, np.array(b), np.array(0.0)).reshape(3)
    start[1] = 0.0
    script = (
        {"kind": "approach", "target": "fly", "speed": float(ap["speed_m_s"])} if side else None
    )
    world.ents["person"] = Entity(id="person", kind="person", asset=ap["asset"], pos=start,
                                  yaw=0.0, radius=0.3, height=1.8, solid=False, script=script)
    if side is None:
        return [p for p in world.poses() if p["id"] == "person"]
    dt = float(config.load("eyeworld")["v1"]["frame_ms"]) / 1000.0
    n = int(round(float(ap["window_ms"]) / (dt * 1000.0)))
    traj = []
    for _ in range(n):
        world.step(dt, None)
        traj.append(next(p for p in world.poses() if p["id"] == "person"))
    return traj


def me_start(world, vcfg: dict, colliders: list[dict], body_r: float) -> dict:
    """내 캐릭터 시작 — 초파리 뇌 캐릭터 몸 기준 (방위 · 거리) · 그쪽을 바라봄 · 물체와의 틈."""
    fly = world.ents["fly"]
    d = body_dir(fly.yaw, np.array(float(vcfg["me_start_bearing_deg"])), np.array(0.0)).reshape(3)
    pos = fly.pos + float(vcfg["me_start_dist_m"]) * d
    pos[1] = 0.0
    to = fly.pos - pos
    others = [*colliders, {"x": float(fly.pos[0]), "z": float(fly.pos[2]), "r": float(fly.radius)}]
    gap = min(float(np.hypot(pos[0] - c["x"], pos[2] - c["z"])) - c["r"] - body_r
              for c in others)
    return {"pos": [float(x) for x in pos], "yaw": float(np.arctan2(to[0], to[2])),
            "clearance_m": round(gap, 2)}


def neuron_rows() -> dict[str, list[dict]]:
    """표 묶음마다 줄(종류 하나 = 줄 하나).

    실제 이름 + 쉬운 이름 · 설명 · 근거 · 확인 수준(config/neurons.yaml).
    """
    from moai_echomind_fly.neuron_names import Names

    n = Names()
    return {g: [{"type": t, **n.describe(t)} for t in ts] for g, ts in n.groups().items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="arena-b")
    ap.add_argument("--no-open", action="store_true", help="브라우저를 열지 않음(먼저 확인할 때)")
    ap.add_argument("--brain", action="store_true", help="초파리 뇌 연결(뇌 둘 · 무거운 잠금)")
    a = ap.parse_args()
    ecfg = config.load("eyeworld")
    rcfg, vcfg = ecfg["render"], ecfg["view"]
    out = paths.runs_dir() / "eyeworld-view"
    out.mkdir(parents=True, exist_ok=True)
    world, scene, _ = scene_mod.load(a.scene)
    rest = with_person(world, scene, None)
    poses = [p for p in world.poses() if p["id"] != "person"] + rest
    approach = {}
    for side in ("left", "right"):
        w, sc, _ = scene_mod.load(a.scene)
        approach[side] = with_person(w, sc, side)
    fly = world.ents["fly"]
    rig = world.eye_rig("fly")
    eye = ((np.asarray(rig[0]["pos"]) + np.asarray(rig[1]["pos"])) / 2).tolist()
    all_assets = scene["assets"]
    ground = dict(scene.get("ground") or {})
    tex = ground.pop("texture", None)
    if tex:
        ground["texture_url"] = f"/assets/{all_assets[tex]['path']}"
    n = int(ecfg["v1"]["face_px"])
    fe = FrontendV2(graph_mod.load(), n, 10.0)
    sampler = {}
    for s in ("L", "R"):  # 낱눈 기둥마다 큐브맵 픽셀 무게(앞단과 같은 표본 · 5° 흐림)
        smp, c = fe.samplers[s], fe.cols[s]
        sampler[s] = {
            "k": int(smp.idx.shape[1]),
            "idx": smp.idx.reshape(-1).astype(int).tolist(),
            "w": np.round(smp.w.reshape(-1), 5).tolist(),
            "az": np.round(c.az, 2).tolist(),
            "el": np.round(c.el, 2).tolist(),
        }
    colliders = [  # 놓인 물체만(초파리 뇌 캐릭터 · 내 캐릭터는 페이지가 움직이며 서로 밀어냄)
        {"x": float(e.col_center()[0]), "z": float(e.col_center()[2]), "r": float(e.radius)}
        for e in world.ents.values()
        if e.kind == "prop" and e.solid
    ]
    b = ecfg["world"]["body"]
    head = {
        "scene": a.scene,
        "version": scene["version"],
        "assets": {k: f"/assets/{v['path']}" for k, v in all_assets.items()
                   if v.get("kind", "model") == "model"},
        "texture_index": texture_index(),
        "poses": poses,
        "person_rest": rest[0],
        "approach": {**approach, "dt_ms": float(ecfg["v1"]["frame_ms"])},
        "avatar_asset": ecfg["v1"]["approach"]["asset"],
        "ground": {**ground, "size_m": float(ground.get("size_m", 60.0))},
        "sky": scene.get("sky"),
        "light": scene.get("light"),
        "eye": eye,
        "rig": rig,
        "face_px": n,
        "sampler": sampler,
        "eye_every_ms": float(vcfg["eye_every_ms"]),
        "fly_pos": fly.pos.tolist(),
        "fly_yaw": float(fly.yaw),
        "fly_radius": float(ecfg["view_brain"]["fly_body"]["radius_m"]),
        "colliders": colliders,
        "me_start": me_start(world, vcfg, colliders, float(b["radius_m"])),
        "body": {"walk_m_s": float(b["walk_max_m_s"]), "run_m_s": float(b["run_max_m_s"]),
                 "radius_m": float(b["radius_m"]), "eye_sep_m": float(b["eye_sep_m"]),
                 "turn_max_deg_s": float(b["turn_max_deg_s"]),
                 "eye_m": float(b["height_m"]) * float(b["eye_frac"])},
        "brain": None,
        "fly_body": ecfg["view_brain"]["fly_body"],   # 10-18 초파리 크기(보기 창만)
        "eye_rgb": [float(x) for x in vcfg["eye_rgb"]],
        "view_marker": {k: float(v) for k, v in vcfg["marker"].items()},
    }
    link = None
    guard = contextlib.ExitStack()
    if a.brain:  # 뇌 연결(탐색 · 판정 아님) — 뇌 둘을 프로세스로 · 무거운 잠금(한 번에 하나)
        from moai_echomind_fly.eyeworld.brainlink import BrainLink
        from moai_echomind_fly.guard import heavy

        extra = float(ecfg["view_brain"]["extra_gb"])
        rep = guard.enter_context(heavy("view-brain", extra_gb=extra))
        print(f"[guard] 메모리 여유 {rep.free_percent}%", flush=True)

        vb = ecfg["view_brain"]
        link = BrainLink()
        head["brain"] = {"port": int(vb["port"]), "dt_ms": float(vb["dt_ms"]),
                         "dial": vb["dial"], "motor": vb["motor"],
                         "brains": list(vb["brains"]), "window_ms": float(vb["window_ms"]),
                         "touch": vb["touch"],
                         "neurons": neuron_rows()}
    (out / f"{a.scene}.json").write_text(json.dumps(head, ensure_ascii=False), encoding="utf-8")
    port = int(rcfg["view_port"])
    srv = StaticServer(port=port, table={**mounts(), "/view/": out})
    srv.start()
    stop = None
    if link is not None:
        loop = asyncio.new_event_loop()
        stop = asyncio.Event()
        threading.Thread(target=loop.run_forever, daemon=True).start()
        asyncio.run_coroutine_threadsafe(link.serve(head["brain"]["port"], stop), loop)
    url = f"http://127.0.0.1:{port}/web/view.html?scene={a.scene}"
    print(f"[view] {url} · 물체 {len(poses)} · 기둥 L {len(sampler['L']['az'])}"
          f" · R {len(sampler['R']['az'])} · 내 시작 틈 {head['me_start']['clearance_m']} m",
          flush=True)
    if not a.no_open:
        subprocess.run(["open", url], check=False)
    try:
        time.sleep(60.0 * float(rcfg["view_serve_min"]))
    except KeyboardInterrupt:
        pass
    finally:
        srv.stop()
        if link is not None:
            link.close()
        guard.close()
        print("[view] 서버 닫음", flush=True)


if __name__ == "__main__":
    main()
