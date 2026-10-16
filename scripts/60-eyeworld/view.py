"""보기 창(R-1177 F-008 최소판 · 요구자 10-17 "띄워서 보여줘") — 브라우저 탭으로 장면을 둘러본다.

판정 실행이 아니다(뇌 · 앞단 판정 없음). 하는 일:
① 장면(+ V1 의 사람 · 다가옴 대본)을 runs/eyeworld-view/<장면>.json 으로 적음(자세 · 에셋 주소)
② 초파리가 받는 그림 두 장(낱눈 기둥 밝기 모자이크 · 5° 흐림) — 저장된 렌더 그림에서(새로 안 그림)
③ 정적 서버(127.0.0.1 · 읽기 전용 · 보기 포트 config render.view_port)를 띄우고 기본 브라우저로 엶
④ config render.view_serve_min 분 뒤(또는 Ctrl-C) 서버를 닫음.
사용: python scripts/60-eyeworld/view.py [--scene arena-b] [--frames runs/v1-check-…]
"""

from __future__ import annotations

import argparse
import json
import struct
import subprocess
import time
import zlib
from pathlib import Path

import numpy as np

from moai_echomind_fly import config, paths
from moai_echomind_fly import graph as graph_mod
from moai_echomind_fly.eye.frontend import FrontendV2
from moai_echomind_fly.eyeworld import scene as scene_mod
from moai_echomind_fly.eyeworld.static import StaticServer, mounts
from moai_echomind_fly.eyeworld.world import Entity, body_dir


def png(path: Path, img: np.ndarray) -> None:
    """회색 u8 [H, W](줄 0 = 위) → PNG."""
    h, w = img.shape
    raw = b"".join(b"\x00" + img[r].tobytes() for r in range(h))

    def chunk(t: bytes, d: bytes) -> bytes:
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)

    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )


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


def mosaic(fe: FrontendV2, frame: np.ndarray) -> np.ndarray:
    """두 눈 기둥 밝기(5° 흐림) → 방위 +180~−180 · 고도 +70~−70 의 가장 가까운 기둥 그림(1°)."""
    az = np.arange(180.0, -180.0, -1.0)
    el = np.arange(70.0, -70.0, -1.0)
    A, E = np.meshgrid(az, el)
    d = body_dir(0.0, A, E).reshape(-1, 3)
    dirs, lum = [], []
    for e, s in enumerate("LR"):
        c = fe.cols[s]
        dirs.append(body_dir(0.0, c.az, c.el))
        lum.append(fe.samplers[s].sample(frame[e]))
    dirs_a, lum_a = np.concatenate(dirs), np.concatenate(lum)
    out = np.zeros(len(d))
    for i in range(0, len(d), 4000):
        cos = d[i : i + 4000] @ dirs_a.T
        k = cos.argmax(1)
        near = cos[np.arange(len(k)), k] > np.cos(np.radians(4.0))  # 기둥 간격 약 5° 안
        out[i : i + 4000] = np.where(near, lum_a[k], -1.0)
    img = np.where(out < 0, 0, np.clip(out, 0, 1) * 230 + 25).reshape(A.shape)
    return np.repeat(np.repeat(img.astype(np.uint8), 3, 0), 3, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="arena-b")
    ap.add_argument("--frames", default=config.load("eyeworld")["f0"]["frames_from"])
    a = ap.parse_args()
    rcfg = config.load("eyeworld")["render"]
    out = paths.runs_dir() / "eyeworld-view"
    out.mkdir(parents=True, exist_ok=True)
    world, scene, _ = scene_mod.load(a.scene)
    rest = with_person(world, scene, None)
    poses = [p for p in world.poses() if p["id"] != "person"] + rest
    approach = {}
    for side in ("left", "right"):
        w, sc, _ = scene_mod.load(a.scene)
        approach[side] = with_person(w, sc, side)
    rig = world.eye_rig("fly")
    eye = ((np.asarray(rig[0]["pos"]) + np.asarray(rig[1]["pos"])) / 2).tolist()
    all_assets = scene["assets"]
    ground = dict(scene.get("ground") or {})
    tex = ground.pop("texture", None)
    if tex:
        ground["texture_url"] = f"/assets/{all_assets[tex]['path']}"
    fly = world.ents["fly"]
    flyeye = []
    fr_dir = paths.root() / a.frames
    meta = json.loads((fr_dir / "meta.json").read_text(encoding="utf-8"))
    if meta.get("scene") == a.scene:
        z = np.load(fr_dir / "frames.npz")
        fe = FrontendV2(graph_mod.load(), int(config.load("eyeworld")["v1"]["face_px"]), 10.0)
        for k, (idx, cap) in enumerate([(0, "사람이 왼쪽 6 m(방위 +30°)에 있을 때"),
                                        (-1, "1 초 뒤 — 사람이 왼쪽 1.5 m 까지 달려왔을 때")]):
            f = out / f"{a.scene}-flyeye-{k + 1}.png"
            png(f, mosaic(fe, z["app_left"][idx]))
            flyeye.append({"url": f"/view/{f.name}", "caption": f"{cap} · 그림 {a.frames}"})
    head = {
        "scene": a.scene,
        "version": scene["version"],
        "assets": {k: f"/assets/{v['path']}" for k, v in all_assets.items()
                   if v.get("kind", "model") == "model"},
        "poses": poses,
        "person_rest": rest[0],
        "approach": {**approach, "dt_ms": float(config.load("eyeworld")["v1"]["frame_ms"])},
        "ground": ground,
        "sky": scene.get("sky"),
        "light": scene.get("light"),
        "eye": eye,
        "fly_pos": fly.pos.tolist(),
        "fly_yaw": float(fly.yaw),
        "fly_radius": float(fly.radius),
        "flyeye": flyeye or [{"url": "", "caption": "이 장면의 저장된 그림 없음"}] * 2,
    }
    (out / f"{a.scene}.json").write_text(json.dumps(head, ensure_ascii=False), encoding="utf-8")
    port = int(rcfg["view_port"])
    srv = StaticServer(port=port, table={**mounts(), "/view/": out})
    srv.start()
    url = f"http://127.0.0.1:{port}/web/view.html?scene={a.scene}"
    print(f"[view] {url} · 그림 {[f['url'] for f in flyeye]}", flush=True)
    subprocess.run(["open", url], check=False)
    try:
        time.sleep(60.0 * float(rcfg["view_serve_min"]))
    except KeyboardInterrupt:
        pass
    finally:
        srv.stop()
        print("[view] 서버 닫음", flush=True)


if __name__ == "__main__":
    main()
