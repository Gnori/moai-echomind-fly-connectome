"""V1 렌더러 재기(R-1177 · P render Task 2 · 3 · 판정 아님).

실리나 · 같은 상태 → 같은 그림 · 방향 · 비용을 본다. 경기장 arena-a 를 싣고,
왼눈 기준 (방위 +40°, 고도 +10°) · 오른눈 기준 (−60°, −5°) 6 m 에 빛나는 표지를 둔다.
면 크기 32 · 64 로 세 번씩 그려 해시가 같은지 · 가장 밝은 픽셀 방향과 표지 방향의 각 오차 ·
한 번 시간 ·
크롬 메모리(우리 그룹 RSS 합)를 잰다. 면 그림을 PNG 로 남긴다(사람 확인용).
사용: python scripts/60-eyeworld/v1_render_probe.py [--gpu] → runs/eyeworld-render-<시각>/probe.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import subprocess
import time
import zlib
from pathlib import Path

import numpy as np

from moai_echomind_fly.eye.compound import brightest_dir, face_pixel_dirs
from moai_echomind_fly.eyeworld import scene as scene_mod
from moai_echomind_fly.eyeworld.render import Renderer
from moai_echomind_fly.eyeworld.world import body_dir
from moai_echomind_fly.guard import heavy

MARKERS = [("L", 40.0, 10.0), ("R", -60.0, -5.0)]
DIST_M = 6.0


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


def group_rss_gb(pgid: int) -> float:
    out = subprocess.run(["ps", "-A", "-o", "pgid=,rss="], capture_output=True, text=True).stdout
    kb = sum(
        int(r) for g, r in (ln.split() for ln in out.splitlines() if ln.strip()) if int(g) == pgid
    )
    return kb / 1024 / 1024


def angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    return float(
        np.degrees(np.arccos(np.clip(a @ b / np.linalg.norm(a) / np.linalg.norm(b), -1, 1)))
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", action="store_true", help="WebGPU(기본 WebGL2)")
    a = ap.parse_args()
    world, scene, checked = scene_mod.load("arena-a")
    rig = world.eye_rig("fly")
    yaw = world.ents["fly"].yaw
    markers, expect = [], {}
    for side, az, el in MARKERS:
        eye = next(r for r in rig if r["side"] == side)
        d = body_dir(yaw, np.array(az), np.array(el)).reshape(3)
        markers.append(
            {"pos": list(np.asarray(eye["pos"]) + DIST_M * d), "radius": 0.25, "color": [1, 1, 1]}
        )
        expect[side] = d
    res: dict = {"scene": scene["version"], "assets": checked, "markers": MARKERS, "runs": []}
    with heavy("eyeworld-render", extra_gb=5.0) as guard:
        print(f"[guard] 메모리 여유 {guard.free_percent}%", flush=True)
        r = Renderer(gpu=a.gpu)
        try:
            t0 = time.time()
            ready = r.start()
            res["ready"] = {**ready, "s": round(time.time() - t0, 1)}
            print(f"[probe] ready {ready} · {time.time() - t0:.1f} s", flush=True)
            t0 = time.time()
            sr = r.load_scene(scene, world.poses(), markers)
            res["scene_ready"] = {**sr, "s": round(time.time() - t0, 1)}
            print(f"[probe] scene {sr} · {time.time() - t0:.1f} s", flush=True)
            for n in (32, 64):
                hashes, times = [], []
                for k in range(3):
                    t0 = time.time()
                    img, head = r.render(k, world.poses(), rig, n)
                    times.append(round(time.time() - t0, 3))
                    hashes.append(hashlib.sha256(img.tobytes()).hexdigest()[:16])
                errs = {}
                for e, eye in enumerate(rig):
                    dirs = face_pixel_dirs(eye["faces"], n)
                    errs[eye["side"]] = angle_deg(brightest_dir(img[e], dirs), expect[eye["side"]])
                for e, eye in enumerate(rig):
                    for f, face in enumerate(eye["faces"]):
                        png(
                            r.run_dir / f"face{n}-{eye['side']}-{face['name']}.png", img[e, f][::-1]
                        )
                run = {
                    "face_px": n,
                    "hash": hashes,
                    "same": len(set(hashes)) == 1,
                    "sec": times,
                    "page_render_ms": head["render_ms"],
                    "page_read_ms": head["read_ms"],
                    "marker_err_deg": errs,
                    "chrome_rss_gb": round(group_rss_gb(r.chrome.pid), 2),
                }
                res["runs"].append(run)
                print(
                    f"[probe] {n}px · 같음 {run['same']} · 시간 {times} s"
                    f" · 페이지 {head['render_ms']} ms(읽기 {head['read_ms']})"
                    f" · 표지 오차 {errs} · 크롬 {run['chrome_rss_gb']} GB",
                    flush=True,
                )
        finally:
            changed = r.stop()
            res["readonly_changed"] = changed
    (r.run_dir / "probe.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print(f"[probe] {r.run_dir}")


if __name__ == "__main__":
    main()
