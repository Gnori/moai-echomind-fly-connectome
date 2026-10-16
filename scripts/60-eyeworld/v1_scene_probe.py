"""V1 장면 진단(10-17 · 판정 아님) — 사람이 그림에 나오나 · 잔디 텍스처 · 몸 돎의 그림 변화.

V1-a 에서 사람 다가옴 그림이 걸음 사이 변화 0 이었다. 사람을 몸 앞 3 m 에 세워 렌더러 진단(info)과
있을 때 · 없을 때 그림 차이를 본다. 왼눈 앞 · 아래 면을 PNG 로 남긴다.
사용: python scripts/60-eyeworld/v1_scene_probe.py → runs/eyeworld-render-<시각>/scene-probe.json
"""

from __future__ import annotations

import json

import numpy as np
from v1_render_probe import png

from moai_echomind_fly import config
from moai_echomind_fly.eyeworld import scene as scene_mod
from moai_echomind_fly.eyeworld.render import Renderer
from moai_echomind_fly.eyeworld.world import Command, Entity
from moai_echomind_fly.guard import heavy


def main() -> None:
    v1 = config.load("eyeworld")["v1"]
    ap = v1["approach"]
    world, scene, _ = scene_mod.load(v1["scene"])
    scene["assets"][ap["asset"]] = {
        "path": ap["asset_path"],
        "license": ap["license"],
        "source": ap["source"],
    }
    world.ents["person"] = Entity(
        id="person",
        kind="person",
        asset=ap["asset"],
        pos=np.array([0.0, 0.0, 3.0]),
        yaw=np.pi,
        radius=0.3,
        height=1.8,
        solid=False,
    )
    res: dict = {}
    with heavy("eyeworld-scene-probe", extra_gb=5.0):
        r = Renderer()
        try:
            res["ready"] = r.start()
            res["scene"] = r.load_scene(scene, world.poses())
            res["info"] = r.info("person")
            n = 32
            near, _ = r.render(0, world.poses(), world.eye_rig("fly"), n)
            far_poses = [
                dict(p, pos=[0.0, 0.0, -500.0]) if p["id"] == "person" else p for p in world.poses()
            ]
            far, _ = r.render(1, far_poses, world.eye_rig("fly"), n)
            res["near_minus_far_mean_abs"] = float(np.abs(near.astype(float) - far).mean())
            res["near_minus_far_front_L"] = float(
                np.abs(near[0, 0].astype(float) - far[0, 0]).mean()
            )
            for name, img in (("near", near), ("far", far)):
                png(r.run_dir / f"probe-{name}-L-front.png", img[0, 0][::-1])
                png(r.run_dir / f"probe-{name}-L-down.png", img[0, 5][::-1])
            seq = []
            for k in range(10):
                world.step(0.01, {"fly": Command(turn=90.0)})
                img, _ = r.render(2 + k, far_poses, world.eye_rig("fly"), n)
                seq.append(img.astype(float))
            res["rotation_step_mean_abs"] = float(np.mean(np.abs(np.diff(np.stack(seq), axis=0))))
        finally:
            res["readonly_changed"] = r.stop()
    (r.run_dir / "scene-probe.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                k: res[k]
                for k in (
                    "scene",
                    "info",
                    "near_minus_far_mean_abs",
                    "near_minus_far_front_L",
                    "rotation_step_mean_abs",
                )
            },
            ensure_ascii=False,
            default=str,
        )
    )
    print(f"[scene-probe] {r.run_dir}")


if __name__ == "__main__":
    main()
