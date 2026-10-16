"""arena-b 만들기(R-1177 · F0-2 '장면 필요' 뒤 · 결과 전) — arena-a + 눈높이 안쪽 고리.

F0-c(runs/f0-frontend-20261017-002617): 경기장 가로 국소 대비 중앙값 0.0096 — 자연 장면 0.09~0.23
(Dyakova 2015 · 노트 18 §5) 의 약 1/10. 둘레 나무가 12~25 m 에 줄기가 가늘고(5° 해상도에서 지워짐)
바위는 눈(1.5 m)보다 낮다. 그래서 2.5~14 m 에 눈높이 물체를 더한다(에코월드 CC0 식물 · 읽기 전용).
- 다가옴 길(몸 방위 ±30° · 사람 출발 6 m)은 비운다: 방위 차 < corridor_deg 인 자리에는 놓지 않음.
- 큰 소나무(키 20 m · 수관 반지름 약 4 m)는 뒤쪽 반구(|방위| ≥ 100°)에만.
- 잎 텍스처가 Codex 생성인 에셋(oak_mongolian 등)은 쓰지 않는다(잔디 텍스처를 고를 때와 같은 기준).
- 몸에서 min_dist 안 · 다른 물체와 겹침(충돌 반지름 합 + gap)은 다시 뽑음.
목표는 자연 장면 범위이지 그 이상이 아니다(넘치면 그대로 보고).
사용: python scripts/60-eyeworld/make_arena_b.py → config/eyeworld_scenes/arena-b.yaml(덮어씀)
"""

from __future__ import annotations

import json

import numpy as np
import yaml

from moai_echomind_fly.eyeworld import scene as scene_mod

SEED = 11772
CORRIDOR_DEG = 15.0  # 다가옴 방위(±30°)에서 이만큼 안쪽은 비움
MIN_DIST = 2.0
GAP = 0.2
NEW_ASSETS = {
    "evergreen_shrub_mound": "village/foliage/evergreen_shrub_mound",
    "mixed_native_hedge": "village/foliage/mixed_native_hedge",
    "fir_sapling_medium": "village/foliage/fir_sapling_medium",
    "pine_tree_01": "village/foliage/pine_tree_01",
}
LOD = {
    "evergreen_shrub_mound": "lod1",
    "mixed_native_hedge": "lod1",
    "fir_sapling_medium": "lod2",
    "pine_tree_01": "lod3",
}
# (이름 머리, 에셋, 개수, 거리 m, 충돌 반지름 m, 키 m, 뒤쪽만)
GROUPS = [
    ("shrub", "evergreen_shrub_mound", 14, (2.5, 6.0), 0.47, 1.11, False),
    ("hedge", "mixed_native_hedge", 10, (4.0, 8.0), 1.20, 1.60, False),
    ("fir", "fir_sapling_medium", 10, (7.0, 14.0), 0.30, 7.73, False),
    ("pine", "pine_tree_01", 4, (7.0, 12.0), 0.40, 20.4, True),
]


def bearing(x: float, z: float) -> float:
    """몸(yaw 0 · 앞 +z · 왼 +x) 기준 방위 °(+ 왼쪽)."""
    return float(np.degrees(np.arctan2(x, z)))


def main() -> None:
    a = yaml.safe_load(scene_mod.scene_path("arena-a").read_text(encoding="utf-8"))
    root = scene_mod.assets_root()
    assets = dict(a["assets"])
    for key, base in NEW_ASSETS.items():
        rec = json.loads((root / base / "meta.json").read_text(encoding="utf-8"))
        src = rec["source"]
        assets[key] = {
            "path": f"{base}/{LOD[key]}.glb",
            "record": f"{base}/meta.json",
            "license": scene_mod._record_license(rec),
            "source": "에코월드 자체 제작" if src["source"] == "made" else src["source"],
        }
    ents = list(a["entities"])
    placed = [(np.asarray(e["pos"], float)[[0, 2]], float(e.get("radius", 0.35))) for e in ents]
    rng = np.random.default_rng(SEED)
    for head, asset, n, (d0, d1), r, h, back in GROUPS:
        k = 0
        while k < n:
            d = float(rng.uniform(d0, d1))
            th = float(rng.uniform(100.0, 180.0) * rng.choice([-1, 1]) if back else
                       rng.uniform(-180.0, 180.0))
            x, z = d * np.sin(np.radians(th)), d * np.cos(np.radians(th))
            if min(abs(th - 30.0), abs(th + 30.0)) < CORRIDOR_DEG or d < MIN_DIST:
                continue
            p = np.array([x, z])
            if any(np.linalg.norm(p - q) < r + rq + GAP for q, rq in placed):
                continue
            k += 1
            placed.append((p, r))
            ents.append({
                "id": f"{head}{k}",
                "kind": "prop",
                "asset": asset,
                "pos": [round(float(x), 2), 0, round(float(z), 2)],
                "radius": r,
                "height": h,
                "yaw": round(float(rng.uniform(-np.pi, np.pi)), 2),
            })
    out = {
        "version": "scene-arena-b-v1",
        "assets": assets,
        "ground": a["ground"],
        "sky": a["sky"],
        "light": a["light"],
        "entities": ents,
    }
    head = (
        "# arena-b(R-1177 · 10-17 · F0-2 '장면 필요' 뒤 · 결과 전)"
        " — arena-a 그대로 + 눈높이 안쪽 고리.\n"
        "# 만든 것: scripts/60-eyeworld/make_arena_b.py(씨앗 11772 · 다가옴 길 ±15° 비움 ·"
        " 큰 소나무는 뒤쪽만) — 손으로 고치지 않는다.\n"
        "# 에셋 경로는 에코월드 3D 저장소 assets/ 기준(읽기 전용)."
        " 허가는 원본 기록(meta.json)에서 읽은 값.\n"
    )
    text = head + yaml.safe_dump(out, allow_unicode=True, sort_keys=False, width=200)
    scene_mod.scene_path("arena-b").write_text(text, encoding="utf-8")
    world, _, checked = scene_mod.load("arena-b")
    print(f"[arena-b] 개체 {len(world.ents)} · 에셋 {len(checked)} · 허가 "
          f"{sorted({v['license'] for v in checked.values()})}")


if __name__ == "__main__":
    main()
