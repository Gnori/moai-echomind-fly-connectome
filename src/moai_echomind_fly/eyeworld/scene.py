"""장면 불러오기 — config/eyeworld_scenes/<이름>.yaml → World(설계 문서 I 2.7).

에셋 허가는 **원본 기록**(에코월드 assets 의 meta.json · source.json)에서 다시 읽어
장면에 적힌 값과 맞춰 본다(ASSETS.json 의 허가 칸은 틀려 있어 믿지 않는다 — 조사 10-16).
기록의 허가 칸이 없으면 장면의 `source` 설명이 있어야 하고,
허가가 승인 목록(설정 world.allowed_licenses) 밖이면 거부한다.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from .. import config, paths
from .safety import SafetyError, repo_path
from .world import Body, Entity, World


def scene_path(name: str) -> Path:
    return paths.root() / "config" / "eyeworld_scenes" / f"{name}.yaml"


def assets_root() -> Path:
    return repo_path(config.load("eyeworld")["render"]["world3d_root"]) / "assets"


def _record_license(rec: dict) -> str | None:
    for k in ("license",):
        if k in rec:
            return str(rec[k])
    src = rec.get("source")
    if isinstance(src, dict) and "license" in src:
        return str(src["license"])
    return None


def verify_assets(scene: dict, root: Path | None = None) -> dict[str, dict]:
    """에셋마다 파일 · 원본 기록 허가를 확인 → {키: {path, license, record_license, sha16}}."""
    root = assets_root() if root is None else root
    allowed = set(config.load("eyeworld")["world"]["allowed_licenses"])
    out = {}
    for key, a in scene.get("assets", {}).items():
        f = root / a["path"]
        if not f.is_file():
            raise SafetyError(f"에셋 {key} 파일 없음: {f}")
        rec_lic = None
        if a.get("record"):
            rec = json.loads((root / a["record"]).read_text(encoding="utf-8"))
            rec_lic = _record_license(rec)
        if rec_lic is not None and rec_lic != str(a["license"]):
            raise SafetyError(f"에셋 {key} 허가 불일치: 장면 {a['license']} · 기록 {rec_lic}")
        if rec_lic is None and not a.get("source"):
            raise SafetyError(f"에셋 {key} 허가 기록 없음 · 장면 source 설명도 없음")
        if str(a["license"]) not in allowed:
            raise SafetyError(f"에셋 {key} 허가 {a['license']} 는 승인 목록 밖")
        h = hashlib.sha256(f.read_bytes()).hexdigest()[:16]
        out[key] = {
            "path": a["path"],
            "license": a["license"],
            "record_license": rec_lic,
            "source": a.get("source"),
            "sha16": h,
        }
    return out


def load(name: str, seed: int = 0, check_assets: bool = True) -> tuple[World, dict, dict]:
    """장면 → (World, 장면 사전, 에셋 확인 결과)."""
    scene = yaml.safe_load(scene_path(name).read_text(encoding="utf-8"))
    checked = verify_assets(scene) if check_assets else {}
    bcfg = config.load("eyeworld")["world"]["body"]
    ents = []
    for e in scene["entities"]:
        body = Body(**bcfg) if e.get("body") == "default" else None
        ents.append(
            Entity(
                id=e["id"],
                kind=e["kind"],
                asset=e.get("asset"),
                pos=np.asarray(e["pos"], dtype=float),
                yaw=float(e.get("yaw", 0.0)),
                radius=float(e.get("radius", body.radius_m if body else 0.3)),
                height=float(e.get("height", body.height_m if body else 1.0)),
                scale=float(e.get("scale", 1.0)),
                solid=bool(e.get("solid", True)),
                script=dict(e["script"]) if e.get("script") else None,
                anim=e.get("anim"),
                body=body,
                col_offset=tuple(e.get("col_offset", (0.0, 0.0))),
            )
        )
    return World(ents, seed=seed), scene, checked
