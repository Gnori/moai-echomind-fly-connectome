"""F0 — 앞단 v3 만(뇌 없음 · R-1177 · 노트 18 §7).

결과 전 잠금: config/eyeworld.yaml f0(rule f0-a) · frontend_v3.
F0-1 대비 곡선(계산 줄무늬) · F0-2 경기장 가로 대비(보고만) · F0-3 렌더 제자리 돎(V1-b 그림)
· F0-4 좌우 대응(V1-b 사람 다가옴 그림 · 계산 원판). 같은 자극의 v2 값을 나란히 적는다.
그림은 새로 그리지 않는다(V1-b frames.npz · 해시를 그 meta.json 과 대조).
사용: python scripts/60-eyeworld/f0_frontend.py → runs/f0-frontend-<시각>/f0.json
"""

from __future__ import annotations

import hashlib
import json
import time

import numpy as np

from moai_echomind_fly import config, paths
from moai_echomind_fly import graph as graph_mod
from moai_echomind_fly.eye.compound import face_pixel_dirs
from moai_echomind_fly.eye.frontend import FrontendV2, FrontendV3, body_faces, neighbours
from moai_echomind_fly.eyeworld.world import body_dir
from moai_echomind_fly.guard import heavy

CONDS = ["rot-left", "rot-right", "app-left", "app-right"]


def grating(az: np.ndarray, contrast: float, lam: float, speed: float, k: int, dt: float):
    """몸 둘레 세로 사인 줄무늬 — 위에서 보아 시계(방위 줄어듦) · 두 눈 같은 그림 [2, 6, N, N]."""
    lum = 0.5 + 0.5 * contrast * np.sin(2 * np.pi * (az + speed * k * dt / 1000.0) / lam)
    face = np.round(255 * lum).astype(np.uint8)
    return np.stack([face, face])


def means(fe: FrontendV2, vals: np.ndarray) -> dict[str, float]:
    """T4/T5 값 [t45] → 눈 · 종류 · 아형 평균."""
    t = fe.t45
    return {
        f"{s}_{k}{sub}": float(vals[(t.side == s) & (t.kind == k) & (t.sub == sub)].mean())
        for s in "LR"
        for k in ("T4", "T5")
        for sub in "abcd"
    }


def run_grating(fe, az, contrast, lam, speed, t_ms, dt) -> dict[str, float]:
    """줄무늬를 흘려 뒤 절반의 자르기 전 출력(÷ ref) 평균."""
    fe.reset()
    steps = int(t_ms / dt)
    acc = []
    for k in range(steps):
        fe.step(grating(az, contrast, lam, speed, k, dt))
        if k >= steps // 2:
            acc.append(fe.last_raw.copy())
    return means(fe, np.mean(acc, axis=0))


def drews(c: float, c50: float, p: float) -> float:
    return c**p / (c50**p + c**p)


def f0_1(fes: dict, cfg: dict, n: int, dt: float) -> dict:
    d = face_pixel_dirs(body_faces(), n)
    az = np.degrees(np.arctan2(d[..., 0], d[..., 2]))  # + = 왼쪽
    lam, sp, t_ms = float(cfg["lambda_deg"]), float(cfg["speed_deg_s"]), float(cfg["t_ms"])
    dw = cfg["drews_t4"]
    out: dict = {"curve": {}, "dsi": {}}
    for ver, fe in fes.items():
        pref = {}
        for c in cfg["contrasts"]:
            m = run_grating(fe, az, float(c), lam, sp, t_ms, dt)
            pref[float(c)] = (m["R_T4a"] + m["L_T4b"]) / 2
        top = pref[1.0]
        out["curve"][ver] = {
            str(c): {"out": v, "ratio": v / top if top > 0 else 0.0} for c, v in pref.items()
        }
        dsi = {}
        for v in cfg["dsi"]["speeds_deg_s"]:
            m = run_grating(fe, az, float(cfg["dsi"]["contrast"]), lam, float(v), t_ms, dt)
            r = (m["R_T4a"] - m["R_T4b"]) / max(m["R_T4a"] + m["R_T4b"], 1e-12)
            lft = (m["L_T4b"] - m["L_T4a"]) / max(m["L_T4a"] + m["L_T4b"], 1e-12)
            vert = (m["R_T4c"] + m["R_T4d"]) / max(m["R_T4a"] + m["R_T4b"], 1e-12)
            dsi[str(v)] = {"R": r, "L": lft, "R_vert_over_horiz": vert}
        out["dsi"][ver] = dsi
    top_d = drews(1.0, float(dw["c50"]), float(dw["p"]))
    want = {str(float(c)): drews(float(c), float(dw["c50"]), float(dw["p"])) / top_d
            for c in cfg["contrasts"]}
    got = {k: v["ratio"] for k, v in out["curve"]["v3"].items()}
    diffs = {k: got[k] - want[k] for k in want}
    at = cfg["min_ratio_at"]
    dmin = float(cfg["dsi"]["min"])
    dsi_ok = all(min(x["R"], x["L"]) >= dmin for x in out["dsi"]["v3"].values())
    out.update(
        drews_ratio=want,
        diff=diffs,
        pass_curve=all(abs(x) <= float(cfg["max_abs_diff"]) for x in diffs.values()),
        pass_at=got[str(float(at["contrast"]))] >= float(at["ratio"]),
        pass_dsi=dsi_ok,
    )
    out["pass"] = bool(out["pass_curve"] and out["pass_at"] and out["pass_dsi"])
    return out


def f0_2(fe: FrontendV3, frame: np.ndarray, cfg: dict) -> dict:
    """첫 그림의 가로 국소 대비(눈높이 띠) — 보고만."""
    band = float(cfg["band_el_deg"])
    res = {}
    for e, s in enumerate("LR"):
        c = fe.cols[s]
        lum = fe.samplers[s].sample(frame[e])
        mean = np.maximum(fe.weber[s].w @ lum, fe.weber[s].floor)
        nb = neighbours(c)["b"]
        post, ant = nb[0][:, 0], nb[1][:, 0]
        ok = (ant >= 0) & (post >= 0) & (np.abs(c.el) <= band)
        ch = (lum[ant[ok]] - lum[post[ok]]) / (2 * mean[ok])
        web = fe.weber[s](lum)[np.abs(c.el) <= band]
        res[s] = {
            "n_cols": int(ok.sum()),
            "median_abs": float(np.median(np.abs(ch))),
            "rms": float(np.sqrt(np.mean(ch**2))),
            "frac_over_0_08": float(np.mean(np.abs(ch) > 0.08)),
            "weber_median_abs": float(np.median(np.abs(web))),
            "lum_std_all_cols": float(lum.std()),  # V1-b 가 적은 값(하늘/땅 섞임 · 비교 숫자 아님)
        }
    med = float(np.median([res[s]["median_abs"] for s in "LR"]))
    res["median_both"] = med
    res["scene_needed"] = med < float(cfg["scene_needed_below"])
    return res


def subtype_hz(fe: FrontendV2, frames: np.ndarray, warm: int) -> dict[str, float]:
    """그림 차례 → 눈마다 a · b · c · d(T4 + T5 함께) 평균 발화율(Hz · 앞 warm 걸음 뺌)."""
    fe.reset()
    acc = []
    for k, f in enumerate(frames):
        r = fe.step(f)[: len(fe.t45)]
        if k >= warm:
            acc.append(r)
    m = np.mean(acc, axis=0)
    t = fe.t45
    return {
        f"{s}_{sub}": float(m[(t.side == s) & (t.sub == sub)].mean())
        for s in "LR"
        for sub in "abcd"
    } | {"all": float(m.mean()), "frac_over_10hz": float(np.mean(m > 10.0))}


def f0_3(fes: dict, frames: dict, cfg: dict, warm: int) -> dict:
    out: dict = {}
    for ver, fe in fes.items():
        out[ver] = {c: subtype_hz(fe, frames[c], warm) for c in ("rot-left", "rot-right")}
    ok = {}
    for cond, ip, ot in (("rot-left", "R", "L"), ("rot-right", "L", "R")):
        m = out["v3"][cond]
        ok[cond] = bool(
            m[f"{ip}_a"] >= float(cfg["min_hz"])
            and m[f"{ip}_a"] >= float(cfg["min_ratio"]) * m[f"{ip}_b"]
            and m[f"{ot}_b"] > m[f"{ot}_a"]
        )
    out["ok"] = ok
    out["pass"] = all(ok.values())
    return out


def lum_change(fe: FrontendV2, frames: np.ndarray) -> dict[str, float]:
    """눈마다 기둥 밝기 변화 Σ_t Σ_기둥 |L(t) − L(0)|."""
    res = {}
    for e, s in enumerate("LR"):
        base = fe.samplers[s].sample(frames[0][e])
        res[s] = float(sum(np.abs(fe.samplers[s].sample(f[e]) - base).sum() for f in frames[1:]))
    return res


def eye_rates(fe: FrontendV2, frames: np.ndarray) -> dict[str, float]:
    """눈마다 T5 · 소엽 세포 평균 발화율(그림 차례 전체)."""
    fe.reset()
    acc = np.mean([fe.step(f) for f in frames], axis=0)
    n45 = len(fe.t45)
    t, lob = fe.t45, fe.lob
    return {
        f"{s}_T5": float(acc[:n45][(t.side == s) & (t.kind == "T5")].mean()) for s in "LR"
    } | {f"{s}_lobula": float(acc[n45:][lob.side == s].mean()) for s in "LR"}


def f0_4(fes: dict, frames: dict, cfg: dict, n: int) -> dict:
    fe = fes["v3"]
    out: dict = {"render_approach": {}, "disk": {}, "rates": {}}
    for side, cond in (("L", "app-left"), ("R", "app-right")):
        ch = lum_change(fe, frames[cond])
        out["render_approach"][cond] = ch | {"ipsi_share": ch[side] / max(ch["L"] + ch["R"], 1e-12)}
        out["rates"][cond] = {ver: eye_rates(f, frames[cond]) for ver, f in fes.items()}
    faces = body_faces()
    dirs = face_pixel_dirs(faces, n)
    gray = np.full((6, n, n), 128, np.uint8)
    for side, key in (("L", "left"), ("R", "right")):
        az, el = cfg["disk"]["centers_deg"][key]
        cen = body_dir(0.0, np.array(float(az)), np.array(float(el))).reshape(3)
        inside = (dirs @ cen) >= np.cos(np.radians(float(cfg["disk"]["radius_deg"])))
        dk = np.where(inside, 0, 128).astype(np.uint8)
        ch = lum_change(fe, np.stack([np.stack([gray, gray]), np.stack([dk, dk])]))
        out["disk"][key] = ch | {"ipsi_share": ch[side] / max(ch["L"] + ch["R"], 1e-12)}
    shares = [v["ipsi_share"] for v in out["render_approach"].values()] + [
        v["ipsi_share"] for v in out["disk"].values()
    ]
    out["pass"] = bool(all(x >= float(cfg["min_ipsi_share"]) for x in shares))
    return out


def main() -> None:
    ecfg = config.load("eyeworld")
    cfg = ecfg["f0"]
    n, dt = int(cfg["face_px"]), float(cfg["frame_ms"])
    src = paths.root() / cfg["frames_from"]
    out_dir = paths.runs_dir() / f"f0-frontend-{time.strftime('%Y%m%d-%H%M%S')}"
    out_dir.mkdir(parents=True)
    z = np.load(src / "frames.npz")
    frames = {c: z[c.replace("-", "_")] for c in CONDS}
    want = json.loads((src / "meta.json").read_text(encoding="utf-8"))["frames_sha16"]
    got = {c: hashlib.sha256(frames[c].tobytes()).hexdigest()[:16] for c in CONDS}
    if got != want:
        raise SystemExit(f"그림 해시가 다름: {got} 대 {want}")
    res: dict = {
        "rule": cfg["rule"],
        "frontend": ecfg["frontend_v3"]["version"],
        "base": ecfg["frontend"]["version"],
        "frames_from": cfg["frames_from"],
        "frames_sha16": got,
    }
    with heavy("f0-frontend", extra_gb=3.0) as guard:
        print(f"[guard] 메모리 여유 {guard.free_percent}%", flush=True)
        t0 = time.time()
        g = graph_mod.load()
        fes = {"v2": FrontendV2(g, n, dt), "v3": FrontendV3(g, n, dt)}
        v3 = fes["v3"]
        assert isinstance(v3, FrontendV3)
        res["v3_c50"] = v3.c50
        res["ref"] = {k: f.ref for k, f in fes.items()}
        res["setup_s"] = round(time.time() - t0, 1)
        print(f"[f0] 준비 {res['setup_s']} s · c50 {v3.c50:.4g} · ref {res['ref']}", flush=True)
        warm = int(round(float(cfg["warmup_ms"]) / dt))
        res["F0_1"] = f0_1(fes, cfg["f0_1"], n, dt)
        print(f"[f0] F0-1 {res['F0_1']['pass']} · 몫 {res['F0_1']['curve']['v3']}", flush=True)
        res["F0_2"] = f0_2(v3, frames["rot-left"][0], cfg["f0_2"])
        print(f"[f0] F0-2 {res['F0_2']}", flush=True)
        res["F0_3"] = f0_3(fes, frames, cfg["f0_3"], warm)
        print(f"[f0] F0-3 {res['F0_3']['pass']} · v3 {res['F0_3']['v3']}", flush=True)
        res["F0_4"] = f0_4(fes, frames, cfg["f0_4"], n)
        print(f"[f0] F0-4 {res['F0_4']['pass']} · {res['F0_4']}", flush=True)
    res["pass"] = {k: res[k.upper()]["pass"] for k in cfg["pass"]}
    res["all_pass"] = all(res["pass"].values())
    res["cost_s"] = round(time.time() - t0, 1)
    text = json.dumps(res, ensure_ascii=False, indent=2)
    (out_dir / "f0.json").write_text(text, encoding="utf-8")
    print(f"[f0] 판정 {res['pass']} · 모두 {res['all_pass']} · {res['cost_s']} s · {out_dir}")


if __name__ == "__main__":
    main()
