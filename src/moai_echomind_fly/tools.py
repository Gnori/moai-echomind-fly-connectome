"""도구 본체 (R-3340 F-008 · I §1.5 · §2) — MCP 서버와 Qwen 실험자가 같은 함수를 부른다.

원칙(연구 노트 05): 계산은 시뮬레이터가, 말은 언어 모델이. 언어 모델에는 **요약만** 준다 —
원본 배열(수십만 뉴런)은 `runs/<run_id>/spikes.npz` 에 남는다.
상한은 지침이 아니라 이 코드가 막는다.
"""

from __future__ import annotations

import difflib
import json
from collections import Counter
from functools import lru_cache
from typing import Any

import numpy as np

from . import config, paths
from . import graph as graph_mod
from .guard import GuardRefused, heavy
from .runs import RunRecorder
from .sim import Engine, LIFParams

SUPERCLASS_DESCENDING = "descending_neuron"
SUPERCLASSES_MOTOR = ("vnc_motor", "cb_motor")


class ToolError(Exception):
    """I §2.2 의 에러. 메시지 첫 줄이 코드다."""

    def __init__(self, code: str, message: str, **detail: Any) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.detail = detail

    def to_text(self) -> str:
        extra = json.dumps(self.detail, ensure_ascii=False) if self.detail else ""
        return f"{self.code}\n{self}\n{extra}".rstrip()


@lru_cache(maxsize=1)
def _graph(name: str) -> graph_mod.Graph:
    return graph_mod.load(name)


def _cfg() -> dict[str, Any]:
    return config.load("tool")


def _types_index(g: graph_mod.Graph) -> dict[str, np.ndarray]:
    order = np.argsort(g.columns["type"], kind="stable")
    types = g.columns["type"][order]
    uniq, start = np.unique(types, return_index=True)
    bounds = list(start[1:]) + [len(order)]
    return {t: order[s:e] for t, s, e in zip(uniq, start, bounds) if t}


@lru_cache(maxsize=1)
def _index(name: str) -> dict[str, np.ndarray]:
    return _types_index(_graph(name))


def _resolve_types(names: list[str], what: str) -> np.ndarray:
    g_name = _cfg()["graph"]
    index = _index(g_name)
    out = []
    for name in names:
        if name not in index:
            near = difflib.get_close_matches(name, list(index), n=5, cutoff=0.5)
            raise ToolError("unknown_type", f"{what}: 없는 종류 이름 '{name}'", suggestions=near)
        out.append(index[name])
    return np.unique(np.concatenate(out)) if out else np.zeros(0, dtype=np.int64)


def find_neurons(query: str, field: str = "type", limit: int = 20) -> list[dict[str, Any]]:
    """주석 칸(type · class · superclass)으로 뉴런 종류를 찾는다(대소문자 무시 · 부분 일치)."""
    if field not in ("type", "class", "superclass"):
        raise ToolError("over_limit", "field 는 type · class · superclass 중 하나", allowed=[
            "type", "class", "superclass"])
    g = _graph(_cfg()["graph"])
    col = g.columns[field]
    q = query.lower()
    hit = np.flatnonzero(np.char.find(np.char.lower(col.astype(str)), q) >= 0)
    groups: dict[str, list[int]] = {}
    for i in hit:
        groups.setdefault(g.columns["type"][i] or "(이름 없음)", []).append(int(i))
    rows = []
    for typ, idx in sorted(groups.items(), key=lambda kv: -len(kv[1]))[: max(1, min(limit, 100))]:
        signs = Counter(int(s) for s in g.sign[idx])
        rows.append({"type": typ, "neurons": len(idx),
                     "superclass": Counter(g.columns["superclass"][idx]).most_common(1)[0][0],
                     "class": Counter(g.columns["class"][idx]).most_common(1)[0][0],
                     "sign": signs.most_common(1)[0][0]})
    return rows


def _validate(stimulate: list[str], rate_hz: float, duration_ms: int, silence: list[str]
              ) -> tuple[np.ndarray, np.ndarray]:
    cfg = _cfg()
    if not stimulate:
        raise ToolError("over_limit", "자극할 종류가 없다")
    if not (0 < duration_ms <= cfg["max_duration_ms"]):
        raise ToolError("over_limit", "duration_ms 범위 밖", max=cfg["max_duration_ms"])
    if not (0 < rate_hz <= cfg["max_rate_hz"]):
        raise ToolError("over_limit", "rate_hz 범위 밖", max=cfg["max_rate_hz"])
    stim = _resolve_types(stimulate, "stimulate")
    sil = _resolve_types(silence, "silence")
    g = _graph(cfg["graph"])
    allowed = set(cfg["allowed_superclasses"])
    bad = sorted({str(t) for t, sc in zip(g.columns["type"][stim], g.columns["superclass"][stim])
                  if sc not in allowed})
    if bad:
        raise ToolError("not_allowed", "자극 허용 목록 밖의 종류", types=bad,
                        allowed_superclasses=sorted(allowed))
    if len(stim) > cfg["max_stim_neurons"]:
        raise ToolError("over_limit", f"자극 뉴런 {len(stim)} 개", max=cfg["max_stim_neurons"])
    if len(sil) > cfg["max_silence_neurons"]:
        raise ToolError("over_limit", f"침묵 뉴런 {len(sil)} 개", max=cfg["max_silence_neurons"])
    return stim, sil


def run_experiment(stimulate: list[str], rate_hz: float = 150.0, duration_ms: int = 200,
                   silence: list[str] | None = None, seed: int = 0, purpose: str = ""
                   ) -> dict[str, Any]:
    """종류를 포아송으로 켜고(Shiu 방식) 뇌 전체를 돌린 뒤 요약을 돌려준다."""
    silence = list(silence or [])
    stim, sil = _validate(list(stimulate), float(rate_hz), int(duration_ms), silence)
    cfg = _cfg()
    g = _graph(cfg["graph"])
    params = LIFParams.from_config()
    request = {"stimulate": list(stimulate), "rate_hz": rate_hz, "duration_ms": duration_ms,
               "silence": silence, "seed": seed, "purpose": purpose}
    try:
        with heavy("tool-run") as guard:
            rec = RunRecorder("run", request)
            rec.log(f"[guard] 메모리 여유 {guard.free_percent}%")
            eng = Engine(g, params, seed=seed)
            eng.set_poisson(stim, float(rate_hz))
            if len(sil):
                eng.silence(sil)
            eng.run(float(duration_ms))
            rec.record_setup(eng, g, {"sim": params.version, "tool": cfg["version"]})
            rec.record_spikes(eng, stim_idx=stim, silence_idx=sil)
            rec.honesty(
                wiring=["자극한 종류에서 뇌 전체로 퍼진 발화 전부"],
                added=[f"자극 = 포아송 {rate_hz} Hz(Shiu 2024 방식 · 사건마다 68.75 mV)",
                       f"w_scale={params.w_scale} ({params.version})",
                       *(["침묵 = 나가는 연결 세기 0"] if silence else [])])
            summary = summarize(rec.run_id)
            rec.write_json("summary.json", summary)
            rec.log(f"[run] {rec.run_id} · 발화 {summary['total_spikes']:,} · {rec.wall_s:.1f} s")
    except GuardRefused as err:
        raise ToolError(err.code, err.message) from None
    return summary


def _run_dir(run_id: str):
    d = paths.runs_dir() / run_id
    if not run_id or "/" in run_id or ".." in run_id or not (d / "spikes.npz").exists():
        raise ToolError("unknown_run", f"없는 run_id '{run_id}'")
    return d


def summarize(run_id: str) -> dict[str, Any]:
    """`spikes.npz` 에서 다시 센 요약(I §1.5). 모든 숫자는 원본에서 나온다."""
    d = _run_dir(run_id)
    req = json.loads((d / "request.json").read_text())
    prm = json.loads((d / "params.json").read_text())
    gname = json.loads((d / "graph.json").read_text())["name"]
    g = _graph(gname)
    cfg = _cfg()
    with np.load(d / "spikes.npz") as z:
        step, neuron, dt = z["step"], z["neuron"], float(z["dt_ms"])
        stim = z["stim_idx"] if "stim_idx" in z else np.zeros(0, np.int64)
        sil = z["silence_idx"] if "silence_idx" in z else np.zeros(0, np.int64)
    duration = float(req["duration_ms"])
    stim_mask = np.zeros(g.n, bool)
    stim_mask[stim] = True
    non_stim = ~stim_mask[neuron]
    types = g.columns["type"]
    sc = g.columns["superclass"]

    def per_type(sel: np.ndarray, limit: int) -> list[dict[str, Any]]:
        spk_types = types[neuron[sel]]
        counts = Counter(spk_types.tolist())
        rows = []
        for typ, n_spk in counts.most_common(limit):
            m = sel & (types[neuron] == typ)
            idx_typ = np.flatnonzero(types == typ)
            rows.append({"type": typ or "(이름 없음)", "superclass": str(sc[idx_typ[0]]),
                         "neurons_fired": int(len(np.unique(neuron[m]))),
                         "neurons_total": int(len(idx_typ)), "spikes": int(n_spk),
                         "first_spike_ms": round(float(step[m].min() * dt), 2)})
        return rows

    is_desc = sc[neuron] == SUPERCLASS_DESCENDING
    is_motor = np.isin(sc[neuron], SUPERCLASSES_MOTOR)
    return {
        "run_id": run_id,
        "duration_ms": duration,
        "dt_ms": dt,
        "graph": gname,
        "params_version": prm["versions"]["sim"],
        "stimulated": {"types": req["stimulate"], "neurons": int(len(stim)),
                       "rate_hz": req["rate_hz"]},
        "silenced": {"types": req.get("silence", []), "neurons": int(len(sil))},
        "spiking_neurons": int(len(np.unique(neuron[non_stim]))),
        "total_spikes": int(non_stim.sum()),
        "stimulated_spikes": int((~non_stim).sum()),
        "population_rate_hz": round(float(non_stim.sum() / max(g.n - len(stim), 1)
                                          / (duration / 1000.0)), 6),
        "top_types": per_type(non_stim, cfg["top_types"]),
        "descending": per_type(non_stim & is_desc, cfg["max_descending"]),
        "motor": per_type(non_stim & is_motor, cfg["max_motor"]),
        "baseline": "silent network: 0 spikes without stimulus (tests/test_sim_parity.py)",
        "raw": f"runs/{run_id}/spikes.npz",
        "honesty": _honesty_lines(d),
    }


def _honesty_lines(d) -> list[str]:
    f = d / "honesty.md"
    if not f.exists():
        return []
    lines = f.read_text(encoding="utf-8").splitlines()
    return [ln for ln in lines if ln and not ln.startswith("# ")]


def get_run(run_id: str) -> dict[str, Any]:
    d = _run_dir(run_id)
    f = d / "summary.json"
    return json.loads(f.read_text()) if f.exists() else summarize(run_id)


def compare_runs(run_a: str, run_b: str, limit: int = 20) -> dict[str, Any]:
    a, b = get_run(run_a), get_run(run_b)
    ca = _type_spikes(run_a)
    cb = _type_spikes(run_b)
    rows = [{"type": t, "spikes_a": ca.get(t, 0), "spikes_b": cb.get(t, 0),
             "diff": cb.get(t, 0) - ca.get(t, 0)} for t in set(ca) | set(cb)]
    rows.sort(key=lambda r: -abs(r["diff"]))
    return {"run_a": run_a, "run_b": run_b, "total_a": a["total_spikes"],
            "total_b": b["total_spikes"], "types": rows[:limit]}


def _type_spikes(run_id: str) -> dict[str, int]:
    d = _run_dir(run_id)
    g = _graph(json.loads((d / "graph.json").read_text())["name"])
    with np.load(d / "spikes.npz") as z:
        neuron = z["neuron"]
        stim = z["stim_idx"] if "stim_idx" in z else np.zeros(0, np.int64)
    neuron = neuron[~np.isin(neuron, stim)]
    return dict(Counter(g.columns["type"][neuron].tolist()))


def list_runs(limit: int = 20) -> list[dict[str, Any]]:
    rows = []
    for d in sorted(paths.runs_dir().glob("*-run-*"), reverse=True)[:limit]:
        s = d / "summary.json"
        if not s.exists():
            continue
        summ = json.loads(s.read_text())
        rows.append({"run_id": d.name, "stimulate": summ["stimulated"]["types"],
                     "duration_ms": summ["duration_ms"], "total_spikes": summ["total_spikes"]})
    return rows
