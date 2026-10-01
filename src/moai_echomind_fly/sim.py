"""LIF 엔진 (R-3340 F-003 · I §1.3).

Shiu et al. 2024 의 방정식(MIT 참조 구현 philshiu/Drosophila_brain_model 의 model.py)을
PyTorch 로 다시 구현했다. 코드를 복사하지 않았고, 한 스텝의 순서를 Brian2 기본 일정
(groups → thresholds → synapses → resets)과 같게 해서 Brian2 판과 1 스텝 안에서 맞게 했다.

    dv/dt = (v0 − v + g − A) / t_mbr     (unless refractory)
    dg/dt = −g / tau                     (unless refractory)
    dA/dt = −A / tau_A                   (발화 적응 — 기본 꺼짐, Shiu 에 없음)
    발화: v > v_th · 리셋: v = v_reset, g = 0, A += b_adapt · 불응 t_rfc
    연결: 발화가 t_dly 뒤 도착하면 g_post += sign × count × w_syn × w_scale
          — 단, **불응 중인 뉴런에 도착한 입력은 버린다.** Brian2 는 `(unless refractory)` 변수에
            조건부 쓰기(set_conditional_write(not_refractory))를 걸어 적분뿐 아니라
            시냅스 쓰기도 막는다. Shiu 모델이 실제로 이렇게 돈다
            (2026-09-22 Brian2 2.10.1 로 확인 · tests/test_sim_parity.py).
    자극: 포아송 사건마다 v += poisson_kick(Shiu: w_syn × 250) · 자극 뉴런의 불응기 0
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import asdict, dataclass, field, fields
from typing import Any

import numpy as np
import torch

from . import config
from .graph import Graph


@dataclass(frozen=True)
class LIFParams:
    version: str = "sim-v0"
    v0_mV: float = -52.0
    v_reset_mV: float = -52.0
    v_th_mV: float = -45.0
    t_mbr_ms: float = 20.0
    tau_syn_ms: float = 5.0
    t_rfc_ms: float = 2.2
    t_dly_ms: float = 1.8
    w_syn_mV: float = 0.275
    w_scale: float = 1.0
    poisson_kick_mV: float = 68.75
    dt_ms: float = 0.1
    tau_adapt_ms: float = 100.0
    adaptation: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    ipsi_release: tuple[dict[str, Any], ...] = field(default_factory=tuple)

    @classmethod
    def from_config(cls, name: str = "sim", **override: Any) -> LIFParams:
        cfg = config.load(name)
        known = {f.name for f in fields(cls)}
        values = {k: v for k, v in cfg.items() if k in known}
        values["adaptation"] = tuple(cfg.get("adaptation") or ())
        values["ipsi_release"] = tuple(cfg.get("ipsi_release") or ())
        values.update(override)
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["adaptation"] = list(self.adaptation)
        d["ipsi_release"] = list(self.ipsi_release)
        return d


def _decay_mix(dt: float, tau_in: np.ndarray | float, t_m: np.ndarray | float):
    """입력(시간 상수 tau_in)이 막(t_m)으로 새는 정확한 한 걸음 계수. tau_in == t_m 이면 극한."""
    tau_in, t_m = np.asarray(tau_in, dtype=np.float64), np.asarray(t_m, dtype=np.float64)
    a, b = np.exp(-dt / t_m), np.exp(-dt / tau_in)
    same = np.isclose(tau_in, t_m, rtol=0, atol=1e-9)
    diff = np.where(same, 1.0, tau_in - t_m)
    return np.where(same, dt / t_m * a, tau_in / diff * (b - a))


def resolve_device(name: str = "cpu") -> torch.device:
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS 를 쓸 수 없다 — --device cpu 로 돌린다")
    return torch.device(name)


def ipsi_scale(graph: Graph, rule: dict[str, Any]) -> np.ndarray:
    """연결마다 곱할 값 — `rule["class"]` 뉴런에서 **같은 쪽(L/R)** 뉴런으로 가는 연결 × 배율.

    같은 쪽 방출 비대칭(sim-v2 · Gaudry 2013). 연결 수 · 부호 · 상대는 그대로다.
    섞은 배선에도 같은 규칙이 걸린다(출발은 그대로 · 도착이 섞였으므로 "같은 쪽"도 섞인 도착 기준).
    """
    from .loop.signals import side_of  # 늦게 — signals 가 sim 을 부른다

    side = side_of(graph)
    src = graph.src()
    from_class = np.isin(graph.columns["class"][src], list(rule["class"]))
    lr = np.isin(side[src], ["L", "R"])
    same = from_class & lr & (side[src] == side[graph.dst])
    out = np.ones(len(src))
    out[same] = float(rule["ipsi_factor"])
    return out


class Engine:
    """뇌 전체 LIF. `run(ms)` 로 이어 달린다(시각 유지)."""

    def __init__(
        self,
        graph: Graph,
        params: LIFParams,
        *,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        seed: int = 0,
        cell=None,
    ) -> None:
        """`cell`(cellparams.CellParams · sim-v3)이 있으면 종류마다 막 시간 상수 · 문턱과
        종류 쌍 배율을 쓴다. 없으면 지금과 같은 길(스칼라)."""
        if device == "mps" and dtype == torch.float64:
            raise ValueError("MPS 는 float64 를 지원하지 않는다")
        self.graph = graph
        self.p = params
        self.dev = resolve_device(device)
        self.dtype = dtype
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        n = graph.n
        self.n = n

        dt = params.dt_ms
        self.n_dly = int(round(params.t_dly_ms / dt))
        self.n_rfc_default = int(round(params.t_rfc_ms / dt))
        t_m, tau, tau_a = params.t_mbr_ms, params.tau_syn_ms, params.tau_adapt_ms
        a = math.exp(-dt / t_m)
        b = math.exp(-dt / tau)
        a_adapt = math.exp(-dt / tau_a)
        self.k_a = a
        self.k_b = b
        self.k_g = tau / (tau - t_m) * (b - a)
        self.k_adapt = a_adapt
        self.k_A = tau_a / (tau_a - t_m) * (a_adapt - a)
        self.v_th = params.v_th_mV
        if cell is not None:
            cell.check(graph)
            tau_n, vth_n = cell.per_neuron(graph, t_m, params.v_th_mV)

            def tens(x: np.ndarray) -> torch.Tensor:
                return torch.as_tensor(x, dtype=dtype, device=self.dev)

            self.k_a = tens(np.exp(-dt / tau_n))
            self.k_g = tens(_decay_mix(dt, tau, tau_n))
            self.k_A = tens(_decay_mix(dt, tau_a, tau_n))
            self.v_th = tens(vth_n)

        self.row_ptr = torch.as_tensor(graph.row_ptr, dtype=torch.int64, device=self.dev)
        self.dst = torch.as_tensor(graph.dst.astype(np.int64), device=self.dev)
        w = graph.sign[graph.src()].astype(np.float64) * graph.count
        for rule in params.ipsi_release:
            w *= ipsi_scale(graph, rule)
        if cell is not None:
            w *= cell.edge_scale()          # 종류 쌍 배율(α ≥ 0 · 부호 불변)
        self.w = torch.as_tensor(w * params.w_syn_mV * params.w_scale, dtype=dtype, device=self.dev)

        self.n_rfc = torch.full((n,), self.n_rfc_default, dtype=torch.int64, device=self.dev)
        self.b_adapt = torch.zeros(n, dtype=dtype, device=self.dev)
        self.adapt_on = False
        for entry in params.adaptation:
            idx = graph.select(type=entry["type"])
            self.set_adaptation(idx, float(entry["b_adapt_mV"]))

        self.stim_idx = torch.zeros(0, dtype=torch.int64, device=self.dev)
        self.stim_rate = np.zeros(0)
        self.rec_idx = torch.zeros(0, dtype=torch.int64, device=self.dev)
        self.acc_idx = torch.zeros(0, dtype=torch.int64, device=self.dev)
        self.reset()

    # ------------------------------------------------------------------ 설정
    def reset(self) -> None:
        """상태를 쉬는 값으로 되돌린다(자극·기록 설정과 침묵은 유지)."""
        p, n = self.p, self.n
        self.v = torch.full((n,), p.v0_mV, dtype=self.dtype, device=self.dev)
        self.g = torch.zeros(n, dtype=self.dtype, device=self.dev)
        self.A = torch.zeros(n, dtype=self.dtype, device=self.dev)
        self.rfc_until = torch.zeros(n, dtype=torch.int64, device=self.dev)
        self.queue: deque[torch.Tensor] = deque(
            torch.zeros(0, dtype=torch.int64, device=self.dev) for _ in range(self.n_dly)
        )
        self.step = 0
        self._spk_steps: list[np.ndarray] = []
        self._spk_neurons: list[np.ndarray] = []
        self._volt: list[torch.Tensor] = []
        self.acc_sum = torch.zeros(self.acc_idx.numel(), dtype=torch.float64, device=self.dev)
        self.acc_steps = 0

    def set_poisson(self, idx: np.ndarray, rate_hz: np.ndarray | float) -> None:
        """포아송 자극. 자극 뉴런은 불응기 0(Shiu). 빈 배열이면 자극을 끈다."""
        idx = np.asarray(idx, dtype=np.int64)
        rate = np.broadcast_to(np.asarray(rate_hz, dtype=np.float64), idx.shape).copy()
        self.n_rfc[self.stim_idx] = self.n_rfc_default
        self.stim_idx = torch.as_tensor(idx, device=self.dev)
        self.stim_rate = rate
        self.n_rfc[self.stim_idx] = 0

    def set_rates(self, rate_hz: np.ndarray | float) -> None:
        """자극 뉴런은 그대로 두고 발화율만 바꾼다."""
        self.stim_rate = np.broadcast_to(
            np.asarray(rate_hz, dtype=np.float64), self.stim_rate.shape).copy()

    def set_adaptation(self, idx: np.ndarray, b_adapt_mV: float) -> None:
        self.b_adapt[torch.as_tensor(np.asarray(idx, dtype=np.int64), device=self.dev)] = b_adapt_mV
        self.adapt_on = bool((self.b_adapt != 0).any())

    def set_edge_weights(self, edges: np.ndarray, values: np.ndarray) -> None:
        """연결 세기 일부를 바꾼다(가소성 · plasticity.py — KC → MBON 배율만 · 부호 불변)."""
        self.w[torch.as_tensor(np.asarray(edges, dtype=np.int64), device=self.dev)] = (
            torch.as_tensor(np.asarray(values), dtype=self.dtype, device=self.dev))

    def silence(self, idx: np.ndarray) -> None:
        """뉴런의 나가는 연결 세기를 0 으로(뉴런은 남고 영향만 사라진다 — Shiu)."""
        for i in np.asarray(idx, dtype=np.int64):
            self.w[int(self.graph.row_ptr[i]) : int(self.graph.row_ptr[i + 1])] = 0

    def accumulate_voltage(self, idx: np.ndarray) -> None:
        """이 뉴런들의 막전위를 스텝마다 더해 둔다(문턱 아래 신호 — C 단계 채널 벡터)."""
        self.acc_idx = torch.as_tensor(np.asarray(idx, dtype=np.int64), device=self.dev)
        self.acc_sum = torch.zeros(self.acc_idx.numel(), dtype=torch.float64, device=self.dev)
        self.acc_steps = 0

    def take_voltage_mean(self) -> np.ndarray:
        """지난 `take` 이후 평균 막전위(mV)를 돌려주고 비운다."""
        mean = (self.acc_sum / max(self.acc_steps, 1)).cpu().numpy()
        self.acc_sum.zero_()
        self.acc_steps = 0
        return mean

    def spikes_since(self, step: int) -> tuple[np.ndarray, np.ndarray]:
        """스텝 `step` 이후의 발화(창 단위로 읽기)."""
        steps, neurons = [], []
        for s_arr, n_arr in zip(reversed(self._spk_steps), reversed(self._spk_neurons)):
            if s_arr[0] < step:
                break
            steps.append(s_arr)
            neurons.append(n_arr)
        if not steps:
            return np.zeros(0, np.int32), np.zeros(0, np.int32)
        return np.concatenate(steps[::-1]), np.concatenate(neurons[::-1])

    def record_voltage(self, idx: np.ndarray) -> None:
        self.rec_idx = torch.as_tensor(np.asarray(idx, dtype=np.int64), device=self.dev)

    # ------------------------------------------------------------------ 달리기
    def run(self, ms: float, *, kicks: np.ndarray | None = None) -> None:
        """ms 만큼 달린다.

        Args:
            ms: 모의 시간.
            kicks: (시험용) [steps, len(stim_idx)] 불리언 — 주면 포아송 대신 이 사건표를 쓴다.
        """
        steps = int(round(ms / self.p.dt_ms))
        k = len(self.stim_rate)
        if kicks is None:
            prob = self.stim_rate * self.p.dt_ms / 1000.0
            kicks = self.rng.random((steps, k)) < prob if k else np.zeros((steps, 0), bool)
        if kicks.shape != (steps, k):
            raise ValueError(f"kicks 모양 {kicks.shape} ≠ {(steps, k)}")
        kick_t = torch.as_tensor(kicks, device=self.dev)
        kick_mv = self.p.poisson_kick_mV
        rec = self.rec_idx.numel() > 0
        acc = self.acc_idx.numel() > 0
        volt = (torch.empty((steps, self.rec_idx.numel()), dtype=self.dtype, device=self.dev)
                if rec else None)
        p = self.p
        v0, v_th, v_reset = p.v0_mV, self.v_th, p.v_reset_mV
        has_stim = k > 0 and bool(kicks.any())
        for t in range(steps):
            s = self.step
            # 1) 상태 갱신 — 불응 중이 아닌 뉴런만(Shiu 의 unless refractory)
            active = self.rfc_until <= s
            u = self.v - v0
            u_new = u * self.k_a + self.g * self.k_g
            if self.adapt_on:
                u_new = u_new - self.A * self.k_A
                self.A = self.A * self.k_adapt
            self.v = torch.where(active, u_new + v0, self.v)
            self.g = torch.where(active, self.g * self.k_b, self.g)
            # 2) 문턱
            spiked = torch.nonzero((self.v > v_th) & active).squeeze(1)
            # 3) 전달 — t_dly 스텝 전에 발화한 뉴런의 나가는 연결만
            if self.n_dly:
                arriving = self.queue.popleft()
                self.queue.append(spiked)
            else:
                arriving = spiked
            if arriving.numel():
                self._deliver(arriving, active)
            if has_stim:
                row = kick_t[t]
                self.v.index_add_(0, self.stim_idx, row.to(self.dtype) * kick_mv)
            # 4) 리셋
            if spiked.numel():
                self.v[spiked] = v_reset
                self.g[spiked] = 0
                if self.adapt_on:
                    self.A[spiked] += self.b_adapt[spiked]
                self.rfc_until[spiked] = s + self.n_rfc[spiked]
                self._spk_steps.append(np.full(spiked.numel(), s, dtype=np.int32))
                self._spk_neurons.append(spiked.to("cpu").numpy().astype(np.int32))
            if rec:
                volt[t] = self.v[self.rec_idx]
            if acc:
                self.acc_sum += self.v[self.acc_idx].to(torch.float64)
                self.acc_steps += 1
            self.step += 1
        if rec:
            self._volt.append(volt.to("cpu"))

    def _deliver(self, src_idx: torch.Tensor, active: torch.Tensor) -> None:
        starts = self.row_ptr[src_idx]
        counts = self.row_ptr[src_idx + 1] - starts
        total = int(counts.sum())
        if total == 0:
            return
        first = torch.cumsum(counts, 0) - counts
        edges = torch.arange(total, device=self.dev) + torch.repeat_interleave(
            starts - first, counts, output_size=total)
        tgt = self.dst[edges]
        # 불응 중인 도착 뉴런에는 쓰지 않는다(Brian2 조건부 쓰기 — 모듈 설명)
        self.g.index_add_(0, tgt, self.w[edges] * active[tgt].to(self.dtype))

    # ------------------------------------------------------------------결과
    @property
    def t_ms(self) -> float:
        return self.step * self.p.dt_ms

    def spikes(self) -> tuple[np.ndarray, np.ndarray]:
        """(발화 스텝, 뉴런 인덱스) — 지금까지 전부."""
        if not self._spk_steps:
            return np.zeros(0, np.int32), np.zeros(0, np.int32)
        return np.concatenate(self._spk_steps), np.concatenate(self._spk_neurons)

    def voltage(self) -> np.ndarray:
        """기록한 전압 [steps, len(rec_idx)] (mV)."""
        if not self._volt:
            return np.zeros((0, self.rec_idx.numel()))
        return torch.cat(self._volt).numpy()
