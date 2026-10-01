"""회로 학습 2 단계 — 함수형 LIF + 희소 대리 기울기 + 잘림 BPTT + 체크포인트(F-006 · I v2 §7.2).

`sim.Engine` 과 **같은 순서**(갱신 → 문턱 → 지연 도착 전달 → 포아송 → 리셋)를 제자리 쓰기 없이
다시 적어 autograd 가 지나가게 했다. 배우는 값은 `cellparams.CellParams` 와 같다
(종류별 tau · 문턱 · 쌍 배율 α).

- 정방향은 정확하다: 발화 = 계단 함수 그대로 → **엔진과 발화가 같다**(시험 · float64).
- 뒤로는 문턱 근처에서 빠른 시그모이드 대리 도함수(Zenke & Ganguli 2018). 리셋 경로는 끊는다.
- **기울기 범위(grad_scope)**: 범위 안 뉴런만 대리 기울기를 쓰고, 범위 밖 뉴런의 상태는 걸음마다
  끊는다 — 대리 기울기 계산은 범위 뉴런 수에 비례(뇌 전체 18 만에 비례하지 않게).
- 전달은 엔진처럼 **도착한 발화의 나가는 연결만**(걸음마다 연결 전체를 곱하지 않는다 · I v2 금지).
- 체크포인트: `checkpoint_steps` 걸음 묶음마다 다시 계산(√T 기억). 잘림은 창 사이에서 `detach`.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.utils.checkpoint import checkpoint

from . import cellparams as cp
from .graph import Graph
from .sim import Engine, LIFParams


class _Spike(torch.autograd.Function):
    """앞: x > 0 → 1 · 뒤: 1 / (1 + β|x|)²."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, beta: float) -> torch.Tensor:
        ctx.save_for_backward(x)
        ctx.beta = beta
        return (x > 0).to(x.dtype)

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        (x,) = ctx.saved_tensors
        return grad / (1.0 + ctx.beta * x.abs()) ** 2, None


def _mix(dt: float, tau_in: float, t_m: torch.Tensor) -> torch.Tensor:
    """sim._decay_mix 의 torch 판(미분 가능 · 두 시간 상수가 같으면 극한)."""
    a = torch.exp(-dt / t_m)
    b = float(np.exp(-dt / tau_in))
    same = (t_m - tau_in).abs() < 1e-9
    denom = torch.where(same, torch.ones_like(t_m), tau_in - t_m)
    return torch.where(same, dt / t_m * a, tau_in / denom * (b - a))


class FuncLIF:
    """미분 가능한 뇌 — 매개변수 `tau` · `vth` · `alpha`(잎 텐서 · requires_grad)."""

    def __init__(self, graph: Graph, lif: LIFParams, cell: cp.CellParams, *,
                 stim_idx: np.ndarray, readout_idx: np.ndarray,
                 grad_scope: np.ndarray | None = None, beta: float = 1.0,
                 dtype: torch.dtype = torch.float64, device: str = "cpu") -> None:
        if lif.adaptation:
            raise ValueError("적응(adaptation)은 함수형 걸음에 아직 없다 — sim 설정에 적응이 있다")
        cell.check(graph)
        base = Engine(graph, lif, dtype=dtype, device=device)   # 연결 세기(ipsi · w_scale 포함)
        self.g, self.p, self.dt, self.dtype, self.dev = graph, lif, lif.dt_ms, dtype, base.dev
        self.n = graph.n
        self.n_dly = base.n_dly
        self.row_ptr, self.dst = base.row_ptr, base.dst
        self.w_base = base.w.detach().clone()
        self.pair_id = torch.as_tensor(cell.pair_id.astype(np.int64), device=self.dev)
        self.in_pair = self.pair_id >= 0
        k = cell.neuron_type(graph)
        self.ntype = torch.as_tensor(np.where(k >= 0, k, len(cell.types)), device=self.dev)
        self.tau = torch.tensor(cell.tau_ms, dtype=dtype, device=self.dev, requires_grad=True)
        self.vth = torch.tensor(cell.v_th_mV, dtype=dtype, device=self.dev, requires_grad=True)
        self.alpha = torch.tensor(cell.alpha, dtype=dtype, device=self.dev, requires_grad=True)
        self.k_b = float(np.exp(-self.dt / lif.tau_syn_ms))
        n_rfc = np.full(self.n, int(round(lif.t_rfc_ms / self.dt)), dtype=np.int64)
        n_rfc[np.asarray(stim_idx, dtype=np.int64)] = 0          # 자극 뉴런 불응 0(엔진과 같음)
        self.n_rfc = torch.as_tensor(n_rfc, device=self.dev)
        self.stim_idx = torch.as_tensor(np.asarray(stim_idx, dtype=np.int64), device=self.dev)
        self.readout = torch.as_tensor(np.asarray(readout_idx, dtype=np.int64), device=self.dev)
        scope = np.zeros(self.n, bool)
        if grad_scope is not None:
            scope[np.asarray(grad_scope, dtype=np.int64)] = True
        self.scope = torch.as_tensor(scope, device=self.dev)
        self.scope_idx = torch.nonzero(self.scope).squeeze(1)
        self.beta = float(beta)

    # ------------------------------------------------------------- 매개변수 → 뉴런 · 연결
    def coefs(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        p = self.p
        tau_all = torch.cat([self.tau, torch.tensor([p.t_mbr_ms], dtype=self.dtype,
                                                    device=self.dev)])[self.ntype]
        vth_all = torch.cat([self.vth, torch.tensor([p.v_th_mV], dtype=self.dtype,
                                                    device=self.dev)])[self.ntype]
        k_a = torch.exp(-self.dt / tau_all)
        k_g = _mix(self.dt, p.tau_syn_ms, tau_all)
        a_e = torch.where(self.in_pair, self.alpha[self.pair_id.clamp(min=0)],
                          torch.ones((), dtype=self.dtype, device=self.dev))
        return k_a, k_g, vth_all, self.w_base * a_e

    # ------------------------------------------------------------- 상태
    def initial_state(self) -> tuple[torch.Tensor, ...]:
        v = torch.full((self.n,), self.p.v0_mV, dtype=self.dtype, device=self.dev)
        g = torch.zeros(self.n, dtype=self.dtype, device=self.dev)
        rfc = torch.zeros(self.n, dtype=torch.int64, device=self.dev)
        buf = [torch.zeros(self.n, dtype=self.dtype, device=self.dev) for _ in range(self.n_dly)]
        return (v, g, rfc, torch.zeros((), dtype=torch.int64), *buf)

    @staticmethod
    def detach(state: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
        """잘림 BPTT — 창 사이에서 기울기 길을 끊는다(값은 그대로)."""
        return tuple(t.detach() for t in state)

    # ------------------------------------------------------------- 걸음
    def _steps(self, kicks: torch.Tensor, coefs, raster: list | None, *state):
        v, g, rfc, step = state[0], state[1], state[2], state[3]
        buf = list(state[4:])
        k_a, k_g, vth, w = coefs
        p = self.p
        counts = torch.zeros(len(self.readout), dtype=self.dtype, device=self.dev)
        for t in range(kicks.shape[0]):
            s = int(step) + t
            active = rfc <= s
            u_new = (v - p.v0_mV) * k_a + g * k_g
            v = torch.where(active, u_new + p.v0_mV, v)
            g = torch.where(active, g * self.k_b, g)
            x = v - vth
            hard = (x > 0) & active
            spk = hard.to(self.dtype)                                    # 범위 밖은 기울기 없음
            if self.scope_idx.numel():
                si = self.scope_idx
                sur = _Spike.apply(x[si], self.beta) * active[si].to(self.dtype)
                spk = spk.index_put((si,), sur)
            if self.n_dly:
                arriving_val = buf.pop(0)
                buf.append(spk)
            else:
                arriving_val = spk
            src = torch.nonzero(arriving_val.detach() > 0).squeeze(1)
            if src.numel():
                starts = self.row_ptr[src]
                cnt = self.row_ptr[src + 1] - starts
                total = int(cnt.sum())
                if total:
                    first = torch.cumsum(cnt, 0) - cnt
                    edges = torch.arange(total, device=self.dev) + torch.repeat_interleave(
                        starts - first, cnt, output_size=total)
                    tgt = self.dst[edges]
                    val = (w[edges] * torch.repeat_interleave(arriving_val[src], cnt,
                                                              output_size=total)
                           * active[tgt].to(self.dtype))
                    g = g.index_add(0, tgt, val)
            if kicks.shape[1]:
                v = v.index_add(0, self.stim_idx, kicks[t].to(self.dtype) * p.poisson_kick_mV)
            v = torch.where(hard, torch.full_like(v, p.v_reset_mV), v)        # 리셋 경로는 끊음
            g = torch.where(hard, torch.zeros_like(g), g)
            rfc = torch.where(hard, s + self.n_rfc, rfc)
            if self.scope_idx.numel():                                  # 범위 밖 상태 끊기
                v = torch.where(self.scope, v, v.detach())
                g = torch.where(self.scope, g, g.detach())
            counts = counts + spk[self.readout]
            if raster is not None:
                idx = torch.nonzero(hard).squeeze(1)
                if idx.numel():
                    raster.append((np.full(idx.numel(), s, np.int32), idx.cpu().numpy()))
        step = step + kicks.shape[0]
        return (counts, v, g, rfc, step, *buf)

    def run(self, state: tuple[torch.Tensor, ...], kicks: np.ndarray, *,
            checkpoint_steps: int = 0, record: bool = False
            ) -> tuple[tuple[torch.Tensor, ...], torch.Tensor, tuple[np.ndarray, np.ndarray]]:
        """`kicks`[걸음, 자극 수] 만큼 달린다 → (새 상태, 읽기 뉴런 발화 수, 발화 기록)."""
        kt = torch.as_tensor(np.asarray(kicks, dtype=bool), device=self.dev)
        coefs = self.coefs()
        raster: list | None = [] if record else None
        total = torch.zeros(len(self.readout), dtype=self.dtype, device=self.dev)
        chunk = checkpoint_steps or kt.shape[0]
        for a in range(0, kt.shape[0], chunk):
            part = kt[a:a + chunk]
            if checkpoint_steps and torch.is_grad_enabled() and not record:
                out = checkpoint(self._steps, part, coefs, None, *state, use_reentrant=False)
            else:
                out = self._steps(part, coefs, raster, *state)
            total = total + out[0]
            state = tuple(out[1:])
        if raster:
            steps = np.concatenate([r[0] for r in raster])
            neurons = np.concatenate([r[1] for r in raster]).astype(np.int32)
        else:
            steps, neurons = np.zeros(0, np.int32), np.zeros(0, np.int32)
        return state, total, (steps, neurons)

    def params(self) -> list[torch.Tensor]:
        return [self.tau, self.vth, self.alpha]

    def to_cell(self, template: cp.CellParams) -> cp.CellParams:
        from dataclasses import replace

        return replace(template, tau_ms=self.tau.detach().cpu().numpy().copy(),
                       v_th_mV=self.vth.detach().cpu().numpy().copy(),
                       alpha=self.alpha.detach().clamp(min=0).cpu().numpy().copy(),
                       meta=dict(template.meta))


def rule() -> dict[str, Any]:
    from . import config

    return dict(config.load("fit")["sgbptt"])
