"""MCP 도구 서버 (R-3340 F-008 · I §2) — stdio. 도구 본체는 `tools.py`.

클로드 코드에 붙이기:
    claude mcp add fly-connectome -- uv run --directory <저장소> fly-connectome mcp
"""

from __future__ import annotations

from typing import Any

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError as McpToolError

from . import tools

INSTRUCTIONS = """\
수컷 초파리 중추신경계(MaleCNS v1.0, 뉴런 165,122 · 연결 2,556만)를
Shiu 2024 LIF 방정식으로 돌리는 시뮬레이터.
너는 실험자고 초파리 회로는 피험체다. 종류 이름은 find_neurons 로 먼저 찾는다.
run_experiment 는 요약만 돌려준다 — 원본은 runs/<run_id>/spikes.npz 에 있다.
요약의 숫자를 고쳐 쓰지 않는다.
이 모델은 쉬는 상태가 침묵이라 "억제를 푸는" 경로(빛 ON 경로 · 먹기)는 재현하지 못한다.
"""

server = MCPServer("fly-connectome", instructions=INSTRUCTIONS)


def _call(fn, *args: Any, **kwargs: Any) -> Any:
    try:
        return fn(*args, **kwargs)
    except tools.ToolError as err:
        raise McpToolError(err.to_text()) from None


@server.tool()
def find_neurons(query: str, field: str = "type", limit: int = 20) -> list[dict[str, Any]]:
    """뉴런 종류 찾기. field = type | class | superclass (대소문자 무시 · 부분 일치)."""
    return _call(tools.find_neurons, query, field, limit)


@server.tool()
def run_experiment(stimulate: list[str], rate_hz: float = 150.0, duration_ms: int = 200,
                   silence: list[str] | None = None, seed: int = 0,
                   purpose: str = "") -> dict[str, Any]:
    """종류들을 포아송(rate_hz)으로 켜고 뇌 전체를 duration_ms 동안 돌린다.

    silence 는 그 종류의 나가는 연결을 끈다.

    상한(코드로 막음): duration_ms ≤ 1000 · 자극 뉴런 ≤ 3000 · 침묵 ≤ 5000 · rate_hz ≤ 300.
    돌려주는 것: 발화 뉴런 수 · 상위 20 종류 · 하강 뉴런 · 운동 뉴런 · run_id (원본 배열은 파일).
    """
    return _call(tools.run_experiment, stimulate, rate_hz, duration_ms, silence, seed, purpose)


@server.tool()
def get_run(run_id: str) -> dict[str, Any]:
    """지난 실행의 요약."""
    return _call(tools.get_run, run_id)


@server.tool()
def compare_runs(run_a: str, run_b: str) -> dict[str, Any]:
    """두 실행의 종류별 발화 차이(큰 순 20)."""
    return _call(tools.compare_runs, run_a, run_b)


@server.tool()
def list_runs(limit: int = 20) -> list[dict[str, Any]]:
    """최근 실험 목록."""
    return _call(tools.list_runs, limit)


def main() -> None:
    server.run(transport="stdio")
