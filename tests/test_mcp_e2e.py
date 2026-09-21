"""F-008: 실제 stdio MCP 클라이언트로 서버를 띄워 도구를 끝까지 부른다(클로드 코드와 같은 규약)."""

from __future__ import annotations

import json
import os
import sys

import anyio
import pytest
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

pytestmark = [pytest.mark.data, pytest.mark.heavy]


def _text(result) -> str:
    return "".join(getattr(c, "text", "") for c in result.content)


def test_stdio_server_runs_an_experiment_end_to_end(tmp_path):
    params = StdioServerParameters(
        command=sys.executable, args=["-m", "moai_echomind_fly.cli", "mcp"],
        env={**os.environ, "FLY_RUNS_DIR": str(tmp_path)})

    async def scenario() -> dict:
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                names = {t.name for t in (await session.list_tools()).tools}
                assert {"find_neurons", "run_experiment", "get_run", "compare_runs",
                        "list_runs"} <= names
                found = await session.call_tool("find_neurons", {"query": "DNp01"})
                assert "DNp01" in _text(found)
                bad = await session.call_tool("run_experiment",
                                              {"stimulate": ["DNp01"], "duration_ms": 5000})
                assert bad.is_error and "over_limit" in _text(bad)
                ok = await session.call_tool("run_experiment", {
                    "stimulate": ["DNp01"], "duration_ms": 100,
                    "purpose": "DNp01 을 켜면 다리 운동뉴런까지 반응하나?"})
                assert not ok.is_error
                return ok.structured_content or json.loads(_text(ok))

    summary = anyio.run(scenario)
    assert (tmp_path / summary["run_id"] / "spikes.npz").exists()
    assert "motor" in summary and "descending" in summary
