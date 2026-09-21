"""Qwen 실험자 (R-3340 F-009) — Qwen 이 **우리 MCP 서버에 클라이언트로 붙어** 도구를 스스로 부른다.

클로드 코드가 쓰는 것과 같은 통로(stdio MCP)다.
언어 모델은 실험자, 초파리 회로는 피험체(연구 노트 05 · 07).
기록: runs/<run_id>-experiment/transcript.json — 대화 · 도구 호출 · 도구 결과 · 인용한 run_id 검사.
"""

from __future__ import annotations

import json
import os
import re
import sys
from typing import Any

import anyio
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

from . import config, paths
from .guard import OWNER_ENV, GuardRefused, heavy
from .loop.qwen import QwenLM, parse_tool_calls, text_before_tool_call
from .runs import RunRecorder

RUN_ID_RE = re.compile(r"\d{8}-\d{6}-run-[0-9a-f]{4}")


def _result_text(res) -> str:
    if res.structured_content is not None:
        return json.dumps(res.structured_content, ensure_ascii=False)
    return "".join(getattr(c, "text", "") for c in res.content)


async def _converse(qwen: QwenLM, question: str, rec: RunRecorder, max_turns: int
                    ) -> dict[str, Any]:
    params = StdioServerParameters(
        command=sys.executable, args=["-m", "moai_echomind_fly.cli", "mcp"],
        env={**os.environ, OWNER_ENV: str(os.getpid())})
    transcript: list[dict[str, Any]] = []
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            init = await session.initialize()
            tools = [{"type": "function", "function": {
                "name": t.name, "description": t.description or "", "parameters": t.input_schema}}
                for t in (await session.list_tools()).tools]
            messages: list[dict[str, Any]] = [
                {"role": "system", "content": init.instructions or ""},
                {"role": "user", "content": question}]
            answer = ""
            for turn in range(max_turns):
                gen = qwen.generate(qwen.render(messages, tools=tools))
                calls = parse_tool_calls(gen.text)
                transcript.append({"turn": turn, "assistant": gen.text, "tool_calls": calls,
                                   "prompt_tokens": gen.prompt_tokens, "gen_tokens": gen.gen_tokens,
                                   "prefill_s": round(gen.prefill_s, 2),
                                   "decode_s": round(gen.decode_s, 2)})
                rec.log(f"[turn {turn}] 입력 {gen.prompt_tokens} 토큰 · "
                        f"출력 {gen.gen_tokens} 토큰 · "
                        f"{gen.prefill_s + gen.decode_s:.1f} s · 도구 {[c['name'] for c in calls]}")
                if not calls:
                    answer = gen.text.strip()
                    break
                tool_calls = [{"type": "function", "function": c} for c in calls]
                messages.append({"role": "assistant", "content": text_before_tool_call(gen.text),
                                 "tool_calls": tool_calls})
                for c in calls:
                    res = await session.call_tool(c["name"], c["arguments"])
                    body = _result_text(res)
                    transcript.append({"turn": turn, "tool": c["name"], "arguments": c["arguments"],
                                       "is_error": bool(res.is_error), "result": body})
                    messages.append({"role": "tool", "name": c["name"], "content": body})
    return {"answer": answer, "transcript": transcript}


def _run_id_of(body: str) -> str | None:
    try:
        obj = json.loads(body)
    except ValueError:
        return None
    return obj.get("run_id") if isinstance(obj, dict) else None


def run(question: str) -> dict[str, Any]:
    cfg = config.load("llm")
    try:
        with heavy("qwen-experimenter", extra_gb=float(cfg["weights_gb"])) as guard:
            rec = RunRecorder("experiment", {"question": question, "llm": cfg["label"],
                                             "llm_version": cfg["version"]})
            rec.log(f"[guard] 메모리 여유 {guard.free_percent}% · Qwen 올리는 중")
            qwen = QwenLM(cfg)
            rec.log(f"[qwen] {qwen.label} · 불러오기 {qwen.load_s:.1f} s")
            out = anyio.run(_converse, qwen, question, rec, int(cfg["experimenter_max_turns"]))
            cited = sorted(set(RUN_ID_RE.findall(out["answer"])))
            called = sorted({_run_id_of(t["result"]) for t in out["transcript"]
                             if t.get("tool") == "run_experiment" and not t["is_error"]} - {None})
            missing = [r for r in cited if not (paths.runs_dir() / r).exists()]
            result = {"question": question, "llm": qwen.label, "answer": out["answer"],
                      "runs_called": called, "runs_cited": cited, "cited_missing": missing,
                      "turns": 1 + max((t["turn"] for t in out["transcript"]), default=0)}
            rec.write_json("transcript.json", out["transcript"])
            rec.write_json("summary.json", result)
            rec.log(f"[answer] {out['answer']}")
            rec.log(f"기록: {rec.dir}")
            return result
    except GuardRefused as err:
        return {"error": str(err)}
