"""F-009: Qwen3.8 채팅 틀의 도구 호출 형식 읽기(가중치 없이 도는 시험)."""

from __future__ import annotations

from moai_echomind_fly.loop.qwen import parse_tool_calls, text_before_tool_call

SAMPLE = """먼저 DNp01 을 켜 보겠습니다.
<tool_call>
<function=run_experiment>
<parameter=stimulate>
["DNp01"]
</parameter>
<parameter=duration_ms>
200
</parameter>
<parameter=purpose>
다리 운동뉴런 반응 확인
</parameter>
</function>
</tool_call>"""


def test_parse_tool_call_with_json_and_text_values():
    calls = parse_tool_calls(SAMPLE)
    assert calls == [{"name": "run_experiment", "arguments": {
        "stimulate": ["DNp01"], "duration_ms": 200, "purpose": "다리 운동뉴런 반응 확인"}}]
    assert text_before_tool_call(SAMPLE) == "먼저 DNp01 을 켜 보겠습니다."


def test_no_tool_call_means_final_answer():
    assert parse_tool_calls("DNp01 을 켜도 운동 뉴런은 반응하지 않았다.") == []
