# fly-connectome-closed-loop — tool 검증 보고 (W) · B 단계 도구 서버

## 메타데이터

- **요구사항**: 3340 — [./20260921-233859-3340_fly-connectome-closed-loop_req.md](./20260921-233859-3340_fly-connectome-closed-loop_req.md)
- **대응 P**: [./20260921-233859-3340_fly-connectome-closed-loop-tool_dev-plan.md](./20260921-233859-3340_fly-connectome-closed-loop-tool_dev-plan.md)
- **관련 I**: [./20260921-233859-3340_fly-connectome-closed-loop_interface-design.md](./20260921-233859-3340_fly-connectome-closed-loop_interface-design.md)
- **작성일**: 2026-09-22
- **결과**: 조건부 통과(잔여 이슈 있음) — F-008 · F-009 인수 기준 충족. 클로드 코드 자체에 붙이는 것은 요구자 손(5장)
- **저장소 커밋**: `ec1ba83`(F-008) · `49c4c99`(F-009)

---

## 1. 변경 파일 목록

| 파일 | 변경 | 요약 |
|---|---|---|
| `src/moai_echomind_fly/tools.py` | 추가 | 도구 본체 · 검증 · 상한 · 요약(원본에서 다시 셈) |
| `src/moai_echomind_fly/mcp_server.py` · `.mcp.json` | 추가 | MCP SDK 2.x 서버(stdio) · 프로젝트 등록 |
| `src/moai_echomind_fly/experimenter.py` · `loop/qwen.py` · `config/llm.yaml` | 추가 | Qwen 직접 로드 · 도구 호출 읽기 · 실험자 |
| `config/tool.yaml` | 추가 | 상한 · 허용 superclass |
| `src/moai_echomind_fly/{guard,runs,paths,cli,sim}.py` | 수정 | 부모 잠금 안 실행 · 로그 표준 에러 · `FLY_RUNS_DIR` · 명령 `run`·`mcp`·`experiment` · 창 막전위 평균 |
| `tests/{test_tools,test_mcp_e2e,test_qwen_parse}.py` · `tests/test_guard.py` | 추가·수정 | 13 시험 추가 |

## 2. 테스트 실행 로그 (원문)

```
$ uv run pytest -p no:warnings tests/test_tools.py -k limits -rA
PASSED tests/test_tools.py::test_limits_are_enforced_in_code[kwargs0-over_limit]     # 길이 5000 ms
PASSED tests/test_tools.py::test_limits_are_enforced_in_code[kwargs1-over_limit]     # 1000 Hz
PASSED tests/test_tools.py::test_limits_are_enforced_in_code[kwargs2-over_limit]     # 자극 3,374
PASSED tests/test_tools.py::test_limits_are_enforced_in_code[kwargs3-over_limit]     # 침묵 5,152
PASSED tests/test_tools.py::test_limits_are_enforced_in_code[kwargs4-not_allowed]    # DH44(cb_endocrine)
PASSED tests/test_tools.py::test_limits_are_enforced_in_code[kwargs5-unknown_type]   # DNp1

$ uv run pytest -v -p no:warnings tests/test_tools.py tests/test_mcp_e2e.py tests/test_qwen_parse.py tests/test_guard.py
tests/test_tools.py::test_find_neurons_groups_by_type PASSED
tests/test_tools.py::test_unknown_type_suggests_near_names PASSED
tests/test_tools.py::test_summary_numbers_recount_from_raw PASSED
tests/test_mcp_e2e.py::test_stdio_server_runs_an_experiment_end_to_end PASSED
tests/test_qwen_parse.py::test_parse_tool_call_with_json_and_text_values PASSED
tests/test_qwen_parse.py::test_no_tool_call_means_final_answer PASSED
tests/test_guard.py::test_low_memory_refused PASSED
tests/test_guard.py::test_extra_weights_counted PASSED
tests/test_guard.py::test_busy_when_live_process_holds_lock PASSED
tests/test_guard.py::test_stale_lock_is_cleared_and_released PASSED
tests/test_guard.py::test_child_runs_inside_owner_lock PASSED
============================== 17 passed in 3.17s ==============================

$ uv run pytest            # 저장소 전체
32 passed in 13.59s
$ uv run ruff check src tests
All checks passed!
```

## 3. 모듈 동작 확인 기록

### 3.1 Qwen 가중치 직접 로드 (F-013 첫 인수 기준을 여기서 먼저 쟀다)

```
load 8.1s · peak 28.6 GB · active 28.6 GB
hidden 5120 layers 64 vocab 248320 tie False
tokens 40 max |logit diff| (tokens vs same embeddings): 0.0
```

토큰 대신 **같은 토큰의 임베딩**을 넣으면 로짓이 정확히 같다 → 자리 표시 칸에 초파리 벡터를 끼울 수 있다(C 단계).

### 3.2 Qwen 실험자 (F-009) — `runs/20260922-003320-experiment-1b15/`

질문: *"거대섬유(DNp01)를 켜면 다리 운동 뉴런까지 반응하나? 도구로 실험해서 확인하고, 근거로 쓴 run_id 를 인용해 한국어로 답해줘."*

```
[qwen] Qwen3.8-27B (orcarouter Uncensored MLX 8bit 파생판) · 불러오기 9.1 s
TURN 0 decode 65 tok 10.02 s · prefill 1132 tok 7.44 s   → find_neurons(DNp01) · find_neurons(leg motor)=[]
TURN 1 …  → find_neurons(leg)=[] · find_neurons(motor)
TURN 2 …  → find_neurons(MN) · find_neurons(motor, superclass)
TURN 3 decode 250 tok 25.84 s · prefill 5179 tok 15.85 s → run_experiment(DNp01 150Hz) · run_experiment(DNp01 150Hz, silence DNp01)
TURN 4 … → run_experiment(DNp01 300Hz)
TURN 5 … → find_neurons(TTMn)
TURN 6 decode 512 tok 48.44 s · prefill 6961 tok 17.66 s → 최종 답(512 토큰 상한에서 잘림 → 1024 로 올림)
runs_called = runs_cited = [20260922-003503-run-ce11, 20260922-003505-run-d919, 20260922-003535-run-2f81] · cited_missing = []
```

| Qwen 이 설계한 실험 | 결과(시뮬레이터 요약) |
|---|---|
| DNp01 150 Hz · 200 ms | 운동 뉴런 0 · DNp11 1 회 |
| 대조: 같은 자극 + DNp01 출력 침묵 | 발화 0 |
| DNp01 300 Hz | DNp11 5 회 · TTMn 2 회(첫 80.3 ms) |

**숫자는 요약 그대로 옮겼다. 해석에는 틀린 곳이 있다** — TTMn 을 "경골-대퇴(tibia-trochanter) 운동 뉴런으로 추정"했으나
실제는 도약 근육(tergotrochanteral) 운동 뉴런이다. 원인을 "억제 해제"로 짐작했으나 실제 원인은
GF→TTMn 이 **전기 시냅스(커넥톰에 없음)** 라는 것이다(FLYCNS · W core 5장). "계산은 시뮬레이터, 말은 언어 모델"(연구 노트 05)의 뜻이 여기 있다.

속도: 해독 약 10 tok/s(512 tok / 48.4 s) · 입력 읽기 약 400 tok/s(6,961 tok / 17.7 s).

### 3.3 시나리오를 직접 돌린 결과 (같은 질문 · CLI)

```
run 20260922-002911-run-fa18 · 자극 DNp01 2개 150 Hz · 발화 뉴런 1 · 발화 1
하강: [('DNp11', 1, 48.6)] · 운동: []
```

## 4. 정본 문서 동기화 내역

- 불필요 — EchoMind 런타임과 연결되지 않는 형제 저장소다(R 1.6). 저장소 README 에 `run` · `mcp` · `experiment` 쓰는 법은 C 단계 끝에 한꺼번에 보탠다.

## 5. 잔여 이슈

1. **클로드 코드 자체에 붙여 돌리는 것**(R F-008 마지막 인수 기준)은 이 세션 안에서 새 MCP 서버를 붙일 수 없어 **요구자 손이 필요**하다.
   같은 규약(stdio MCP)을 실제 클라이언트로 끝까지 돌린 E2E 시험(2장 `test_stdio_server_runs_an_experiment_end_to_end`)과 Qwen 실험자(3.2)로 대신 검증했다.
   붙이는 법: 저장소에서 클로드 코드를 열면 `.mcp.json` 이 잡힌다 · 또는 `claude mcp add fly-connectome -- uv run --directory <저장소> fly-connectome mcp`.
2. Qwen 해석 오류(3.2) — 모델의 한계로 기록. 서버 지침에 "요약의 숫자를 고쳐 쓰지 않는다"를 두었고, 숫자는 지켰다.
3. 편집기(pyright)가 MCP 2.x 필드(`input_schema` · `is_error`)를 모른다고 경고 — 편집기 쪽 환경 문제(실행은 2.2.0 에서 확인).
