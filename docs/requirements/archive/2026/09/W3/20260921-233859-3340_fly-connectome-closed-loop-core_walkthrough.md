# fly-connectome-closed-loop — core 검증 보고 (W) · A 단계 시뮬레이터 코어

## 메타데이터

- **요구사항**: 3340 — [./20260921-233859-3340_fly-connectome-closed-loop_req.md](./20260921-233859-3340_fly-connectome-closed-loop_req.md)
- **대응 P**: [./20260921-233859-3340_fly-connectome-closed-loop-core_dev-plan.md](./20260921-233859-3340_fly-connectome-closed-loop-core_dev-plan.md)
- **관련 I**: [./20260921-233859-3340_fly-connectome-closed-loop_interface-design.md](./20260921-233859-3340_fly-connectome-closed-loop_interface-design.md)
- **작성일**: 2026-09-22 00:23:49
- **결과**: 조건부 통과(잔여 이슈 있음) — A 관문(도망 반사 채점 + 섞은 배선 침묵)은 통과. F-006(Could) 미구현 · 청각 실측이 생체 기록과 다름(5장)
- **저장소 커밋**: `moai-echomind-fly-connectome` `1c58688`

> W는 **대응 P의 구현 완료를 증명하는 유일한 증거**다 (/DEV_PROCESS.md §W).
> 원칙: 서술("테스트했음")은 증거가 아니다 — **로그와 실행 기록만 증거다.**

---

## 1. 변경 파일 목록 (새 저장소 — 전부 추가)

| 파일 | 요약 |
|---|---|
| `pyproject.toml` · `uv.lock` · `.gitignore` · `pyrightconfig.json` | 3.14 · hatchling · ruff 100 · data/runs 제외 |
| `README.md` · `NOTICE` | 용어 · 출처(MaleCNS CC BY 4.0 · Shiu MIT · 인용 수치 출처) |
| `config/sources.yaml` | 공식 파일 3개 · GCS 공식 MD5 |
| `config/sim.yaml` | Shiu 매개변수 + 출처 · **sim-v1**(w_scale 0.25 잠금) |
| `config/loom.yaml` | Turner 2022 상대 세기 13종 · **loom-v1** |
| `config/guard.yaml` | 메모리 여유 40 % · 잠금 파일 |
| `src/moai_echomind_fly/{paths,config,guard,data,graph,sim,protocols,runs,cli}.py` | F-001 · F-002 · F-003 · F-007 · 가드 |
| `src/moai_echomind_fly/bench/{cli,escape,limits}.py` | F-004(채점 · fit-v1 규칙 · 재실행) · F-005 |
| `tests/{conftest,test_data_verify,test_graph_data,test_sim_parity,test_escape_replay,test_guard}.py` | 19 시험 |

---

## 2. 테스트 실행 로그 (원문)

```
$ uv run pytest -v -p no:warnings
tests/test_data_verify.py::test_verify_accepts_matching_file PASSED
tests/test_data_verify.py::test_verify_rejects_corrupted_file PASSED
tests/test_data_verify.py::test_verify_rejects_wrong_size PASSED
tests/test_data_verify.py::test_sources_are_the_three_official_files PASSED
tests/test_escape_replay.py::test_replay_from_record_gives_same_spikes PASSED
tests/test_escape_replay.py::test_score_trials_counts_hits_latency_and_per_cell PASSED
tests/test_escape_replay.py::test_wilson_interval_bounds PASSED
tests/test_graph_data.py::test_counts_match_research_note_08 PASSED
tests/test_graph_data.py::test_kenyon_cells_are_excitatory PASSED
tests/test_graph_data.py::test_weight_is_sign_times_count_csr_sorted PASSED
tests/test_graph_data.py::test_rewired_preserves_in_and_out_degree PASSED
tests/test_guard.py::test_low_memory_refused PASSED
tests/test_guard.py::test_extra_weights_counted PASSED
tests/test_guard.py::test_busy_when_live_process_holds_lock PASSED
tests/test_guard.py::test_stale_lock_is_cleared_and_released PASSED
tests/test_sim_parity.py::test_engine_matches_brian2_shiu_equations PASSED
tests/test_sim_parity.py::test_silent_without_stimulus PASSED
tests/test_sim_parity.py::test_silence_removes_influence PASSED
tests/test_sim_parity.py::test_same_seed_same_spikes_cpu PASSED
============================= 19 passed in 12.60s ==============================

$ uv run ruff check src tests
All checks passed!
```

**Brian2 대조가 처음엔 실패했다**(우리 839 발화 · Brian2 817). 원인을 끝까지 좇았다:
갈라진 첫 뉴런(131번)의 궤적을 보니 불응 중(212~232 스텝)에 도착한 입력을 우리는 쌓았고 Brian2 는 버렸다.
Brian2 는 `(unless refractory)` 변수에 `set_conditional_write(not_refractory)` 를 걸어 **시냅스 쓰기도 막는다**
(`brian2/groups/neurongroup.py:841-848`). 엔진을 그렇게 고친 뒤 **발화 (스텝, 뉴런) 목록이 완전히 같다.**

---

## 3. 모듈 동작 확인 기록

### 3.1 데이터 받기 (F-001) — `runs/download.log`

```
[data] 공식 MD5 일치: body-annotations-male-cns-v1.0-minconf-0.5.feather
[data] 공식 MD5 일치: body-neurotransmitters-male-cns-v1.0.feather
[data] 공식 MD5 일치: connectome-weights-male-cns-v1.0-minconf-0.5.feather
2177e246…a9a3b2  body-annotations-…feather
95c92892…879621  body-neurotransmitters-…feather
e35da783…56afc1  connectome-weights-…feather      ← 대조용 값(mps-malecns-model)과도 같다
```

### 3.2 그래프 (F-002) — `runs/graph.log` · `/usr/bin/time -l`

```
[graph] 주석 211,577 행 → Traced 165,122
[graph] 연결 원본 151,856,684 행 → Traced 사이 25,563,197
       22.44 real        21.72 user         0.62 sys
          3484909568  maximum resident set size
nt_counts: acetylcholine 103718 · glutamate 29296 · gaba 22055 · histamine 5910 · unclear 3100 · missing 502 · dopamine 392 · octopamine 101 · serotonin 48
sign_counts: -1 57261 · 0 4143 · 1 103718 · kenyon_cells 4064 → 전부 +1 · self_loops 101 · 겹친 쌍 0
```

연구 노트 08 과 대조: 전부 일치. 08 이 적지 않았던 **NT 행 없는 502** 와 superclass **"그 밖" 798**(08 은 "약 300")을 다시 셌다 → 연구 노트 고침(4장).

### 3.3 LIF 엔진 속도 (F-003) — 뇌 전체 · 1,000 ms · 다가옴 자극

```
cpu: 1,000 ms 모의 4.3 s · 발화 11,213 · 발화한 뉴런 2,463
mps: 1,000 ms 모의 9.0 s · 발화 11,213 · 발화한 뉴런 2,463
CPU == MPS 발화 기록: True (차이 발화 수: 0 )
```

목표(60 s 이하) 충족. 이 부하(발화가 드문 뇌)에서는 CPU 가 빠르다 → 기본 장치 CPU.

### 3.4 w_scale 맞추기 (fit-v1 — 결과 보기 전에 적은 규칙) · 20 시행

```
w_scale | 통과 | GF 반응 | 반응당 | 잠복 ms | 뇌 전체 Hz
    0.2 | X    | 0.35 | 1.14 | 39.9 | 0.0018
   0.25 | O    | 0.60 | 1.25 | 36.0 | 0.0025
    0.3 | O    | 0.75 | 1.67 | 33.9 | 0.0032
   0.35 | X    | 0.85 | 2.29 | 31.5 | 0.0041
    0.4 | X    | 0.95 | 2.66 | 29.0 | 0.0049
    0.5 | X    | 1.00 | 3.62 | 23.5 | 0.0075
   0.75 | X    | 1.00 | 6.15 | 12.3 | 0.0429
    1.0 | X    | 1.00 | 6.15 | 9.5 | 0.1042
통과: [0.25, 0.3] · 가장 긴 이어진 구간: [0.25, 0.3] · 고른 값: 0.25
```

### 3.5 도망 반사 채점 (F-004 · A 관문) · 40 시행 · sim-v1 · loom-v1

```
[guard] 메모리 여유 83% ≥ 40%
[real] GF 반응 0.65 (95% 0.50–0.78) · 반응당 1.38 · 잠복 33.8 ms · 뇌 전체 0.0027 Hz · 발화 27,135 · 106 s · 20260922-001555-escape-38c9
[rewired] GF 반응 0.00 (95% 0.00–0.09) · 반응당 0.00 · 잠복 nan ms · 뇌 전체 0.0000 Hz · 발화 20,627 · 107 s · 20260922-001742-escape-9016
  통과  gf_spikes_per_response_1_to_2
  통과  gf_response_prob_0.5_to_1
  통과  gf_latency_10_60ms
  통과  cns_quiet
  통과  null_silent
```

FLYCNS(참고): 반응 0.68 · 반응당 1.06 · 잠복 37 ms. 우리는 **GF 발화 적응 없이** 통과(FLYCNS 는 30 mV 적응을 더했다).
섞은 배선에서는 자극 뉴런 밖 발화가 **0** — 신호가 아무 데도 가지 않는다.
⚠️ 반응 확률의 95 % 하한은 0.495 로 문턱 0.5 에 걸친다(채점은 점추정 기준 — FLYCNS 와 같음).

### 3.6 한계 측정 (F-005) — `runs/20260922-002041-limits-267c/summary.json`

| 측정 | 결과 | 생체·예상 |
|---|---|---|
| 광수용체 R1–R6 1,394 개 · 150 Hz · 300 ms | 라미나 L1–L5 · 시각엽 · 그 뒤 **발화 0** | 예상대로(침묵 LIF 는 ON 경로 이중 반전을 못 냄) → 시각 입구 LC/LPLC 결정(R 1.3-1) 확인 |
| 청각 115 개(subclass auditory) · 150 Hz | GF **발화 22** · 최대 탈분극 3.9 mV(평균) | ⚠️ **생체와 다르다**(정답: 문턱 아래) |
| 　└ JO→GF 직접 연결 27 개 끔 | GF 발화 0 · 최대 탈분극 0.9 mV | 이 접촉이 원인 |
| 미각 271 개(labellar bristle · taste peg · pharyngeal sensillum) → MN9 | 발화 0 · 최대 탈분극 0.005 mV | 예상대로 재현 실패(억제를 푸는 경로) |

### 3.7 재실행 (F-007)

`test_replay_from_record_gives_same_spikes` — 실행 기록(request · params · graph)만으로 다시 돌린 발화 = 원래 발화(CPU).

---

## 4. 정본 문서 동기화 내역

| 문서 | 무엇 |
|---|---|
| 연구 노트 `README.md`(당시 `docs/research/fly-connectome/` · 2026-09-23 fly 저장소로 옮김) | 목차 "여섯 가지" → 일곱 가지 |
| 연구 노트 `06` | 판단 주기 "1~2 초" 네 곳 → "수 초(5장)" · §2.2 에 광수용체 경로 실측(발화 0)과 구현 방향 |
| 연구 노트 `08` | "그 밖 약 300" → **798**(빈 칸 516 포함) · NT 행 없는 **502** |
| 연구 노트 `09` · `10` | 청각 → 거대섬유는 "발화"가 아니라 문턱 아래 탈분극 |
| 그림 `03` · `05` | "1~2초" → "수 초" |
| HTML 사본 | `build.py` 로 다시 만듦 — "깨진 링크 없음" |

moai-echomind 의 코드 정본 문서(runtime-architecture 등)는 **불필요** — 이 저장소는 EchoMind 런타임과 연결되지 않는다(R 1.6).

---

## 5. 잔여 이슈

1. **청각 → 거대섬유가 생체 기록과 다르다**(화학 시냅스만의 모델에서 GF 가 발화). 이 접촉은 전기 + 화학 혼합 시냅스인데 EM 에서 화학으로 잡혔다.
   FLYCNS 는 전기 시냅스(합 3 mV)로 바꿔 맞췄다 → **F-006(Could)** 로 남긴다. C 단계의 청각 채널을 쓸 때 이 한계를 정직 목록에 적는다.
2. F-006(전기 시냅스 · TTMn 1:1) 미구현(Could).
3. GF 반응 확률 95 % 하한 0.495(3.5).
4. 편집기(pyright)가 새 가상환경을 못 찾던 경고 → `pyrightconfig.json` 추가로 해소.
