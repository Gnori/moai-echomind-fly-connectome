# fly-tiered-control — loop 검증 보고 (W)

## 메타데이터

- **요구사항**: 1027 — [./20261001-070937-1027_fly-tiered-control_req.md](./20261001-070937-1027_fly-tiered-control_req.md)
- **대응 P**: [./20261001-070937-1027_fly-tiered-control-loop_dev-plan.md](./20261001-070937-1027_fly-tiered-control-loop_dev-plan.md)
- **관련 I**: 생략(R 3.3 축약 — 한 모듈 loop + 설정 파일 하나)
- **작성일**: 2026-10-14(실행 · 판정은 2026-10-01 ~ 10-03)
- **범위**: P Task 1~5 = F-001(층 구조 · 게이트) · F-002(빠른 정책) · F-003(호출 조건 · 문턱 잠금) · F-004(성적표 scorecard-v4) · F-005(1층 반사) · F-007(조사 반영 control-v2 · fast-v2)
- **결과**: **조건부 통과** — 층 구조로 **Qwen 호출 −82 % · 에피소드 벽시계 −51 %** 이고 먹이 도착은 늘 부름보다 낮아지지 않았다(0.575 대 0.350).
  그러나 ① **같은 호출률 무작위 · 주기와 구별되지 않는다**(0.575 대 0.550 · p 0.51 · 0.50) → "호출을 줄였다"까지만 · "언제 부를지를 잘 골랐다"는 쓰지 않는다
  ② 진짜 대 재배선(층 구조 쌍) 순위 100 % · **p 0.12 → 못 넘음**(R 2.0-6 그대로) ③ 1층 반사 규칙은 잠근 설정에서 꺼 두어(`reflex_threat_min: null`) 성적표에서 한 번도 쓰이지 않았다.

> 원칙: 서술은 증거가 아니다 — 아래 로그 · 파일 · sha256 만 증거다. 파일은 `runs/`(git 밖 · 백업 `~/Backups/moai-echomind-fly-connectome`).
> ⚠️ 투영기는 train-v6 — R-7469 씨앗 100 성적표에서 대조군을 이긴 판이라 **결과를 보고 고른 것**이다.

---

## 1. 변경 파일 목록

| 파일 | 변경 유형 | 요약 |
|---|---|---|
| `src/moai_echomind_fly/loop/tiered.py` | 추가 | `Gate`(조건 다섯 · 분위수 문턱 · 최소 간격 · 냄새 맡기 동안 안 부름 · `conditions`) · `TieredPolicy`(반사 → 빠른 정책 → 사건이면 Qwen) · always · never · gated · random · periodic 모드 · 참값을 쓰지 않음 |
| `src/moai_echomind_fly/loop/fast_policy.py` | 추가 | fast-v1 → **fast-v2**(능선 회귀 · 돌기 셋 · 걷기 · 위험 머리 · 확신도 · 에피소드 단위 맞춤 70 % / 보정 30 % · 전방 모형 놀람 · `simulate` · `calibrate`) |
| `src/moai_echomind_fly/loop/scorecard.py` | 수정 | scorecard-v4 — 층 구성(`-gated` · `fast-only` · `-random-matched` · `-periodic-matched`) · 결정 기록 칸 추가만(`tier` · `trigger` · `fast_conf` · 신호 · `z` · `fast_action`) · `gating_verdict` · **바탕 판정에서 층 구성 빼기**(`9df30d1`) |
| `config/control.yaml` | 추가 | control-v1 → **control-v2** · 문턱 잠금(결과 전 `8756f86`) |
| `src/moai_echomind_fly/cli.py` | 수정 | `fast-policy fit | eval | calibrate` |
| `scripts/40-eval/scorecard-tiered.sh` · `scripts/50-analysis/useful_calls.py` | 추가 | 층 구성 성적표 · 쓸모 있는 호출 분석 |
| `src/moai_echomind_fly/dashboard.py`(doc 세션) | 수정 | 에피소드 쪽 층 · 계기 표시 |
| `tests/test_tiered.py` | 추가 | 아래 2장 |

커밋: `1729afa`(틀 · 빠른 정책) · `e0cf3ee`(scorecard-v4) · `3c482f6`(control-v2 · fast-v2 — 연구 노트 15 `ded1099` 반영) · **`8756f86`(문턱 잠금 · 결과 전)** · `db7a48d`(스크립트 · 쓸모 있는 호출) · `2d96a1c`(재기) · `9df30d1`(판정 고침 · 결과) · `2adb2b5`(대시보드 · doc 세션).

---

## 2. 테스트 실행 로그 (원문 · 2026-10-14)

```
$ uv run ruff check src tests
All checks passed!

$ uv run pytest -q -m "not heavy"
197 passed, 3 deselected, 18 warnings in 13.45s

$ uv run pytest -q -m "not heavy" tests/test_tiered.py tests/test_dashboard_debug.py
24 passed in 0.52s
```

`tests/test_tiered.py`(17): `test_always_calls_llm_every_decision_like_today` · `test_never_is_fast_only` · `test_gated_triggers_goal_gap_ambiguous_conflict_surprise` ·
`test_min_interval_and_sniff_suppress_calls` · `test_conditions_can_be_switched_off_before_results` · `test_gated_refuses_unlocked_thresholds` · `test_random_and_periodic_match_a_rate` ·
**`test_truth_is_never_read_by_the_tiers`** · `test_hold_keeps_last_llm_action_and_reflex_skips_llm` · `test_fast_policy_recovers_planted_rule_and_round_trips` · `test_labels_follow_teacher_branches` ·
**`test_scorecard_parses_tier_configs_and_keeps_old_names`** · `test_scorecard_summary_call_stats_and_gated_pairing` · **`test_decision_row_carries_tier_fields_only_when_present`**(기존 구성은 칸 그대로 · 회귀) ·
`test_forward_model_recovers_planted_dynamics_and_flags_a_jump` · `test_matched_rate_and_gating_verdict` · **`test_base_judge_ignores_tier_configs`**(`9df30d1` 고침).
추론 시간(< 5 ms)은 시험이 아니라 실측(3.1).

---

## 3. 모듈 동작 확인 기록

### 3.1 빠른 정책 (F-002 · Task 2 · 5)

fast-v2 · fly-loop-v4 평가 1,708 결정(`runs/fast-policy-fly-loop-v4-<뇌>-fast-v2/`):

| | 진짜 | 재배선 s0 · s1 · s2 | 찍기 |
|---|---|---|---|
| 돌기(셋) | **0.666** | 0.572 · 0.569 · 0.577 | 0.434 |
| 멈춤 | 0.962 | 0.952~0.953 | — |
| 위험 | 0.955 | 0.934~0.935 | 0.935 |

추론 0.005 ms/결정(목표 < 5 ms) · 네 뇌 맞추기 3.2 s · 0.7 GB · 확신도 구간별 정확도 0.52 → 0.86(애매함 조건의 근거).

### 3.2 문턱 잠금 (F-003 · T3)

`calibrate`(보정 60 에피소드 · 격자 432 · 예산 0.30)의 **진짜 뇌 제안을 모든 뇌에 같은 분위수로**: 간격 상한 5 · 최소 간격 2 · 놀람 0.99 · 애매함 0.05 · 충돌 0.99 — 결과 전 `8756f86`(요구자 위임 10-01).
오프라인 추정: 진짜 호출률 0.298 · 교사 일치 0.858 · 빠른 정책만 0.731 · 같은 호출률 무작위 기댓값 0.811.

### 3.3 성적표 (F-004 · T4)

씨앗 2000~2039 · 11 구성 · 440 에피소드 · 2026-10-02 21:22 → 10-03 07:11 · `runs/scorecard-20261002-212213`(scorecard.json sha256 `b5692ca234a53eb6` · 정본 판정 `scorecard.rejudged.json` sha256 `6f9d63610c4e25db`).

| 구성 | 먹이 도착 | 회피 | Qwen 호출(결정 대비) | Qwen 몫 벽시계 | 에피소드 벽시계 합 |
|---|---|---|---|---|---|
| 늘 부름 vector | 0.350 | 0.682 | 633/633 (1.0) | 1,636 s | 96.8 분 |
| 빠른 정책만 fast-only | 0.475 | 0.727 | 0 | 0 | 51.8 분 |
| **사건 때만 vector-gated** | **0.575**(0.425~0.725) | 0.682 | 97/532 (**0.182**) | 193 s(−88 %) | **47.3 분(−51 %)** |
| 같은 호출률 무작위 | 0.550 | 0.682 | 106/556 (0.191) | 212 s | 49.5 분 |
| 같은 호출률 주기 | 0.550 | 0.636 | 80/559 (0.143) | 159 s | 48.7 분 |
| 재배선 gated s0 · s1 · s2 | 0.500 · 0.450 · 0.475 | 0.41 · 0.68 · 0.59 | 0.187~0.196 | — | — |
| 재배선 fast-only s0 · s1 · s2 | 0.475 · 0.450 · **0.625** | 0.73 · 0.55 · 0.59 | 0 | — | — |

판정(`scorecard.rejudged.json`):
- `gating`(gating-v1): gated 0.575 대 무작위 0.550(p 0.506) · 주기 0.550(p 0.497) → `gated_beats` 둘 다 false · 문구 "호출을 줄였다까지만" · PGR −0.8(늘 부름 < 빠른 정책만이라 분모가 음수 · 뜻 없음).
- `judgement_gated`: 순위 100 % · p 0.120 → `vector_beats_rewired` false · `judgement_fast_only`: 순위 33 % · p 0.707 → false.
- 늘 부름 0.350 은 R-7469 같은 씨앗 40 과 같다(재현).

쓸모 있는 호출(`useful_calls.py` · 교사와 같음: Qwen 대 그 결정의 빠른 정책): 애매함 7 번 0.714 대 0.000 · 시간 상한 49 번 0.551 대 0.735(오히려 나쁨) · 목표 40 번 0.675 대 0.700 · 충돌 1 · 놀람 0.

### 3.4 1층 반사 (F-005 · Task 3)

규칙: 빠른 정책 위험 머리 ≥ `reflex_threat_min` 이면 Qwen 없이 회피 · 먹이 닿음은 미각 채널이 관문 밖이라 **회로 신호 없음**(도착은 몸 쪽 끝 조건).
**잠근 설정은 `reflex_threat_min: null`(끔)** — 성적표 결정 기록에 `tier = reflex` 결정 **0**(vector-gated 532 · 재배선 gated 580~612) → 틀린 반사 비율은 **잴 수 없음**.
참고(이 W 에서 셈 · 판정 아님): vector-gated 위험 점수 ≥ 0.5 결정 14 · 다가옴 있는 결정 168 · 회로 도망 반사(디코더) 구간이 있는 결정은 늘 부름 1 · 층 구성 0.

### 3.5 인수 기준 대조 (R-1027 §2.2)

| 기능 | 인수 · 성공 기준 | 결과 |
|---|---|---|
| F-001 | `tier` · `trigger` 가 결정 기록에 남음 | ✅(scorecard-v4 decisions.jsonl) |
| F-001 | "늘 부름"이면 지금 성적표와 같은 결과 | ✅ 회귀 시험 · 늘 부름 0.350 = R-7469 같은 씨앗 |
| F-002 | 돌기 일치율 · 확신도 보정 곡선(진짜 대 재배선 셋) · 추론 < 5 ms | ✅ 3.1 |
| F-003 | 조건마다 호출 수 · 쓸모 있는 호출 비율 · 문턱이 결과 전 커밋에 | ✅(`8756f86` · 3.3) |
| F-004 (a) | 호출 ≤ 30 % 이면서 도착 · 회피가 늘 부름보다 95 % 구간 넘게 안 떨어짐 | ✅ 0.182 · 도착 0.575 ≥ 0.350 · 회피 0.682 = 0.682 |
| F-004 (b) | gated 가 재배선 gated 셋보다 높음 · p < 0.05 | ❌ 순위 100 % · p 0.12 → R 2.0-6 |
| F-004 (c) | 벽시계 추정(호출 × 지연 + 결정 × 빠른 정책 시간)이 늘 부름의 1/3 이하 | ✅ Qwen 몫 193 / 1,636 s = 0.12 · (몸 · 뇌 포함 실제 에피소드 벽시계는 0.49) |
| F-004 | 세 줄이 성적표 JSON · 대시보드 에피소드 쪽에 | ✅(`gating` · 요약 칸 · `2adb2b5`) |
| F-005 | 반사로 처리된 결정 수 · 틀린 반사 비율 | ⚠️ 규칙은 구현 · 잠근 설정에서 꺼 둠 → 0 결정 · 잴 수 없음 |
| F-006(Could) | 실시간 모의 | 안 함 |
| F-007 | 조사 → 문턱 설계 근거 | ✅ 연구 노트 15(`ded1099`) → control-v2 · fast-v2 |

---

## 4. 정본 문서 동기화 내역

| 문서 | 내용 | 커밋 |
|---|---|---|
| 대시보드 에피소드 쪽 · 10-01 코드 문서 따라잡기 | 층 · 계기 표시(doc 세션) | `2adb2b5` |
| 연구 노트 15 | 깊은 판단 호출 조사(research 세션) | `ded1099` |
| 학습 안내서 00장 · `docs/model-v1.md` 6 절 | 확인된 효과 "호출 −82 %" · 에코마인드로 가져갈 것(두 층 제어) | `6d5375e` · `3b24e56` |
| CLAUDE.md "지금까지" | 두 층 제어 요지 · "호출을 줄였다"까지만 | (10-03) · `e5edee0` |
| ⏳ 남음 | 연구 노트 06 · 11 · 학습 안내서 12장 동기화(R 7 장) — doc 세션에 다음 일감으로 | — |

---

## 5. 잔여 이슈

1. **"잘 골랐다"는 증명 안 됨** — 같은 호출률 무작위 · 주기와 도착이 같다. 사건 조건 가운데 쓸모 있는 것은 애매함(7 번)뿐이었고 시간 상한은 오히려 나빴다.
2. **진짜 대 재배선 못 넘음**(층 구조 쌍 p 0.12 · 빠른 정책만 쌍은 재배선 s2 가 위) — R 2.0-6 문구 규칙 그대로.
3. **늘 부름이 빠른 정책만보다 낮다**(0.350 대 0.475) — 이 과제 · 이 씨앗에서 Qwen 을 매번 부르면 도착이 낮아졌다. PGR 은 뜻이 없다.
4. **1층 반사는 성적표에서 꺼져 있었다** — 반사의 효과는 이 R 에서 재지 않았다(도망 반사는 회로 · 디코더가 그대로 맡음).
5. 투영기 train-v6 은 결과를 보고 고른 판 · 씨앗 40 · 오프라인 문턱 일치는 교사가 몬 기록 기준이라 성적표에서 그대로 나오지 않는다.
6. F-006 실시간 모의(Could)는 하지 않았다.
