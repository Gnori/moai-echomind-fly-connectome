# fly-dashboard-debug — tool 검증 보고 (W)

## 메타데이터

- **요구사항**: 5777 — [./20260926-171115-5777_fly-dashboard-debug_req.md](./20260926-171115-5777_fly-dashboard-debug_req.md)
- **대응 P**: 생략 — R 4장 체크리스트로 대체
- **관련 I**: 생략 — R 3.3(파일 형식 하나)
- **작성일**: 2026-09-26
- **결과**: **통과** — 세 기능 구현 · 실제 v5 파일로 화면 확인. 단 성적표 **결정 표는 다음 성적표부터** 보인다(v5 에는 결정 기록이 없음 · R §5 에 적은 그대로).

---

## 1. 변경 파일 목록

| 파일 | 변경 유형 | 요약 | 커밋 · 누가 |
|---|---|---|---|
| `src/moai_echomind_fly/loop/scorecard.py` | 수정 | scorecard-v3 — `decision_row` · 구성마다 `<구성>.decisions.jsonl`(판정은 v2 그대로) | `e069579` · 조율 세션 |
| `src/moai_echomind_fly/dashboard_debug.py` | 추가 | `samples.html`(캡션 표본) · `episodes.html`(에피소드 · 결정) — 표준 라이브러리만 · 원본이 같으면 다시 안 읽음 | `25587c2` · doc 세션 |
| `src/moai_echomind_fly/dashboard.py` | 수정 | 두 쪽을 같은 주기로 생성 · 판정 표에 "진짜만 맞음/틀림" 링크 · 성적표 판정 줄이 `judgement` 키를 못 읽던 것 고침 | `25587c2` |
| `config/dashboard.yaml` | 수정 | `debug:` 쪽 이름 · 저장본 · 상한(표본 5,000 · 결정 20,000 · 원문 200 자) | `25587c2` |
| `tests/test_dashboard_debug.py` · `tests/test_loop_parts.py` | 추가 | 표본 5 · 결정 기록 칸 1 | 두 커밋 |

---

## 2. 테스트 실행 로그 (원문 · 2026-09-26)

```
$ uv run ruff check src tests
All checks passed!

$ uv run pytest -q -m "not heavy"
106 passed, 3 deselected, 18 warnings in 6.50s

$ uv run pytest -q tests/test_dashboard_debug.py tests/test_dashboard.py
13 passed in 0.70s
```

---

## 3. 모듈 동작 확인 기록 (통합 E2E — 실제 v5 파일 · 떠 있는 서버 8770)

```
http://127.0.0.1:8770/  → /runs/dashboard/index.html
runs/dashboard/samples.html   200  344,355 B
runs/dashboard/episodes.html  200   26,615 B
대시보드 프로세스 RSS 37,616 KB(한도 50 MB · 이전 32~36 MB)
```

| 인수 기준 | 결과 |
|---|---|
| F-001 · 376 표본 × 뇌 4 · 칸별 맞음 수 = 파일 정확도 | ✅ 10 칸 × 뇌 4 × 저장본 2 전부 일치(doc 세션) · 조율 세션 재확인: 진짜 물체 있음 303 = 0.806 × 376 |
| F-001 · "진짜만 맞음" 수가 판정 차이와 맞음 | ✅ 물체 있음 진짜 303 대 재배선 274 · 274 · 278(진짜만 맞음 27 · 진짜만 틀림 0) · 다리 216 대 190 · 187 · 147(22 · 8) · 재배선별 (진짜✓ 그 뇌✗) − (진짜✗ 그 뇌✓) = 맞음 수 차이 |
| F-001 · 메모리 50 MB 이하 | ✅ 37 MB · 가짜 결정 5,600 개 최악 40.3 MB(처음 64.6 → 흘려 쓰기로 고침) |
| F-002 · 결정 기록 칸 | ✅ 시험 `test_scorecard_decision_row_has_the_fields_the_dashboard_reads` · 형식 변경(`loom_mm` 뺌)은 R 3.3 에 적음 |
| F-002 · 요약 파일 하위 호환 | ✅ `<구성>.jsonl` 형식 그대로 — v5 성적표를 episodes.html 이 그대로 읽음 |
| F-002 · 파일 크기 | ⏳ 추정만(결정당 약 1 KB × 약 5,600 결정 ≈ 5.6 MB) — 다음 성적표에서 실측 |
| F-003 · v5 40 × 7 표 · 도착 수 = scorecard.json | ✅ 도착 3 · 3 · 38 · 3 · 1 · 3 · 0 일치 · 결정 파일 없음 → "결정 기록 전(scorecard-v2)" 표시 |
| F-003 · 결정 표 · 좌우 불일치 표시 | ✅ 가짜 파일 시험 · 브라우저 확인(Qwen 원문 속 `</script>` 는 글자로만) |

---

## 4. 정본 문서 동기화 내역

| 문서 | 내용 |
|---|---|
| R-5777 3.3 | 결정 기록 형식 — 구현 중 변경 적음 |
| doc 세션 인수인계 | `11c2207` |
| `README.md` 대시보드 절 | ⏳ 남음 — 두 쪽 이름 한 줄(가벼움 · 다음 doc 일감에 묶음) |

---

## 5. 잔여 이슈

1. 결정 표는 **다음 성적표부터** — v5 를 결정 단위로 보려면 성적표를 다시 돌려야 한다(약 13 시간 · 무거움 · 이 R 범위 밖).
2. 결정 기록 파일 크기 실측(다음 성적표).
3. `README.md` 대시보드 절 한 줄.
