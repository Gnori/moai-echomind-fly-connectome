# fly-dashboard-state

## 메타데이터

- **ID**: 7347
- **작성일**: 2026-09-24 11:51:38
- **상태**: 확정
  <!-- 허용값(enum, 자유 서술 금지): 접수 | 작성 | 상세화 | 확정 | 구현중 | 구현완료(검증대기) | 검증완료 | 아카이브 | 보류 -->
- **우선순위**: 중간
- **규모**: simple (간단)
- **유형**: 기능개선 (enhance)
- **관련 산출물**: P: (생략 — 2장 체크리스트로 대체) / W: (간이형 — 4장에 내장, 별도 문서 작성 시 링크)

> 표준: [/DEV_PROCESS.md](../../DEV_PROCESS.md) — simple 규모는 R(본 문서)+간이 W만 필수 (게이트 G1·G4)

---

> **[유형 가이드 — 기능개선]** 1장 작성 시 반드시 다룰 것:
> - **As-Is**: 현행 동작 (코드/화면 기준 사실 — 추측 금지)
> - **To-Be**: 개선 후 동작과 그 차이
> - **영향 범위**: 이 동작에 의존하는 다른 기능·API·사용자 습관, **하위 호환성** 유지 여부
> - **결정 번복 확인**: 기존 결정을 변경하면 해당 요구사항 폴더(또는 레거시 문서)를 번복 대상으로 링크 (DEV_PROCESS 공통규칙 2)

---

## 0. 요구자 말 (2026-09-24 11:4x · G1)

> *"이 대시보드의 데이터는 코드로 하기보다 데이터 파일로 가지고 있는 게 좋지 않을까?"* · *"그리고 그 데이터는 갱신하고"* ·
> *"DB 는 아니어도 데이터 파일로 결과를 가지고 있고 주기적으로 돌면서 업데이트 하고"* · 범위 선택 **"1 · 2 둘 다 v5 전에"**.

관련: [R-7928](./20260924-002928-7928_fly-dashboard_req.md)(dashboard-v1 · 검증완료) 의 개선 — 번복 대상은 R-7928 §3 "읽기만 · 로그 파싱" 중 **로그 파싱** 부분.

## 1. 문제 정의

**현재 상황(As-Is)**:
- 대시보드 코드가 `runs/full-v4-*.log` · `runs/projector-*/train.log` 를 **정규식으로 긁어** 표를 만들고, 만든 자료는 HTML 로 찍고 버린다.
- 그래서 ① 로그 문구가 바뀌면 화면이 깨지고 ② 세션 · 스크립트가 상태를 알려면 HTML 이나 로그를 다시 긁어야 하고
  ③ 프로세스를 밖에서 끊으면 "끝" 줄이 없어 도는 중으로 보인다(2026-09-24 11:3x 실제 — 재배선 s0 중단 뒤 "돌고 있다"고 표시).

**해결 방안(To-Be)** — DB 없이 **자료 파일** · **주기 갱신**:
1. **만드는 쪽이 상태를 직접 쓴다** — 학습(`train`)이 `runs/projector-<뇌>-<판>/status.json` 을 시작 · 25 걸음마다 · 회전 끝 · 끝/멈춤에 갱신
   (뇌 · 판 · 규칙 · 단계 · 걸음/전체 걸음 · 손실 · s/걸음 · 메모리 · 검증 손실 · 시작/갱신/끝 시각 · 상태 running/done/aborted/refused · 끝 코드 · Qwen 지문 같음).
   전체 순서 스크립트(`bin/run_full_v5.sh`)는 단계마다 `runs/full-v5-<시각>.jsonl`(한 줄 = 한 사건: 시작/끝/중단/전체 끝 · 명령 · 여유 % · 끝 코드).
   평가 · 성적표 · 관문 · 판정은 이미 json.
2. **대시보드는 자료 파일을 모아 `runs/dashboard/state.json` 으로 들고 있고**(`--serve --every N` 이 N 초마다 갱신), HTML 은 그 파일에서 그린다.
   status.json · jsonl 이 없는 옛 폴더(v1~v4)는 지금의 로그 파싱으로 **대체 읽기**(하위 호환).

**영향 범위**:
- `src/moai_echomind_fly/loop/status.py`(신규 · 원자적 JSON 쓰기) · `loop/train_cli.py`(status.json — 학습 **규칙은 그대로** · 기록만 더함) ·
  `bin/run_full_v5.sh`(jsonl) · `src/moai_echomind_fly/dashboard.py`(state.json · json 우선 · 로그 대체) · `config/dashboard.yaml`(dashboard-v2 · `state_file`) · 시험.
- 하위 호환: v1~v4 폴더 · v4 로그는 그대로 보인다. `--serve` 주소 · 포트 같음.
- ⚠️ v5 학습 **시작 전**에 끝낸다 — 도는 중에 학습 코드를 바꾸면 뇌 4 개가 다른 코드로 돈다.

---

## 2. 작업 체크리스트

### 2.1 구현
- [ ] `loop/status.py` `StatusFile.update(**칸)` — 합쳐서 임시 파일 → 이름 바꾸기(원자적) · `updated` 시각
- [ ] `train` 이 status.json 을 쓴다(시작 · 자료 · 25 걸음 · 회전 끝 · 끝 · 멈춤 · 막음) — `total_steps` = n × (캡션 회전 + 행동 회전) + 되새김 수
- [ ] `run_full_v5.sh` `run()` · 시작 · 중단 · 전체 끝 → `.jsonl`
- [ ] 대시보드: `collect()` 가 status.json · jsonl 을 먼저 읽고(없으면 로그) → `state.json` 저장 → `render(state)`
- [ ] `config/dashboard.yaml` dashboard-v2 · `state_file` · globs 에 jsonl

### 2.2 검증
- [ ] 가벼운 시험: status.json 왕복 · 가짜 runs/ 에 status.json + jsonl 만 있을 때 state.json 과 HTML 에 값이 들어감 · 옛 로그만 있을 때도 됨
- [ ] `ruff` · 기존 대시보드 시험 통과
- [ ] 실제 `--serve` 로 state.json 이 30 초마다 갱신되는지(갱신 시각) · v5 학습 status.json 이 걸음마다 바뀌는지

---

## 3. 범위 제외 (Out of Scope)

- DB · 원격 · 2단계(결정 단위 디버깅) · 옛 폴더(v1~v4)에 status.json 을 소급 생성하는 일

---

## 4. 검증 결과 (간이 W) — 상태를 "검증완료"로 바꾸려면 필수

**검증 명령**:
```bash
uv run ruff check src tests && uv run pytest tests/test_dashboard.py tests/test_status.py -q
uv run fly-connectome dashboard          # runs/dashboard/state.json · index.html
```

**결과**: (구현 뒤 채움)
