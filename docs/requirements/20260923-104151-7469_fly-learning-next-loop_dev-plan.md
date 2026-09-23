# fly-learning-next — loop 구현 플랜 (P)

## 헤더

- **요구사항**: 7469 — [./20260923-104151-7469_fly-learning-next_req.md](./20260923-104151-7469_fly-learning-next_req.md)
- **인터페이스 계약(I)**: [./20260923-104151-7469_fly-learning-next_interface-design.md](./20260923-104151-7469_fly-learning-next_interface-design.md) (승인 v1 · 구현 중 변경 위임)
- **모듈**: loop — `src/moai_echomind_fly/loop/*` · `datasets.py` · `config/{task,channels,compress}.yaml`
- **작성일**: 2026-09-23
- **Goal**: 판정 기준(stats-v1) → 3차원 입력 · 채널 v2 → 자료 v2 → 압축 · 관문 → 작은 투영기(train-v4) 순으로, 진짜 배선이 재배선 앙상블을 앞서는지 **판정 가능한** 실험을 만든다.
- **Tech Stack**: Python 3.14 · numpy · PyTorch(회로) · MLX(Qwen · 투영기) · FlyGym 1.x(몸 · 3.12)
- **선행 P 의존성**: core P 의 Task 1(재배선 범위) · Task 2(냄새 좌우 측정) · Task 3(LC 시야 대응) 이 이 P 의 Task 2 · 3 보다 먼저

> 코드 정본은 커밋의 파일이고, 이 문서는 **순서와 검증 절차**를 적는다. 무거운 실행은 작게 잰 뒤 요구자 확인.

---

## Task 1: 판정 기준 stats-v1 (F-001)

- [x] `loop/stats.py` — 에피소드 블록 순열(라벨을 에피소드 단위로 섞음) · Holm · 에피소드 부트스트랩 구간 · 판 상수 `STATS_RULE`
- [x] `linear_probe` — 자료의 모든 재배선 뇌 · 셀(채널|칸)마다 I §3.2 형식 · `judge: false` 채널 제외 · `--permutations`
- [x] `datasets.load` — `rewired-s<K>` 받기 · v1 의 `rewired` = s0
- [x] 시험(`tests/test_stats.py` 5 개): 심은 신호는 p 작고 잡음은 크다 · Holm 단조 · 블록 순열이 에피소드를 넘지 않는다
- [x] 실행: `fly-loop-v1` → `runs/linear-probe-20260923-105625.json`(CPU 32 s) — **통과 2 칸**: 시각 → 물체 거리 0.771/0.581 · 움직임 0.781/0.571(Holm p 0.047).
  시각 → 물체 쪽(p 0.025) · 몸 → 다리(p 0.005)는 Holm 뒤 탈락. ⚠️ 재배선 1 개라 앙상블 순위는 사실상 1:1 비교 — v2 에서 다시.

## Task 2: 3차원 세계 · 시야 지도 입력 · 채널 v2 (F-008 1~3 · F-003 반영)

- [x] `config/task.yaml` → task-v2(물체 고도 −30~60° · `vision.mode: retinotopic` · σ 25°) · 옛 판 `config/archive/task-v1.yaml` · `config.load(name, version=)` — **지형은 미룸**(FlyGym 1.x OdorArena 는 평지)
- [x] 몸: 물체가 시작 고도에서 곧게 내려온다(3차원 거리로 크기) · 자세 `fly_euler_rad`(FlyGym ZYX) · 머리 높이 기록
- [x] 인코더 `retinotopic`: LC 마다 고도 가우시안(좌표 없는 LC 는 1) · 시험 `test_retinotopic_encoder_prefers_lc_at_object_elevation`
- [x] `config/channels.yaml` → channels-v2(`space` = EPG · PEN · PEG · Δ7 · PFN · hΔB · FC2 · PFL3 · HS · VS · 단위 instance) — 후각 · 판정 제외는 sim-v2 · stats-v2 로(I v1.2)
- [x] 참값에 3차원 칸(`looming_elevation_deg` · `heading_deg` · `pos_xy` · `body_euler_deg`) — 캡션 서식은 그대로
- [x] 시험: 위 물체 → 고도 높은 LC 가 더 셈 · task-v2 가 v1 의 뽑기를 지킨다. (뒤섞은 대응표 대조는 자료 v2 판정 때)

## Task 3: 자료 v2 (F-002)

- [x] `collect --rewired-seeds` · `stim.npz`(묶기에도) · `collect-probe`
- [x] 작은 측정 `runs/collect-probe-20260923-111821.json`: 뇌 6 · **결정당 12.0 s**(몸 3.1 + 뇌 9.0) · 일꾼 하나 3.4~3.6 GB(최대 4.3) · 추정 학습 300 에피소드 3.3 h + 평가 40 에피소드 0.45 h(일꾼 3)
- [ ] **요구자 확인** → 수집 → `dataset pack fly-loop-v2` · 백업

## Task 4: 압축 · 관문 (F-004)

- [x] `loop/compress.py` · `config/compress.yaml`(compress-v1 — 규칙 먼저: PCA · 설명 분산 0.95 · k 16~64) · `compress fit/gate` · 시험
- [ ] 자료 v2 에 실행

## Task 5: 작은 투영기 · 칸 토큰 손실 (F-005)

- [x] train-v4 코드(압축 입력 · 채널당 토큰 1 · 칸 가중 1 / 틀 0.1) · 시험(칸 구간 · 가중 손실)
- [x] 성적표 재배선 앙상블 — `--configs vector,…,rewired-s0,rewired-s1,rewired-s2,no_llm` · `judge()`(scorecard-v2: 먹이 도착이 재배선 전부보다 높고 씨앗별 쌍 부호 뒤집기 p < 0.05 · 재배선 하나면 옛 규칙) · 시험
- [ ] `train-probe` → **요구자 확인** → 학습(진짜 + 재배선 3) · 평가 · 성적표(씨앗 40)

## 최종 검증

- `uv run ruff check src tests` · `uv run pytest -m "not heavy"` · 각 Task 의 실행 기록을 W 에
