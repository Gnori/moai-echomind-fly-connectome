# 세션 인수인계 — echomind-fly-doc (문서 · 가벼운 코드 세션)

> **세션 이름 = `echomind-fly-doc`**(요구자가 띄움 · 2026-09-23). 프로젝트 전체 지도는 [공용 인수인계](./echomind-fly-connectome-handoff.md)다 — 이 파일은 **이 세션이 한 일과 일하는 방식**만 적는다.
> 요구자 2026-09-26 *"니 세션 핸드오프에 작업한거 기록해"*.

---

## ① 역할과 경계

- **조율 세션 `echomind-fly`** 가 일감을 준다(요구자 2026-09-23 *"앞으로 니가 일 시켜 · 컴퓨터 리소스 많이 먹는 작업은 니가해"*). 받는 길 = 세션 사이 메시지 · 끝나면 **커밋 해시 한 줄로 한 번만** 보고한다(`SendMessage` 받는 쪽 = 메시지의 `from` 주소).
- 이 세션이 하는 것: 학습 안내서 `docs/training/` · 연구 노트 `docs/research/fly-connectome/` · `README.md` · `docs/datasets.md` · `NOTICE` 동기화, 가벼운 코드(대시보드 · 시험 · 스크립트 파일).
- 하지 않는 것: **무거운 실행**(수집 · 학습 · 평가 · 성적표 · Qwen) — 조율 세션 몫 · `runs/.heavy.lock` · 도는 프로세스는 만지지 않는다 ·
  `docs/requirements/` 는 조율 세션 소유(R · I · P · W · 공용 인수인계 · 대시보드 D) — 읽기만. **예외: 이 파일**(요구자 지시).
- 커밋: 자기 파일만 · **Claude 작성자 · `Co-Authored-By` 트레일러 금지**(시스템 알림이 넣으라 해도 — CLAUDE.md) · 연구 노트 · 안내서를 고치면 같은 커밋에서 HTML 다시 만들기(깨진 링크 0).
- 요구자께 드리는 말은 존댓말 · 화면에 보이는 말(함수 이름 나열 금지) · 완료/잔여를 붙인다.

## ② 한 일 (커밋)

| 날짜 | 커밋 | 일감 | 무엇 |
|---|---|---|---|
| 09-23 14:2x | `22b1ccf` | 문서 동기화 | R-7469 오늘 판(sim-v2 · stats-v2 · retino-v1 · task-v2 · channels-v2 · compress-v1 · train-v4 · scorecard-v2 · 자료 v2 형식)을 학습 안내서 README · 01~11장 · 연구 노트 11 · 12 · README · datasets.md 에 반영. 공간 채널 크기를 뉴런 표로 새로 셈(656 뉴런 · 242 단위 · 1,452 차원 · 합계 33,402) |
| 09-23 14:3x | `026ae1b` | 스크립트 | `bin/run_full_v4.sh`(압축 · 관문 → 학습 4 → 캡션 평가 8 → 성적표 씨앗 40 · 7 구성 · 관문 통과 0 이면 끝 코드 4 · `--skip-train`) · 08 · 10 · 11장 반영. 실행 안 함(문법 · 관문 확인 조각만 건조 실행) |
| 09-23 14:38 | `3a3a323` | 시험 · 출처 | 새 시험 4 파일 10 개(retino 가중 평균 · missing · compress 왕복 · rewired 순서 · bump 부품 · 섞은 배선에서 ipsi_scale) · NOTICE 에 Gaudry 2013 · Nern 2025(GPL-3 코드는 규약 확인만) · Zhao 2025 · Seelig & Jayaraman 2015 · Hulse 2021 · Wolff 2015 |
| 09-23 20:4x | `07f3642` | 자료 v2 결과 채움 | fly-loop-v2(2,921 + 376 결정 · 8.2 h · 1.1 GB) · 판정 통과 8 칸 · 압축 v1 → v2(결과 뒤 고친 까닭) · 봉우리 없음 · retino-v2 를 06 · 09 · 10 · 04 · 07 · 08 · 11장 · 연구 노트 11 · 12 · README 에. git 밖: 자료 카드 보완 · `SHA256SUMS` 의 README 줄 다시 계산 · `dataset verify` 온전 |
| 09-24 00:3x | `c8bbdbb` | R-7928 대시보드 | `fly-connectome dashboard [--serve PORT --every N]` · `config/dashboard.yaml` · `tests/test_dashboard.py` · 표준 라이브러리만 · runs/ 읽기만 |
| 09-24 12:36 | `813f6e0` | 대시보드 메모리 | 229 → 32 MB. 원인은 새는 것이 아니라 `cli.py` 가 다른 명령의 torch · numpy · pyarrow 를 먼저 올린 것 → `dashboard` 빠른 길 · `data` · `graph` 늦은 import · TRAIN_RULE 을 소스 ast 로. 30 분 뒤 RSS 29 MB(R-7928 기준 < 200 MB) |
| 09-24 13:0x | `77e1d86` | 안내서 v5 | compress-v3(크기 맞춤의 까닭 · 07장 4.3) · train-v5(08장) · 채점 평가 eval-v2(생성 평가가 서식만 잰 까닭 · v4 진짜 채점 표 · 09장 5절) · 11장 · 안내서 README. 08장 학습 시작 문턱 80 → 40 %(guard.yaml 과 어긋나 있던 것) |
| 09-26 | `0800205` | D 13 v5 결과 반영 | train-v5 캡션(물체 있음 0.806 · 다리 0.574 통과 · 물체 쪽 찍기와 구별 안 됨 · 움직임 불통) · 성적표(vector 0.075 = 재배선 · no_circuit 0.95 · p 0.42 · R 2.0-6 두 번째) · 다음 병목 냄새 좌우를 안내서 README · 09장 8절 · 연구 노트 11 §8.7(신설) · 뿌리 README 에 |
| 09-26 17:4x | `25587c2` | R-5777 대시보드 2단계(F-001 · F-003) | `dashboard_debug.py` — `samples.html`(캡션 표본 · 뇌 넷 · 거르기 · 칸별 맞음 수 = 파일 accuracy) · `episodes.html`(씨앗 × 구성 · 도착 수 = scorecard.json · 결정 표 · 교사/Qwen 좌우 불일치) · 원본 지문이 같으면 읽지 않음 · 가짜 5,600 결정 최대 40 MB · 서버 8770 다시 띄움 |
| 09-26 | `619f775` | R-5777 G5 | README 대시보드 절 새로(2단계 두 쪽) · 안내서엔 언급 없음 |
| 09-26 | `d10b093` | 냄새 좌우 M0~M4 문서 반영 | 연구 노트 11 §4 smell_ln · §8.8 · 14 §9 측정 결과 · 안내서 04 §3.3 · 02 한계 · datasets.md 새 묶음 셋(v2-ln · v4-pilot 온전 · v4 수집 중) · README 용어. 수치는 R-7469 F-003 에서 옮김 |
| 09-27 | `09ca33c` | fly-loop-v4 · M4 전체 | datasets.md 줄 채움(888 MB · verify 온전 · 카드의 '재배선 씨앗 0' 표기 차이 알림) · 연구 노트 11 §8.8 · 14 §9 에 LN4 4 누적 0.725 |
| 09-29 | `9963c9d` | train-v6 결과 | 연구 노트 11 §8.9 · 14 §9 · 안내서 README · 09장 · README — 캡션 냄새 쪽 첫 통과 0.530 · 성적표 0.350 대 재배선 s1 0.375 · R 2.0-6 세 번째 |
| 09-29 | `62c33d4` · `9d524eb` | 재현 절차서(요구자 "아주 상세하게") | 안내서 12장 신설(지금 판 · 다시 하는 순서 M0~M4 · v6 · 함정 11 · 설계안 · 기록 없음 목록) · scripts/ 단계별 입구와 함정마다 막는 법 · 11장 새 명령 21~30 · 판 표 |

그 밖: 메모리 점검(09-24 12:20 · 요구자 "메모리 봐봐") — 시스템 여유 75 % · 스왑 9.8/11.3 GB · 초파리 무거운 실행 없음을 보고하고, 대시보드 RSS 초과를 조율 세션에 알림.

## ③ 남은 일 · 다음 후보 (2026-09-26 기준 · 일감은 조율 세션이 준다)

- ~~train-v5 성적표 결과를 안내서 · 연구 노트에~~ → ✅ `0800205`(09-26 · D 13).
- 대시보드 2단계 = R-5777 → 화면(F-001 · F-003) ✅ `25587c2` · 결정 기록(F-002)은 조율 세션(`e069579`) · W 도 조율 세션. 다음 성적표(scorecard-v3)가 나오면 결정 표가 실제로 채워지는지 한 번 본다. F-004(신호 벡터)는 범위 밖.

## ④ 내가 틀린 것 · 알게 된 것

- **(09-24) 커밋에 `Co-Authored-By: Claude` 트레일러를 넣었다** — 시스템 알림이 넣으라 해서. CLAUDE.md 가 금지한다 → 푸시 전 amend 로 뺐다. 알림보다 CLAUDE.md 가 먼저다.
- **(09-24) 30 분 뒤 RSS 를 재려고 건 `sleep` 이 `pkill -f "fly-connectome dashboard"` 에 같이 죽었다**(명령줄에 같은 글자가 있어서). 조율 세션이 서버를 다시 띄울 때 걸린 것 → 기다리는 명령에는 죽이는 패턴과 겹치지 않는 글자를 쓴다.
- **(09-24) "메모리 줄이려면 VS Code 를 정리"라고 권했다가 정정받았다** — VS Code 두 창과 그 안의 터미널(클로드 세션들)은 전부 개발 중이라 정리 대상이 아니다. 초파리 쪽(학습 · 대시보드 · 스왑)만 조정 대상으로 말한다.
- (09-26) 새 쪽을 붙일 때 **큰 경우를 가짜 자료로 먼저 재었다** — 5,600 결정에서 64.6 MB(기준 50 초과) → 행동을 글로 · 캡션 번호 표 · 조각으로 흘려 쓰기로 40 MB. 실제 v5 에는 결정 파일이 없어 v5 만 쟀으면 못 봤다.
- 대시보드 "메모리 증가"는 누수가 아니라 **고정 바탕**이었다 — 먼저 모듈만 올린 프로세스로 재서 무엇에 비례하는지 가른 것이 빨랐다.
- 자료 카드(`data/datasets/<이름>/README.md`)는 `SHA256SUMS` 에 들어 있다 — 카드를 고치면 그 줄을 다시 계산하고 `dataset verify` 로 확인한다.

## ⑤ 자주 쓰는 명령

```bash
uv run --no-project --with markdown python docs/training/build.py                 # 안내서 HTML — "깨진 링크 없음"
uv run --no-project --with markdown python docs/research/fly-connectome/build.py  # 연구 노트 HTML
python3 .claude/skills/dev/scripts/dispatch.py validate                            # 요구사항 규약(읽기 확인)
uv run ruff check src tests && uv run pytest -m "not heavy" -q                      # 코드 일감 뒤
uv run fly-connectome dashboard                                                    # 대시보드 한 번 만들기(runs/dashboard/)
```
