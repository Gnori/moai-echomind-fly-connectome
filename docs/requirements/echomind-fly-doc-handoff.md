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

그 밖: 메모리 점검(09-24 12:20 · 요구자 "메모리 봐봐") — 시스템 여유 75 % · 스왑 9.8/11.3 GB · 초파리 무거운 실행 없음을 보고하고, 대시보드 RSS 초과를 조율 세션에 알림.

## ③ 남은 일 · 다음 후보 (2026-09-26 기준 · 일감은 조율 세션이 준다)

- **train-v5 성적표 결과를 안내서 · 연구 노트에** — 조율 세션이 `9c1f053` 에서 10장 · R · 공용 인수인계는 적었다(vector 0.075 = 재배선 · no_circuit 0.95 · R 2.0-6 발동 · 캡션 판정은 이김).
  아직 옛 상태인 곳: 안내서 README "지금 상태"(v5 학습 중으로 적혀 있음) · 09장 8절 판정 순서의 "v5 는 지금 여기(학습 중)" · 연구 노트 11 결과 절 · 뿌리 README 상태 줄.
- 대시보드 2단계(결정 단위 디버깅)는 따로 R 이 생기면.

## ④ 내가 틀린 것 · 알게 된 것

- **(09-24) 커밋에 `Co-Authored-By: Claude` 트레일러를 넣었다** — 시스템 알림이 넣으라 해서. CLAUDE.md 가 금지한다 → 푸시 전 amend 로 뺐다. 알림보다 CLAUDE.md 가 먼저다.
- **(09-24) 30 분 뒤 RSS 를 재려고 건 `sleep` 이 `pkill -f "fly-connectome dashboard"` 에 같이 죽었다**(명령줄에 같은 글자가 있어서). 조율 세션이 서버를 다시 띄울 때 걸린 것 → 기다리는 명령에는 죽이는 패턴과 겹치지 않는 글자를 쓴다.
- **(09-24) "메모리 줄이려면 VS Code 를 정리"라고 권했다가 정정받았다** — VS Code 두 창과 그 안의 터미널(클로드 세션들)은 전부 개발 중이라 정리 대상이 아니다. 초파리 쪽(학습 · 대시보드 · 스왑)만 조정 대상으로 말한다.
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
