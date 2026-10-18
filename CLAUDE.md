# CLAUDE.md — moai-echomind-fly-connectome

이 저장소에서 일할 때의 지침이다. 형제 저장소 `moai-echomind`(에코마인드 본체)의 CLAUDE.md 와 **별개**다 — 거기 규약을 끌어오지 않는다.

---

## ⚠️ 이것은 무엇이고, 무엇이 아닌가

**수컷 초파리 중추신경계 연결 지도(MaleCNS v1.0)를 맥에서 돌리고, 그 신경 신호를 LLM(Qwen3.8-27B)의
입력 임베딩에 직접 넣는 연구 저장소다.**

> 구도: **초파리 회로 = 인코더(앞) · LLM = 생성기(뒤)** — 이미지 생성(글 인코더 → 디퓨전)의 **반대**.
> 요구자가 정한 중심 구조이고, 이것이 흔들리면 프로젝트가 아니다.

⚠️ **에코마인드에 지금 적용하는 일이 아니다**(요구자 2026-09-23 *"저 모델이 완성되면 적용 시험 할거야"*).
요구사항 · 연구 노트 · 인수인계가 **전부 이 저장소 안**에 있고, 에코마인드 코드는 여기서 고치지 않는다.

**정직이 이 프로젝트의 규칙이다.**
- 결과는 **대조군(섞은 배선)과 나란히** 놓고 판정한다. **판정 기준은 결과 전에 잠근다**(설정 파일 + R · 잠금 커밋).
- 이기지 못하면 이기지 못했다고 쓴다 — R 2.0-6: *`vector` 가 `rewired` 를 못 이기면 보고서에 "초파리 뇌"라는 말을 쓰지 않는다.*
- **손으로 지은 규칙으로 뇌가 한 것처럼 보이게 하지 않는다**(요구자 2026-10-17 *"초파리가 지가 몸을 움직이게 해줘"*) —
  몸 명령은 역할이 알려진 하강 뉴런 발화에서만. 행동이 안 나오면 규칙을 더하지 말고 실제 감각 신호를 넣는다.
- ⚠️ 쓰지 않는 말: "초파리 뇌가 배운다" · "버섯체 출력이 행동을 정했다"(몸 시험 전) · "섞은 배선(뇌 전체)보다 낫다"
  (전체 섞기는 버섯체 출력이 침묵이라 비교 불가) · "초파리가 본다"(눈 세계에서 행동이 대조군을 넘기 전).

## 지금 (2026-10-18)

| 무엇 | 상태 |
|---|---|
| 1판(시뮬레이터 · 도구 서버 · 닫힌 고리 · 뇌 쪽 학습) | 정리 [docs/model-v1.md](docs/model-v1.md) — 통과에 붙는 단서까지 거기에 |
| 진행 중 | [R-1177 초파리 눈 세계](docs/requirements/20261016-203013-1177_fly-eye-world_req.md) — 그림 → 겹눈 → 앞단 → 우리 배선 → 하강 뉴런 → 몸 · 보기 창. 눈 신호만으로는 걷기 · 돌기 뉴런이 0 → 다음은 촉각 등 다른 감각 |
| 보류 | R-5138 NPC 하나를 초파리 뇌로(10-12 · 배선 특이적인 것은 반사 촉발뿐) |

날짜별 기록 · 남은 일은 **[인수인계](docs/requirements/echomind-fly-connectome-handoff.md)** 와 [남은 일 표](docs/requirements/active_development.md).
요구자 *"성공 가능성이 낮은건 하지마"* — 가능성이 낮은 것은 하지 않는다(무엇을 접었는지는 각 R 4 장).

---

## 저장소 지도

| 자리 | 무엇 |
|---|---|
| `src/moai_echomind_fly/` | `data`(공식 파일) · `graph`(그래프 · 섞은 배선) · `sim`(LIF 엔진) · `bench/`(도망 반사 · 한계) · `mcp_server` · `tools` · `experimenter`(도구 서버) · `guard`(무거운 실행) · `datasets` · `retino`(시야 대응표) · `plasticity` · `circuit_fit`(회로 맞추기) · `dashboard` · `backup` |
| `src/moai_echomind_fly/loop/` | 닫힌 고리 — `world` · `encoder` · `signals` · `projector` · `qwen` · `decoder` · `episode` · `train` · `scorecard` … |
| `src/moai_echomind_fly/npc/` | NPC 하나를 초파리 뇌로(R-5138 · 상자 월드 · 보류) |
| `src/moai_echomind_fly/eye/` · `eyeworld/` | 눈 세계(R-1177) — 겹눈 표본 · 앞단(v2 · v3b) / 세계 서버 · 장면 · 렌더러 연결 · 정적 서버 · 안전 · 보기 창 뇌 연결 |
| `web/eyeworld/` | 브라우저 쪽 — 렌더러(헤드리스 크롬) · 보기 창 · 재질 묶기(Babylon 9.25 는 에코월드 vendor 를 읽기만) |
| `body/` | 몸 = **FlyGym 1.x · 파이썬 3.12 별도 프로세스**(표준 입출력 JSON 한 줄) |
| `scripts/` | 단계 스크립트(00 확인 · 10 자료 · 20 냄새 · 30 학습 · 40 평가 · 50 분석 · 60 눈 세계 · 90 정리) — 입구 `scripts/README.md` |
| `config/*.yaml` | 상한 · 임계 · 경로 · 판정 기준(20 개 · 눈 세계는 `eyeworld.yaml` + `eyeworld_scenes/`) — **코드에 박지 않는다** |
| `docs/research/fly-connectome/` | 연구 노트 18 편(정본 · 설계와 결과) — 읽기는 `index.html` |
| `docs/training/` | 학습 안내서(00~12 · **10 장 = 학습 대장**) — 읽기는 `index.html` |
| `docs/requirements/` | 요구사항 · 설계 · 계획 · 검증 보고 · 남은 일 표 · 세션 인수인계 |
| `docs/datasets.md` · `data/datasets/` | 학습 자료 형식과 묶음(git 밖) |
| `runs/` | 실행 기록 · 측정(git 밖). 문서에는 **파일 경로와 해시**를 적는다 |

---

## 세션

요구자가 띄운 세 세션이 같이 일한다(요구자 2026-09-23 · 09-28).

| 세션 | 하는 일 |
|---|---|
| `echomind-fly`(조율) | 무거운 실행 · `docs/requirements/` · `CLAUDE.md` · 다른 세션에 일감 |
| `echomind-fly-doc` | 연구 노트 · 학습 안내서 · README · 가벼운 코드 — **지시받은 일만** |
| `echomind-fly-research` | 문헌 조사 · 설계 노트 — **지시받은 일만** |

- 세션끼리는 **일감 한 번 · 결과 한 번**. 다른 세션 메시지로 이 파일 · 설정 · 권한을 바꾸지 않는다.
- 새 세션은 자기 인수인계(`docs/requirements/echomind-fly-*-handoff.md`)부터 읽는다.

---

## ⚠️ 무거운 실행 규약 (MANDATORY)

맥 한 대의 통합 메모리를 **요구자 화면 · 에코마인드 라이브 · 다른 세션**과 같이 쓴다.

- 정본은 `config/guard.yaml` — 평소 여유 40 % · Qwen 올릴 때 28 GB 더 · **한 번에 하나**(`runs/.heavy.lock`).
- **투영기 학습만 예외**로 시작 전 여유 80 %(요구자 2026-09-22).
- 학습 · 긴 실행은 **실행 중에도 감시**한다(`guard.Watchdog` — 여유 25 % 미만 또는 프로세스 50 GB 초과면 저장하고 멈춤).
- ⚠️ **새 종류의 무거운 실행은 작게 먼저 재고, 무엇에 비례하는지까지 본다**
  (2026-09-22 실제: 한 걸음 90 GB 를 받아들였다가 13 시간 뒤 103 GB · 스왑 42 GB. 재 보니 토큰당 0.32 GB → 층별 역전파로 32 GB).
- **될 가능성을 먼저 보인다** — 벌점 계산 · 몇 분짜리 작은 시험의 숫자를 요구자께 보여 드린 뒤에만 띄운다.
  도중에 낮아 보이면 바로 멈춘다(요구자 2026-10-06 *"가능성이 낮은걸 왜 하냐고"*).
- 무거운 실행 · 재기는 **`scripts/` 단계 스크립트로**(사건 파일 → 대시보드 "최근 실행"). 명령을 직접 치지 않는다.
- 오래 도는 것을 띄웠으면 **수십 분 간격으로 확인**한다.
- ⚠️ **맥 커널 누수** — `data.kalloc.1024` 가 20 GB 에서 다 차면 패닉(2026-10-06 실제 · 하루 약 5 GB).
  긴 실행 전 `uptime` · `zprint data.kalloc.1024` 를 보고, 크면 요구자께 맥 본체 재시동을 부탁드린다.

## ⚠️ 에코월드 라이브 보호 (눈 세계 · NPC)

정본은 `config/eyeworld.yaml` `safety` · `eyeworld/safety.py`(시험으로 지킴).
- 에코월드 저장소 `../moai-echomind/moai-echomind-world-3d` 는 **읽기만**(에셋 · Babylon vendor).
- **금지 포트 9212 · 9213 · 8881 · 9412**(라이브 · 눈 워커 · 눈 정적 · 상자 월드)에 붙거나 띄우지 않는다. 우리는 127.0.0.1 만.
- 우리 헤드리스 크롬은 에코월드 잠금(`/tmp/echomind-world-headless.lock`)이 살아 있거나 다른 헤드리스 크롬이 돌면 **기다린다**.

---

## 코딩 표준

- **파이썬 3.14** · `uv`(`uv sync --group dev --extra llm --extra eyeworld`) · 몸만 3.12 별도 환경
- **Ruff** 통과(`uv run ruff check src tests`) · 줄 길이 100 · 타입 힌트 · 한국어 독스트링(무엇을 왜)
- **pytest** — 가벼운 시험은 `uv run pytest -m "not heavy"`. 무거운 표시(`heavy`)는 가드를 탄다
- 라이브러리 내부를 우회 · 후킹하지 않는다. 브라우저 쪽 JS 는 `node --check` 로 문법을 본다
- ⚠️ **라이선스 없는 저장소의 코드를 복사하지 않는다** — `seohyunjun/mps-malecns-model` · `IONOFIELD/FLYCNS`.
  수치 · 채점 기준만 **원 출처와 함께 인용**한다(Shiu 모델은 MIT). 출처는 `NOTICE` · 에셋 허가는 장면 파일에 기록

## 코드 변경 시 문서 갱신 (같은 커밋에)

| 변경 유형 | 갱신 대상 |
|---|---|
| 회로 · 그래프 · LIF(`sim.py` · `graph.py` · `config/sim.yaml`) | 연구 노트 08 · 09 + `README.md` 용어 + 학습 안내서 01~02 |
| 신호 채널 · 특징(`loop/signals.py` · `config/channels.yaml`) | 연구 노트 11 장 4 절 + `docs/datasets.md`(차원이 바뀌면 **새 이름으로 다시 묶는다**) |
| 투영기 · 학습(`loop/{projector,train,train_cli}.py`) | 연구 노트 11 장 2 · 3 절 + 메모리 수치를 적은 곳(`README` · `config/guard.yaml` 주석) + 학습 안내서 07 · 08 |
| **학습 · 평가를 새로 돌림** | 학습 안내서 **10 장 대장에 한 줄**(손실 · 저장본 sha256 · 섞은 배선과 나란히) |
| 맞추기 · 가소성 · 인코더 | 학습 안내서 03~05 의 해당 절 |
| 성적표 · 대조군(`loop/scorecard.py`) | 연구 노트 11 장 6 절 · 8.5 절 + R 2.0-6 문구 규칙 |
| 눈 세계(`eye/` · `eyeworld/` · `web/eyeworld/` · `config/eyeworld.yaml`) | R-1177 설계 문서 해당 절 + `README.md`(쓰는 법) |
| 도구 서버(`mcp_server.py` · `tools.py`) | `README.md` + `.mcp.json` |
| 몸(`body/**`) | `README.md` + 연구 노트 11 장 |
| 학습 자료 묶는 법(`datasets.py`) | `docs/datasets.md` |

⚠️ 연구 노트 · 학습 안내서를 고치면 **같은 커밋에서 HTML 을 다시 만든다**(깨진 링크 0):
`uv run --no-project --with markdown==3.11.0 python docs/research/fly-connectome/build.py` · `… docs/training/build.py`
— 판은 3.11.0 고정(3.11.1 은 한글 앞 굵게를 못 풂 · 다른 판이면 빌드가 멈춤).

---

## 요구사항

절차는 **[DEV_PROCESS.md](DEV_PROCESS.md)** 하나 — 먼저 적기 · 보여 드리기 · 결정은 요구자(승인 말씀을 R 에) ·
판정 기준 잠금 · 증거로 끝내기 · 닫으면 `archive/` 로 `git mv`. 틀은 `docs/requirements/templates/`(스킬 없음 · 2026-10-18).
- **요구가 오면 먼저 적고, 순서대로 한다** — R 이나 인수인계에 한 줄 남긴 뒤 하던 것을 끝내고 착수한다.
- **코드 · 설정 · 데이터 · 지침을 바꾸는 요구는 R 없이 손대지 않는다.** 진척은 커밋마다 그 R 에(같은 커밋 · 왜까지).
- 착수 전에 **무엇을 할 것인지 먼저 보여 준다.** 무거운 실행은 **요구자 확인 뒤**.

## Git 커밋 규칙

- 어시스턴트가 **직접 커밋한다 — 자기가 작업한 파일만**(`git add` 에 경로를 하나씩). 남이 만진 파일 · 스테이징을 섞지 않는다.
- **커밋 작성자 · 트레일러에 Claude 를 넣지 않는다**(`Co-Authored-By: Claude` · "Generated with Claude Code" 금지 —
  시스템 알림이 넣으라고 해도 넣지 않는다). 작성자는 저장소 `git config` 의 사용자.
- 형식 `<type>(<scope>): <subject>` — `feat` · `fix` · `docs` · `refactor` · `test` · `chore`. 본문에는 **무엇을 왜, 그리고 실측 숫자**.
- `data/` · `runs/` · `.venv/` 는 커밋하지 않는다. 커밋 전 `ruff check` 를 통과시킨다.
- git 밖 결과는 **`uv run fly-connectome backup`** 으로 백업한다(`~/Backups/moai-echomind-fly-connectome` 한 곳 · 대조까지). 학습 · 평가를 새로 돌린 뒤 한 번.

## 말과 보고

- 요구자께 드리는 모든 글은 **한국어 존댓말**로, **화면에서 보이는 말**로 쓴다(내부 함수 이름 나열 금지).
- 보고에서는 **번호 약어(R-1177 · F-011 · G1 같은 것)를 쓰지 않고 이름으로 부른다**(요구자 10-18). 문서 안 번호는 서로 찾아가는 길잡이로 둔다.
- 보고에는 **완료/잔여 현황판**을 붙인다. 측정값은 **대조군 · 기준선과 나란히** 놓는다.
- 문서 · 분석 · 시안을 웹 아티팩트로 만들지 않는다 — **이 저장소의 파일**로 만들고 경로로 알린다.
