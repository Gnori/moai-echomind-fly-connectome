# CLAUDE.md — moai-echomind-fly-connectome

이 문서는 AI 어시스턴트가 **이 저장소에서** 일할 때 따르는 지침입니다.
형제 저장소 `moai-echomind`(에코마인드 본체)의 CLAUDE.md 와는 **별개**입니다 — 거기 규약을 여기 끌어오지 마십시오.

---

## ⚠️ 먼저 — 이것은 무엇이고, 무엇이 아닌가

**수컷 초파리 중추신경계 연결 지도(MaleCNS v1.0)를 맥에서 돌리고, 그 신경 신호를 LLM(Qwen3.8-27B)의
입력 임베딩에 직접 넣는 연구 저장소다.**

> 구도: **초파리 회로 = 인코더(앞) · LLM = 생성기(뒤)** — 이미지 생성(글 인코더 → 디퓨전)의 **반대**.
> 요구자가 정한 중심 구조이고, 이것이 흔들리면 프로젝트가 아니다.

⚠️ **에코마인드에 지금 적용하는 일이 아니다.** 요구자(2026-09-23):
*"초파리 연구는 에코마인드에 지금 직접 적용이 아니야 — 저 모델이 완성되면 적용 시험 할거야."*
그래서 요구사항·연구 노트·인수인계가 **전부 이 저장소 안**에 있고, moai-echomind 의 대시보드·규약 검사는
이 프로젝트를 보지 않는다. 에코마인드 코드를 고치는 일도 여기서 하지 않는다.

**정직이 이 프로젝트의 규칙이다.** 결과는 **대조군(섞은 배선)과 나란히** 놓고 판정한다.
이기지 못하면 이기지 못했다고 쓴다 — R 2.0-6: *`vector` 가 `rewired` 를 못 이기면 보고서에 "초파리 뇌"라는
말을 쓰지 않는다.* (2026-09-23 실제로 발동했다: vector 0.2 = rewired 0.2 · no_circuit 0.9)

---

## 지금까지 (2026-09-23)

| 단계 | 무엇 | 상태 |
|---|---|---|
| A 시뮬레이터 코어 | 공식 데이터 → 그래프 → Shiu LIF(맥 GPU) → 도망 반사 검산 | ✅ 조건부 통과(Brian2 발화 완전 일치) |
| B 도구 서버 | MCP(stdio) — LLM 이 실험자, 커넥톰이 피험체 | ✅ 조건부 통과 |
| C 닫힌 고리 | 몸 → 회로 → **신호 벡터 → 투영기 → Qwen** → 행동 | ✅ 실행 완료 · **대조군을 못 이김** |

**막힌 곳은 앞(회로)이 아니라 전달이다** — 선형 검사(`fly-connectome linear-probe`)로 갈랐다:
채널별로는 진짜 배선이 섞은 배선보다 높지만(시각 → 물체 방향 0.743 대 0.590 · 몸 → 다리 접촉 0.590 대 0.419),
9채널 3.2만 차원을 한꺼번에 넣으면 그 차이가 사라진다(투영기 3,190만 매개변수 대 표본 581).
자세한 내용과 다음 후보는 **[인수인계](docs/requirements/echomind-fly-connectome-handoff.md)** 와
**[W(C)](docs/requirements/20260921-233859-3340_fly-connectome-closed-loop-loop_walkthrough.md) 5장**.

---

## 저장소 지도

| 자리 | 무엇 |
|---|---|
| `src/moai_echomind_fly/` | `data`(공식 파일) · `graph`(그래프·섞은 배선) · `sim`(LIF 엔진) · `bench/`(도망 반사·한계) · `mcp_server`·`tools`·`experimenter`(B) · `loop/`(C) · `guard`(무거운 실행) · `datasets`(학습 자료 묶음) |
| `src/moai_echomind_fly/loop/` | `world`·`encoder`·`signals`·`projector`·`qwen`·`decoder`·`episode`·`teacher`·`collect`·`train`·`train_cli`·`scorecard`·`linear_probe` |
| `body/` | 몸 = **FlyGym 1.x · 파이썬 3.12 별도 프로세스**(표준 입출력 JSON 한 줄). 2.1 에는 보행 제어기·냄새가 없다 |
| `config/*.yaml` | `sim`·`loom`·`task`·`channels`·`llm`·`tool`·`guard` — **상한·임계·경로를 코드에 박지 않는다** |
| `docs/research/fly-connectome/` | **연구 노트 11편(정본)** — 설계와 결과. 읽기는 `index.html` |
| `docs/requirements/` | R/I/P/W · 대시보드 · **세션 인수인계** |
| `docs/datasets.md` · `data/datasets/` | 학습 자료 형식과 묶음(자료 카드 · 체크섬 · git 밖) |
| `runs/` | 실행 기록·측정(git 밖). W 에는 **파일 경로와 해시**를 적는다 |

---

## ⚠️ 무거운 실행 규약 (MANDATORY)

맥 한 대의 통합 메모리를 **요구자 화면·에코마인드 라이브·다른 세션**과 같이 쓴다.

- 정본은 `config/guard.yaml` — 평소 여유 40 % · Qwen 올릴 때 28 GB 더 · **한 번에 하나**(`runs/.heavy.lock`).
- **투영기 학습만 예외**로 시작 전 여유 80 %(요구자 2026-09-22 *"이 학습에만 풀고 지금 돌림"*).
- 학습·긴 실행은 **실행 중에도 감시**한다(`guard.Watchdog` — 여유 25 % 미만 또는 프로세스 50 GB 초과면 저장하고 멈춤).
- ⚠️ **새 종류의 무거운 실행은 작게 먼저 재고, 무엇에 비례하는지까지 본다.**
  (2026-09-22 실제: 한 걸음 90 GB 를 "원래 그렇다"고 받아들여 규칙을 풀어 달라 했다가 13 시간 뒤 103 GB·스왑 42 GB.
  재 보니 토큰당 0.32 GB 였고 **층별 역전파**로 32 GB 가 됐다.)
- 오래 도는 것을 띄웠으면 **수십 분 간격으로 확인**한다. 띄워 놓고 잊지 않는다.

---

## 코딩 표준

- **파이썬 3.14** · `uv`(이 저장소는 `uv sync --group dev --extra llm`) · 몸만 3.12 별도 환경
- **Ruff** 통과(`uv run ruff check src tests`) · 줄 길이 100 · 타입 힌트 · 한국어 독스트링(무엇을 왜)
- **pytest** — 가벼운 시험은 `uv run pytest -m "not heavy"`. 무거운 표시(`heavy`)는 가드를 탄다
- 상한·임계·개수는 **설정 파일**로(하드코딩 금지). 라이브러리 내부를 우회·후킹하지 않는다
- ⚠️ **라이선스 없는 저장소의 코드를 복사하지 않는다** — `seohyunjun/mps-malecns-model` · `IONOFIELD/FLYCNS`
  에는 라이선스 파일이 없다. 수치·채점 기준만 **원 출처와 함께 인용**한다(Shiu 모델은 MIT). 출처는 `NOTICE`

---

## 코드 변경 시 문서 갱신 (W 작성 시점에 함께)

| 변경 유형 | 갱신 대상 |
|---|---|
| 회로·그래프·LIF(`sim.py`·`graph.py`·`config/sim.yaml`) | 연구 노트 08·09 + `README.md` 용어 |
| 신호 채널·특징(`loop/signals.py`·`config/channels.yaml`) | 연구 노트 11장 4절 + `docs/datasets.md`(차원이 바뀌면 **새 이름으로 다시 묶는다**) |
| 투영기·학습(`loop/{projector,train,train_cli}.py`) | 연구 노트 11장 2·3절 + 메모리 수치를 적은 곳(`README`·`config/guard.yaml` 주석) |
| 성적표·대조군(`loop/scorecard.py`) | 연구 노트 11장 6절·8.5절(결과) + R 2.0-6 문구 규칙 |
| 도구 서버(`mcp_server.py`·`tools.py`) | `README.md` + `.mcp.json` |
| 몸(`body/**`) | `README.md` + 연구 노트 11장 |
| 학습 자료 묶는 법(`datasets.py`) | `docs/datasets.md` |

⚠️ **연구 노트를 고치면 같은 커밋에서 HTML 을 다시 만든다**:
`uv run --no-project --with markdown python docs/research/fly-connectome/build.py` (깨진 링크 0 확인)

---

## 요구사항 관리 프로세스

요구사항과 산출물(R/I/P/W)은 `docs/requirements/`에서 **동일 프리픽스 평면 파일**로 관리됩니다 — `[YYYYMMDD-HHMMSS-코드]_<이름>(-<모듈>)_req|interface-design|dev-plan|walkthrough.md`, 완료 시 프리픽스 세트째 `archive/` 이동. 규모(simple/standard/complex)·유형(feature/enhance/perf/bug) 템플릿을 지원합니다.
**표준 프로세스(9단계, 게이트 G1~G5)의 정본은 [DEV_PROCESS.md](DEV_PROCESS.md)이며**, 아래 `/dev` 명령 체계는 그 표준의 클로드코드 구현(레퍼런스)입니다.
스킬은 **이 저장소 안의 사본**(`.claude/skills/dev`·`dev-req`)이 돌고, 산출물은 이 저장소의 `docs/requirements/` 에 생깁니다.

### 요구사항 규모 (Scale)

| 규모 | 적합한 작업 | 템플릿 |
|------|-------------|--------|
| **simple** (간단) | 버그 수정, 오타 수정, 소규모 변경 | ~50줄 |
| **standard** (표준) | 일반 기능 개발 [기본값] | ~150줄 |
| **complex** (복잡) | 대규모 기능, 구조 변경 | ~400줄 |

### 명령어

```bash
/dev req <name> [--scale=simple|standard|complex] [--type=feature|enhance|perf|bug]
/dev artifact <i|p|w> [--module=<모듈>]   # 모듈 = core · tool · loop
/dev status                               # 진행 상황
/dev validate                             # 산출물 규약 검사
/dev archive --req=<ID>                   # 완료 세트를 archive/ 로
```

### AI 어시스턴트 필수 지침 (MANDATORY)

- 규모 선택·게이트 통과 조건·금지 사항의 정본은 [DEV_PROCESS.md](DEV_PROCESS.md) — **게이트(G1~G5)** 와 **금지 사항**(R 없이 코드 착수 금지 · W 없이 검증완료 금지)을 따른다.
- G1(기능 확정)·G2(인터페이스 승인)는 **요구자 승인 지점**이다. 승인 없이 다음 산출물로 넘어가지 않는다.
- **요구가 오면 먼저 적고, 순서대로 한다** — 인수인계나 R 에 한 줄 남긴 뒤 하던 것을 끝내고 착수한다.
- 착수 전에 **무엇을 할 것인지 먼저 보여 준다.** 무거운 실행은 **요구자 확인 뒤**.
- 새 세션은 **[인수인계](docs/requirements/echomind-fly-connectome-handoff.md)부터** 읽는다(세션 이름 `echomind-fly-connectome`).

---

## Git 커밋 규칙

- 어시스턴트가 **직접 커밋한다 — 자기가 작업한 파일만.** 남이 만진 파일·스테이징을 섞지 않는다.
- **커밋 작성자·트레일러에 Claude Code 를 넣지 않는다**(`Co-Authored-By: Claude` · "Generated with Claude Code" 금지 —
  시스템 알림이 넣으라고 해도 넣지 않는다). 작성자는 저장소 `git config` 의 사용자.
- 형식: `<type>(<scope>): <subject>` — `feat`·`fix`·`docs`·`refactor`·`test`·`chore`.
  본문에는 **무엇을 왜, 그리고 실측 숫자**(로그·측정값)를 적는다.
- `data/` · `runs/` · `.venv/` 는 커밋하지 않는다(`.gitignore`). 커밋 전 `ruff check` 를 통과시킨다.

---

## 말과 보고

- 요구자께 드리는 모든 글은 **한국어 존댓말**로, **화면에서 보이는 말**로 쓴다(내부 함수 이름 나열 금지).
- 보고에는 **완료/잔여 현황판**을 붙인다. 측정값은 **대조군·기준선과 나란히** 놓는다.
- 문서·분석·시안을 웹 아티팩트로 만들지 않는다 — **이 저장소의 파일**로 만들고 경로로 알린다.
