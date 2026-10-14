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

## 지금까지 (2026-10-14)

| 단계 | 무엇 | 상태 |
|---|---|---|
| A 시뮬레이터 코어 | 공식 데이터 → 그래프 → Shiu LIF(맥 GPU) → 도망 반사 검산 | ✅ 조건부 통과(Brian2 발화 완전 일치) |
| B 도구 서버 | MCP(stdio) — LLM 이 실험자, 커넥톰이 피험체 | ✅ 조건부 통과 |
| C 닫힌 고리 | 몸 → 회로 → **신호 벡터 → 투영기 → Qwen** → 행동 | ✅ 실행 · R 2.0-6 은 train-v2 · v5 · v6(씨앗 40) · v7 에서 발동 · **train-v6 씨앗 100 에서 처음 통과**(vector 0.46 · 재배선 0.19/0.33/0.42 · p 0.0048) |
| D 뇌 쪽 학습 | 회로 맞추기(F-006) · 버섯체 가소성(F-007) · 촉각엽 고침 · 가치 읽기 | ✅ 조건부(R-7469 검증완료) — 진짜 배선 대조 여섯 · 부분 섞기 · 가치 읽기 A1(몸 없이) 통과 · 몸 행동은 하지 않음 |

⚠️ 통과에 붙는 단서(문서에 함께 쓴다): 같은 자료의 train-v7 은 씨앗 100 에서 못 넘음(p 0.056) ·
vector 는 참값 글(no_circuit 0.47)과 같다(회로 신호가 참값만큼 쓰였다는 뜻이지 그보다 낫다는 뜻 아님) · 재배선은 셋.

그 뒤(10-01~03): 두 층 제어(R-1027 · Qwen 은 사건 때만) — 호출 −82 % · 벽시계 −51 % · 도착 유지 이상이나
같은 호출률 무작위와 구별 안 됨("호출을 줄였다"까지만). 시야 방위 대응표 대조 — 못 넘음(물체 쪽은 좌우 눈으로 실림).
회로 학습(F-006) — 도망 범위는 이미 만족 · KC 희소성 1.0 → 4.8 %(재배선도 4.7 % — 배선 특이적 아님).

그 뒤(10-04~14): 버섯체 출력(MBON) 침묵 풀림(식초에 진짜 32/97 대 재배선 0/97) → 냄새 특이성은 촉각엽 번짐
(발화 PN 겹침 0.83)에 막힘 → 흥분성 LN → PN ×0.5(강한 uPN 57 → 21 %) · KC 다시 맞춤 · η 0.002 →
**진짜 배선 대조 여섯 통과**(짝지은 냄새 −71 % · 안 지은 −19 % · 처벌 뒤집힘 −85 %) · 부분 섞기(그 갈래만)
PN→KC p 0.002 · KC→MBON p 0.010(차 작음) · 가치 읽기 A1(몸 없이) 주 기준 넷 통과.
NPC 하나를 초파리 뇌로(R-5138)는 10-12 보류(배선 특이적인 것은 반사 촉발뿐). 1판 정리는 [docs/model-v1.md](docs/model-v1.md).
⚠️ 쓰지 않는 말: "초파리 뇌가 배운다" · "버섯체 출력이 행동을 정했다"(몸 A2 전) · "섞은 배선(뇌 전체)보다 낫다"
(전체 섞기는 MBON 침묵이라 비교 불가). R-7469 는 10-14 **검증완료** — 성공 가능성이 낮은 것(몸 행동 A2 · 같은 절차 대조 · 대리 기울기 · 공간 목표 · 촉각엽 1판)은 **하지 않음**(요구자 *"성공 가능성이 낮은건 하지마"*).
자세한 내용과 다음 일은 **[인수인계](docs/requirements/echomind-fly-connectome-handoff.md)** 와
**[R-7469](docs/requirements/20260923-104151-7469_fly-learning-next_req.md)** · **[R-1027](docs/requirements/20261001-070937-1027_fly-tiered-control_req.md)**.

---

## 저장소 지도

| 자리 | 무엇 |
|---|---|
| `src/moai_echomind_fly/` | `data`(공식 파일) · `graph`(그래프·섞은 배선) · `sim`(LIF 엔진) · `bench/`(도망 반사·한계) · `mcp_server`·`tools`·`experimenter`(B) · `loop/`(C) · `guard`(무거운 실행) · `datasets`(학습 자료 묶음) |
| `src/moai_echomind_fly/loop/` | `world`·`encoder`·`signals`·`projector`·`qwen`·`decoder`·`episode`·`teacher`·`collect`·`train`·`train_cli`·`scorecard`·`linear_probe` |
| `body/` | 몸 = **FlyGym 1.x · 파이썬 3.12 별도 프로세스**(표준 입출력 JSON 한 줄). 2.1 에는 보행 제어기·냄새가 없다 |
| `scripts/` | 단계 스크립트(00 확인 · 10 자료 · 20 냄새 · 30 학습 · 40 평가 · 50 분석 · 90 정리 · `common.sh`) — 입구는 `scripts/README.md` |
| `config/*.yaml` | `sim`·`loom`·`task`·`channels`·`llm`·`tool`·`guard`·`backup` — **상한·임계·경로를 코드에 박지 않는다** |
| `docs/research/fly-connectome/` | **연구 노트 11편(정본)** — 설계와 결과. 읽기는 `index.html` |
| `docs/training/` | **학습 안내서** — 1부 초파리 뇌(01~05) · 2부 Qwen 연결(06~10) · 부록(11). 무엇을 어떤 순서로 하나 + **학습 대장(10장)**. 읽기는 `index.html` |
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
- **될 가능성을 먼저 보인다** — 벌점 계산 · 몇 분짜리 작은 시험의 숫자를 요구자께 보여 드린 뒤에만 띄운다.
  도중에 낮아 보이면 바로 멈춘다(요구자 2026-10-06 *"가능성이 낮은걸 왜 하냐고"*).
- 무거운 실행 · 재기는 **`scripts/` 단계 스크립트로**(사건 파일 → 대시보드 "최근 실행"). 명령을 직접 치지 않는다.
- ⚠️ **맥 커널 누수** — `data.kalloc.1024` 가 20 GB 에서 다 차면 패닉(2026-10-06 실제 · 하루 약 5 GB).
  긴 실행 전 `uptime` · `zprint data.kalloc.1024` 를 보고, 크면 요구자께 맥 본체 재시동을 부탁드린다.

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
| 투영기·학습(`loop/{projector,train,train_cli}.py`) | 연구 노트 11장 2·3절 + 메모리 수치를 적은 곳(`README`·`config/guard.yaml` 주석) + 학습 안내서 07·08장 |
| **학습을 새로 돌림**(투영기 · 평가 · 성적표) | 학습 안내서 **10장 대장에 한 줄**(손실 · 저장본 sha256 · 섞은 배선과 나란히) |
| 뇌 쪽(그래프·LIF·맞추기·인코더·채널) | 학습 안내서 01~04장의 해당 절 |
| 성적표·대조군(`loop/scorecard.py`) | 연구 노트 11장 6절·8.5절(결과) + R 2.0-6 문구 규칙 |
| 도구 서버(`mcp_server.py`·`tools.py`) | `README.md` + `.mcp.json` |
| 몸(`body/**`) | `README.md` + 연구 노트 11장 |
| 학습 자료 묶는 법(`datasets.py`) | `docs/datasets.md` |

⚠️ **연구 노트를 고치면 같은 커밋에서 HTML 을 다시 만든다**:
`uv run --no-project --with markdown==3.11.0 python docs/research/fly-connectome/build.py` (깨진 링크 0 확인)
학습 안내서도 같다: `uv run --no-project --with markdown==3.11.0 python docs/training/build.py` · 판은 3.11.0 고정(3.11.1 은 한글 앞 굵게를 못 풂 · 다른 판이면 빌드가 멈춤)

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
- `data/` · `runs/` · `.venv/` 는 커밋하지 않는다(`.gitignore` — 큰 산출물 확장자 · 비밀 · 도구 흔적도 막아 둠). 커밋 전 `ruff check` 를 통과시킨다.
- git 밖 결과는 **`uv run fly-connectome backup`** 으로 백업한다(`~/Backups/moai-echomind-fly-connectome` 한 곳 · 대조까지 — 요구자 2026-10-05 USB 는 쓰지 않음). 학습 · 평가를 새로 돌린 뒤 한 번.

---

## 말과 보고

- 요구자께 드리는 모든 글은 **한국어 존댓말**로, **화면에서 보이는 말**로 쓴다(내부 함수 이름 나열 금지).
- 보고에는 **완료/잔여 현황판**을 붙인다. 측정값은 **대조군·기준선과 나란히** 놓는다.
- 문서·분석·시안을 웹 아티팩트로 만들지 않는다 — **이 저장소의 파일**로 만들고 경로로 알린다.
