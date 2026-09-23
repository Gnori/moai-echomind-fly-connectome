# moai-echomind-fly-connectome

수컷 초파리 중추신경계 연결 지도(**MaleCNS v1.0**)를 맥에서 돌리고, 그 신경 신호를 **LLM(Qwen3.8-27B)의 입력 임베딩에 직접** 넣는 실험 저장소.
요구사항 R-3340 · 인터페이스 계약(I) · 인수인계는 이 저장소의 `docs/requirements/` 에 있다(2026-09-23 moai-echomind 에서 옮겨 왔다).

**학습 안내서** — 초파리 뇌를 만들고 · 맞추고 · 검산하는 법(1부)과 Qwen 에 잇는 투영기 학습(2부)을 하는 순서대로:
[docs/training/](docs/training/README.md)(읽기는 `index.html` · 고친 뒤 `uv run --no-project --with markdown python docs/training/build.py`).

**연구 노트 11편은 이 저장소 안에 있다** — [docs/research/fly-connectome/](docs/research/fly-connectome/README.md)
(읽기는 `index.html` · 2026-09-23 에 moai-echomind 에서 옮겨 왔다).
고친 뒤에는 `uv run --no-project --with markdown python docs/research/fly-connectome/build.py` 로 HTML 을 다시 만든다.

> 구도: **초파리 회로 = 인코더(앞) · LLM = 생성기(뒤)** — 이미지 생성(글 인코더 → 디퓨전)의 반대.

## 단계

| 단계 | 무엇 | 관문 |
|---|---|---|
| A 시뮬레이터 코어 | 공식 데이터 → 그래프 → Shiu LIF(맥 GPU) → 알려진 반사로 검산 | 도망 반사 채점 통과 · 섞은 배선 대조군 침묵 |
| B 도구 서버 | MCP(stdio) — LLM 이 실험자, 커넥톰이 피험체 | 클로드 코드가 실험 하나를 끝까지 |
| C 닫힌 고리 | 몸(FlyGym) → 초파리 회로 → **신호 벡터 → 투영기 → Qwen** → 행동 → 몸 | 글 요약 · 대조군 대비 성적표 |

## 용어

| 말 | 뜻 |
|---|---|
| Traced | 주석 파일의 `status == "Traced"` 뉴런 165,122 개 — 기본 뉴런 집합 |
| 부호 | `consensus_nt` 로 붙인 +1(아세틸콜린) · −1(GABA·글루탐산·히스타민) · 0(조절 물질·미정) |
| 섞은 배선(rewired) | 연결의 도착 뉴런만 무작위로 섞은 그래프 — 뉴런마다 들어오고 나가는 연결 수는 같다. "초파리 배선 덕인가"를 가르는 대조군 |
| 채널 | LLM 에 넣는 신경 신호 묶음(시각 · 촉각 · 청각 · 후각 · 미각 · 구애/성 회로 · 가치 · 몸 · 운동) |
| 투영기(projector) | 채널 벡터 → Qwen 은닉 5,120 차원 소프트 토큰. 학습되는 것은 이것뿐 |
| 정직 목록 | 결과마다 "배선이 낸 것 / 코드가 더한 것"을 나눈 기록 |

## 쓰는 법

```bash
uv sync --group dev                 # 파이썬 3.14
uv run fly-connectome download      # 공식 파일 3개(1.1 GB) · 공식 MD5 대조
uv run fly-connectome graph         # data/graph/traced-v1 (+ --rewired-seed 0)
uv run fly-connectome bench escape  # 도망 반사 채점 — 무거움(가드: 메모리 40 % · 한 번에 하나)
uv run pytest                       # 시험
uv run fly-connectome collect --name main --seeds 0-59 --workers 3   # C: 학습 자료 모으기
uv run fly-connectome dataset pack --name fly-loop-v1               # 오래 둘 자리에 묶기(docs/datasets.md)
uv run fly-connectome train --brain real                            # 투영기 학습(Qwen 얼림 · 최대 약 32 GB)
bin/run_full.sh                                                     # 학습 둘 → 캡션 평가 → 성적표
```

학습 자료는 `data/datasets/<이름>/` 에 묶어 둔다 — [docs/datasets.md](docs/datasets.md).

`data/` 와 `runs/` 는 커밋하지 않는다. 출처는 [NOTICE](NOTICE).
