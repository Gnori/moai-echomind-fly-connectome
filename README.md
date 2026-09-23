# moai-echomind-fly-connectome

수컷 초파리 중추신경계 연결 지도(**MaleCNS v1.0**)를 맥에서 돌리고, 그 신경 신호를 **LLM(Qwen3.8-27B)의 입력 임베딩에 직접** 넣는 실험 저장소.
요구사항 R-3340(C 단계 결과) · **R-7469(다음 학습 · 진행 중)** · 인터페이스 계약(I) · 인수인계는 이 저장소의 `docs/requirements/` 에 있다(2026-09-23 moai-echomind 에서 옮겨 왔다).

> **지금(2026-09-23)**: C 단계 결과는 진짜 배선이 섞은 배선을 못 이겼다(먹이 도착 0.2 대 0.2 · 신호에는 정보가 있고 전달이 묻는다).
> 다음 판은 판정 기준(`stats-v2`) · 냄새 좌우(`sim-v2`) · 3차원 입력(`task-v2` · `retino-v1` · `channels-v2`) · 채널별 압축 관문(`compress-v1`) · 작은 투영기(`train-v4`) · 앙상블 성적표(`scorecard-v2`)까지 **코드가 끝났고 자료 v2 를 모으는 중**이다 — [학습 안내서 10장](docs/training/10-학습-기록과-다음.md).

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
| 섞은 배선 · 재배선(rewired) | 연결의 도착 뉴런만 무작위로 섞은 그래프 — 뉴런마다 들어오고 나가는 연결 수는 같다. "초파리 배선 덕인가"를 가르는 대조군 · `rewired-s<K>` 는 씨앗 K · 판정은 **앙상블**(여럿의 분포)과 비교한다 |
| 채널 | LLM 에 넣는 신경 신호 묶음(시각 · 촉각 · 청각 · 후각 · 미각 · 구애/성 회로 · 가치 · 몸 · 운동 · **공간**(channels-v2 · 중심복합체)) |
| 시야 자리(retino-v1) | 시각 투사 뉴런마다 연결 지도로 계산한 **고도** — 입력 기둥 뉴런 좌표의 시냅스 가중 평균. 방위(앞뒤 축)는 미확정 |
| 투영기(projector) | 채널 벡터 → Qwen 은닉 5,120 차원 소프트 토큰. 학습되는 것은 이것뿐(train-v2 2층 MLP 3,187만 · train-v4 압축 벡터 → 선형 수십만) |
| 압축기 · 관문(compress-v1) | 채널마다 PCA 로 16~64 차원(학습 나눔으로만 맞춤 · 기울기 없음) · 줄인 벡터로 판정을 통과한 채널만 투영기에 넣는다 |
| 판정(stats-v2) | 진짜 배선이 "앞선다" = 재배선 앙상블 상위 5 % · 에피소드 부호 뒤집기 순열 Holm p < 0.05 · 찍기보다 높음. 운동 채널 · 냄새 쪽 칸은 판정 밖 |
| 정직 목록 | 결과마다 "배선이 낸 것 / 코드가 더한 것"을 나눈 기록 |
| 같은 쪽 방출(ipsi_release) | `sim-v2` — 후각 수용 뉴런의 같은 쪽 연결만 세기 × 1.4(Gaudry 2013). 연결 수는 그대로 |
| 판 번호 | 설정 · 규칙을 잠그는 이름(`sim-v2` · `task-v2` · `channels-v2` · `retino-v1` · `stats-v2` · `compress-v1` · `train-v4` · `scorecard-v2`). 옛 설정 판은 `config/archive/<판>.yaml` |
| 학습 안내서 | [docs/training/](docs/training/README.md) · 다음 학습 정의 [R-7469](docs/requirements/20260923-104151-7469_fly-learning-next_req.md) |

## 쓰는 법

```bash
uv sync --group dev                 # 파이썬 3.14
uv run fly-connectome download      # 공식 파일 3개(1.1 GB) · 공식 MD5 대조
uv run fly-connectome graph         # data/graph/traced-v1 (+ --rewired-seed 0 · 여럿은 --rewired-seeds 0-4)
uv run fly-connectome bench escape  # 도망 반사 채점 — 무거움(가드: 메모리 40 % · 한 번에 하나)
uv run pytest                       # 시험(56)
uv run fly-connectome retinotopy build                              # 시각 투사 뉴런의 시야 고도(retino-v1 · CPU 1 s)
uv run fly-connectome inspect laterality --types ORN_DM1,ORN_VA2    # 감각 입구의 좌우 비율(CPU)
uv run fly-connectome collect-probe --seeds 0-1 --rewired-seeds 0-4 # 자료 모으기 작은 측정(저장 없음)
uv run fly-connectome collect --name v2main --seeds 0-299 --rewired-seeds 0-4 --workers 3   # 학습 자료 v2(재배선 5)
uv run fly-connectome dataset pack --name fly-loop-v2 --train v2main --eval v2eval          # 오래 둘 자리에 묶기(docs/datasets.md)
uv run fly-connectome linear-probe --dataset fly-loop-v2            # 판정(stats-v2 · CPU)
uv run fly-connectome inspect bump --dataset fly-loop-v2            # 나침반 봉우리 검사(CPU)
uv run fly-connectome compress fit --dataset fly-loop-v2 && uv run fly-connectome compress gate --dataset fly-loop-v2   # 압축 · 관문
uv run fly-connectome train-probe --steps 12                        # 학습 작은 측정(Qwen · 여유 80 %)
uv run fly-connectome train --brain real                            # 투영기 학습(train-v4 · Qwen 얼림) · --brain rewired-s0 …
uv run fly-connectome scorecard --seeds 2000-2039 --configs vector,text,no_circuit,rewired-s0,rewired-s1,rewired-s2,no_llm
bin/run_full.sh                                                     # (v2 순서) 학습 둘 → 캡션 평가 → 성적표
uv run fly-connectome backup                                        # git 밖 결과를 두 곳에 복사 · 대조
```

학습 자료는 `data/datasets/<이름>/` 에 묶어 둔다 — [docs/datasets.md](docs/datasets.md).

`data/` 와 `runs/` 는 커밋하지 않는다. 출처는 [NOTICE](NOTICE).
