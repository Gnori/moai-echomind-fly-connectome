# moai-echomind-fly-connectome

수컷 초파리 중추신경계 연결 지도(**MaleCNS v1.0**)를 맥에서 돌리고, 그 신경 신호를 **LLM(Qwen3.8-27B)의 입력 임베딩에 직접** 넣는 실험 저장소.
요구사항 R-3340(C 단계 결과) · **R-7469(다음 학습 · 진행 중)** · 인터페이스 계약(I) · 인수인계는 이 저장소의 `docs/requirements/` 에 있다(2026-09-23 moai-echomind 에서 옮겨 왔다).

> **지금(2026-10-14)** — 닫힌 고리(1판)는 10-02 그대로이고, 그 뒤는 대부분 **뇌만**(닫힌 고리와 따로) 쟀다.
>
> **닫힌 고리(10-02)**: train-v6(자료 fly-loop-v4 · 냄새 맡기 구간 과제 task-v4 · 입력 시각 · 후각 국소 뉴런 smell_ln · 공간)과 train-v7(v6 + 캡션 되새김 0.5)까지 돌았다.
> **캡션**: 물체 있음과 함께 **냄새 쪽이 처음 통과**했다(v6 0.530 · v7 0.566 · 섞은 배선 0.46~0.52 · 찍기 0.461).
> **성적표(행동)**: v6 투영기를 씨앗 100 으로 다시 재자 **처음으로 섞은 배선 대조를 통과**했다 — 먹이 도착 vector 0.46 · 섞은 배선 0.19/0.33/0.42 · p 0.0048 → "초파리 배선 벡터가 섞은 배선보다 나았다".
> ⚠️ 단서 셋과 함께만 읽는다: ① 같은 자료 · 입력의 train-v7 은 씨앗 100 에서 p 0.056 으로 못 넘었다 ② vector 0.46 은 참값 글(회로 없음) 0.47 과 같은 수준 — 회로 신호가 참값만큼 쓰였다는 뜻이지 그보다 낫다는 뜻이 아니다 ③ 섞은 배선은 셋뿐 · s2 0.42 는 구간이 겹친다.
> **두 층 제어(R-1027 · 10-03)**: Qwen 은 사건 때만 부른다 — 호출 −82 % · 벽시계 −51 % · 도착 유지 이상. 그러나 같은 호출률 무작위와 구별되지 않았다(p 0.51) → "호출을 줄였다"까지만.
> **뇌만 · 버섯체(10-03 → 10-14)**: KC 희소성 1.0 → 4.8 %(섞은 배선도 4.7 % — 배선 특이적 아님) · MBON 냄새 반응은 진짜만(30/97 대 0/97) · 촉각엽 흥분 연결 ×0.5(모형 수정) 뒤 도파민 짝짓기가 냄새 · 구획 특이적 약화를 냈다(진짜 대조 여섯 통과 · 짝지은 냄새 −71 % 대 안 지은 냄새 −19 %). 뇌 전체 섞기는 MBON 이 침묵해 비교할 수 없어 **한 갈래만 섞은 부분 섞기**(같은 매개변수)와 견줬다 — 특이성 SI 진짜 0.513 대 PN → KC 섞기 0.35 · 0.41 · 0.49(통과) · KC → MBON 섞기 0.46 · 0.48 · 0.47(통과 · 차 작음). **가치 읽기 A1**(몸 없이): MBON 발화를 신경전달물질 부호로 더한 가치가 보상 짝짓기 뒤 +114(10/10) · 처벌 −124 · 무작위 부호 20 벌 모두 아래.
> 쓸 수 있는 말: "진짜 PN → KC 배선이 (그 갈래만) 섞은 배선보다 냄새 특이적 연합 약화를 낸다(같은 매개변수)" · "회로가 만든 가치를 손으로 정한 읽기가 (몸 없이) 짝지은 냄새 쪽으로 옮긴다". 쓰지 않는 말: "초파리 뇌가 배운다" · "버섯체 출력이 행동을 정했다"(몸 A2 전). ⚠️ 단서: 같은 매개변수 대조다(섞은 뇌마다 다시 맞추는 작은 시험은 기준을 못 넘어 본 학습 안 함) · A1 은 KC → MBON 섞기(122~162 · 날 값)보다 작다 · 몸 · 행동으로는 아직 잇지 않았다 — [학습 안내서 03장 6.4 · 6.5](docs/training/03-뇌-맞추기와-검산.md) · [연구 노트 11 §8.13](docs/research/fly-connectome/11-구현-초파리-신호를-Qwen-에.md).
> **1판 정리(2026-10-07)**: 되는 것 · 안 되는 것 · 쓸 수 있는 말 · 비용 · 에코마인드 적용 전에 정할 것 — [docs/model-v1.md](docs/model-v1.md).
> 문구 규칙(R 2.0-6)은 train-v2 · v5 · v6(씨앗 40) · v7 에서 발동했고 v6 씨앗 100 에서 처음 통과했다 — [학습 안내서 10장](docs/training/10-학습-기록과-다음.md).

**학습 안내서** — 초파리 뇌를 만들고 · 맞추고 · 검산하는 법(1부)과 Qwen 에 잇는 투영기 학습(2부)을 하는 순서대로:
[docs/training/](docs/training/README.md)(읽기는 `index.html` · 고친 뒤 `uv run --no-project --with markdown==3.11.0 python docs/training/build.py`).

**연구 노트 11편은 이 저장소 안에 있다** — [docs/research/fly-connectome/](docs/research/fly-connectome/README.md)
(읽기는 `index.html` · 2026-09-23 에 moai-echomind 에서 옮겨 왔다).
고친 뒤에는 `uv run --no-project --with markdown==3.11.0 python docs/research/fly-connectome/build.py` 로 HTML 을 다시 만든다.

> 구도: **초파리 회로 = 인코더(앞) · LLM = 생성기(뒤)** — 이미지 생성(글 인코더 → 디퓨전)의 반대.

## 단계

| 단계 | 무엇 | 관문 |
|---|---|---|
| A 시뮬레이터 코어 | 공식 데이터 → 그래프 → Shiu LIF(맥 GPU) → 알려진 반사로 검산 | 도망 반사 채점 통과 · 섞은 배선 대조군 침묵 |
| B 도구 서버 | MCP(stdio) — LLM 이 실험자, 커넥톰이 피험체 | 클로드 코드가 실험 하나를 끝까지 |
| C 닫힌 고리 | 몸(FlyGym) → 초파리 회로 → **신호 벡터 → 투영기 → Qwen** → 행동 → 몸 | 글 요약 · 대조군 대비 성적표 |
| D NPC(R-5138 · `src/moai_echomind_fly/npc/` · `config/npc.yaml`) | 몸 자리에 **사람 NPC 하나**를 끼움 — 뇌 · 인코더 · 디코더는 그대로 · 세계는 가짜 세계 → 상자 월드(라이브 9212 금지) · 단계 S0 연결 → S1 반사 → S2 목표 → S3 함께 → S4 깊은 판단만 LLM → S5 대화 | 단계마다 진짜 대 섞은 배선(R 2.0-6) · **10-12 S1 에서 보류**(반사 촉발만 배선 특이적 · [1판 정리 6 절](docs/model-v1.md)) |

## 용어

| 말 | 뜻 |
|---|---|
| Traced | 주석 파일의 `status == "Traced"` 뉴런 165,122 개 — 기본 뉴런 집합 |
| 부호 | `consensus_nt` 로 붙인 +1(아세틸콜린) · −1(GABA·글루탐산·히스타민) · 0(조절 물질·미정) |
| 섞은 배선 · 재배선(rewired) | 연결의 도착 뉴런만 무작위로 섞은 그래프 — 뉴런마다 들어오고 나가는 연결 수는 같다. "초파리 배선 덕인가"를 가르는 대조군 · `rewired-s<K>` 는 씨앗 K · 판정은 **앙상블**(여럿의 분포)과 비교한다 |
| 채널 | LLM 에 넣는 신경 신호 묶음(시각 · 촉각 · 청각 · 후각 · 미각 · 구애/성 회로 · 가치 · 몸 · 운동 · **공간**(channels-v2 · 중심복합체) · **후각 국소 뉴런 `smell_ln`**(channels-v3)) |
| smell_ln | 안테나엽 국소 뉴런 4 종(26 뉴런 · 48 차원) — 약한 냄새 좌우 차(5~10 %)를 1 s 이상 두면 가른다. 후각 투사 뉴런은 거의 다 잃는다(연구 노트 14) |
| 냄새 맡기 구간(task-v4) | 에피소드 처음 4 결정(1.2 s) 몸을 세워 두는 구간 — 좌우 차가 일정하게 유지돼 국소 뉴런이 가를 수 있게. 먹이 4~7 mm · 옆 방위 45~90° 와 함께 |
| 냄새 센서(task-v2.1) | FlyGym 1.x 센서 2 · 3 = 왼 · 오른 더듬이. 그 전의 0 · 1 은 작은턱수염이었다(자료 fly-loop-v2 · v5 결과는 그 기준) |
| 시야 자리(retino-v1 · v2) | 시각 투사 뉴런마다 연결 지도로 계산한 **고도**(v1) · **방위**(v2 · 선형 근사 · 과제 task-v5 `retinotopic2`) — 입력 기둥 뉴런 좌표의 시냅스 가중 평균. 방위 대응표를 섞은 대조는 못 넘었다(10-03 · p 0.064) |
| 투영기(projector) | 채널 벡터 → Qwen 은닉 5,120 차원 소프트 토큰. 학습되는 것은 이것뿐(train-v2 2층 MLP 3,187만 · train-v4 압축 벡터 → 선형 수십만 · 1판 train-v6 선형 91.6 만) |
| 압축기 · 관문(compress-v1 → v3) | 채널마다 PCA 로 줄임(v3 = 64 고정 + 채널마다 평균 크기 1 · 학습 나눔으로만 맞춤 · 기울기 없음) · 줄인 벡터로 판정을 통과한 채널만 투영기에 넣는다 |
| 판정(stats-v2 → v3 → v4) | 진짜 배선이 "앞선다" = 재배선 앙상블 상위 5 % · 에피소드 부호 뒤집기 순열 Holm p < 0.05 · 찍기보다 높음. 자료의 과제 판으로 자동(`stats.rule_for_dataset`): 옛 자료 stats-v2(운동 · 냄새 쪽 칸 판정 밖) · task-v4 이상 **stats-v3**(+ 냄새 쪽 칸 · 찍기 대비 p < 0.05) · task-v6 부터 **stats-v4**(+ 선형 검사 절편 · 10-14 잠금 · 지금까지 판정은 그대로) |
| 정직 목록 | 결과마다 "배선이 낸 것 / 코드가 더한 것"을 나눈 기록 |
| 같은 쪽 방출(ipsi_release) | `sim-v2` — 후각 수용 뉴런의 같은 쪽 연결만 세기 × 1.4(Gaudry 2013). 연결 수는 그대로 |
| sim-v3 · CellParams | 회로 학습 자리(2026-10-01 · `cellparams.py`) — 범위 안 **세포 종류마다** 막 시간 상수 · 문턱 + **종류 쌍마다** 연결 배율 α ≥ 0. 연결 수 · 부호는 그대로 · `cell` 이 없으면 sim-v2 와 같은 길 · 처음 값이면 발화 완전 일치(시험) |
| scope(범위) | 무엇을 배우나를 이름으로 잠근 것(`config/fit.yaml`) — `escape-small` = 다가옴 LC/LPLC 13 종 + DNp01 · 02 · 04 · 11 · 출력 쌍 10 · 매개변수 44 · 버섯체 `kc-sparse-a · b · c` → `kc-mbon-a · b · c` → `kc-mbon-al`(촉각엽 고침 고정). 회로 학습 1 단계는 CMA-ES(기울기 없음 · 10-03 부터 실행 · 작은 시험 `fit-small` 먼저) |
| KC · MBON · DAN | 버섯체(냄새 기억 자리) — KC(케니언 세포 약 4,000)가 냄새를 드물게 부호화 · MBON(출력 97) · DAN(도파민 뉴런 · 상 · 벌 신호) |
| 가소성(도파민 짝짓기) | KC 와 도파민이 같이 켜지면 그 KC → MBON 연결이 약해지는 규칙(문헌에서 가져온 가정 · `config/plasticity.yaml`) · CS+ = 짝지은 냄새 · CS− = 짝 안 지은 냄새 · η = 배우는 빠르기(이 뇌에서 보정 · 0.002) |
| 촉각엽 고침(al-fix) | 흥분성(콜린성) 국소 뉴런 → 투사 뉴런 화학 연결 ×0.5 — 냄새마다 PN 이 넓게 켜지던 번짐을 줄인 **모형 수정**(실제는 주로 전기 시냅스 · 연구 노트 16 §11) |
| 부분 섞기(`pnkc-s<K>` · `kcmbon-s<K>`) | 한 갈래(PN → KC 또는 KC → MBON)만 섞고 나머지는 진짜 그대로 둔 뇌(`graph.brain_graph`) — 전체 섞기는 MBON 이 침묵해 가소성을 비교할 수 없어서. 종류가 그대로라 진짜로 맞춘 매개변수를 얹는다(**같은 매개변수** 대조) · 섞은 뇌마다 다시 맞추면 **같은 절차** |
| 특이성 SI | (CS+ 보상 구획 감소) − (CS− 보상 구획 감소) — 짝지은 냄새만 골라 약해졌나 |
| 가치 읽기(vr-a) | MBON 발화에 신경전달물질로 정한 부호(글루탐산 −1 · 아세틸콜린 · GABA +1)를 곱해 더한 가치 V — 몸 없이 잰 것이 A1 · 몸에 잇는 A2 는 대기 |
| 결과 전 잠금 | 실행 전에 판정 기준 · 고르는 규칙을 설정 파일에 적고 커밋한 것 — 결과를 본 뒤 바꾸지 않는다(바꾸면 새 판 이름) |
| NPC 몸 | `npc/body.py` `NpcBody` — `Loop` 의 몸(FlyGym) 자리에 끼우는 같은 모양(`reset` · `step` · `close`). 감각은 FlyGym 과 같은 식으로 NPC 위치에서 계산 · 구동 → 속도 · 회전 · 반사 → 옆으로 비켜서기 |
| 가짜 세계 · 상자 월드 | 가짜 세계 = 시험용 2D(점 · 원 · 직선 이동 · `npc/fake_world.py`) · 상자 월드 = 에코마인드 월드 서버의 시험용 사본(다른 포트 · 라이브 DB 안 씀). **라이브 월드(9212)에는 붙지 않는다**(`forbidden_ports`) |
| lockstep | 세계를 결정 하나(0.3 s)씩만 진행 — 뇌가 실시간보다 느려서(약 16 배) 세계가 기다린다 |
| 축척 0.15 m/mm | 초파리 1 mm = NPC 0.15 m(선언 · 초파리 몸 약 3 mm ↔ 사람 어깨 폭 약 0.45 m) · 구동 1.0 = 12 mm/s = 1.8 m/s |
| 층 구조(R-1027) | 1 반사 → 2 빠른 정책(초파리 신호 선형 읽기) → 3 Qwen 은 **사건 때만**(목표 · 간격 · 충돌 · 애매함 · 놀람). 문턱 `config/control.yaml` control-v2 · 대조 = 같은 호출률 무작위 · 주기 |
| 판 번호 | 설정 · 규칙을 잠그는 이름(지금 `sim-v2` · `task-v5` · `channels-v3` · `retino-v2` · `stats-v3` · `compress-v3` · `train-v6` · `scorecard-v3` · `control-v2` · `npc-v4` · 회로 학습 `sim-v3` · 가소성 잠금 `calib-a` · `pr-a` · `vr-a`). 옛 설정 판은 `config/archive/<판>.yaml` · 무엇을 바꾸면 무엇을 다시 하나는 [학습 안내서 11장 §3](docs/training/11-명령-모음과-용어.md) |
| 학습 안내서 | [docs/training/](docs/training/README.md) · 다음 학습 정의 [R-7469](docs/requirements/20260923-104151-7469_fly-learning-next_req.md) |

## 쓰는 법

```bash
uv sync --group dev                 # 파이썬 3.14
uv run fly-connectome download      # 공식 파일 3개(1.1 GB) · 공식 MD5 대조
uv run fly-connectome graph         # data/graph/traced-v1 (+ --rewired-seed 0 · 여럿은 --rewired-seeds 0-4)
uv run fly-connectome bench escape  # 도망 반사 채점 — 무거움(가드: 메모리 40 % · 한 번에 하나)
uv run pytest                       # 시험(200 · 가벼운 것만은 -m "not heavy" 197)
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
bin/run_full_v4.sh                                                  # v4 전체: 압축 · 관문 → 학습 4 → 캡션 평가 8 → 성적표(--skip-train 은 평가만)
bin/run_full.sh                                                     # (v2 순서) 학습 둘 → 캡션 평가 → 성적표
uv run fly-connectome reflex-probe                                  # 반사 진단(정면 다가옴에 GF 가 켜지나 · 시야 모드 × 상한) · scripts/20-odor/reflex-probe.sh
uv run fly-connectome npc-run --stage s1                            # NPC 하나를 초파리 뇌로(R-5138) · scripts/40-eval/npc-run.sh s1 40
uv run fly-connectome fit --scope kc-mbon-al --method cmaes --brain real   # 회로 학습(sim-v3 · CMA-ES · 무거움) · 작은 시험 먼저 scripts/30-train/fit-small.sh
uv run fly-connectome plasticity-controls --cell-from <회로 학습 폴더> --brain real   # 가소성 대조(뇌만) · --brain pnkc-s0 · kcmbon-s0 = 부분 섞기
scripts/20-odor/valence-a1.sh                                       # 가치 읽기 A1(몸 없이) · 스크립트 전체는 학습 안내서 11장 §2
uv run fly-connectome backup                                        # git 밖 결과(data/datasets · runs)를 ~/Backups/moai-echomind-fly-connectome 한 곳에 복사 · 대조
```

**대시보드**(R-7928 · R-5777 · 읽기 전용 · 127.0.0.1 만 · 표준 라이브러리만) — `uv run fly-connectome dashboard --serve 8770 --every 30` → http://127.0.0.1:8770/ .
`runs/dashboard/` 에 진행 · 결과 한 장(`index.html` · 자료 `state.json`)과 2단계 두 쪽을 같은 주기로 만든다:
`samples.html` 캡션 표본(채점 평가 표본마다 진짜 · 재배선 셋 예측 나란히 · 칸 · 진짜만 맞음/틀림 · 에피소드로 거르기) ·
`episodes.html` 성적표 에피소드 · 결정(씨앗 × 구성 → 결정 표 · 교사와 Qwen 의 도는 쪽이 다르면 표시 — 결정 표는 scorecard-v3 이후 성적표부터 · 층 구조 기록(scorecard-v4)이면 층 · 계기 칸과 LLM 호출률). 값 · 상한은 `config/dashboard.yaml`.

학습 자료는 `data/datasets/<이름>/` 에 묶어 둔다 — [docs/datasets.md](docs/datasets.md).

`data/` 와 `runs/` 는 커밋하지 않는다. 출처는 [NOTICE](NOTICE).
