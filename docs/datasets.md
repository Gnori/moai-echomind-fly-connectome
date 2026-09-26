# 학습 자료 (R-3340 F-017 · R-7469 F-002)

자료 자체는 **git 에 올리지 않는다**(`data/` 는 gitignore). 이 문서는 어디에 무엇이 있고 어떻게 쓰는지의 기록이다.

## 자리

| 자리 | 무엇 | 오래 두나 |
|---|---|---|
| `runs/dataset-<이름>-w<N>/` | `collect` 가 일꾼마다 남기는 조각 | ❌ 임시(실행 기록 자리) |
| **`data/datasets/<이름>/`** | 조각을 학습(train) · 평가(eval)로 합쳐 묶은 것 + 자료 카드 + 체크섬 | ✅ **다음에도 쓸 자리** |

## 묶음 목록

| 이름 | 학습 | 평가 | 재배선 | 모은 코드 | 설정 판 | 비고 |
|---|---|---|---|---|---|---|
| `fly-loop-v1` | 581 결정(씨앗 0~59 · 60 에피소드) | 105 결정(씨앗 1000~1011) | s0 하나 | `4ac4635` | sim-v1 · loom-v1 · task-v1 · channels-v1 | 2026-09-22 · 교사 60/60 도착 · 75 MB |
| **`fly-loop-v2`** | **2,921** 결정(씨앗 0~299 · 300 에피소드) | **376** 결정(씨앗 1000~1039 · 40) | **s0~s4 다섯** | `80b21b5` | **sim-v2 · loom-v1 · task-v2 · channels-v2 · retino-v1** | 2026-09-23 19:43 묶음 · 교사 300/300 · 40/40 도착 · 물체 175 · 23 · 부딪힘 1 · 0 · `stim.npz` · **1.1 GB** · 체크섬 온전 · 수집 11:21 → 19:33(8.2 h · 일꾼 3) · ⚠️ **냄새 센서는 작은턱수염(0 · 1)** 이었다(task-v2 · 2026-09-26 바로잡음) |
| `fly-loop-v2-ln` | 1,189 결정(씨앗 0~119 · 120) | 376 결정(씨앗 1000~1039 · 40) | s0 하나 | `9b969ec` | sim-v2 · loom-v1 · task-v2.1 · **channels-v3** · retino-v2 | 2026-09-26 20:37 묶음 · **`fly-loop-v2` 의 감각 입력(`stim.npz`)을 뇌만 다시 돌린 것**(몸 · 세계는 v2 그대로 — 센서도 v2 의 작은턱수염 값) · 11 채널 33,450 차원(`smell_ln` 48) · 연구 노트 14 M3 · 179 MB · 체크섬 온전 |
| `fly-loop-v4-pilot` | 529 결정(씨앗 0~59 · 60) | 330 결정(씨앗 1000~1039 · 40) | s0 하나 | `1c7857e` | sim-v2 · loom-v1 · **task-v4** · channels-v3 · retino-v2 | 2026-09-26 22:31 묶음 · 작은 시험(연구 노트 14 M4) · 냄새 센서 더듬이(2 · 3) · 냄새 맡기 구간(처음 4 결정 몸 세움) · 먹이 4~7 mm · 옆 방위 45~90° · 교사 58/60 · 40/40 도착 · 물체 39 · 21 · 부딪힘 13 · 6 · 100 MB · 체크섬 온전 |
| **`fly-loop-v4`** | **1,692** 결정(씨앗 0~199 · 200 에피소드) | **1,708** 결정(씨앗 1000~1199 · 200) | **s0~s2 셋** | `66e5a8d` | sim-v2 · loom-v1 · **task-v4** · channels-v3 · retino-v2 | 2026-09-27 02:09 묶음 · 수집 09-26 22:34 → 09-27 02:11(3 h 36 m · `bin/run_m4_full.sh`) · 냄새 센서 더듬이 · 냄새 맡기 구간 · 교사 197/200 · 199/200 도착 · 물체 121 · 126 · 부딪힘 36 · 34 · 11 채널 33,450 차원 · `stim.npz` · 888 MB · 체크섬 온전 · ⚠️ 자료 카드 · `meta.json` 머리는 "섞은 배선 씨앗 0" 이지만 파일은 `features-rewired-s0~s2.npz` 셋 |

각 묶음의 `README.md`(자료 카드)에 파일 형식 · 채널 차원 · 교사 규칙 · 불러오는 법 · 다시 만드는 법이 있다.

## 형식 (v2 · I-7469 §3.1 — v1 과 다른 점을 표시)

| 파일 | 형식 | v1 과 다른 점 |
|---|---|---|
| `features-real.npz` | 채널 → float32 [결정, 차원] · v1 9 채널 31,950 · v2 10 채널 33,402 · **v3(channels-v3) 11 채널 33,450** | 공간 채널 `space`(1,452) · v3 는 후각 국소 뉴런 `smell_ln`(48) |
| `features-rewired-s<K>.npz` | 같음 · 재배선 씨앗마다 하나 | v1 은 `features-rewired.npz` 하나(= s0). **읽기는 둘 다 받는다** |
| `stim.npz` | `rates` float32 [결정, 구간 3, 자극 뉴런] · `stim_idx` | **새로** — 감각 입력을 남겨 나중에 다른 뇌에 다시 넣을 수 있게(재배선을 늘릴 때 몸을 다시 안 돌린다) |
| `samples.jsonl` | 결정마다 `seed` · `decision` · `t_ms` · `truth` · `caption` · `teacher_action` · `reflex_bins` | `truth` 에 3차원 칸 `looming_elevation_deg` · `heading_deg` · `pos_xy` · `body_euler_deg` |
| `episodes.jsonl` | 에피소드마다 도착 · 시간 · 물체 · 부딪힘 | 같음 |
| `meta.json` | 판 번호 · 코드 커밋 · 채널 단위 이름 · 감각 대응표 · 교사 규칙 | `rewired_seeds` 목록 · 판에 `retino` |
| `README.md` · `SHA256SUMS` | 자료 카드 · 체크섬 | 같음 |

## 쓰는 법

```bash
uv run fly-connectome dataset list                 # 묶음 목록
uv run fly-connectome dataset verify fly-loop-v1   # 체크섬 대조 — "온전"
```

```python
from moai_echomind_fly import datasets
feats, rows = datasets.load("fly-loop-v1", "train", "real")        # "rewired"(= s0) · "eval"
feats, rows = datasets.load("fly-loop-v2", "eval", "rewired-s3")   # 재배선 씨앗 3
datasets.rewired_brains("fly-loop-v2")                             # ["rewired-s0", …, "rewired-s4"]
```

학습 · 평가 명령(`train` · `eval-caption` · `train-probe`)은 `--dataset <묶음 이름>`(기본 **`fly-loop-v2`** · 2026-09-23 부터)으로 **묶음을 직접 읽는다**(2026-09-23 부터 ·
묶음이 없으면 예전처럼 `runs/dataset-<이름>` 조각). 읽은 묶음과 그 `SHA256SUMS` 해시가 투영기 `meta.json` 의 `data` 에 남는다.
판정 · 관문 · 봉우리 검사(`linear-probe` · `compress fit/gate` · `inspect bump`)도 같은 `--dataset` 을 받는다.

## 새로 만들기

```bash
uv run fly-connectome collect-probe --seeds 0-1 --rewired-seeds 0-4                                   # 작은 측정 먼저(저장 없음)
uv run fly-connectome collect --name v2main --seeds 0-299     --rewired-seeds 0-4 --workers 3 --threads 4
uv run fly-connectome collect --name v2eval --seeds 1000-1039 --rewired-seeds 0-4 --workers 3 --threads 4
uv run fly-connectome dataset pack --name fly-loop-v2 --train v2main --eval v2eval --generated-with <커밋>
uv run fly-connectome backup
```

⚠️ 설정 판(`config/*.yaml` 의 version)이나 채널 표가 바뀌면 특징 차원이 달라진다 — **새 이름**으로 묶고 위 표에 한 줄 더한다.
같은 이름은 덮어쓰지 않는다(`pack` 이 거부한다). 옛 판은 `config/archive/<판>.yaml` 에 남아 있어 옛 자료를 다시 돌릴 수 있다.
