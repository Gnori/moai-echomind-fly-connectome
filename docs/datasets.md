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
| `fly-loop-v2` | 약 3 천 결정(씨앗 0~299 · 300 에피소드) | 약 350 결정(씨앗 1000~1039 · 40) | **s0~s4 다섯** | (묶을 때 HEAD) | **sim-v2 · loom-v1 · task-v2 · channels-v2 · retino-v1** | ⏳ **2026-09-23 수집 중**(11:21 시작 · 조각 `v2main` · `v2eval`) — 묶은 뒤 이 줄을 채운다 |

각 묶음의 `README.md`(자료 카드)에 파일 형식 · 채널 차원 · 교사 규칙 · 불러오는 법 · 다시 만드는 법이 있다.

## 형식 (v2 · I-7469 §3.1 — v1 과 다른 점을 표시)

| 파일 | 형식 | v1 과 다른 점 |
|---|---|---|
| `features-real.npz` | 채널 → float32 [결정, 차원] · v1 9 채널 31,950 · v2 10 채널 33,402 | 공간 채널 `space`(1,452) |
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
