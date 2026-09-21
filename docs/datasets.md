# 학습 자료 (R-3340 F-017)

자료 자체는 **git 에 올리지 않는다**(`data/` 는 gitignore). 이 문서는 어디에 무엇이 있고 어떻게 쓰는지의 기록이다.

## 자리

| 자리 | 무엇 | 오래 두나 |
|---|---|---|
| `runs/dataset-<이름>-w<N>/` | `collect` 가 일꾼마다 남기는 조각 | ❌ 임시(실행 기록 자리) |
| **`data/datasets/<이름>/`** | 조각을 학습(train) · 평가(eval)로 합쳐 묶은 것 + 자료 카드 + 체크섬 | ✅ **다음에도 쓸 자리** |

## 묶음 목록

| 이름 | 학습 | 평가 | 모은 코드 | 설정 판 | 비고 |
|---|---|---|---|---|---|
| `fly-loop-v1` | 581 결정(씨앗 0~59 · 60 에피소드) | 105 결정(씨앗 1000~1011) | `4ac4635` | sim-v1 · loom-v1 · task-v1 · channels-v1 | 2026-09-22 · 교사 60/60 도착 · 75 MB |

각 묶음의 `README.md`(자료 카드)에 파일 형식 · 채널 차원 · 교사 규칙 · 불러오는 법 · 다시 만드는 법이 있다.

## 쓰는 법

```bash
uv run fly-connectome dataset list                 # 묶음 목록
uv run fly-connectome dataset verify fly-loop-v1   # 체크섬 대조 — "온전"
```

```python
from moai_echomind_fly import datasets
feats, rows = datasets.load("fly-loop-v1", "train", "real")    # "rewired" · "eval"
```

## 새로 만들기

```bash
uv run fly-connectome collect --name main --seeds 0-59 --workers 3 --threads 4
uv run fly-connectome collect --name eval --seeds 1000-1011 --workers 3 --threads 4
uv run fly-connectome dataset pack --name fly-loop-v2 --train main --eval eval --generated-with <커밋>
```

⚠️ 설정 판(`config/*.yaml` 의 version)이나 채널 표가 바뀌면 특징 차원이 달라진다 — **새 이름**으로 묶고 위 표에 한 줄 더한다.
같은 이름은 덮어쓰지 않는다(`pack` 이 거부한다).
