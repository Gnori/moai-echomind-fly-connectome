"""겹눈 표본 — 큐브맵(눈마다 6 면) → 낱눈 방향마다 밝기(설계 문서 I 3.2 · P eye Task 1).

면 픽셀 방향: 면의 forward f · up u · 오른쪽 r = f × u(오른손 · 화면 오른쪽).
픽셀 (줄 i · 줄 0 = 아래, 칸 j) → d = norm(f + x·r + y·u)
  x = (j + ½)/N·2 − 1 · y = (i + ½)/N·2 − 1 (90° 면).
낱눈 밝기 = 받아들이는 각 가우시안(σ) 무게 평균 — 무게는 시작 때 한 번 계산.
"""

from __future__ import annotations

import numpy as np


def face_pixel_dirs(faces: list[dict], n: int) -> np.ndarray:
    """눈 하나의 6 면 → 픽셀 방향 [6, N, N, 3](세계 좌표 · 단위)."""
    c = (np.arange(n) + 0.5) / n * 2.0 - 1.0
    y, x = np.meshgrid(c, c, indexing="ij")  # [i 줄(아래 → 위), j 칸(왼 → 오)]
    out = np.zeros((len(faces), n, n, 3))
    for k, f in enumerate(faces):
        fw, up = np.asarray(f["forward"], float), np.asarray(f["up"], float)
        r = np.cross(fw, up)
        d = fw[None, None, :] + x[..., None] * r + y[..., None] * up
        out[k] = d / np.linalg.norm(d, axis=-1, keepdims=True)
    return out


def brightest_dir(img: np.ndarray, dirs: np.ndarray) -> np.ndarray:
    """눈 하나의 밝기 [6, N, N] 에서 가장 밝은 픽셀 방향(방향 확인용)."""
    k = int(np.argmax(img))
    return dirs.reshape(-1, 3)[k]


class Sampler:
    """낱눈 방향(세계 좌표 · [M, 3]) → 큐브맵 픽셀 무게(희소: 각 낱눈의 상위 K 픽셀)."""

    def __init__(self, faces: list[dict], n: int, sigma_deg: float, top_k: int = 64) -> None:
        self.n = n
        self.dirs = face_pixel_dirs(faces, n).reshape(-1, 3)
        self.sigma = np.radians(sigma_deg)
        self.top_k = top_k
        self.idx: np.ndarray | None = None
        self.w: np.ndarray | None = None

    def fit(self, omm_dirs: np.ndarray) -> Sampler:
        cosang = np.clip(omm_dirs @ self.dirs.T, -1.0, 1.0)  # [M, 6N²]
        k = min(self.top_k, cosang.shape[1])
        idx = np.argpartition(-cosang, k - 1, axis=1)[:, :k]
        ang = np.arccos(np.take_along_axis(cosang, idx, axis=1))
        w = np.exp(-0.5 * (ang / self.sigma) ** 2)
        self.idx, self.w = idx, w / w.sum(axis=1, keepdims=True)
        return self

    def sample(self, img: np.ndarray) -> np.ndarray:
        """눈 하나의 밝기 [6, N, N](u8 또는 실수) → 낱눈 밝기 [M](0~1)."""
        flat = img.reshape(-1).astype(np.float64) / 255.0
        return (flat[self.idx] * self.w).sum(axis=1)
