"""겹눈 표본(R-1177 · P eye Task 1) — 면 픽셀 방향 · 낱눈 무게."""

import numpy as np

from moai_echomind_fly.eye.compound import Sampler, brightest_dir, face_pixel_dirs
from moai_echomind_fly.eyeworld.world import Body, Entity, World

FACES = World([Entity(id="a", kind="agent", asset=None, pos=np.zeros(3), body=Body())]
              ).eye_rig("a")[0]["faces"]


def test_face_centres_and_right_axis():
    d = face_pixel_dirs(FACES, 2)
    centre = d.mean(axis=(1, 2))
    for k, f in enumerate(FACES):
        assert np.allclose(centre[k] / np.linalg.norm(centre[k]), f["forward"], atol=1e-9)
    front = d[0]                                   # 앞(+z) 면 · 오른쪽 = f × u = −x
    assert front[0, 1, 0] < front[0, 0, 0]         # 칸이 오른쪽으로 갈수록 x 가 작아짐
    assert front[1, 0, 1] > front[0, 0, 1]         # 줄이 위로 갈수록 y 가 커짐


def test_cubemap_covers_all_directions_once():
    d = face_pixel_dirs(FACES, 16).reshape(-1, 3)
    probe = np.random.default_rng(0).normal(size=(500, 3))
    probe /= np.linalg.norm(probe, axis=1, keepdims=True)
    best = (probe @ d.T).max(axis=1)
    assert best.min() > np.cos(np.radians(6))      # 어느 방향이든 가까운 픽셀이 있음


def test_sampler_picks_the_bright_direction():
    n = 32
    img = np.zeros((6, n, n), dtype=np.uint8)
    img[2, 20, 7] = 255                            # 왼쪽 면 한 픽셀
    dirs = face_pixel_dirs(FACES, n)
    target = dirs[2, 20, 7]
    omm = np.stack([target, -target, np.array([0.0, 0.0, 1.0])])
    s = Sampler(FACES, n, sigma_deg=2.0).fit(omm)
    v = s.sample(img)
    assert v[0] > 0.05 and v[1] == 0.0 and v[2] < v[0]
    assert np.allclose(brightest_dir(img, dirs), target)
