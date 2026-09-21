"""시험 공통 — 공식 파일·그래프가 없으면 `data` 표시 시험은 건너뛴다(사유를 남긴다)."""

from __future__ import annotations

import pytest

from moai_echomind_fly import paths


def _graph_ready() -> bool:
    return (paths.graph_dir() / "traced-v1.edges.npz").exists()


def pytest_collection_modifyitems(config, items):
    if _graph_ready():
        return
    skip = pytest.mark.skip(
        reason="data/graph/traced-v1 없음 — `fly-connectome download && fly-connectome graph` 먼저")
    for item in items:
        if "data" in item.keywords:
            item.add_marker(skip)
