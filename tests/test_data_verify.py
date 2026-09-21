"""F-001: 크기·공식 MD5 가 다르면 실패한다(받은 파일을 남기지 않는 길의 핵심)."""

from __future__ import annotations

import pytest

from moai_echomind_fly import data


def _src(tmp_path, content: bytes) -> data.SourceFile:
    import base64
    import hashlib

    return data.SourceFile("k", "f.bin", "http://x", len(content),
                           base64.b64encode(hashlib.md5(content).digest()).decode(), "")


def test_verify_accepts_matching_file(tmp_path):
    f = tmp_path / "f.bin"
    f.write_bytes(b"connectome")
    sha = data.verify(_src(tmp_path, b"connectome"), f)
    assert len(sha) == 64


def test_verify_rejects_corrupted_file(tmp_path):
    f = tmp_path / "f.bin"
    f.write_bytes(b"connectomE")  # 같은 크기 · 다른 내용
    with pytest.raises(ValueError, match="MD5"):
        data.verify(_src(tmp_path, b"connectome"), f)


def test_verify_rejects_wrong_size(tmp_path):
    f = tmp_path / "f.bin"
    f.write_bytes(b"conn")
    with pytest.raises(ValueError, match="크기"):
        data.verify(_src(tmp_path, b"connectome"), f)


def test_sources_are_the_three_official_files():
    names = [s.name for s in data.sources()]
    assert len(names) == 3 and all(n.endswith(".feather") for n in names)
