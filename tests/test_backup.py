"""백업 — 복사 · 대조 · 없는 볼륨 건너뛰기 · 지우지 않기(가벼운 시험 · 임시 폴더)."""

from __future__ import annotations

from pathlib import Path

from moai_echomind_fly import backup


def _cfg(dest: Path) -> dict:
    return {"version": "t", "items": ["data/datasets", "runs"], "exclude": [".heavy.lock"],
            "destinations": [{"path": str(dest), "why": "시험"},
                             {"path": "/Volumes/__없는_볼륨__/b", "why": "없음"}]}


def test_backup_copies_verifies_and_keeps_old_files(tmp_path):
    root, dest = tmp_path / "repo", tmp_path / "bak"
    (root / "data/datasets/a").mkdir(parents=True)
    (root / "data/datasets/a/x.npz").write_bytes(b"123")
    (root / "runs/p").mkdir(parents=True)
    (root / "runs/p/final.safetensors").write_bytes(b"w" * 1000)
    (root / "runs/.heavy.lock").write_text("1")
    res = backup.backup(_cfg(dest), root, log=lambda m: None)
    assert res[0]["bad"] == [] and res[0]["files"] == 2
    assert res[1]["skipped"] is True                                # 꽂혀 있지 않은 볼륨
    assert (dest / "runs/p/final.safetensors").read_bytes() == b"w" * 1000
    assert not (dest / "runs/.heavy.lock").exists()                 # 잠금 파일은 뺀다
    assert "runs/p/final.safetensors" in (dest / "MANIFEST.sha256").read_text()
    (root / "runs/p/final.safetensors").unlink()                     # 원본에서 지워도
    backup.backup(_cfg(dest), root, log=lambda m: None)
    assert (dest / "runs/p/final.safetensors").exists()             # 백업에는 남는다
