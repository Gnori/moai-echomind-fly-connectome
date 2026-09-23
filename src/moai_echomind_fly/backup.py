"""백업 — git 밖에 있는 비싼 결과(학습 자료 묶음 · 실행 기록)를 다른 자리에 복사한다.

`data/` 와 `runs/` 는 gitignore 로 빠져 있어 이 맥에만 있다. 대상과 둘 곳은 `config/backup.yaml`.
복사는 `rsync -rlt`(권한은 옮기지 않는다 — USB 가 exFAT 이라)이고 **지우지 않는다**:
원본에서 지운 파일도 백업에는 남는다. 복사 뒤 파일마다 SHA-256 을 원본과 대조하고,
둘 곳 뿌리에 `MANIFEST.sha256` 과 `BACKUP.md`(언제 · 어느 커밋 · 몇 개)를 남긴다.
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import time
from pathlib import Path
from typing import Any

from . import config, paths


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _files(root: Path, items: list[str], exclude: set[str]) -> list[Path]:
    """백업 대상 파일(저장소 뿌리 기준 상대 경로)."""
    out = []
    for item in items:
        base = root / item
        if base.is_file():
            out.append(Path(item))
            continue
        for p in sorted(base.rglob("*")):
            if p.is_file() and p.name not in exclude:
                out.append(p.relative_to(root))
    return out


def available(dest: Path) -> bool:
    """둘 곳을 쓸 수 있나 — `/Volumes/<이름>` 아래면 그 볼륨이 꽂혀 있어야 한다."""
    parts = dest.parts
    if len(parts) > 2 and parts[1] == "Volumes":
        return Path("/", "Volumes", parts[2]).is_dir()
    return True


def _git_head(root: Path) -> str:
    try:
        return subprocess.run(["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def backup(cfg: dict[str, Any] | None = None, root: Path | None = None,
           log=print) -> list[dict[str, Any]]:
    """설정의 둘 곳마다 복사하고 대조한다. 둘 곳마다 결과(복사 수 · 다른 파일)를 돌려준다."""
    cfg = cfg or config.load("backup")
    root = root or paths.root()
    exclude = set(cfg.get("exclude") or [])
    files = _files(root, cfg["items"], exclude)
    mb = sum((root / f).stat().st_size for f in files) / 1e6
    log(f"[backup] 대상 {len(files)} 개 · {mb:,.0f} MB · 원본 해시 계산")
    src_sha = {f: _sha(root / f) for f in files}
    results = []
    for d in cfg["destinations"]:
        dest = Path(d["path"]).expanduser()
        if not available(dest):
            log(f"[backup] 건너뜀 {dest} — 볼륨이 꽂혀 있지 않다")
            results.append({"dest": str(dest), "skipped": True})
            continue
        t0 = time.time()
        for item in cfg["items"]:
            (dest / item).parent.mkdir(parents=True, exist_ok=True)
            cmd = ["rsync", "-rlt", *(f"--exclude={e}" for e in sorted(exclude)),
                   f"{root / item}/", f"{dest / item}/"]
            subprocess.run(cmd, check=True)
        bad = [str(f) for f in files
               if not (dest / f).is_file() or _sha(dest / f) != src_sha[f]]
        (dest / "MANIFEST.sha256").write_text(
            "".join(f"{src_sha[f]}  {f}\n" for f in files), encoding="utf-8")
        (dest / "BACKUP.md").write_text(
            f"# moai-echomind-fly-connectome 백업\n\n"
            f"- 마지막 복사: {time.strftime('%Y-%m-%d %H:%M:%S')} · "
            f"저장소 커밋 `{_git_head(root)}`\n"
            f"- 대상: {', '.join(cfg['items'])} · 파일 {len(files)} 개 · "
            f"원본과 다른 파일 {len(bad)} 개\n"
            f"- 까닭: {d.get('why', '')}\n"
            f"- 되살리기: `rsync -rlt <이 폴더>/runs/ <저장소>/runs/` (data/datasets 도 같다) → "
            f"`shasum -a 256 -c MANIFEST.sha256` 를 저장소 뿌리에서\n"
            f"- 복사는 지우지 않는다 — 원본에서 지운 파일도 여기에는 남는다\n",
            encoding="utf-8")
        ok = len(files) - len(bad)
        log(f"[backup] {dest} · {time.time() - t0:.0f} s · 대조 {ok}/{len(files)}"
            + (f" · ⚠️ 다른 파일 {bad[:5]}" if bad else " · 모두 같음"))
        results.append({"dest": str(dest), "skipped": False, "files": len(files), "bad": bad})
    return results


def main(args: argparse.Namespace) -> int:
    res = backup()
    done = [r for r in res if not r["skipped"]]
    return 0 if done and not any(r["bad"] for r in done) else 1


def register(sub) -> None:
    p = sub.add_parser("backup",
                       help="git 밖의 비싼 결과(학습 자료 · runs)를 config/backup.yaml 의 자리로")
    p.set_defaults(func=main)
