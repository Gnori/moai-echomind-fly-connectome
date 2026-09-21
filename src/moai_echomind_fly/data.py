"""공식 파일 받기와 대조 (R-3340 F-001).

무결성 기준은 구글 클라우드 스토리지가 주는 MD5(공식)다. 받으면서 SHA-256 도 계산해
`data/raw/SHA256SUMS` 에 남기고, 대조용 값(mps-malecns-model 이 계산한 것)과 다르면 경고한다.
"""

from __future__ import annotations

import base64
import hashlib
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from . import config, paths

_BLOCK = 8 * 1024 * 1024


@dataclass(frozen=True)
class SourceFile:
    key: str
    name: str
    url: str
    size: int
    md5_b64: str
    sha256_reference: str


def sources() -> list[SourceFile]:
    cfg = config.load("sources")
    return [
        SourceFile(
            key=key,
            name=f["name"],
            url=cfg["base_url"] + f["name"],
            size=int(f["bytes"]),
            md5_b64=f["md5_b64"],
            sha256_reference=f.get("sha256_reference", ""),
        )
        for key, f in cfg["files"].items()
    ]


def digests(path: Path) -> tuple[str, str]:
    """파일의 (MD5 base64, SHA-256 16진)."""
    md5, sha = hashlib.md5(), hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(_BLOCK), b""):
            md5.update(block)
            sha.update(block)
    return base64.b64encode(md5.digest()).decode(), sha.hexdigest()


def verify(src: SourceFile, path: Path) -> str:
    """크기와 공식 MD5 를 대조하고 SHA-256 을 돌려준다. 다르면 ValueError."""
    size = path.stat().st_size
    if size != src.size:
        raise ValueError(f"{src.name}: 크기 {size} ≠ 공식 {src.size}")
    md5, sha = digests(path)
    if md5 != src.md5_b64:
        raise ValueError(f"{src.name}: MD5 {md5} ≠ 공식 {src.md5_b64}")
    return sha


def download(directory: Path | None = None, *, log=print) -> dict[str, str]:
    """공식 파일 3개를 받는다. 이미 있고 대조가 맞으면 받지 않는다.

    Returns:
        파일 이름 → SHA-256.

    Raises:
        ValueError: 받은 파일이 공식 크기·MD5 와 다를 때(그 파일은 남기지 않는다).
    """
    directory = directory or paths.raw_dir()
    directory.mkdir(parents=True, exist_ok=True)
    result: dict[str, str] = {}
    for src in sources():
        target = directory / src.name
        if target.exists():
            try:
                result[src.name] = verify(src, target)
                log(f"[data] 있음 · 공식 MD5 일치: {src.name}")
                continue
            except ValueError as err:
                log(f"[data] 있는 파일이 다르다 — 다시 받는다: {err}")
                target.unlink()
        part = target.with_name(target.name + ".part")
        log(f"[data] 받는 중: {src.name} ({src.size / 1e6:.1f} MB)")
        try:
            with urllib.request.urlopen(src.url, timeout=120) as resp, part.open("wb") as out:
                got, next_report = 0, 0.1
                for block in iter(lambda: resp.read(_BLOCK), b""):
                    out.write(block)
                    got += len(block)
                    if got / src.size >= next_report:
                        log(f"[data]   {got / src.size:5.1%}")
                        next_report += 0.1
            result[src.name] = verify(src, part)
            part.replace(target)
        finally:
            part.unlink(missing_ok=True)
        log(f"[data] 공식 MD5 일치: {src.name}")
    for src in sources():
        ref = src.sha256_reference
        if ref and result.get(src.name) != ref:
            print(f"[data] ⚠️ SHA-256 이 대조용 값과 다르다: {src.name}", file=sys.stderr)
    (directory / "SHA256SUMS").write_text(
        "".join(f"{sha}  {name}\n" for name, sha in sorted(result.items())), encoding="utf-8"
    )
    return result
