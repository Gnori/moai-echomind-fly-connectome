#!/usr/bin/env python3
"""
요구사항 아카이브 이동 — 프리픽스 세트째 archive/<YYYY>/<MM>/W<N>/ 로 이동한다.

사용법:
    python3 archive_requirement.py --req=<ID 또는 프리픽스 일부>   # 활성 → 아카이브
    python3 archive_requirement.py --reorganize                    # 기존 평면 아카이브를 주차 구조로 재배치
    python3 archive_requirement.py --reorganize --dry-run          # 이동 계획만 출력

구조 규약 (DEV_PROCESS.md):
    archive/<YYYY>/<MM>/W<N>/  — 주차는 프리픽스 날짜 기준 ((일-1)//7)+1
    (1~7일=W1, 8~14일=W2, 15~21일=W3, 22~28일=W4, 29~31일=W5)

규칙:
    - 프리픽스 세트(R/I/P/W) 전체를 함께 이동한다 (세트 분리 금지)
    - W(파일 또는 R 내장 간이형) 없이 이동 금지 (게이트 G5)
    - git 추적 파일은 git mv로 이동 (이력 보존)
"""

import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from claude_md_utils import get_project_root

PREFIX_RE = re.compile(r"^(\d{4})(\d{2})(\d{2})-\d{6}-\w+_")


def week_of_month(day: int) -> int:
    return (day - 1) // 7 + 1


def target_dir(archive_root: Path, filename: str):
    m = PREFIX_RE.match(filename)
    if not m:
        return None
    year, month, day = m.group(1), m.group(2), int(m.group(3))
    return archive_root / year / month / f"W{week_of_month(day)}"


def git_tracked(project_root: Path, f: Path) -> bool:
    r = subprocess.run(["git", "ls-files", "--error-unmatch", str(f)],
                       cwd=project_root, capture_output=True)
    return r.returncode == 0


def move(project_root: Path, src: Path, dst_dir: Path, dry_run: bool):
    dst = dst_dir / src.name
    rel_src = src.relative_to(project_root)
    rel_dst = dst.relative_to(project_root)
    if dry_run:
        print(f"  [계획] {rel_src} → {rel_dst}")
        return
    dst_dir.mkdir(parents=True, exist_ok=True)
    if git_tracked(project_root, src):
        subprocess.run(["git", "mv", str(src), str(dst)], cwd=project_root, check=True)
    else:
        src.rename(dst)
    print(f"  ✅ {rel_src} → {rel_dst}")


def archive_active_set(project_root: Path, req_key: str, dry_run: bool) -> int:
    """활성 요구사항 세트를 아카이브로 이동한다."""
    req_dir = project_root / "docs" / "requirements"
    archive_root = req_dir / "archive"

    reqs = [f for f in req_dir.glob("*_req.md") if req_key in f.name]
    if len(reqs) != 1:
        print(f"❌ '--req={req_key}' 매칭 R 문서 {len(reqs)}건 — 정확히 1건이어야 합니다.")
        for f in reqs:
            print(f"   - {f.name}")
        return 1
    req = reqs[0]
    prefix = req.name.split("_")[0]
    siblings = sorted(req_dir.glob(f"{prefix}_*"))

    # G5 가드: W(파일 또는 내장) 확인
    text = req.read_text(encoding="utf-8")
    has_w = any("_walkthrough" in s.name for s in siblings) or \
        bool(re.search(r"^##.*검증", text, re.M))
    if not has_w:
        print(f"❌ W(검증 보고) 없음 — G5 위반. 이동 중단: {req.name}")
        return 1

    dst_dir = target_dir(archive_root, req.name)
    print(f"📦 {prefix} 세트 {len(siblings)}건 → {dst_dir.relative_to(project_root)}")
    for s in siblings:
        move(project_root, s, dst_dir, dry_run)
    if not dry_run:
        print("\n⚠️  후속: active_development.md에서 해당 행을 제거하고 R 상태=아카이브인지 확인하세요 (/dev validate).")
    return 0


def reorganize(project_root: Path, dry_run: bool) -> int:
    """archive/ 평면 파일을 <YYYY>/<MM>/W<N>/ 구조로 재배치한다 (멱등)."""
    archive_root = project_root / "docs" / "requirements" / "archive"
    flat = [f for f in archive_root.iterdir() if f.is_file() and f.suffix == ".md"]
    if not flat:
        print("✅ 평면 파일 없음 — 이미 재배치됨")
        return 0
    moved = skipped = 0
    for f in sorted(flat):
        dst_dir = target_dir(archive_root, f.name)
        if dst_dir is None:
            print(f"  ⚠️ 프리픽스 해석 불가 — 건너뜀: {f.name}")
            skipped += 1
            continue
        move(project_root, f, dst_dir, dry_run)
        moved += 1
    print(f"\n{'계획' if dry_run else '완료'}: 이동 {moved}건, 건너뜀 {skipped}건")
    return 0


def main():
    dry_run = "--dry-run" in sys.argv
    req_key = None
    for a in sys.argv[1:]:
        if a.startswith("--req="):
            req_key = a.split("=", 1)[1]

    project_root = get_project_root()
    if "--reorganize" in sys.argv:
        return reorganize(project_root, dry_run)
    if req_key:
        return archive_active_set(project_root, req_key, dry_run)
    print(__doc__)
    return 1


if __name__ == "__main__":
    sys.exit(main())
