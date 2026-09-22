#!/usr/bin/env python3
"""
CLAUDE.md 관리 유틸리티

여러 스킬에서 공유하는 CLAUDE.md 백업, 읽기, 섹션 업데이트 기능
"""

import re
import shutil
from datetime import datetime
from pathlib import Path


def get_project_root() -> Path:
    """프로젝트 루트 디렉토리를 반환합니다.

    하위 디렉토리에서 실행해도(코덱스 등 스킬 없이 CLI 직접 호출) 루트를 정확히 찾도록,
    스크립트 위치(<root>/.claude/skills/dev-req/scripts/)에서 역산하고,
    구조가 다르면 cwd에서 위로 올라가며 저장소 마커(.git/AGENTS.md)를 찾는다.
    """
    script_root = Path(__file__).resolve().parents[4]
    if (script_root / ".git").exists() or (script_root / "AGENTS.md").exists():
        return script_root
    cur = Path.cwd()
    for candidate in [cur, *cur.parents]:
        if (candidate / ".git").exists() or (candidate / "AGENTS.md").exists():
            return candidate
    return cur


def get_backup_dir() -> Path:
    """백업 디렉토리 경로를 반환합니다."""
    backup_dir = get_project_root() / ".claude" / ".backup"
    backup_dir.mkdir(parents=True, exist_ok=True)
    return backup_dir


def backup_claude_md() -> str:
    """
    CLAUDE.md를 백업합니다.

    Returns:
        백업 파일 경로 (백업 안 했으면 빈 문자열)
    """
    claude_md = get_project_root() / "CLAUDE.md"

    if not claude_md.exists():
        return ""

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup_path = get_backup_dir() / f"CLAUDE.md.{timestamp}.bak"

    shutil.copy(claude_md, backup_path)
    return str(backup_path)


def read_claude_md() -> str:
    """
    CLAUDE.md 내용을 읽습니다.

    Returns:
        파일 내용 (없으면 빈 문자열)
    """
    claude_md = get_project_root() / "CLAUDE.md"

    if not claude_md.exists():
        return ""

    return claude_md.read_text(encoding="utf-8")


def write_claude_md(content: str) -> None:
    """
    CLAUDE.md에 내용을 저장합니다.

    Args:
        content: 저장할 내용
    """
    claude_md = get_project_root() / "CLAUDE.md"
    claude_md.write_text(content, encoding="utf-8")


def find_section(content: str, section_title: str) -> tuple:
    """
    특정 섹션의 시작과 끝 위치를 찾습니다.

    Args:
        content: 전체 내용
        section_title: 섹션 제목 (## 제외)

    Returns:
        (시작 위치, 끝 위치) 튜플, 없으면 (-1, -1)
    """
    # ## 섹션제목 패턴 찾기
    pattern = rf'^## {re.escape(section_title)}\s*$'
    match = re.search(pattern, content, re.MULTILINE)

    if not match:
        return (-1, -1)

    start = match.start()

    # 다음 ## 섹션 찾기
    next_section = re.search(r'^## ', content[match.end():], re.MULTILINE)

    if next_section:
        end = match.end() + next_section.start()
    else:
        end = len(content)

    return (start, end)


def update_section(content: str, section_title: str, section_content: str) -> str:
    """
    특정 섹션을 업데이트하거나 추가합니다.

    Args:
        content: 기존 CLAUDE.md 내용
        section_title: 섹션 제목 (## 제외)
        section_content: 새 섹션 내용 (## 제목 포함)

    Returns:
        업데이트된 전체 내용
    """
    start, end = find_section(content, section_title)

    if start == -1:
        # 섹션이 없으면 끝에 추가
        if content and not content.endswith("\n\n"):
            if content.endswith("\n"):
                content += "\n"
            else:
                content += "\n\n"
        return content + section_content
    else:
        # 섹션이 있으면 교체
        return content[:start] + section_content + content[end:]


def create_minimal_claude_md(sections: list) -> str:
    """
    최소한의 CLAUDE.md를 생성합니다.

    Args:
        sections: 섹션 내용 리스트

    Returns:
        생성된 내용
    """
    header = """# CLAUDE.md

이 문서는 AI 어시스턴트(Claude)를 위한 프로젝트 가이드입니다.

"""
    return header + "\n\n".join(sections)


def ensure_claude_md_with_section(section_title: str, section_content: str) -> str:
    """
    CLAUDE.md가 있으면 섹션 추가/업데이트, 없으면 새로 생성합니다.
    백업도 자동으로 수행합니다.

    Args:
        section_title: 섹션 제목 (## 제외)
        section_content: 섹션 내용 (## 제목 포함)

    Returns:
        백업 파일 경로 (백업 안 했으면 빈 문자열)
    """
    existing_content = read_claude_md()
    backup_path = ""

    if existing_content:
        # 백업
        backup_path = backup_claude_md()
        # 섹션 업데이트
        new_content = update_section(existing_content, section_title, section_content)
    else:
        # 새로 생성
        new_content = create_minimal_claude_md([section_content])

    write_claude_md(new_content)
    return backup_path
