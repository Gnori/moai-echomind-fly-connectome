#!/usr/bin/env python3
"""
진행 상황 리포트 생성 스크립트

사용법:
    python progress_report.py [--requirement_file=<file>] [--format=<format>]
"""

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

from claude_md_utils import get_project_root


def find_active_requirement(requirements_dir: Path) -> Path:
    """활성 요구사항 파일을 찾습니다."""
    # active_development.md에서 활성 요구사항 찾기
    active_dev_file = requirements_dir / "active_development.md"

    if not active_dev_file.exists():
        return None

    content = active_dev_file.read_text(encoding="utf-8")

    # 활성 요구사항 섹션만 검색 (완료/아카이브 표 오매칭 방지)
    archive_section = content.find("## 완료된")
    if archive_section != -1:
        content = content[:archive_section]

    # 테이블에서 첫 번째 요구사항 추출
    # 패턴: | [폴더/R.md](./폴더/R.md) | 상태 | ... (구형 평면 파일 [파일명.md]도 지원)
    pattern = r'\|\s*\[([^\]]+\.md)\]'
    match = re.search(pattern, content)

    if match:
        filename = match.group(1)
        requirement_file = requirements_dir / filename
        if requirement_file.exists():
            return requirement_file

    return None


def parse_checklist(content: str) -> dict:
    """체크리스트를 파싱하여 진행률을 계산합니다."""
    # 체크리스트 아이템 찾기
    # 패턴: - [ ] 또는 - [x]
    total_items = len(re.findall(r'- \[[x ]\]', content))
    completed_items = len(re.findall(r'- \[x\]', content))

    progress = 0
    if total_items > 0:
        progress = int((completed_items / total_items) * 100)

    return {
        "total": total_items,
        "completed": completed_items,
        "progress": progress
    }


def generate_progress_report(requirement_file: str = "", report_format: str = "markdown") -> dict:
    """
    진행 상황 리포트를 생성합니다.

    Args:
        requirement_file: 요구사항 파일명 (비어있으면 활성 요구사항)
        report_format: 리포트 포맷 (markdown/json)

    Returns:
        리포트 정보 딕셔너리
    """
    project_root = get_project_root()
    requirements_dir = project_root / "docs" / "requirements"

    # 요구사항 파일 찾기
    if requirement_file:
        req_file = requirements_dir / requirement_file
    else:
        req_file = find_active_requirement(requirements_dir)

    if not req_file or not req_file.exists():
        print("❌ 요구사항 파일을 찾을 수 없습니다.")
        if requirement_file:
            print(f"💡 지정한 파일: {requirement_file}")
        else:
            print("💡 활성 요구사항이 없습니다.")
        print("   먼저 요구사항을 생성하세요:")
        print("   /dev-req create <요구사항명>")
        return {}

    # 파일 읽기
    content = req_file.read_text(encoding="utf-8")

    # 메타데이터 파싱
    requirement_name = extract_title(content)
    metadata = extract_metadata(content)
    checklist_stats = parse_checklist(content)

    # 동일 프리픽스 산출물(I/P/W) 스캔 ({프리픽스}_{이름}_req.md → {프리픽스}_*.md)
    artifacts = []
    if req_file.name.endswith("_req.md"):
        prefix = req_file.name.split("_", 1)[0]
        for item in sorted(req_file.parent.glob(f"{prefix}_*.md")):
            if item.name != req_file.name:
                artifacts.append(item.name)

    # 리포트 데이터
    report_data = {
        "requirement_name": requirement_name,
        "file": str(req_file.relative_to(requirements_dir)),
        "artifacts": artifacts,
        "status": metadata.get("status", "알 수 없음"),
        "priority": metadata.get("priority", "알 수 없음"),
        "created_at": metadata.get("created_at", "알 수 없음"),
        "last_updated": metadata.get("last_updated", "알 수 없음"),
        "checklist": checklist_stats,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    # 포맷에 따라 출력
    if report_format == "json":
        print(json.dumps(report_data, ensure_ascii=False, indent=2))
    else:
        print_markdown_report(report_data)

    return report_data


def extract_title(content: str) -> str:
    """문서 제목을 추출합니다."""
    lines = content.split("\n")
    for line in lines:
        if line.startswith("# ") and not line.startswith("## "):
            return line.replace("# ", "").strip()
    return "알 수 없음"


def extract_metadata(content: str) -> dict:
    """메타데이터를 추출합니다."""
    metadata = {}
    lines = content.split("\n")

    for line in lines:
        if "**작성일**:" in line:
            metadata["created_at"] = line.split(":", 1)[1].strip()
        elif "**상태**:" in line:
            metadata["status"] = line.split(":", 1)[1].strip()
        elif "**우선순위**:" in line:
            metadata["priority"] = line.split(":", 1)[1].strip()
        elif "**마지막 업데이트**:" in line:
            metadata["last_updated"] = line.split(":", 1)[1].strip()

    return metadata


def print_markdown_report(data: dict):
    """마크다운 형식으로 리포트를 출력합니다."""
    checklist = data["checklist"]

    print("📊 개발 진행 상황 리포트")
    print("")
    print(f"📋 요구사항: {data['requirement_name']}")
    print(f"📄 파일: {data['file']}")
    print(f"📅 작성일: {data['created_at']}")
    print(f"🔄 상태: {data['status']}")
    print(f"⭐ 우선순위: {data['priority']}")
    print(f"⏱️  마지막 업데이트: {data['last_updated']}")
    print("")

    # 진행률 바 생성
    progress = checklist["progress"]
    bar_length = 30
    filled = int(bar_length * progress / 100)
    bar = "█" * filled + "░" * (bar_length - filled)

    print(f"📈 전체 진행률: {progress}% ({checklist['completed']}/{checklist['total']} 완료)")
    print(f"   [{bar}]")
    print("")

    if progress == 0:
        print("🚀 상태: 시작 전")
        print("💡 다음 단계: 요구사항 분석 및 상세화 시작")
    elif progress < 30:
        print("🔍 상태: 초기 단계")
        print("💡 계속 진행: 요구사항 분석 및 기능 정의 완료하기")
    elif progress < 60:
        print("⚙️  상태: 개발 진행 중")
        print("💡 계속 진행: 핵심 기능 구현 및 테스트")
    elif progress < 90:
        print("🎯 상태: 거의 완료")
        print("💡 마무리: 테스트 및 문서화 완료하기")
    else:
        print("✅ 상태: 완료 단계")
        print("💡 최종 검토: 모든 인수 기준 확인 및 배포 준비")

    print("")
    print("📝 상세 내역:")
    print(f"   ✅ 완료: {checklist['completed']}개")
    print(f"   ⏳ 남음: {checklist['total'] - checklist['completed']}개")
    print("")
    artifacts = data.get("artifacts", [])
    if artifacts:
        print("📦 산출물 (동일 프리픽스):")
        for name in artifacts:
            print(f"   - {name}")
        has_w = any(name.endswith("_walkthrough.md") for name in artifacts)
        if not has_w:
            print("   ⚠️  W(검증 보고) 없음 — W 없이 '검증완료' 전환 금지 (G4)")
        print("")
    print(f"📄 요구사항 파일 열기:")
    print(f"   docs/requirements/{data['file']}")
    print("")
    print(f"⏰ 리포트 생성 시각: {data['generated_at']}")


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(
        description="개발 진행 상황 리포트를 생성합니다."
    )
    parser.add_argument(
        "--requirement_file",
        type=str,
        default="",
        help="요구사항 파일명 (비어있으면 활성 요구사항)"
    )
    parser.add_argument(
        "--format",
        type=str,
        default="markdown",
        choices=["markdown", "json"],
        help="리포트 포맷 (기본값: markdown)"
    )

    args = parser.parse_args()

    # 진행 상황 리포트 생성
    result = generate_progress_report(args.requirement_file, args.format)

    if not result:
        sys.exit(1)


if __name__ == "__main__":
    main()
