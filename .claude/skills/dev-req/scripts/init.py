#!/usr/bin/env python3
"""
요구사항 관리 구조 초기화 스크립트

사용법:
    python init.py [--force]
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

from claude_md_utils import ensure_claude_md_with_section, get_project_root


def get_skill_template_dir() -> Path:
    """스킬 템플릿 디렉토리 경로를 반환합니다."""
    skill_dir = Path(__file__).parent.parent
    return skill_dir / "templates"


def load_template(template_path: Path, variables: dict) -> str:
    """템플릿을 로드하고 변수를 치환합니다."""
    if not template_path.exists():
        return None

    content = template_path.read_text(encoding="utf-8")
    for key, value in variables.items():
        content = content.replace(f"{{{{{key}}}}}", str(value))
    return content


def load_claude_md_section_template() -> str:
    """CLAUDE.md 섹션 템플릿을 로드합니다."""
    template_dir = get_skill_template_dir()
    template_path = template_dir / "claude_md_section.md.template"

    if not template_path.exists():
        return None

    return template_path.read_text(encoding="utf-8")


def get_requirements_section() -> str:
    """CLAUDE.md에 추가할 요구사항 관리 섹션을 생성합니다."""
    # 템플릿 로드 시도
    template_content = load_claude_md_section_template()

    if template_content:
        return template_content

    # 템플릿이 없으면 기본 내용 생성 (fallback)
    return """## 요구사항 관리 프로세스

요구사항은 `docs/requirements/`에서 관리되며, 5단계 개발 프로세스를 따릅니다.

### 명령어

```bash
/dev init                              # 요구사항 관리 구조 초기화
/dev req <name>                        # 요구사항 명세서 생성
/dev status                            # 진행 상황 리포트
```

"""


def check_existing_setup() -> dict:
    """기존 설정을 확인합니다."""
    project_root = get_project_root()
    requirements_dir = project_root / "docs" / "requirements"

    result = {
        "requirements_dir_exists": requirements_dir.exists(),
        "existing_files": [],
    }

    if requirements_dir.exists():
        for item in requirements_dir.iterdir():
            if item.is_file():
                result["existing_files"].append(item.name)

    return result


def create_requirements_structure(force: bool = False) -> bool:
    """요구사항 관리 구조를 생성합니다."""
    project_root = get_project_root()
    requirements_dir = project_root / "docs" / "requirements"

    # 기존 설정 확인
    existing = check_existing_setup()

    if existing["requirements_dir_exists"] and not force:
        print("⚠️  docs/requirements/ 디렉토리가 이미 존재합니다.")
        print("💡 덮어쓰려면 --force 옵션을 사용하세요.")
        return False

    print("\n📁 요구사항 관리 구조 생성 중...\n")

    # 디렉토리 생성
    requirements_dir.mkdir(parents=True, exist_ok=True)
    (requirements_dir / "templates").mkdir(exist_ok=True)
    (requirements_dir / "archive").mkdir(exist_ok=True)

    # 현재 날짜
    creation_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    variables = {"CREATION_DATE": creation_date}

    # 템플릿 디렉토리 경로
    template_dir = get_skill_template_dir() / "requirements"

    # README.md 생성
    readme_template = template_dir / "README.md.template"
    readme_content = load_template(readme_template, variables)
    if readme_content:
        (requirements_dir / "README.md").write_text(readme_content, encoding="utf-8")
        print("   ✅ README.md")
    else:
        print("   ⚠️  README.md (템플릿 없음)")

    # active_development.md 생성
    active_template = template_dir / "active_development.md.template"
    active_content = load_template(active_template, variables)
    if active_content:
        (requirements_dir / "active_development.md").write_text(active_content, encoding="utf-8")
        print("   ✅ active_development.md")
    else:
        print("   ⚠️  active_development.md (템플릿 없음)")

    # 템플릿 파일들 복사 (R 규모별 3종 + 산출물 I/P/W 3종)
    templates_subdir = template_dir / "templates"

    for template_path in sorted(templates_subdir.glob("*.md.template")):
        dest = requirements_dir / "templates" / template_path.name
        dest.write_text(template_path.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"   ✅ templates/{template_path.name}")

    return True


def update_claude_md() -> str:
    """CLAUDE.md에 요구사항 관리 섹션을 추가/업데이트합니다."""
    print("\n📄 CLAUDE.md 업데이트 중...")

    section_content = get_requirements_section()
    backup_path = ensure_claude_md_with_section(
        "요구사항 관리 프로세스",
        section_content
    )

    if backup_path:
        print(f"   📦 백업 완료: {backup_path}")

    print("   ✅ 요구사항 관리 프로세스 섹션 추가/업데이트 완료")

    return backup_path


def init_requirements(force: bool = False) -> bool:
    """요구사항 관리를 초기화합니다."""
    print("\n🚀 요구사항 관리 구조 초기화\n")

    # 1. 구조 생성
    if not create_requirements_structure(force):
        return False

    # 2. CLAUDE.md 업데이트
    update_claude_md()

    # 완료 메시지
    print("\n✅ 요구사항 관리 구조 초기화 완료!")

    print("\n📋 생성된 구조:")
    print("   docs/requirements/")
    print("   ├── README.md")
    print("   ├── active_development.md          # 진행 대시보드 (D)")
    print("   ├── templates/                     # R 규모별 3종 + I/P/W 3종")
    print("   ├── [YYYYMMDD-HHMMSS-코드]_<이름>_req.md  # 요구사항(R) — I/P/W는 동일 프리픽스")
    print("   └── archive/                       # 완료 시 프리픽스 세트째 이동")

    print("\n💡 다음 단계:")
    print("   1. 새로운 요구사항 생성: /dev req <요구사항명>  → [프리픽스]_<이름>_req.md 생성")
    print("   2. 산출물 생성: /dev artifact <i|p|w> [--module=<모듈명>]")

    return True


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(
        description="요구사항 관리 구조를 초기화합니다."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="기존 파일 덮어쓰기"
    )

    args = parser.parse_args()

    success = init_requirements(args.force)

    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
