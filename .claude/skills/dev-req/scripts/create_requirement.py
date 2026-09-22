#!/usr/bin/env python3
"""
요구사항 명세서 생성 스크립트

사용법:
    python create_requirement.py <requirement_name> [--priority=<priority>] [--scale=<scale>]
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

from claude_md_utils import get_project_root


# 규모별 템플릿 정의
SCALE_TEMPLATES = {
    "simple": {
        "name": "간단",
        "file": "simple.md.template",
        "description": "버그 수정, 소규모 변경",
        "keywords": ["버그", "수정", "오타", "fix", "bug", "patch", "hotfix", "typo"],
    },
    "standard": {
        "name": "표준",
        "file": "standard.md.template",
        "description": "일반적인 기능 개발",
        "keywords": ["추가", "구현", "개발", "feature", "add", "implement", "create"],
    },
    "complex": {
        "name": "복잡",
        "file": "complex.md.template",
        "description": "대규모 기능, 아키텍처 변경",
        "keywords": ["아키텍처", "마이그레이션", "리팩토링", "통합", "architecture", "migration", "refactor", "integration", "redesign"],
    },
}

SCALE_NAMES = {
    "simple": "간단 (simple)",
    "standard": "표준 (standard)",
    "complex": "복잡 (complex)",
}

# 요구사항 유형 정의 — 규모(프로세스 게이팅)와 직교하는 내용 가이드 축.
# 공통 골격(메타·상태·게이트·산출물 계약)은 규모 템플릿이 담당하고,
# 유형은 1장 분석의 관점(가이드 블록)만 바꾼다.
TYPE_DEFS = {
    "feature": {
        "name": "신규기능",
        "keywords": ["신규", "추가", "구축", "개발", "feature", "add", "new", "create", "구현"],
        "section": """
> **[유형 가이드 — 신규기능]** 1장 작성 시 반드시 다룰 것:
> - **사용자 시나리오**: 누가, 어떤 상황에서, 무엇을 하려는가
> - **신규 메뉴/화면 여부**와 인증·접근제어 반영 계획 (게이트웨이 JWT 등)
> - **기존 기능과의 경계**: 어떤 기존 흐름에 편입되는가, 완전 독립인가

---
""",
    },
    "enhance": {
        "name": "기능개선",
        "keywords": ["개선", "변경", "확장", "고도화", "enhance", "improve", "update", "리뉴얼"],
        "section": """
> **[유형 가이드 — 기능개선]** 1장 작성 시 반드시 다룰 것:
> - **As-Is**: 현행 동작 (코드/화면 기준 사실 — 추측 금지)
> - **To-Be**: 개선 후 동작과 그 차이
> - **영향 범위**: 이 동작에 의존하는 다른 기능·API·사용자 습관, **하위 호환성** 유지 여부
> - **결정 번복 확인**: 기존 결정을 변경하면 해당 요구사항 폴더(또는 레거시 문서)를 번복 대상으로 링크 (DEV_PROCESS 공통규칙 2)

---
""",
    },
    "perf": {
        "name": "성능개선",
        "keywords": ["성능", "속도", "최적화", "튜닝", "perf", "performance", "optimize", "지연", "메모리"],
        "section": """
> **[유형 가이드 — 성능개선]** 1장 작성 시 반드시 다룰 것:
> - **현재 지표**: 실측값 + 측정 방법/환경/데이터 규모 (측정 로그를 R 또는 W에 첨부)
> - **목표 지표**: 수치로 (예: p95 응답 2s → 500ms). 수치 없는 성능 목표는 인수기준이 될 수 없다
> - **측정 재현 방법**: 개선 전후 동일 조건으로 비교할 절차
> - **기능 회귀 방지**: 성능을 위해 동작이 달라지는 지점이 있는지 명시

---
""",
    },
    "bug": {
        "name": "버그",
        "keywords": ["버그", "수정", "오류", "오타", "fix", "bug", "patch", "hotfix", "깨짐", "에러"],
        "section": """
> **[유형 가이드 — 버그]** 1장 작성 시 반드시 다룰 것:
> - **재현 절차**: 단계별로, 재현 환경/계정/데이터 포함
> - **기대 동작 vs 실제 동작**
> - **원인 분석**: 증상이 아니라 근본 원인 (파일:라인까지 — 원인 불명이면 "조사 중" 명시)
> - **회귀 범위**: 같은 원인이 영향을 주는 다른 지점, 재발 방지 테스트 계획

---
""",
    },
}

TYPE_NAMES = {k: f"{v['name']} ({k})" for k, v in TYPE_DEFS.items()}


def recommend_type(requirement_name: str) -> str:
    """요구사항 이름을 기반으로 유형을 추천합니다."""
    name_lower = requirement_name.lower()
    for type_key in ("bug", "perf", "enhance", "feature"):
        for keyword in TYPE_DEFS[type_key]["keywords"]:
            if keyword in name_lower:
                return type_key
    return "feature"


def recommend_scale(requirement_name: str) -> str:
    """요구사항 이름을 기반으로 규모를 추천합니다."""
    name_lower = requirement_name.lower()

    # complex 키워드 먼저 검사 (우선순위 높음)
    for keyword in SCALE_TEMPLATES["complex"]["keywords"]:
        if keyword in name_lower:
            return "complex"

    # simple 키워드 검사
    for keyword in SCALE_TEMPLATES["simple"]["keywords"]:
        if keyword in name_lower:
            return "simple"

    # 기본값은 standard
    return "standard"


def get_template_path(scale: str) -> Path:
    """규모에 따른 요구사항 템플릿 경로를 반환합니다."""
    project_root = get_project_root()
    template_file = SCALE_TEMPLATES[scale]["file"]
    return project_root / "docs" / "requirements" / "templates" / template_file


def get_skill_template_path(scale: str) -> Path:
    """스킬 디렉토리의 템플릿 경로를 반환합니다."""
    skill_dir = Path(__file__).parent.parent
    template_file = SCALE_TEMPLATES[scale]["file"]
    return skill_dir / "templates" / "requirements" / "templates" / template_file


def create_requirement_spec(
    requirement_name: str,
    priority: str = "중간",
    scale: str = "standard",
    req_id: str = None,
    req_type: str = "feature"
) -> str:
    """
    요구사항 명세서를 생성합니다.

    Args:
        requirement_name: 요구사항 명칭
        priority: 우선순위 (높음/중간/낮음)
        scale: 규모 (simple/standard/complex) — 프로세스 게이팅 축
        req_id: 도메인코드 (1xxx=에이전트·핵심 기능, 2xxx=도구·연동(MCP), 3xxx=인증·권한·인프라). 생략 시 밀리초 기반
        req_type: 유형 (feature/enhance/perf/bug) — 내용 가이드 축

    Returns:
        생성된 파일의 경로
    """
    project_root = get_project_root()

    # 디렉토리 경로
    requirements_dir = project_root / "docs" / "requirements"

    # 템플릿 파일 찾기 (프로젝트 내 → 스킬 디렉토리 순서)
    template_file = get_template_path(scale)
    if not template_file.exists():
        template_file = get_skill_template_path(scale)

    # 요구사항 파일 생성 (YYYYMMDD-HHMMSS-코드_요구사항명_req.md — 유형 토큰은 파일명 끝)
    # 산출물(I/P/W)도 같은 디렉토리에 동일 프리픽스로 생성된다:
    #   {프리픽스}_{이름}_interface-design.md / {프리픽스}_{이름}-{모듈}_dev-plan.md / {프리픽스}_{이름}-{모듈}_walkthrough.md
    now = datetime.now()
    if req_id:
        code = req_id
    else:
        code = now.strftime("%f")[:4]  # 밀리초 기반 (팀 관습은 --id로 도메인코드 지정 권장)
    timestamp = f"{now.strftime('%Y%m%d-%H%M%S')}-{code}"
    filename = f"{timestamp}_{requirement_name}_req.md"
    output_file = requirements_dir / filename

    # 디렉토리 존재 확인
    if not requirements_dir.exists():
        print("❌ 요구사항 디렉토리가 없습니다.")
        print("💡 먼저 초기화를 실행해주세요: /dev-req init")
        return ""

    # 템플릿 파일 확인
    if not template_file.exists():
        print(f"❌ 템플릿 파일이 없습니다: {template_file}")
        print("💡 먼저 초기화를 실행해주세요: /dev-req init")
        return ""

    # 중복 확인
    if output_file.exists():
        print(f"⚠️  동일한 파일이 이미 존재합니다: {filename}")
        print("💡 다른 이름을 사용하거나 잠시 후 다시 시도해주세요.")
        return ""

    # 템플릿 읽기
    template_content = template_file.read_text(encoding="utf-8")

    # 메타데이터 치환
    created_at = now.strftime("%Y-%m-%d %H:%M:%S")
    content = template_content.replace("{{REQUIREMENT_NAME}}", requirement_name)
    content = content.replace("{{CREATION_DATE}}", created_at)
    content = content.replace("{{STATUS}}", "작성")
    content = content.replace("{{PRIORITY}}", priority)
    content = content.replace("{{REQ_ID}}", code)
    content = content.replace("{{REQ_TYPE}}", TYPE_NAMES[req_type])
    content = content.replace("{{TYPE_SECTION}}", TYPE_DEFS[req_type]["section"])

    # 파일 생성
    output_file.write_text(content, encoding="utf-8")

    # active_development.md 업데이트
    update_active_development(project_root, filename, requirement_name, created_at, scale)

    # 성공 메시지
    scale_info = SCALE_TEMPLATES[scale]
    print("✅ 요구사항 명세서 생성 완료!")
    print(f"📄 파일: {output_file.relative_to(project_root)}")
    print(f"📝 요구사항: {requirement_name}")
    print(f"📊 규모: {SCALE_NAMES[scale]}")
    print(f"🏷️  유형: {TYPE_NAMES[req_type]}")
    print(f"🎯 우선순위: {priority}")
    print(f"⏰ 생성 시각: {created_at}")
    print("")

    # 규모별 다음 단계 안내
    if scale == "simple":
        print("📋 다음 단계:")
        print("1. R.md를 열어 '문제 정의' 섹션 작성")
        print("2. 작업 체크리스트 작성")
        print("3. 구현 및 검증 진행 (간이 W는 R 4장에 내장)")
    elif scale == "standard":
        print("📋 다음 단계:")
        print("1. R.md를 열어 '요구사항 분석' 섹션 작성")
        print("2. 기능 정의 및 설계 진행 (G1: 2장 확정)")
        print("3. 다중 모듈이면 I 생성: /dev artifact i")
        print("4. 모듈별 P 생성: /dev artifact p --module=<모듈명>")
        print("5. 구현 완료 시 W 생성: /dev artifact w --module=<모듈명>")
    else:  # complex
        print("📋 다음 단계:")
        print("1. R.md를 열어 '요구사항 분석' 섹션 상세 작성")
        print("2. 리스크 분석 및 영향 범위 분석 진행")
        print("3. I 생성(필수): /dev artifact i → 승인(G2) 후 모듈별 P/W")
        print("4. 배포 계획 및 롤백 계획 수립")
    print("")
    print(f"📁 산출물(I/P/W)은 같은 디렉토리에 동일 프리픽스({timestamp}_)로 생성됩니다.")

    return str(output_file.relative_to(project_root))


def update_active_development(
    project_root: Path,
    filename: str,
    requirement_name: str,
    created_at: str,
    scale: str
) -> None:
    """active_development.md를 업데이트합니다."""
    active_dev_file = project_root / "docs" / "requirements" / "active_development.md"

    if not active_dev_file.exists():
        print("⚠️  active_development.md 파일이 없습니다.")
        return

    # 파일 읽기
    content = active_dev_file.read_text(encoding="utf-8")

    # 활성 요구사항 테이블 업데이트
    table_marker = "| 요구사항 파일 | 상태 | 진행률 | 시작일 | 예상 완료일 |"
    separator = "|--------------|------|--------|--------|-------------|"

    if table_marker in content and separator in content:
        # 기존 테이블 찾기
        table_start = content.find(table_marker)
        separator_start = content.find(separator, table_start)
        next_section = content.find("\n\n", separator_start)

        if next_section == -1:
            next_section = len(content)

        # 새 행 추가 (규모 정보 포함)
        scale_label = SCALE_NAMES[scale].split()[0]  # "간단", "표준", "복잡"
        new_row = f"| [{filename}](./{filename}) | 작성 ({scale_label}) | 0% | {created_at.split()[0]} | TBD |"

        # 기존 "[없음]" 행 제거 및 새 행 추가
        table_section = content[separator_start:next_section]
        if "| [없음]" in table_section:
            table_section = table_section.replace("| [없음] | - | 0% | - | - |", new_row)
        else:
            table_section = table_section.rstrip() + "\n" + new_row

        # 내용 업데이트
        content = content[:separator_start] + table_section + content[next_section:]

        # 마지막 업데이트 시각 갱신
        update_marker = "**마지막 업데이트**:"
        if update_marker in content:
            update_line_start = content.rfind("\n", 0, content.find(update_marker))
            update_line_end = content.find("\n", content.find(update_marker))
            content = (
                content[:update_line_start + 1] +
                f"**마지막 업데이트**: {created_at}" +
                content[update_line_end:]
            )

        # 파일 쓰기
        active_dev_file.write_text(content, encoding="utf-8")

        print(f"✅ active_development.md 업데이트 완료")


def show_scale_help():
    """규모 선택 도움말을 표시합니다."""
    print("\n📊 요구사항 규모 선택 가이드\n")
    print("┌─────────────┬────────────────────────────┬─────────────────────────┐")
    print("│ 규모        │ 적합한 작업                │ 템플릿 크기             │")
    print("├─────────────┼────────────────────────────┼─────────────────────────┤")
    print("│ simple      │ 버그 수정, 오타 수정       │ ~50줄 (최소)            │")
    print("│ (간단)      │ 소규모 변경, 핫픽스        │                         │")
    print("├─────────────┼────────────────────────────┼─────────────────────────┤")
    print("│ standard    │ 일반 기능 개발             │ ~150줄 (표준)           │")
    print("│ (표준)      │ 새 기능 추가, API 추가     │ [기본값]                │")
    print("├─────────────┼────────────────────────────┼─────────────────────────┤")
    print("│ complex     │ 대규모 기능, 아키텍처 변경 │ ~400줄 (상세)           │")
    print("│ (복잡)      │ 마이그레이션, 리팩토링     │                         │")
    print("└─────────────┴────────────────────────────┴─────────────────────────┘")
    print("\n사용 예시:")
    print("  /dev req 버그수정 --scale=simple")
    print("  /dev req 사용자인증 --scale=standard")
    print("  /dev req 마이크로서비스전환 --scale=complex")
    print("\n💡 --scale 옵션을 생략하면 요구사항 이름을 분석하여 자동 추천합니다.")


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(
        description="요구사항 명세서를 생성합니다."
    )
    parser.add_argument(
        "requirement_name",
        nargs="?",
        type=str,
        help="요구사항 명칭 (예: 사용자인증기능)"
    )
    parser.add_argument(
        "--priority",
        type=str,
        default="중간",
        choices=["높음", "중간", "낮음"],
        help="우선순위 (기본값: 중간)"
    )
    parser.add_argument(
        "--scale",
        type=str,
        default=None,
        choices=["simple", "standard", "complex"],
        help="규모 (simple/standard/complex). 생략 시 자동 추천"
    )
    parser.add_argument(
        "--id",
        type=str,
        default=None,
        dest="req_id",
        help="도메인코드 (1xxx=에이전트·핵심 기능, 2xxx=도구·연동(MCP), 3xxx=인증·권한·인프라). 생략 시 밀리초 기반"
    )
    parser.add_argument(
        "--type",
        type=str,
        default=None,
        dest="req_type",
        choices=["feature", "enhance", "perf", "bug"],
        help="유형 (feature=신규기능/enhance=기능개선/perf=성능개선/bug=버그). 생략 시 자동 추천"
    )
    parser.add_argument(
        "--help-scale",
        action="store_true",
        help="규모 선택 가이드 표시"
    )

    args = parser.parse_args()

    # 규모 도움말
    if args.help_scale:
        show_scale_help()
        return

    # 요구사항 이름 필수
    if not args.requirement_name:
        print("❌ 요구사항 이름을 입력해주세요.")
        print("💡 사용법: /dev req <요구사항명> [--scale=simple|standard|complex]")
        print("💡 규모 가이드: /dev req --help-scale")
        sys.exit(1)

    # 규모 자동 추천
    scale = args.scale
    if scale is None:
        scale = recommend_scale(args.requirement_name)
        print(f"💡 자동 추천 규모: {SCALE_NAMES[scale]}")
        print(f"   (다른 규모를 원하면 --scale 옵션을 사용하세요)")
        print("")

    # 유형 자동 추천
    req_type = args.req_type
    if req_type is None:
        req_type = recommend_type(args.requirement_name)
        print(f"💡 자동 추천 유형: {TYPE_NAMES[req_type]}")
        print(f"   (다른 유형을 원하면 --type 옵션을 사용하세요)")
        print("")

    # 요구사항 명세서 생성
    result = create_requirement_spec(args.requirement_name, args.priority, scale, args.req_id, req_type)

    if not result:
        sys.exit(1)


if __name__ == "__main__":
    main()
