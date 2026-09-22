#!/usr/bin/env python3
"""
산출물(I/P/W) 생성 스크립트 — 요구사항과 같은 디렉토리에 동일 프리픽스로 생성

사용법:
    python create_artifact.py <i|p|w> [--req=<요구사항 파일명/프리픽스 일부>] [--module=<모듈명>]

파일명 체계 (docs/requirements/ 평면 관리, 프리픽스 = 요구사항 식별자, 유형 토큰은 파일명 끝):
    {YYYYMMDD-HHMMSS-코드}_{이름}_req.md                       ← R (create_requirement.py)
    {YYYYMMDD-HHMMSS-코드}_{이름}_interface-design.md          ← I (다중 모듈일 때 1건)
    {YYYYMMDD-HHMMSS-코드}_{이름}-{모듈}_dev-plan.md           ← P (모듈당 1건, --module 필수)
    {YYYYMMDD-HHMMSS-코드}_{이름}-{모듈}_walkthrough.md        ← W (P당 1건, --module 필수)

--req 생략 시 가장 최근 생성된 요구사항(_req.md 파일)을 대상으로 한다.
"""

import argparse
import re
import sys
from datetime import datetime
from pathlib import Path

from claude_md_utils import get_project_root


ARTIFACT_TYPES = {
    "i": "interface",
    "interface": "interface",
    "p": "plan",
    "plan": "plan",
    "w": "walkthrough",
    "walkthrough": "walkthrough",
}

TEMPLATE_FILES = {
    "interface": "interface-design.md.template",
    "plan": "dev-plan.md.template",
    "walkthrough": "walkthrough.md.template",
}

REQ_FILE_PATTERN = re.compile(r"^(\d{8}-\d{6}-[^_]+)_(.+)_req\.md$")


def get_template_path(artifact: str, project_root: Path) -> Path:
    """템플릿 경로를 반환합니다 (프로젝트 사본 우선 → 스킬 내장 폴백)."""
    template_file = TEMPLATE_FILES[artifact]
    project_template = project_root / "docs" / "requirements" / "templates" / template_file
    if project_template.exists():
        return project_template
    return Path(__file__).parent.parent / "templates" / "requirements" / "templates" / template_file


def find_requirement_files(requirements_dir: Path) -> list:
    """요구사항(_req-) 파일 목록을 반환합니다 (최신순, archive 제외)."""
    files = [
        f for f in requirements_dir.iterdir()
        if f.is_file() and REQ_FILE_PATTERN.match(f.name)
    ]
    return sorted(files, key=lambda f: f.name, reverse=True)


def resolve_requirement_file(requirements_dir: Path, req_hint: str) -> Path:
    """--req 힌트(부분 일치) 또는 최신 파일로 요구사항을 결정합니다."""
    files = find_requirement_files(requirements_dir)

    if not files:
        print("❌ 요구사항 파일(*_req.md)이 없습니다.")
        print("💡 먼저 요구사항을 생성하세요: /dev req <이름>")
        return None

    if req_hint:
        matched = [f for f in files if req_hint in f.name]
        if not matched:
            print(f"❌ '{req_hint}'와 일치하는 요구사항 파일이 없습니다.")
            print("💡 후보:")
            for f in files[:10]:
                print(f"   - {f.name}")
            return None
        if len(matched) > 1:
            print(f"⚠️  '{req_hint}'와 일치하는 요구사항이 여러 개입니다. 더 구체적으로 지정하세요:")
            for f in matched:
                print(f"   - {f.name}")
            return None
        return matched[0]

    # 힌트 없음 → 최신 요구사항 (사용자 확인용으로 이름 출력)
    print(f"💡 대상 요구사항(최신): {files[0].name}")
    print("   (다른 요구사항이면 --req=<파일명 일부>로 지정하세요)")
    return files[0]


def parse_req_meta(req_file: Path) -> dict:
    """요구사항 파일명에서 프리픽스·ID·이름을 추출합니다."""
    m = REQ_FILE_PATTERN.match(req_file.name)
    prefix, req_name = m.group(1), m.group(2)
    parts = prefix.split("-")
    req_id = parts[2] if len(parts) >= 3 else prefix
    return {"prefix": prefix, "req_id": req_id, "req_name": req_name}


def create_artifact(artifact: str, req_hint: str = "", module: str = "") -> str:
    """산출물 문서를 요구사항과 같은 디렉토리에 동일 프리픽스로 생성합니다."""
    project_root = get_project_root()
    requirements_dir = project_root / "docs" / "requirements"

    req_file = resolve_requirement_file(requirements_dir, req_hint)
    if req_file is None:
        return ""

    if artifact in ("plan", "walkthrough") and not module:
        print(f"❌ {artifact} 산출물은 --module=<모듈명> 지정이 필수입니다 (모듈당 1건).")
        print("💡 예: --module=<모듈 디렉터리명>")
        return ""

    meta = parse_req_meta(req_file)
    prefix, req_name = meta["prefix"], meta["req_name"]

    # 파일명 결정 ({프리픽스}_{이름}[-{모듈}]_{유형}.md)
    if artifact == "interface":
        filename = f"{prefix}_{req_name}_interface-design.md"
    elif artifact == "plan":
        filename = f"{prefix}_{req_name}-{module}_dev-plan.md"
    else:
        filename = f"{prefix}_{req_name}-{module}_walkthrough.md"

    output_file = requirements_dir / filename
    if output_file.exists():
        print(f"⚠️  이미 존재합니다: {output_file.relative_to(project_root)}")
        return ""

    # W는 대응 P 존재 확인 (P당 1건 — 경고만, 차단하지 않음: simple 규모는 P 생략 가능)
    p_filename = f"{prefix}_{req_name}-{module}_dev-plan.md"
    if artifact == "walkthrough":
        if not (requirements_dir / p_filename).exists():
            print(f"⚠️  대응 P가 없습니다: {p_filename} (simple 규모의 간이 W가 아니라면 P를 먼저 작성하세요)")

    # 템플릿 로드·치환
    template_path = get_template_path(artifact, project_root)
    if not template_path.exists():
        print(f"❌ 템플릿이 없습니다: {template_path}")
        return ""

    i_filename = f"{prefix}_{req_name}_interface-design.md"
    i_link = f"[./{i_filename}](./{i_filename})" if (requirements_dir / i_filename).exists() \
        else "(생략 — R 3장에 축약)"

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    content = template_path.read_text(encoding="utf-8")
    content = content.replace("{{REQUIREMENT_NAME}}", req_name)
    content = content.replace("{{REQ_ID}}", meta["req_id"])
    content = content.replace("{{CREATION_DATE}}", now)
    content = content.replace("{{MODULE_NAME}}", module or "-")
    content = content.replace("{{R_LINK}}", f"[./{req_file.name}](./{req_file.name})")
    content = content.replace("{{I_LINK}}", i_link)
    content = content.replace("{{P_LINK}}", f"[./{p_filename}](./{p_filename})")
    content = content.replace("{{W_FILENAME}}", f"{prefix}_{req_name}-{{모듈}}_walkthrough.md")

    output_file.write_text(content, encoding="utf-8")

    rel = output_file.relative_to(project_root)
    print("✅ 산출물 생성 완료!")
    print(f"📄 파일: {rel}")
    print("")
    print("📋 다음 단계:")
    if artifact == "interface":
        print("1. 계약 내용 작성 후 사용자 승인 → 상태=승인 (G2)")
        print("2. 승인 후 모듈별 P 생성: /dev artifact p --module=<모듈명>")
    elif artifact == "plan":
        print("1. Task→Step 작성 (No Placeholders — Files:/Run:/Expected: 필수)")
        print(f"2. 구현 완료 시 W 생성: /dev artifact w --module={module}")
    else:
        print("1. 테스트 로그 원문·동작 확인 기록 작성")
        print("2. R 메타의 관련 산출물 링크 + R 5장(검증 결과)에 이 W를 링크")
        print("3. 갱신한 정본 문서를 4장에 기록 (DEV_PROCESS.md 공통 규칙 3의 트리거 표 참조)")
    print("")
    print(f"🔗 R 문서: {req_file.name} 의 '관련 산출물'에 링크를 추가하세요.")

    return str(rel)


def main():
    parser = argparse.ArgumentParser(description="산출물(I/P/W)을 요구사항과 같은 디렉토리에 생성합니다.")
    parser.add_argument("artifact_type", nargs="?", type=str,
                        help="산출물 유형: i|interface, p|plan, w|walkthrough")
    parser.add_argument("--req", type=str, default="", dest="req_hint",
                        help="요구사항 파일명 일부 (생략 시 최신 요구사항)")
    parser.add_argument("--module", type=str, default="",
                        help="모듈명 (p/w 필수 — 모듈 디렉터리명)")

    args = parser.parse_args()

    if not args.artifact_type or args.artifact_type.lower() not in ARTIFACT_TYPES:
        print("❌ 산출물 유형을 지정하세요: i(인터페이스 계약) | p(구현 플랜) | w(검증 보고)")
        print("💡 사용법: /dev artifact <i|p|w> [--req=<파일명 일부>] [--module=<모듈명>]")
        sys.exit(1)

    artifact = ARTIFACT_TYPES[args.artifact_type.lower()]
    result = create_artifact(artifact, args.req_hint, args.module)

    if not result:
        sys.exit(1)


if __name__ == "__main__":
    main()
