#!/usr/bin/env python3
"""
개발 프로세스 단축 명령 디스패처

dev-req 스킬로 명령을 위임합니다.

사용법:
    python dispatch.py <command> [args...]
"""

import subprocess
import sys
from pathlib import Path


# 명령어 매핑
COMMAND_MAP = {
    # 요구사항 관리 (dev-req)
    "init": {
        "skill": "dev-req",
        "script": "init.py",
        "description": "요구사항 관리 구조 초기화",
    },
    "req": {
        "skill": "dev-req",
        "script": "create_requirement.py",
        "description": "요구사항 명세서 생성",
    },
    "artifact": {
        "skill": "dev-req",
        "script": "create_artifact.py",
        "description": "산출물(I/P/W) 생성",
    },
    "status": {
        "skill": "dev-req",
        "script": "progress_report.py",
        "description": "진행 상황 리포트",
    },
    "validate": {
        "skill": "dev-req",
        "script": "validate.py",
        "description": "규약 검사 (상태 enum·산출물 세트·D↔R 일치)",
    },
    "archive": {
        "skill": "dev-req",
        "script": "archive_requirement.py",
        "description": "요구사항 세트 아카이브 이동 (archive/<YYYY>/<MM>/W<N>/)",
    },

    # 정본 문서 체계 (dev-gen-project-docs)
    "docs": {
        "skill": "dev-gen-project-docs",
        "script": "init.py",
        "description": "정본 문서 체계",
        "subcommands": {
            "init": "init.py",
            "module": "add_module.py",
            "status": "status.py",
        },
    },
}

# 산출물 유형 단축: /dev i|p|w ... → /dev artifact i|p|w ...
ARTIFACT_ALIASES = {"i", "p", "w"}


def get_skill_script_path(skill_name: str, script_name: str) -> Path:
    """스킬 스크립트 경로를 반환합니다."""
    current_dir = Path(__file__).parent.parent.parent  # .claude/skills/
    skill_dir = current_dir / skill_name / "scripts"
    return skill_dir / script_name


def show_help():
    """도움말을 표시합니다."""
    print("""
🚀 /dev - 개발 프로세스 단축 명령

사용법: /dev <command> [options]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📋 요구사항 관리 (dev-req)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  init                          요구사항 관리 구조 초기화
  req <name> [--id=<코드>] [--scale=<규모>]   요구사항(R) 생성 — [프리픽스]_<이름>_req.md
  artifact <i|p|w> [--req=..] [--module=..]   산출물(I/P/W)을 같은 디렉토리·동일 프리픽스로 생성
  i | p | w [옵션]              artifact 단축형 (예: /dev w --module=<모듈명>)
  status [--format=<format>]    진행 상황 리포트 (산출물 현황 포함)
  validate [--update-dashboard] 규약 검사 (enum·세트 완결성·D↔R 일치) — D 규약위반 섹션 자동 기입
  archive --req=<ID>            완료 세트를 archive/<YYYY>/<MM>/W<N>/로 이동 (G5 가드 포함)
  archive --reorganize          기존 평면 아카이브를 주차 구조로 재배치

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📚 정본 문서 체계 (dev-gen-project-docs)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  docs init [--force]           docs/architecture·domains·glossary 골격 생성
  docs module <모듈명>          모듈 문서 골격 생성 (modules/<모듈>.md)
  docs status [--format=json]   커버리지·플레이스홀더 점검

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
기타
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  help                          이 도움말 표시

예시:
  /dev req 사용자인증기능 --id=1026       요구사항 생성 (…-1026_사용자인증기능_req.md)
  /dev artifact i                        인터페이스 계약(I) 생성 — 다중 모듈 시
  /dev artifact p --module=<모듈명>       모듈 구현 플랜(P) 생성
  /dev artifact w --module=<모듈명>       검증 보고(W) 생성 — P당 1건
  /dev status                            진행 상황 확인

워크플로우 (표준: /DEV_PROCESS.md — 산출물은 docs/requirements/에 동일 프리픽스로 응집):
  1. /dev req <name>                → R 생성 (G1: 2장 확정)
  2. /dev artifact i                → I 생성·승인 (G2, 다중 모듈)
  3. /dev artifact p --module=<m>   → 모듈별 P 작성·구현
  4. /dev artifact w --module=<m>   → W 작성 (G3~G4 — W 없이 검증완료 금지)
  5. /dev status                    → 진행 상황·산출물 확인
  6. /dev validate                  → 규약 검사 → /dev archive --req=<ID> (G5)

전체 명령어:
  /dev-req              요구사항 관리 (init, create, artifact, progress-report, validate, archive)
""")


def dispatch_command(command: str, args: list):
    """명령을 해당 스킬 스크립트로 위임합니다."""
    if command == "help" or command is None:
        show_help()
        return 0

    # 산출물 단축형: /dev i|p|w ... → artifact 스크립트에 유형 인자로 전달
    if command in ARTIFACT_ALIASES:
        args = [command] + args
        command = "artifact"

    # docs 서브커맨드 처리: /dev docs [init|module <이름>|status]
    if command == "docs":
        cmd_info = COMMAND_MAP["docs"]
        sub = args[0] if args else "init"
        if sub not in cmd_info["subcommands"]:
            print(f"❌ 알 수 없는 docs 명령: {sub}")
            print("💡 사용법: /dev docs [init|module <모듈명>|status]")
            return 1
        script_path = get_skill_script_path(cmd_info["skill"], cmd_info["subcommands"][sub])
        if not script_path.exists():
            print(f"❌ 스크립트를 찾을 수 없습니다: {script_path}")
            return 1
        print(f"🔗 정본 문서 {sub} ({cmd_info['skill']})")
        print(flush=True)
        try:
            result = subprocess.run(
                [sys.executable, str(script_path)] + args[1:],
                cwd=Path.cwd()
            )
            return result.returncode
        except Exception as e:
            print(f"❌ 실행 오류: {e}")
            return 1

    # 일반 명령 처리
    if command not in COMMAND_MAP:
        print(f"❌ 알 수 없는 명령: {command}")
        print("💡 /dev help 로 사용 가능한 명령을 확인하세요.")
        return 1

    cmd_info = COMMAND_MAP[command]
    script_path = get_skill_script_path(cmd_info["skill"], cmd_info["script"])

    if not script_path.exists():
        print(f"❌ 스크립트를 찾을 수 없습니다: {script_path}")
        print(f"💡 {cmd_info['skill']} 스킬이 설치되어 있는지 확인하세요.")
        return 1

    print(f"🔗 {cmd_info['description']} ({cmd_info['skill']})")
    print(flush=True)

    try:
        result = subprocess.run(
            [sys.executable, str(script_path)] + args,
            cwd=Path.cwd()
        )
        return result.returncode
    except Exception as e:
        print(f"❌ 실행 오류: {e}")
        return 1


def main():
    """메인 함수"""
    if len(sys.argv) < 2:
        show_help()
        return

    command = sys.argv[1]
    args = sys.argv[2:]

    exit_code = dispatch_command(command, args)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
