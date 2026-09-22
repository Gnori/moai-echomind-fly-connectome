---
name: dev
description: Unified shortcut for development process commands. Delegates to dev-req (requirements + artifacts) with simple shortcuts like /dev init, /dev req, /dev artifact, /dev status, /dev validate, /dev archive
---

# Dev - Development Process Shortcuts

Unified command dispatcher for development process management. Provides simple shortcuts to the `dev-req` (requirements) skill.

> 개발 표준 프로세스의 정본은 [/DEV_PROCESS.md](../../../DEV_PROCESS.md)이며, 이 스킬은 그 표준의 산출물 R(요구사항)·I(인터페이스 계약)·P(구현 플랜)·W(검증 보고)·D(대시보드)를 만드는 레퍼런스 구현이다.
> (2026-07-24: SeedAI 참고판에서 이식 — checkpoint(save/load)·dev-project-auto-doc(/dev doc) 제거, 요구사항 위치 `docs/requirements/`, validate/archive 추가. 舊 `.claude/requirements/`는 동결(레거시))

## Execution (MANDATORY)

모든 `/dev` 명령은 **프로젝트 루트에서 디스패처 스크립트를 실행**하여 처리합니다:

```bash
python3 .claude/skills/dev/scripts/dispatch.py <command> [args...]

# 예시
python3 .claude/skills/dev/scripts/dispatch.py req 사용자인증 --scale=standard
python3 .claude/skills/dev/scripts/dispatch.py w --module=moai-agent
python3 .claude/skills/dev/scripts/dispatch.py validate
```

스크립트는 골격 생성·검사만 담당합니다 — 생성된 문서의 플레이스홀더를 실제 내용으로 채우는 것(R 상세화, P 작성, W 증거 기록)은 어시스턴트가 코드·대화를 분석해 직접 수행해야 합니다.

## Quick Reference

| Shortcut | Full Command | Description |
|----------|--------------|-------------|
| `/dev init` | `/dev-req init` | Initialize requirements structure |
| `/dev req <name>` | `/dev-req create <name>` | Create `[프리픽스]_<이름>_req.md` |
| `/dev artifact <i\|p\|w>` | `/dev-req artifact <i\|p\|w>` | Create I/P/W with same prefix, same directory |
| `/dev i` / `/dev p` / `/dev w` | `/dev-req artifact i\|p\|w` | Artifact shortcuts |
| `/dev status` | `/dev-req progress-report` | Show progress report (+artifact presence) |
| `/dev validate` | `/dev-req validate` | 규약 검사 (상태 enum·산출물 세트·D↔R 일치) |
| `/dev archive --req=<ID>` | `/dev-req archive` | 완료 세트를 `archive/<YYYY>/<MM>/W<N>/`로 이동 |
| `/dev docs init` | `/dev-gen-project-docs init` | 정본 문서 체계(docs/architecture·domains·glossary) 골격 생성 |
| `/dev docs module <이름>` | `/dev-gen-project-docs module` | 모듈 문서 골격 생성 |
| `/dev docs status` | `/dev-gen-project-docs status` | 정본 문서 커버리지·플레이스홀더 점검 |

---

## Commands

### init

Initialize requirements management structure.

```bash
/dev init [--force]
```

Creates `docs/requirements/` with templates (R 3종 + I/P/W 3종) and updates CLAUDE.md.

### req

Create a new requirement spec file.

```bash
/dev req <requirement_name> [--priority=높음|중간|낮음] [--scale=simple|standard|complex] [--type=feature|enhance|perf|bug] [--id=<도메인코드>]
# --scale = 프로세스 게이팅 축 (산출물 수·게이트), --type = 내용 가이드 축 (R 1장 관점 블록)
# 둘 다 생략 시 이름 키워드로 자동 추천
```

### artifact

Create an artifact document with the same prefix in the same directory.

```bash
/dev artifact <i|p|w> [--req=<파일명 일부>] [--module=<모듈명>]
# i = <프리픽스>_<이름>_interface-design.md (다중 모듈 시 1건, 승인(G2) 후 P 진행)
# p = <프리픽스>_<이름>-<모듈>_dev-plan.md  (--module 필수, 모듈당 1건)
# w = <프리픽스>_<이름>-<모듈>_walkthrough.md (--module 필수, P당 1건 — W 없이 검증완료 금지)
```

### status

Show progress report for current requirement.

```bash
/dev status [--format=markdown|json]
```

### validate

DEV_PROCESS 산출물 계약의 기계 검사 가능 항목을 전수 검사.

```bash
/dev validate [--update-dashboard] [--format=json]
```

### archive

완료(G5) 요구사항 세트를 주차 구조로 이동.

```bash
/dev archive --req=<ID> [--dry-run]
/dev archive --reorganize
```

### docs

최신성 정본 문서 체계(docs/architecture·domains·glossary) 관리 — `dev-gen-project-docs` 스킬 위임.
골격 생성 후 **AI가 코드 분석으로 플레이스홀더를 채우는 것이 필수** (해당 스킬 SKILL.md의
'AI 작성 지침' 참조).

```bash
/dev docs init [--force]          # 골격 생성 + 모듈 감지
/dev docs module <모듈명>          # 모듈 문서 골격
/dev docs status [--format=json]  # 커버리지·플레이스홀더 점검
```

---

## Typical Workflow

```bash
# 1. Create requirement (R — G1: 2장 확정)
/dev req user-authentication --scale=standard --id=1026

# 2. Interface contract (다중 모듈 시 — G2 승인)
/dev artifact i

# 3. Per-module plan → implement
/dev artifact p --module=<모듈명>

# 4. Walkthrough (P당 1건 — G3/G4)
/dev artifact w --module=<모듈명>

# 5. Check progress → validate → archive
/dev status
/dev validate
/dev archive --req=1026
```

산출물은 모두 `docs/requirements/`에 동일 프리픽스(`[YYYYMMDD-HHMMSS-코드]_`)로 응집되며, 완료 시 프리픽스 세트째 `archive/`로 이동한다. 舊 `.claude/requirements/`(구 체계)는 동결(레거시) — 신규 작성 금지.

---

## Notes

- All commands delegate to the `dev-req` skill
- `/dev` without arguments shows help
- CLAUDE.md is automatically backed up before modifications
