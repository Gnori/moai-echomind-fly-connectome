---
name: dev-req
description: Automate requirement (R) and artifact (I/P/W) document management following /DEV_PROCESS.md. Use when users need to (1) Initialize requirements structure, (2) Create requirement folders, (3) Create interface/plan/walkthrough artifacts, (4) Generate progress reports
---

# Development Requirements Manager

Automates requirement folder (R) + artifact (I/P/W) creation and progress tracking, following the standard in [/DEV_PROCESS.md](../../../DEV_PROCESS.md).

> 표준 원칙: 프로세스·스킬·도구는 자유, **산출물은 계약**. 이 스킬은 산출물 R(요구사항)·I(인터페이스 계약)·P(구현 플랜)·W(검증 보고)·D(대시보드)를 만드는 레퍼런스 구현이다.
> 코덱스/안티그라비티 사용자는 같은 스크립트를 CLI로 직접 실행할 수 있다 (어느 디렉토리에서 실행해도 루트 기준 동작).
> (2026-07-07: **동일 프리픽스 평면 파일 체계로 전환** — R과 I/P/W를 `docs/requirements/`에 `[YYYYMMDD-HHMMSS-코드]_유형-이름(-모듈).md`로 응집, I/P/W 생성 기능 내장. 舊 체계(`.claude/requirements/`)는 동결(레거시 아카이브))
> (2026-07-03: 요구사항 위치 `docs/requirements/`로 이동, checkpoint 기능 제거, 상태 enum·`--id` 도메인코드 도입)
> (2026-07-24: SeedAI 참고판에서 MOA 계열 프로젝트로 이식)

## Key Features

- **Init Structure**: Initialize requirements directory and templates
- **Requirement Spec**: Create `[프리픽스]_<이름>_req.md` from scale/type-based templates
- **Artifacts (I/P/W)**: Create interface-design / dev-plan / walkthrough documents with the same prefix in the same directory
- **Progress Tracking**: Generate progress reports with checklist + artifact presence analysis

---

## Scripts

### init

Initialize requirements management structure.

```bash
/dev-req init [--force]
# CLI: python3 .claude/skills/dev-req/scripts/init.py
```

**Actions**:
1. Create `docs/requirements/` directory
2. Create scale templates (simple/standard/complex)
3. Create `active_development.md` for tracking
4. Create `README.md` with usage guide
5. **Update CLAUDE.md** (자동 백업 후 섹션 추가)

---

### create

Create requirement specification document (R) from scale-based templates.

```bash
/dev-req create <name> [--priority=높음|중간|낮음] [--scale=simple|standard|complex] [--type=feature|enhance|perf|bug] [--id=<도메인코드>]
# CLI: python3 .claude/skills/dev-req/scripts/create_requirement.py <name> --scale=... --type=... --id=...

# Examples
/dev-req create 사용자인증 --scale=standard --type=feature --id=1026
/dev-req create 메일목록깨짐수정 --type=bug
```

**Scales**:
| Scale | Use Case | Template |
|-------|----------|----------|
| simple | Bug fixes, minor changes | ~50 lines (R 축약형 + 간이 W 내장) |
| standard | Regular feature development (default) | ~150 lines |
| complex | Large features, architecture changes | ~400 lines (+리스크/롤백/회고) |

**Auto-recommendation**: `--scale` 생략 시 이름 키워드로 추천 (버그/수정→simple, 추가/구현→standard, 아키텍처/마이그레이션→complex)

**Types (유형 — 규모와 직교하는 내용 가이드 축)**: 공통 골격(메타·상태·게이트·산출물 계약)은 규모 템플릿이 담당하고, 유형은 R 1장 분석의 관점 가이드 블록을 주입한다. `--type` 생략 시 이름 키워드로 추천.

| Type | 가이드 초점 |
|------|------------|
| `feature` (신규기능) | 사용자 시나리오, 신규 메뉴·접근제어 등록 계획, 기존 기능과의 경계 |
| `enhance` (기능개선) | As-Is/To-Be, 영향 범위·하위 호환성, 결정 번복 링크 |
| `perf` (성능개선) | 현재/목표 지표(수치 필수), 측정 재현 방법, 기능 회귀 방지 |
| `bug` (버그) | 재현 절차, 기대 vs 실제, 근본 원인, 회귀 범위·재발 방지 테스트 |

**도메인코드 (`--id`)**: 1xxx=에이전트·핵심 기능, 2xxx=도구·연동(MCP), 3xxx=인증·권한·인프라. 생략 시 밀리초 기반 코드.

Creates `docs/requirements/[YYYYMMDD-HHMMSS-코드]_<name>_req.md` and updates `active_development.md`. All artifacts (I/P/W) share the same prefix in the same directory.

**상태 enum (자유 서술 금지)**: `접수 | 작성 | 상세화 | 확정 | 구현중 | 구현완료(검증대기) | 검증완료 | 아카이브 | 보류`
— W(검증 보고) 없이 "검증완료"로 변경 금지 ([/DEV_PROCESS.md](../../../DEV_PROCESS.md) 게이트 G4).

---

### artifact

Create artifact documents (I/P/W) **in the same directory with the same prefix as the requirement**. 외부 스킬에 위임하지 않는 자체 기능이다.

```bash
/dev-req artifact <i|p|w> [--req=<파일명 일부>] [--module=<모듈명>]
# CLI: python3 .claude/skills/dev-req/scripts/create_artifact.py <i|p|w> --req=... --module=...
# /dev 단축형: /dev i | /dev p --module=<m> | /dev w --module=<m>

# Examples
/dev artifact i --req=1026                       # <프리픽스>_<이름>_interface-design.md (다중 모듈 시 1건)
/dev artifact p --module=<모듈명>                 # <프리픽스>_<이름>-<모듈>_dev-plan.md (모듈당 1)
/dev artifact w --module=<모듈명>                 # <프리픽스>_<이름>-<모듈>_walkthrough.md (P당 1)
```

| Type | File | 게이트 규칙 |
|------|------|------------|
| `i` | `<프리픽스>_<이름>_interface-design.md` | 상태=승인(G2) 전 P 작성 금지. 단일 모듈이면 생략(R 3장에 축약) |
| `p` | `<프리픽스>_<이름>-<모듈>_dev-plan.md` | `--module` 필수. No Placeholders — Files:/Run:/Expected: 명시 |
| `w` | `<프리픽스>_<이름>-<모듈>_walkthrough.md` | `--module` 필수. P당 1건. 테스트 로그 원문 = 유일한 증거. **갱신한 정본 문서 목록을 W 4장에 단방향 기록** (트리거: DEV_PROCESS.md 공통 규칙 3) (정본→이력 링크 금지) |

`--req` 생략 시 최신 요구사항(_req.md 파일)을 대상으로 한다 (대상 파일명을 출력하므로 확인할 것).

---

### progress-report

Analyze requirement and generate progress report.

```bash
/dev-req progress-report [--requirement_file=<filename>] [--format=markdown|json]
# CLI: python3 .claude/skills/dev-req/scripts/progress_report.py
```

Reports: completion percentage, checklist status, active requirement context.

---

### validate

DEV_PROCESS.md 산출물 계약의 **기계 검사 가능 항목**을 전수 검사한다 (신체계 2026-07-07 이후 건만, 구형 아카이브는 소급 제외).

```bash
/dev validate                       # 검사만 (위반 시 exit 1)
/dev validate --update-dashboard    # + D의 "규약 위반" 섹션 자동 기입
/dev validate --format=json         # JSON 출력
# CLI: python3 .claude/skills/dev-req/scripts/validate.py
```

| 규칙 | 검사 내용 (DEV_PROCESS 대응) |
|------|------------------------------|
| `STATUS-ENUM` | R 상태 필드의 enum 외 자유 서술 (금지 사항) |
| `R-META` | 필수 메타(상태·규모) 누락/해석 불가 |
| `G4-W` | 검증완료/아카이브인데 W 없음 (simple 내장형 허용) |
| `G2-I` | complex + P 2건 이상인데 I 없음 |
| `P-W-PAIR` | 구현완료인데 P에 대응 W 없음 (P당 1건) |
| `P-PLACEHOLDER` | P에 "TBD"/"적절히 처리" (No Placeholders) |
| `W-BACKLINK` / `LINK-STYLE` | W 역링크 누락 / `file:///` 사용 |
| `ARCHIVE-LOC` | 상태=아카이브인데 archive/ 미이동 (G5) |
| `D-STALE-ROW` / `D-R-MISMATCH` / `D-MISSING-ROW` | D↔R 정합성 (D는 R 파생물) |

⚠️ 의미 판단(인수기준 측정가능성, 테스트 로그 진위, 코드-문서 일치)은 기계 검사 대상이 아니다 — 리뷰어(AI/사람) 몫.

---

### archive

완료(G5) 요구사항 세트를 `archive/<YYYY>/<MM>/W<N>/`로 이동한다. 주차는 프리픽스 날짜에서 자동 산출(`(일-1)//7+1`), W 없으면 이동을 거부(G5 가드), git 추적 파일은 `git mv`로 이력을 보존한다.

```bash
/dev archive --req=1026            # 활성 세트(R/I/P/W 전체) → 주차 폴더로 이동
/dev archive --req=1026 --dry-run  # 이동 계획만 출력
/dev archive --reorganize          # 아카이브 평면 잔존 파일을 주차 구조로 재배치 (멱등)
# CLI: python3 .claude/skills/dev-req/scripts/archive_requirement.py ...
```

이동 후 `active_development.md`에서 해당 행 제거 + `/dev validate`로 확인한다. `_` 프리픽스 메타 파일(스냅샷 등)은 아카이브 루트 유지가 허용된다.

---

## Typical Workflow

```bash
# Initial setup (once per project)
/dev-req init

# Start new requirement (R 생성 — G1 게이트: 2장 기능 정의 승인 시 상태=확정)
/dev-req create 사용자인증 --scale=standard --id=1026

# Interface contract (다중 모듈 시, G2 승인 후 P 진행)
/dev artifact i

# Per-module plan & walkthrough
/dev artifact p --module=<모듈명>
/dev artifact w --module=<모듈명>

# During development
/dev-req progress-report
```

모든 산출물(I/P/W)은 요구사항과 같은 디렉토리에 동일 프리픽스로 생성된다. 舊 `.claude/requirements/`(구 체계)는 **동결(레거시 아카이브)** — 신규 작성 금지.

---

## Generated Structure

```
docs/requirements/
├── README.md                     # Requirements management guide
├── active_development.md         # 진행 대시보드 (D)
├── [YYYYMMDD-HHMMSS-코드]_<이름>_req.md                    # 요구사항 (R)
├── [YYYYMMDD-HHMMSS-코드]_<이름>_interface-design.md       # 인터페이스 계약 (I, 다중 모듈 시)
├── [YYYYMMDD-HHMMSS-코드]_<이름>-<모듈>_dev-plan.md        # 모듈별 구현 플랜 (P, 모듈당 1)
├── [YYYYMMDD-HHMMSS-코드]_<이름>-<모듈>_walkthrough.md     # 검증 보고 (W, P당 1)
│                                 # ↑ 동일 프리픽스 = 한 요구사항의 산출물 세트
├── templates/
│   ├── simple|standard|complex.md.template       # R 규모별 3종
│   └── interface-design|dev-plan|walkthrough.md.template  # 산출물 3종
├── sql/                          # 공용 부속 DDL/시드
├── wireframes/                   # 화면 목업 (필요 시 생성)
└── archive/<YYYY>/<MM>/W<N>/     # 완료 시 프리픽스 세트째 이동 — W 없이 이동 금지
                                  # 주차 = 프리픽스 날짜 기준 (일-1)//7+1 — /dev archive가 자동 산출
```

**템플릿 정본**: `docs/requirements/templates/` (프로젝트 사본이 우선 적용됨).
스킬 내부 `templates/requirements/templates/`는 init용 원본이자 폴백 — **수정 시 양쪽을 동기화할 것**.

---

## Configuration

Settings in `skill_config.yaml`:
- `requirements_dir`: `docs/requirements`
- `filename_format`: `[YYYYMMDD-HHMMSS-코드]_{name}(-{모듈})_{유형}.md` (유형: req|interface-design|dev-plan|walkthrough — 파일명 끝)

---

## Shortcut Commands

Use with `/dev` dispatcher:

```bash
/dev init                              # Same as /dev-req init
/dev req <name> [--scale=..] [--id=..] # Same as /dev-req create
/dev artifact <i|p|w> [옵션]           # Same as /dev-req artifact
/dev i | /dev p | /dev w [옵션]        # artifact 단축형
/dev status                            # Same as /dev-req progress-report
/dev validate [--update-dashboard]     # 규약 검사 → D 규약위반 섹션 기입
```

---

## Notes

- Scripts run from any directory (project root auto-detected)
- Multiple requirements can be active; `active_development.md` tracks all
- CLAUDE.md is automatically backed up before modifications
