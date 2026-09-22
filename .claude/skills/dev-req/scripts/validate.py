#!/usr/bin/env python3
"""
dev-req 규약 검사기 — DEV_PROCESS.md 산출물 계약의 기계 검사 가능 항목을 검사한다.

사용법:
    python3 validate.py [--update-dashboard] [--format=markdown|json]

검사 규칙 (DEV_PROCESS.md 대응):
    STATUS-ENUM     R 상태 필드가 enum 외 자유 서술 (금지 사항 7)
    R-META          R 필수 메타(상태·규모) 누락/해석 불가
    G4-W            검증완료/아카이브인데 W(파일·내장) 없음 (게이트 G4)
    G2-I            complex + P 2건 이상인데 I 없음 (규모별 적용 표)
    P-W-PAIR        구현완료 상태인데 P에 대응하는 W 없음 (P당 W 1건)
    P-PLACEHOLDER   P에 "TBD"/"적절히 처리" 포함 (No Placeholders)
    W-BACKLINK      W에 대응 R 링크 없음 (공통 규칙 1 역링크 의무)
    LINK-STYLE      file:/// 절대경로 사용 (상대 경로만 허용)
    ARCHIVE-LOC     상태=아카이브인데 archive/ 미이동 (게이트 G5)
    ARCHIVE-FLAT    아카이브 평면 잔존 — <YYYY>/<MM>/W<N>/ 구조 미준수
    D-STALE-ROW     D에 행이 있으나 활성 파일 없음
    D-R-MISMATCH    D 상태 ≠ R 상태 (D는 R 파생물 — D 계약)
    D-MISSING-ROW   활성 R인데 D에 행 없음

주의: 신체계(2026-07-07 동일 프리픽스 체계) 이후 건만 검사한다.
      의미 판단(인수기준 측정가능성, 테스트 로그 진위 등)은 기계 검사 대상이 아니다.
"""

import json
import re
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from claude_md_utils import get_project_root

# DEV_PROCESS.md 상태 enum (자유 서술 금지)
STATUS_ENUM = [
    "접수", "작성", "상세화", "확정", "구현중",
    "구현완료(검증대기)", "검증완료", "아카이브", "보류",
]
# 신체계 전환일 — 이전 건은 소급 검사 제외 (구형은 "W 소급 작성 없이 종결" 명시됨)
NEW_SYSTEM_DATE = "20260707"
PLACEHOLDER_WORDS = ("TBD", "적절히 처리")


def meta_field(text: str, field: str):
    m = re.search(rf"^\s*-\s*\*\*{field}\*\*\s*[::]\s*(.+)$", text, re.M)
    return m.group(1).strip() if m else None


def prefix_of(name: str):
    m = re.match(r"^(\d{8}-\d{6}-\w+)_", name)
    return m.group(1) if m else None


class Validator:
    def __init__(self, project_root: Path):
        self.req_dir = project_root / "docs" / "requirements"
        self.archive = self.req_dir / "archive"
        self.findings = []

    def add(self, rule: str, target: str, detail: str):
        self.findings.append({"rule": rule, "target": target, "detail": detail})

    # ── R 단위 검사 ──────────────────────────────────────────
    def check_req(self, f: Path, archived: bool):
        text = f.read_text(encoding="utf-8")
        prefix = prefix_of(f.name)
        if not prefix:
            return None
        is_new = prefix.split("-")[0] >= NEW_SYSTEM_DATE
        # 아카이브는 <YYYY>/<MM>/W<N>/ 하위 — 세트는 같은 디렉토리에 응집되므로 부모 기준 탐색
        siblings = list(f.parent.glob(f"{prefix}_*"))

        status = meta_field(text, "상태")
        if status is None:
            self.add("R-META", f.name, "메타에 '상태' 필드 없음")
        elif status not in STATUS_ENUM:
            self.add("STATUS-ENUM", f.name, f"enum 외 자유 서술: '{status}'")

        scale_raw = meta_field(text, "규모") or ""
        scale = next((s for s in ("simple", "standard", "complex") if s in scale_raw), None)
        if scale is None and is_new:
            self.add("R-META", f.name, f"규모 값 해석 불가: '{scale_raw}'")

        # G4: 검증완료/아카이브 → W 필수 (simple은 R 내장 간이형 허용)
        has_w_file = any("_walkthrough" in s.name for s in siblings)
        has_w_embedded = bool(re.search(r"^##.*검증", text, re.M))
        if (status in ("검증완료", "아카이브") or archived) and is_new:
            if not has_w_file and not has_w_embedded:
                self.add("G4-W", f.name, "검증완료/아카이브인데 W(파일·내장) 없음")

        # G2: complex + 다중 P → I 필수
        has_i = any("_interface-design" in s.name for s in siblings)
        n_plans = sum(1 for s in siblings if "_dev-plan" in s.name)
        if scale == "complex" and n_plans >= 2 and not has_i:
            self.add("G2-I", f.name, f"complex + P {n_plans}건인데 I(인터페이스 계약) 없음")

        # P당 W 1건 — 구현완료 이상 상태의 활성 건만
        plan_mods = {re.sub(r"_dev-plan\.md$", "", s.name) for s in siblings if "_dev-plan" in s.name}
        w_mods = {re.sub(r"_walkthrough\.md$", "", s.name) for s in siblings if "_walkthrough" in s.name}
        if is_new and not archived and status and "구현완료" in status:
            for p in sorted(plan_mods - w_mods):
                self.add("P-W-PAIR", f.name, f"P '{p}'에 대응하는 W 없음")

        # P: No Placeholders / W: 역링크·링크 스타일
        for s in siblings:
            if "_dev-plan" in s.name:
                ptext = s.read_text(encoding="utf-8")
                for bad in PLACEHOLDER_WORDS:
                    if bad in ptext:
                        self.add("P-PLACEHOLDER", s.name, f"금지 표현 '{bad}' 포함")
            elif "_walkthrough" in s.name:
                wtext = s.read_text(encoding="utf-8")
                if f.name not in wtext:
                    self.add("W-BACKLINK", s.name, "대응 R 링크 없음 (역링크 의무)")
                if "file://" in wtext:
                    self.add("LINK-STYLE", s.name, "file:/// 절대경로 사용 (상대 경로만 허용)")

        # G5: 상태=아카이브인데 활성 위치
        if not archived and status == "아카이브":
            self.add("ARCHIVE-LOC", f.name, "상태=아카이브인데 archive/로 미이동")

        return status

    # ── D(대시보드) 검사 ─────────────────────────────────────
    def check_dashboard(self, active_reqs: dict):
        d = self.req_dir / "active_development.md"
        if not d.exists():
            self.add("D-MISSING", "active_development.md", "대시보드 파일 없음")
            return
        dtext = d.read_text(encoding="utf-8")
        rows = re.findall(r"^\|\s*\[([^\]]+_req\.md)\][^|]*\|\s*([^|]+)\|", dtext, re.M)
        d_map = {name: status.strip() for name, status in rows}

        for name, d_status in d_map.items():
            if not (self.req_dir / name).exists():
                self.add("D-STALE-ROW", name, "D에 행이 있으나 활성 디렉토리에 파일 없음")
                continue
            r_status = active_reqs.get(name)
            # D는 말미 "(규모·유형)" 부기 하나만 허용 — enum 내 괄호는 보존
            base = re.sub(r"\s*\([^()]*\)\s*$", "", d_status).strip()
            if r_status and base != r_status:
                self.add("D-R-MISMATCH", name, f"D='{d_status}' vs R='{r_status}'")

        for name in active_reqs:
            if name not in d_map:
                self.add("D-MISSING-ROW", name, "활성 R인데 D에 행 없음")

    # ── 실행 ────────────────────────────────────────────────
    def run(self):
        active_reqs = {}
        for f in sorted(self.req_dir.glob("*_req.md")):
            active_reqs[f.name] = self.check_req(f, archived=False)
        n_archived = 0
        for f in sorted(self.archive.rglob("*_req.md")):
            self.check_req(f, archived=True)
            n_archived += 1
        # 아카이브 평면 잔존 검사 — 주차 구조(<YYYY>/<MM>/W<N>/) 밖의 파일 (`_` 프리픽스 메타/스냅샷은 예외)
        for f in sorted(self.archive.iterdir()):
            if f.is_file() and f.suffix == ".md" and not f.name.startswith("_"):
                self.add("ARCHIVE-FLAT", f.name,
                         "아카이브 평면 잔존 — <YYYY>/<MM>/W<N>/ 구조로 이동 필요 (/dev archive --reorganize)")
        self.check_dashboard(active_reqs)
        return len(active_reqs), n_archived


def update_dashboard_section(req_dir: Path, findings: list):
    """D의 '## 규약 위반' 섹션을 검사 결과로 갱신한다."""
    d = req_dir / "active_development.md"
    if not d.exists():
        return False
    text = d.read_text(encoding="utf-8")
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if findings:
        lines = [f"- **[{f['rule']}]** [{f['target']}](./{f['target']}) — {f['detail']}"
                 for f in findings]
        body = "\n".join(lines)
    else:
        body = "- [없음]"
    section = f"## 규약 위반 (검사기 출력 자리)\n\n> 마지막 검사: {stamp} (`/dev validate`)\n\n{body}\n"
    new_text, n = re.subn(r"## 규약 위반.*\Z", section, text, flags=re.S)
    if n == 0:
        new_text = text.rstrip() + "\n\n" + section
    d.write_text(new_text, encoding="utf-8")
    return True


def main():
    update_d = "--update-dashboard" in sys.argv
    fmt = "json" if "--format=json" in sys.argv else "markdown"

    project_root = get_project_root()
    v = Validator(project_root)
    n_active, n_archived = v.run()

    if fmt == "json":
        print(json.dumps({"active": n_active, "archived": n_archived,
                          "violations": v.findings}, ensure_ascii=False, indent=2))
    else:
        print(f"🔍 규약 검사 — 활성 R {n_active}건 + 아카이브 R {n_archived}건 (신체계 기준)")
        print(f"{'❌' if v.findings else '✅'} 위반: {len(v.findings)}건")
        for f in v.findings:
            print(f"  [{f['rule']}] {f['target']}")
            print(f"      → {f['detail']}")

    if update_d:
        if update_dashboard_section(project_root / "docs" / "requirements", v.findings):
            print("\n📋 active_development.md '규약 위반' 섹션 갱신 완료")

    return 1 if v.findings else 0


if __name__ == "__main__":
    sys.exit(main())
