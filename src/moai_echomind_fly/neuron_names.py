"""뉴런 쉬운 이름(R-2719 · 요구자 10-18).

종류 이름 → 사람이 알아볼 이름 · 한 줄 설명 · 근거 · 확인 수준.

역할을 지어내지 않는다: config/neurons.yaml 의 types 에 있는 종류만 역할 이름을 붙이고,
없는 종류는 연결 지도 주석(class → superclass)의 쉬운 이름으로만 보인다(check = 주석).
"""

from __future__ import annotations

from . import config


class Names:
    """설정 하나를 읽어 두고 종류마다 설명을 돌려준다."""

    def __init__(self, cfg: dict | None = None) -> None:
        self.cfg = cfg or config.load("neurons")
        self.types: dict = self.cfg.get("types") or {}
        self.classes: dict = self.cfg.get("classes") or {}
        self.supers: dict = self.cfg.get("superclasses") or {}

    def describe(self, typ: str, superclass: str = "", klass: str = "") -> dict:
        """{name, note, source, check} — 목록에 없으면 주석 분류 이름(check '주석')."""
        if typ in self.types:
            e = self.types[typ]
            return {"name": str(e["name"]), "note": str(e.get("note", "")),
                    "source": str(e.get("source", "")), "check": str(e.get("check", ""))}
        name = self.classes.get(klass) or self.supers.get(superclass) or "분류 없음"
        note = "이 종류 하나의 역할은 목록에 없음 — 연결 지도 주석 분류만"
        return {"name": str(name), "note": note,
                "source": "연결 지도 주석(MaleCNS v1.0)", "check": "주석"}

    def groups(self) -> dict[str, list[str]]:
        return {k: list(v) for k, v in (self.cfg.get("groups") or {}).items()}
