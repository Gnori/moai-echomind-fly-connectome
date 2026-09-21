"""`fly-connectome bench …` 하위 명령."""

from __future__ import annotations

import argparse


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("bench", help="알려진 반사로 검산(무거움 — 가드)")
    bsub = p.add_subparsers(dest="bench", required=True)

    e = bsub.add_parser("escape", help="F-004 도망 반사 채점(다가옴 → LC/LPLC → GF)")
    e.add_argument("--trials", type=int, default=40)
    e.add_argument("--graph", default="traced-v1")
    e.add_argument("--rewired-seed", type=int, default=0, help="섞은 배선 대조군 씨앗")
    e.add_argument("--no-null", action="store_true", help="섞은 배선 대조군을 건너뛴다")
    e.add_argument("--device", default="cpu", choices=["cpu", "mps"])
    e.add_argument("--seed", type=int, default=0)
    e.add_argument("--w-scale", type=float, help="sim.yaml 값 대신(맞추기 전용)")
    e.add_argument("--peak-hz", type=float, help="loom.yaml 값 대신(맞추기 전용)")
    e.set_defaults(func=_escape)

    f = bsub.add_parser("fit", help="w_scale 맞추기 — 미리 적어 둔 규칙(escape.FIT_RULE)")
    f.add_argument("--graph", default="traced-v1")
    f.add_argument("--device", default="cpu", choices=["cpu", "mps"])
    f.add_argument("--seed", type=int, default=0)
    f.set_defaults(func=_fit)

    lim = bsub.add_parser("limits", help="F-005 한계 측정 셋(광수용체 · 청각 · 미각) — 기록")
    lim.add_argument("--graph", default="traced-v1")
    lim.add_argument("--seed", type=int, default=0)
    lim.set_defaults(func=_limits)


def _escape(args: argparse.Namespace) -> int:
    from . import escape

    return escape.main(args)


def _fit(args: argparse.Namespace) -> int:
    from . import escape

    return escape.fit(args)


def _limits(args: argparse.Namespace) -> int:
    from . import limits

    return limits.main(args)
