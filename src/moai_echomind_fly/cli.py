"""`fly-connectome` 명령 (I §1.1)."""

from __future__ import annotations

import argparse
import json
import sys
import time

from . import data, graph, paths


def _cmd_download(args: argparse.Namespace) -> int:
    result = data.download()
    print(json.dumps(result, indent=2))
    return 0


def _cmd_graph(args: argparse.Namespace) -> int:
    t0 = time.time()
    g = graph.compile_graph(args.name)
    print(f"[graph] 저장: {paths.graph_dir()}/{g.name}.* ({time.time() - t0:.0f} s)")
    seeds = list(args.rewired_seed or [])
    for part in (args.rewired_seeds or "").split(","):
        if "-" in part:
            lo, hi = part.split("-")
            seeds += list(range(int(lo), int(hi) + 1))
        elif part:
            seeds.append(int(part))
    for seed in dict.fromkeys(seeds):          # 겹친 씨앗은 한 번만
        r = graph.rewire(g, seed)
        graph.save(r)
        print(f"[graph] 섞은 배선 저장: {r.name}")
    print(json.dumps({k: v for k, v in g.meta.items() if k != "source_sha256"}, ensure_ascii=False,
                     indent=2))
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    from . import tools

    try:
        summary = tools.run_experiment(
            [t for t in args.stim.split(",") if t], args.rate, args.ms,
            [t for t in args.silence.split(",") if t], args.seed, "cli")
    except tools.ToolError as err:
        print(err.to_text(), file=sys.stderr)
        return 2
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def _cmd_experiment(args: argparse.Namespace) -> int:
    from . import experimenter

    result = experimenter.run(args.question)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if "error" not in result else 2


def _cmd_collect(args: argparse.Namespace) -> int:
    from .loop import collect

    return collect.main(args)


def _cmd_collect_probe(args: argparse.Namespace) -> int:
    from .loop import collect

    return collect.probe(args)


def _cmd_mcp(args: argparse.Namespace) -> int:
    from . import mcp_server

    mcp_server.main()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="fly-connectome")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("download", help="공식 파일 3개 받기 · 공식 MD5 대조")
    p.set_defaults(func=_cmd_download)

    p = sub.add_parser("graph", help="그래프 컴파일 → data/graph/")
    p.add_argument("--name", default="traced-v1")
    p.add_argument("--rewired-seed", type=int, action="append",
                   help="섞은 배선도 만든다(여러 번 가능)")
    p.add_argument("--rewired-seeds", default="",
                   help="섞은 배선 씨앗 범위(예: 0-19 또는 0,3,5) — 재배선 앙상블(R-7469 F-001)")
    p.set_defaults(func=_cmd_graph)

    p = sub.add_parser("run", help="실험 한 번 — MCP run_experiment 와 같은 함수")
    p.add_argument("--stim", required=True, help="종류 이름(쉼표로 여러 개)")
    p.add_argument("--rate", type=float, default=150.0)
    p.add_argument("--ms", type=int, default=200)
    p.add_argument("--silence", default="", help="침묵시킬 종류(쉼표)")
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=_cmd_run)

    p = sub.add_parser("collect", help="F-017 학습 자료 모으기 — 교사가 몸을 몬다(무거움)")
    p.add_argument("--name", required=True)
    p.add_argument("--seeds", default="0-9", help="예: 0-49 또는 1,3,5")
    p.add_argument("--graph", default="traced-v1")
    p.add_argument("--rewired-seed", type=int, default=0)
    p.add_argument("--rewired-seeds", default="",
                   help="재배선 여럿(예: 0-4) — 뇌마다 features-rewired-s<K>.npz(R-7469 F-002)")
    p.add_argument("--workers", type=int, default=1, help="씨앗을 나눠 돌릴 자식 프로세스 수")
    p.add_argument("--threads", type=int, default=0, help="일꾼마다 torch 스레드 수(0 = 기본)")
    p.set_defaults(func=_cmd_collect)

    p = sub.add_parser("collect-probe",
                       help="자료 모으기 작은 측정 — 결정당 시간 · 메모리(저장 없음)")
    p.add_argument("--seeds", default="0-1")
    p.add_argument("--rewired-seeds", default="0-4")
    p.add_argument("--graph", default="traced-v1")
    p.set_defaults(func=_cmd_collect_probe)

    p = sub.add_parser("mcp", help="MCP 도구 서버(stdio)")
    p.set_defaults(func=_cmd_mcp)

    p = sub.add_parser("experiment",
                       help="F-009 Qwen 실험자 — Qwen 이 MCP 로 실험을 부른다(무거움)")
    p.add_argument("question")
    p.set_defaults(func=_cmd_experiment)

    from .bench import cli as bench_cli  # 무거운 의존을 필요할 때만 불러온다
    bench_cli.register(sub)
    from .loop import compress, linear_probe, replay, scorecard, train_cli
    train_cli.register(sub)
    scorecard.register(sub)
    linear_probe.register(sub)
    compress.register(sub)
    replay.register(sub)
    from . import datasets
    datasets.register(sub)
    from . import backup
    backup.register(sub)
    from . import inspect as inspect_mod
    inspect_mod.register(sub)
    from . import retino
    retino.register(sub)
    from . import dashboard
    dashboard.register(sub)

    args = parser.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
