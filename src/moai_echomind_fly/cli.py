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
    for seed in args.rewired_seed or []:
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
    p.set_defaults(func=_cmd_graph)

    p = sub.add_parser("run", help="실험 한 번 — MCP run_experiment 와 같은 함수")
    p.add_argument("--stim", required=True, help="종류 이름(쉼표로 여러 개)")
    p.add_argument("--rate", type=float, default=150.0)
    p.add_argument("--ms", type=int, default=200)
    p.add_argument("--silence", default="", help="침묵시킬 종류(쉼표)")
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=_cmd_run)

    p = sub.add_parser("mcp", help="MCP 도구 서버(stdio)")
    p.set_defaults(func=_cmd_mcp)

    from .bench import cli as bench_cli  # 무거운 의존을 필요할 때만 불러온다
    bench_cli.register(sub)

    args = parser.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
