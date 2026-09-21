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

    from .bench import cli as bench_cli  # 무거운 의존을 필요할 때만 불러온다
    bench_cli.register(sub)

    args = parser.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
