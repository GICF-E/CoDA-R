"""命令行入口：查看计划、运行实验或分组、查询进度。"""
from __future__ import annotations

import argparse
from pathlib import Path

from experiment.common.io_util import OUTPUT_ROOT, SMOKE_ROOT, log
from experiment.common.runner import RunOptions, cell_status
from experiment.registry import BY_ID, EXPERIMENTS, GROUPS, expand


def _print_plan() -> None:
    for e in EXPERIMENTS:
        count = f"{e.planned()} cells" if e.planned is not None else ""
        needs = f"needs {', '.join(e.needs)}" if e.needs else ""
        print(f"{e.id:24} {e.group:11} {count:>12}  {e.title}  {needs}")
    print("groups: " + ", ".join(f"{name} ({len(ids)})" for name, ids in GROUPS.items()))


def _print_status(root: Path) -> None:
    for e in EXPERIMENTS:
        state = cell_status(root / e.id)
        planned = e.planned() if e.planned is not None else None
        detail = f"{state['done']}/{planned if planned is not None else '?'} done, {state['error']} failed" \
            if state["exists"] else "not started"
        print(f"{e.id:24} {detail}")


def main() -> None:
    parser = argparse.ArgumentParser(description = "CoDA-R experiments")
    parser.add_argument("exp", nargs = "?", help = "experiment id or group (see --list)")
    parser.add_argument("--list", action = "store_true", help = "show the plan and cell counts")
    parser.add_argument("--status", action = "store_true", help = "show progress under the output root")
    parser.add_argument("--smoke", action = "store_true", help = "tiny budgets; output goes to .tmp/experiment-smoke")
    parser.add_argument("--resume", action = "store_true", help = "skip finished cells; without it the experiment starts over")
    parser.add_argument("--workers", type = int, default = 1, help = "parallel worker processes")
    parser.add_argument("--max-cells", type = int, default = None, help = "run at most this many cells, then stop")
    parser.add_argument("--output-root", type = Path, default = None, help = "root directory; each experiment uses <root>/<id>")
    args = parser.parse_args()
    root = args.output_root or (SMOKE_ROOT if args.smoke else OUTPUT_ROOT)
    if args.status:
        _print_status(root)
        return
    if args.list or args.exp is None:
        _print_plan()
        return
    try:
        ids = expand(args.exp)
    except ValueError as exc:
        parser.error(str(exc))
    for i, exp_id in enumerate(ids, start = 1):
        log(f"({i}/{len(ids)}) {exp_id}: {BY_ID[exp_id].title}" + (" [smoke]" if args.smoke else ""))
        options = RunOptions(out = root / exp_id, smoke = args.smoke, resume = args.resume, max_cells = args.max_cells,
                             workers = args.workers)
        out = BY_ID[exp_id].run(options)
        log(f"finished {exp_id} -> {out}")


if __name__ == "__main__":
    main()
