"""命令行入口：列出可用的图、生成指定的图或全部图，以及分析表 analysis.md。"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from analysis.data import Outputs
from analysis.figures import FIGURES
from analysis.tables import write_tables
from experiment.common.io_util import OUTPUT_ROOT, log

DEFAULT_DEST = Path(__file__).resolve().parent / "outputs"
TABLES = "tables"


def main() -> None:
    parser = argparse.ArgumentParser(description = "CoDA-R analysis and paper figures")
    parser.add_argument("names", nargs = "*", help = f"figures to build (see --list) or '{TABLES}'; default: all")
    parser.add_argument("--list", action = "store_true", help = "list the available figures")
    parser.add_argument("--output-root", type = Path, default = OUTPUT_ROOT, help = "experiment output root to read")
    parser.add_argument("--dest", type = Path, default = DEFAULT_DEST, help = "directory for figures and analysis.md")
    args = parser.parse_args()
    if args.list:
        print("\n".join([*FIGURES, TABLES]))
        return
    names = args.names or [*FIGURES, TABLES]
    unknown = [n for n in names if n not in FIGURES and n != TABLES]
    if unknown:
        parser.error(f"unknown figure(s) {unknown}; see --list")
    out = Outputs(args.output_root)
    for name in names:
        t0 = time.time()
        if name == TABLES:
            log(f"{TABLES}: {write_tables(out, args.dest)} ({time.time() - t0:.0f}s)")
            continue
        stems = FIGURES[name](out, args.dest)
        log(f"{name}: " + (", ".join(f"{s}.pdf" for s in stems) if stems else "skipped (no data)") + f" ({time.time() - t0:.0f}s)")
