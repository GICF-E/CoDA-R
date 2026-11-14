"""读取 experiment/outputs 下各实验的结果；同一实验在一次运行里只读一次。"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from experiment.common.io_util import OUTPUT_ROOT, load_json, load_jsonl
from experiment.common.runner import done_records, latest_records
from experiment.experiments.report import CellResult, load_results

TheoryCell = tuple[dict[str, Any], list[dict[str, Any]]]


def method_key(r: CellResult) -> str:
    """样式与分组用的方法键：主实验、调参、消融、诊断、M4 与规模实验记在 meta['method']，M1–M3 记在 meta['rule']。"""
    return str(r.meta.get("method") or r.meta["rule"])


def _is_training_cell(record: dict[str, Any]) -> bool:
    # 机制实验的 prep 格子（共同底座、外部成员）不是被比较的方法
    payload = record.get("payload", {})
    return "config" in payload and payload.get("role") != "prep"


def group_by(results: list[CellResult], key: Callable[[CellResult], Any]) -> dict[Any, list[CellResult]]:
    out: dict[Any, list[CellResult]] = {}
    for r in results:
        out.setdefault(key(r), []).append(r)
    return out


@dataclass
class Outputs:
    root: Path = OUTPUT_ROOT
    _neural: dict[str, list[CellResult]] = field(default_factory = dict)
    _theory: dict[str, list[TheoryCell]] = field(default_factory = dict)

    def has(self, exp_id: str) -> bool:
        return (self.root / exp_id / "cells.jsonl").is_file()

    def neural(self, exp_id: str) -> list[CellResult]:
        """神经训练格子（已完成的），缺失的实验返回空列表。"""
        if exp_id not in self._neural:
            out = self.root / exp_id
            records = {k: r for k, r in latest_records(out).items() if _is_training_cell(r)} if self.has(exp_id) else {}
            self._neural[exp_id] = load_results(out, records)
        return self._neural[exp_id]

    def by_env_method(self, exp_id: str) -> dict[tuple[str, str], list[CellResult]]:
        return group_by(self.neural(exp_id), lambda r: (r.cfg.env, method_key(r)))

    def theory(self, exp_id: str) -> list[TheoryCell]:
        """理论实验：(payload, 逐 seed 的运行记录) 的列表。"""
        if exp_id not in self._theory:
            out = self.root / exp_id
            records = done_records(latest_records(out)) if self.has(exp_id) else []
            self._theory[exp_id] = [(r["payload"], load_json(out / r["cell_id"] / "summary.json")["runs"]) for r in records]
        return self._theory[exp_id]

    def scale_rows(self) -> dict[tuple[str, str], list[dict[str, Any]]]:
        """scale-eval：{(游戏, 方法): exact.jsonl 的行}，每行是一个 seed 在一个预算比例处的精确评估。"""
        out = self.root / "scale-eval"
        if not self.has("scale-eval"):
            return {}
        return {(r["payload"]["env"], r["payload"]["method"]): load_jsonl(out / r["cell_id"] / "exact.jsonl")
                for r in done_records(latest_records(out))}

    def selected(self, stage: str) -> dict[str, dict[str, dict[str, Any]]]:
        path = self.root / f"tune-{stage}" / "selected.json"
        return load_json(path)["selected"] if path.is_file() else {}
