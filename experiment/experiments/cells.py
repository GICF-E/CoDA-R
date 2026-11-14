"""神经训练格子：RunConfig 与 CellSpec 之间的转换，以及按算法分派的单格入口。"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from experiment.common.runner import CellSpec
from experiment.neural.config import RunConfig, config_from_dict
from experiment.neural.nfsp import run_nfsp
from experiment.neural.psro import run_psro
from experiment.neural.train import run_training


def train_spec(cfg: RunConfig, smoke: bool, **meta: Any) -> CellSpec:
    """meta 是报告用的附加字段（方法名、调参阶段等），也进入哈希。"""
    cfg = cfg.shrunk() if smoke else cfg
    slug = re.sub(r"[^0-9A-Za-z.]+", "-", cfg.label).strip("-")
    return CellSpec(f"{cfg.env}_{slug}_s{cfg.seed}", dict(config = cfg.as_dict(), **meta))


def run_train_cell(payload: dict[str, Any], cell_dir: Path) -> dict[str, Any]:
    cfg = config_from_dict(payload["config"])
    if cfg.algorithm == "psro":
        return run_psro(cfg, cell_dir)
    if cfg.algorithm == "nfsp":
        return run_nfsp(cfg, cell_dir)
    return run_training(cfg, cell_dir)
