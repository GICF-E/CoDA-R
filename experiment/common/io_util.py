"""实验目录与 JSON 记录；全部用 UTF-8 读写。"""
from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = REPO_ROOT / "experiment" / "outputs"
SMOKE_ROOT = REPO_ROOT / ".tmp" / "experiment-smoke"


def _json_default(obj: Any) -> Any:
    # numpy 标量与数组不能直接进 json.dumps
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    content = json.dumps(payload, ensure_ascii = False, indent = 2, default = _json_default)
    # 同目录原子替换，进程中断时保留上一份完整文件
    with tempfile.NamedTemporaryFile(mode = "w", encoding = "utf-8", dir = path.parent, delete = False) as handle:
        temporary = Path(handle.name)
        handle.write(content)
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok = True)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding = "utf-8"))


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding = "utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii = False, default = _json_default) + "\n")
        handle.flush()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding = "utf-8").splitlines() if line.strip()]


def repair_jsonl_tail(path: Path) -> None:
    """只移除中断留下的末尾半行；中间损坏仍报错，不悄悄丢弃完整记录。"""
    if not path.is_file():
        return
    raw = path.read_bytes()
    rows = raw.splitlines(keepends = True)
    for index, line in enumerate(rows):
        try:
            json.loads(line)
        except (ValueError, UnicodeDecodeError):
            if index != len(rows) - 1 or line.endswith(b"\n"):
                raise
            path.write_bytes(b"".join(rows[:index]))
            return
    if raw and not raw.endswith(b"\n"):
        with path.open("ab") as handle:
            handle.write(b"\n")


def write_summary(output_dir: Path, text: str) -> None:
    (output_dir / "summary.md").write_text(text, encoding = "utf-8")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush = True)
