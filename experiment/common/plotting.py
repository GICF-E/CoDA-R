"""延迟加载无界面绘图库，统一保存 PNG/PDF 并及时释放图形对象。"""
from __future__ import annotations

from pathlib import Path
from typing import Any

_PLT: Any | None = None
_PLT_FAILED = False


def get_pyplot() -> Any | None:
    global _PLT, _PLT_FAILED
    if _PLT_FAILED:
        return None
    if _PLT is not None:
        return _PLT
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        _PLT = plt
        return plt
    except ImportError:
        _PLT_FAILED = True
        return None


def subplots(*args: Any, **kwargs: Any) -> tuple[Any | None, Any | None]:
    plt = get_pyplot()
    if plt is None:
        return None, None
    return plt.subplots(*args, **kwargs)


def save_figure(fig: Any, output_dir: Path, stem: str) -> None:
    if fig is None:
        return
    output_dir.mkdir(parents = True, exist_ok = True)
    fig.savefig(output_dir / f"{stem}.png", dpi = 150, bbox_inches = "tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches = "tight")
    plt = get_pyplot()
    if plt is not None:
        plt.close(fig)
