"""按类别统计实际完成的环境对局；预算只数算法性对局。"""
from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import wraps
from typing import ClassVar, ParamSpec, TypeVar

P = ParamSpec("P")
R = TypeVar("R")


@dataclass
class GameLedger:
    counts: dict[str, int] = field(default_factory = dict)
    # 只做记录、不影响算法的对局不计入预算
    DIAGNOSTIC_KINDS: ClassVar[frozenset[str]] = frozenset({"diagnostic"})

    @property
    def total(self) -> int:
        return sum(self.counts.values())

    @property
    def algorithmic_total(self) -> int:
        return sum(n for kind, n in self.counts.items() if kind not in self.DIAGNOSTIC_KINDS)

    def record(self) -> None:
        tag = _kind.get()
        self.counts[tag] = self.counts.get(tag, 0) + 1


_ledger: ContextVar[GameLedger | None] = ContextVar("game_ledger", default = None)
_kind: ContextVar[str] = ContextVar("game_kind", default = "learner")


def count_game() -> None:
    ledger = _ledger.get()
    if ledger is not None:
        ledger.record()


@contextmanager
def game_kind(name: str) -> Iterator[None]:
    # 可嵌套；退出或异常时恢复外层分类
    token = _kind.set(name)
    try:
        yield
    finally:
        _kind.reset(token)


def measured_run(fn: Callable[P, R]) -> Callable[P, R]:
    """为每次运行创建独立账本。"""
    @wraps(fn)
    def wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
        token = _ledger.set(GameLedger())
        try:
            return fn(*args, **kwargs)
        finally:
            _ledger.reset(token)
    return wrapped


def current_ledger() -> GameLedger:
    ledger = _ledger.get()
    if ledger is None:
        raise RuntimeError("run must be wrapped in measured_run")
    return ledger
