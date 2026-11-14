"""CoDA-R 的两个混合输出：当前混合 q = v / t 与外环时间平均 Q（论文 §4.4），按到达概率聚合。"""
from __future__ import annotations

from typing import Any

import numpy as np

from experiment.neural.evals import TableCache, aggregate, exploitability_of_table


class OuterMixtures:
    def __init__(self, game: Any, cache: TableCache):
        self.game = game
        self.cache = cache
        self.total_mass = 0.0
        self.current: Any = None
        self.average: Any = None

    def update(self, latest_table: Any, members: list[tuple[str, Any]], pool_field: list[float]) -> None:
        """在外环结账、入池与淘汰之前，用同一时刻的 (latest, w) 更新；自博弈份额为 1，质量 t = 1 + a。"""
        tables = [latest_table]
        weights = [1.0]
        for (member_id, actor), w in zip(members, pool_field):
            if w > 0.0:
                tables.append(self.cache.get(member_id, actor))
                weights.append(float(w))
        mass = float(sum(weights))
        self.current = aggregate(self.game, tables, np.asarray(weights))
        if self.average is None:
            self.average = self.current
        else:
            # 旧的时间平均与新的当前混合在 realization 上等价，可以递归聚合而不保留全部网络
            self.average = aggregate(self.game, [self.average, self.current], np.array([self.total_mass, mass]))
        self.total_mass += mass

    def evaluate(self) -> dict[str, float]:
        if self.current is None:
            return {}
        return dict(mix_current_expl = exploitability_of_table(self.game, self.current),
                    mix_average_expl = exploitability_of_table(self.game, self.average))
