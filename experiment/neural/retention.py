"""M4 遗忘：从近似纳什出发继续训练时，latest 守不守得住已经学会的东西。

底座是 CFR+ 的近似纳什表格策略（与 M3 相同）。表格模型里 R-NaD / MMD 以纳什为不动点，不会离开；
神经网络共享参数，自博弈只在自己走到的信息集上产生梯度，走不到的地方会被别处的更新带着漂移。
所以漂移按自博弈下的对手到达概率分档，档案（考试用的历史快照）则对有池和无池的方法一视同仁。
"""
from __future__ import annotations

from typing import Any

import numpy as np

from experiment.neural.evals import exact_loss, exploitability_of_table, infoset_states, to_tabular
from experiment.neural.holes import InfosetProfile, Table, infoset_profiles


def reach_band(reach: float, bands: tuple[tuple[float, float], ...]) -> int | None:
    return next((b for b, (lo, hi) in enumerate(bands) if lo <= reach < hi), None)


class RetentionProbe:
    """
    每 every 个 rollout 读一次 latest（只看策略，不打对局）：
    - drift_tv / drift_loss：每个对手到达概率档里，latest 与底座表格的平均总变差，以及每次到达比底座少得的分
      （对手按底座行动，收益归一到 [0, 1]）；只统计 latest 自己经常走到（own >= own_min）且有选择的信息集；
    - probe_exploitability：全树 exploitability；
    - archive_losses：对档案里每个历史版本的精确败率。档案从底座开始，每 archive_every 个 rollout 收进一个 latest，
      只用来考试，不进训练。
    """

    def __init__(self, game: Any, table: Table, bands: tuple[tuple[float, float], ...], own_min: float, every: int,
                 archive_every: int, base: Any):
        states = infoset_states(game)
        self.game = game
        self.table = table
        self.n_actions = int(game.num_distinct_actions())
        self.every = every
        self.archive_every = archive_every
        self.sites: list[list[tuple[str, InfosetProfile, Any, int]]] = [[] for _ in bands]
        for key, prof in infoset_profiles(game, table).items():
            band = reach_band(prof.opponent_reach, bands)
            if band is not None and prof.own >= own_min and prof.counterfactual > 0.0 and len(prof.values) >= 2:
                self.sites[band].append((key, prof, *states[key]))
        self.archive = [to_tabular(game, base)]

    def band_sizes(self) -> list[int]:
        return [len(sites) for sites in self.sites]

    def _drift(self, actor: Any) -> tuple[list[float], list[float]]:
        tv, loss = [], []
        for sites in self.sites:
            if not sites:
                tv.append(float("nan"))
                loss.append(float("nan"))
                continue
            band_tv, band_loss = [], []
            for key, prof, state, player in sites:
                probs = actor.probs(state, player, self.n_actions)
                base = self.table[key]
                band_tv.append(0.5 * sum(abs(probs.get(a, 0.0) - base.get(a, 0.0)) for a in prof.values))
                band_loss.append(prof.visit_value(base) - prof.visit_value(probs))
            tv.append(float(np.mean(band_tv)))
            loss.append(float(np.mean(band_loss)))
        return tv, loss

    def __call__(self, roll: int, actor: Any) -> dict[str, Any]:
        if (roll + 1) % self.every != 0:
            return {}
        latest = to_tabular(self.game, actor)
        tv, loss = self._drift(actor)
        rec = dict(probe_exploitability = exploitability_of_table(self.game, latest), drift_tv = tv, drift_loss = loss,
                   archive_losses = [exact_loss(self.game, latest, member) for member in self.archive])
        if (roll + 1) % self.archive_every == 0:
            self.archive.append(latest)
        return rec
