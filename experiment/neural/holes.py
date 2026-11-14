"""M3 隐藏漏洞：自博弈到达概率受控的漏洞。

底座是 CFR+ 的近似纳什表格策略；在若干信息集上把动作换成最差的那个，就得到一个漏洞。
自博弈只有在对手（也是 latest）走到这些信息集时才会修复漏洞，修复速度受对手动作到达概率限制；
池里的 exploiter 是精确 BR，它会主动走到漏洞那里。到达概率按对手动作的部分分档，机会节点的部分两者都改不了。
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from experiment.neural.envs import information_tensor, legal_actions, legal_mask
from experiment.neural.evals import infoset_states
from experiment.neural.mlp import SmallMLP

Table = dict[str, dict[int, float]]


@dataclass(frozen = True)
class InfosetProfile:
    """自博弈（双方都按同一张表行动）下一个信息集的到达概率与反事实价值；收益已归一到 [0, 1]。"""
    player: int
    # latest 自己的动作到达概率；完美回忆下信息集内所有历史相同
    own: float
    # 对手确定性地走到这个信息集时的机会到达概率：机会到达概率之和除以对手动作序列的种类数
    # （Leduc 里同一信息集的历史只差对手的牌，Goofspiel 的轮流形式里只差对手本轮暗出的牌）
    chance: float
    # 机会 x 对手动作到达概率之和（反事实到达概率）
    counterfactual: float
    # 每个动作的反事实价值 sum_h counterfactual(h) * u_i(h a)
    values: dict[int, float]

    @property
    def opponent_reach(self) -> float:
        """到达概率里对手动作的部分：exploiter 能把它抬到 1，机会节点的部分改不了。"""
        return self.counterfactual / self.chance if self.chance > 0.0 else 0.0

    def visit_value(self, policy: dict[int, float]) -> float:
        """按 policy 行动时每次到达这个信息集的期望得分。"""
        if self.counterfactual <= 0.0:
            return float("nan")
        return sum(policy.get(a, 0.0) * v for a, v in self.values.items()) / self.counterfactual

    def regret(self, action: int, policy: dict[int, float]) -> float:
        """每次到达时，把 policy 换成确定性地走 action 损失的得分。"""
        if self.counterfactual <= 0.0:
            return float("nan")
        return self.visit_value(policy) - self.values[action] / self.counterfactual


def infoset_profiles(game: Any, table: Table) -> dict[str, InfosetProfile]:
    """遍历整棵树一次；零概率动作也要展开，否则拿不到它们的反事实价值。"""
    lo = float(game.min_utility())
    span = float(game.max_utility()) - lo
    acc: dict[str, dict[str, Any]] = {}
    paths: dict[str, set[tuple[int, ...]]] = {}

    # moves[p] 是玩家 p 一路走过的动作序列，用来数同一信息集里对手动作序列的种类
    def recurse(state: Any, own: tuple[float, float], chance: float, moves: tuple[tuple[int, ...], tuple[int, ...]]) -> np.ndarray:
        if state.is_terminal():
            return (np.asarray(state.returns(), dtype = np.float64) - lo) / span
        if state.is_chance_node():
            return sum((p * recurse(state.child(a), own, chance * p, moves) for a, p in state.chance_outcomes()),
                       np.zeros(2, dtype = np.float64))
        player = int(state.current_player())
        key = state.information_state_string(player)
        policy = table[key]
        reach = chance * own[1 - player]
        entry = acc.setdefault(key, dict(player = player, own = own[player], chance = 0.0, counterfactual = 0.0, values = {}))
        entry["chance"] += chance
        entry["counterfactual"] += reach
        paths.setdefault(key, set()).add(moves[1 - player])
        value = np.zeros(2, dtype = np.float64)
        for a in legal_actions(state, player):
            p = float(policy.get(a, 0.0))
            child_own = (own[0] * p, own[1]) if player == 0 else (own[0], own[1] * p)
            child_moves = (moves[0] + (a,), moves[1]) if player == 0 else (moves[0], moves[1] + (a,))
            child = recurse(state.child(a), child_own, chance, child_moves)
            entry["values"][a] = entry["values"].get(a, 0.0) + reach * float(child[player])
            value += p * child
        return value

    recurse(game.new_initial_state(), (1.0, 1.0), 1.0, ((), ()))
    for key, entry in acc.items():
        entry["chance"] /= len(paths[key])
    return {key: InfosetProfile(**entry) for key, entry in acc.items()}


@dataclass(frozen = True)
class Hole:
    infoset: str
    action: int
    player: int
    own: float
    opponent_reach: float
    chance: float
    regret: float

    def as_dict(self) -> dict[str, Any]:
        return dict(infoset = self.infoset, action = self.action, player = self.player, own = self.own,
                    opponent_reach = self.opponent_reach, chance = self.chance, regret = self.regret)


def select_holes(profiles: dict[str, InfosetProfile], table: Table, bands: tuple[tuple[float, float], ...], size: int,
                 own_min: float, regret_min: float) -> list[list[Hole]]:
    """
    每个对手到达概率档 [lo, hi) 里选每次到达损失（regret）最大的 size 个信息集，漏洞动作取反事实价值最低的动作。
    要求 latest 自己经常走到（own >= own_min）、底座几乎不走这个动作，且每次到达至少损失 regret_min。
    """
    candidates: list[Hole] = []
    for key, prof in profiles.items():
        if len(prof.values) < 2 or prof.own < own_min or prof.counterfactual <= 0.0:
            continue
        policy = table[key]
        action = min(prof.values, key = lambda a: prof.values[a])
        regret = prof.regret(action, policy)
        if regret >= regret_min and policy.get(action, 0.0) <= 0.05:
            candidates.append(Hole(key, int(action), prof.player, prof.own, prof.opponent_reach, prof.chance, regret))
    out = []
    for lo, hi in bands:
        inside = sorted((h for h in candidates if lo <= h.opponent_reach < hi), key = lambda h: -h.regret)
        out.append(inside[:size])
    return out


def damaged(table: Table, holes: list[Hole]) -> Table:
    out = copy.deepcopy(table)
    for hole in holes:
        out[hole.infoset] = {a: float(a == hole.action) for a in out[hole.infoset]}
    return out


def distill(net: SmallMLP, game: Any, table: Table, steps: int, lr: float = 1e-3) -> dict[str, float]:
    """
    全批量监督：策略头拟合表格策略（交叉熵），价值头拟合自博弈下每次到达的期望得分，后者让 PPO 一开始就有合理的基线。
    返回策略与表格的最大总变差。
    """
    states = infoset_states(game)
    profiles = infoset_profiles(game, table)
    keys = sorted(states)
    n_actions = int(game.num_distinct_actions())
    obs = torch.tensor([information_tensor(*states[k]) for k in keys], dtype = torch.float32)
    mask = torch.tensor([legal_mask(states[k][0], states[k][1], n_actions) for k in keys], dtype = torch.bool)
    target = torch.tensor([[table[k].get(a, 0.0) for a in range(n_actions)] for k in keys], dtype = torch.float32)
    visit = [profiles[k].visit_value(table[k]) for k in keys]
    value_target = torch.tensor([v if np.isfinite(v) else 0.5 for v in visit], dtype = torch.float32)
    value_mask = torch.tensor([float(np.isfinite(v)) for v in visit], dtype = torch.float32)
    opt = torch.optim.Adam(net.parameters(), lr = lr)
    net.train()
    for _ in range(steps):
        logits, value = net(obs, mask)
        policy_loss = -(target * F.log_softmax(logits, dim = -1)).sum(dim = -1).mean()
        value_loss = (value_mask * (value - value_target) ** 2).sum() / value_mask.sum().clamp(min = 1.0)
        opt.zero_grad()
        (policy_loss + 0.5 * value_loss).backward()
        opt.step()
    net.eval()
    with torch.no_grad():
        probs = torch.softmax(net(obs, mask)[0], dim = -1)
    return dict(max_tv = float(0.5 * (probs - target).abs().sum(dim = -1).max()))


class HoleProbe:
    """每个 rollout 读出 latest 在漏洞信息集上走漏洞动作的平均概率；只看策略，不打对局。"""

    def __init__(self, game: Any, holes: list[dict[str, Any]]):
        states = infoset_states(game)
        self.n_actions = int(game.num_distinct_actions())
        self.targets = [(*states[h["infoset"]], int(h["action"])) for h in holes]

    def __call__(self, roll: int, actor: Any) -> dict[str, Any]:
        probs = [actor.probs(state, player, self.n_actions).get(action, 0.0) for state, player, action in self.targets]
        return dict(hole_mass = float(np.mean(probs)) if probs else float("nan"))
