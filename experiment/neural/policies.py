"""把神经网络、表格策略、脚本 bot 与整局混合适配到同一个对局接口。

每个对手都实现 act（抽一个动作）和 probs（当前信息集的动作分布）；
probs 用于全树展开，所以必须只依赖信息集。整局混合例外，见 MixtureActor。
"""
from __future__ import annotations

from typing import Any, Protocol

import numpy as np
import torch

from experiment.neural.envs import information_tensor, legal_actions, legal_mask
from experiment.neural.mlp import SmallMLP


class Actor(Protocol):
    def act(self, state: Any, player: int, n_actions: int, rng: np.random.Generator) -> int:
        ...

    def probs(self, state: Any, player: int, n_actions: int) -> dict[int, float]:
        ...


def _sample(dist: dict[int, float], rng: np.random.Generator) -> int:
    actions = list(dist)
    p = np.array([dist[a] for a in actions], dtype = np.float64)
    return int(actions[int(rng.choice(len(actions), p = p / p.sum()))])


def _uniform(state: Any, player: int) -> dict[int, float]:
    legal = legal_actions(state, player)
    return {a: 1.0 / len(legal) for a in legal}


class TorchActor:
    """冻结网络的随机策略。"""

    def __init__(self, net: SmallMLP, device: str):
        self.net = net
        self.device = device
        self.net.eval()

    def _logits(self, state: Any, player: int, n_actions: int) -> torch.Tensor:
        obs = torch.tensor(information_tensor(state, player), dtype = torch.float32, device = self.device)
        mask = torch.tensor(legal_mask(state, player, n_actions), dtype = torch.bool, device = self.device)
        with torch.no_grad():
            logits, _value = self.net(obs.unsqueeze(0), mask.unsqueeze(0))
        return logits.squeeze(0)

    def act(self, state: Any, player: int, n_actions: int, rng: np.random.Generator) -> int:
        return _sample(self.probs(state, player, n_actions), rng)

    def probs(self, state: Any, player: int, n_actions: int) -> dict[int, float]:
        p = torch.softmax(self._logits(state, player, n_actions), dim = -1).cpu().numpy()
        legal = legal_actions(state, player)
        total = float(sum(p[a] for a in legal))
        return {a: float(p[a]) / total for a in legal}


class UniformActor:
    def act(self, state: Any, player: int, n_actions: int, rng: np.random.Generator) -> int:
        legal = legal_actions(state, player)
        return legal[int(rng.integers(len(legal)))]

    def probs(self, state: Any, player: int, n_actions: int) -> dict[int, float]:
        return _uniform(state, player)


class TabularActor:
    """按信息集字符串查表；表里没有的信息集取均匀分布。"""

    def __init__(self, table: dict[str, dict[int, float]]):
        self.table = table

    def act(self, state: Any, player: int, n_actions: int, rng: np.random.Generator) -> int:
        return _sample(self.probs(state, player, n_actions), rng)

    def probs(self, state: Any, player: int, n_actions: int) -> dict[int, float]:
        key = str(state.information_state_string(player))
        if key in self.table:
            return dict(self.table[key])
        return _uniform(state, player)


class PreferenceBot:
    """脚本 bot：在合法动作里按固定偏好顺序取第一个可行动作（确定性）。"""

    def __init__(self, preference: list[int] | None = None, highest: bool = False, lowest: bool = False):
        self.preference = preference or []
        self.highest = highest
        self.lowest = lowest

    def _choice(self, state: Any, player: int) -> int:
        legal = legal_actions(state, player)
        for action in self.preference:
            if action in legal:
                return action
        if self.highest:
            return max(legal)
        if self.lowest:
            return min(legal)
        return legal[0]

    def act(self, state: Any, player: int, n_actions: int, rng: np.random.Generator) -> int:
        return self._choice(state, player)

    def probs(self, state: Any, player: int, n_actions: int) -> dict[int, float]:
        return {self._choice(state, player): 1.0}


class MixtureActor:
    """整局混合：每局每个座位只抽一次成员。全树评估必须按到达概率聚合，不能用 probs 逐状态平均。"""

    def __init__(self, actors: list[Any], weights: np.ndarray):
        w = np.maximum(np.asarray(weights, dtype = np.float64), 0.0)
        if len(actors) == 0 or w.size != len(actors) or float(w.sum()) <= 0.0:
            raise ValueError("mixture needs one positive weight per member")
        self.actors = list(actors)
        self.weights = w / float(w.sum())
        self._episode_members: dict[int, int] = {}

    def reset_episode(self) -> None:
        self._episode_members.clear()
        for actor in self.actors:
            if hasattr(actor, "reset_episode"):
                actor.reset_episode()

    def act(self, state: Any, player: int, n_actions: int, rng: np.random.Generator) -> int:
        if player not in self._episode_members:
            self._episode_members[player] = int(rng.choice(len(self.actors), p = self.weights))
        return int(self.actors[self._episode_members[player]].act(state, player, n_actions, rng))

    def probs(self, state: Any, player: int, n_actions: int) -> dict[int, float]:
        raise TypeError("a whole-episode mixture has no per-infoset distribution; aggregate by reach instead")


def frozen_actor(net: SmallMLP, device: str) -> TorchActor:
    for p in net.parameters():
        p.requires_grad_(False)
    return TorchActor(net, device)
