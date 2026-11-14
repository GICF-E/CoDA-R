"""规模实验两个游戏的全树精确评估：3x3 abrupt Dark Hex 与 abrupt Phantom Tic-Tac-Toe（exp-a-spiel，见 third_party/）。

两个游戏各约 2.7e10 个节点、2e7 量级的信息集，OpenSpiel 的表格展开不可行；exp-a-spiel 的遍历器在 C++ 里多线程展开整棵树。
遍历器给每名玩家的每个信息集一行，行里就是 OpenSpiel 的信息状态张量，所以网络批量前向一次就得到整份行为策略。
一个遍历器约需 30 GB 内存，同一时刻只能有一个评估进程；构造它约 70 秒，一次评估约 95 秒（32 线程）。
"""
from __future__ import annotations

from typing import Any

import eas
import numpy as np
import torch

from experiment.neural.envs import load_game
from experiment.neural.mlp import SmallMLP

# env -> (遍历器, 单局状态)
EAS_GAMES: dict[str, tuple[Any, Any]] = {
    "abrupt_dark_hex": (eas.AbruptDhTraverser, eas.AbruptDhState),
    "abrupt_phantom_ttt": (eas.AbruptPtttTraverser, eas.AbruptPtttState),
}
FORWARD_BATCH = 1 << 16
# 行为策略的合法性检查容差（每行和为 1）
ROW_SUM_TOL = 1e-9


def coherence_mismatches(env: str, games: int, rng: np.random.Generator) -> int:
    """随机对局逐步比对 OpenSpiel 与 exp-a-spiel：当前玩家、合法动作、信息状态张量与胜负，返回不一致的局数。"""
    game = load_game(env)
    state_cls = EAS_GAMES[env][1]
    mismatches = 0
    for _ in range(games):
        os_state, eas_state = game.new_initial_state(), state_cls()
        while True:
            if os_state.is_terminal() != eas_state.is_terminal():
                mismatches += 1
                break
            if os_state.is_terminal():
                r = float(os_state.returns()[0])
                os_winner = 0 if r > 0.5 else 1 if r < -0.5 else None
                mismatches += int(os_winner != eas_state.winner())
                break
            legal = [int(a) for a in os_state.legal_actions()]
            same = (int(os_state.current_player()) == eas_state.player()
                    and legal == [i for i, ok in enumerate(eas_state.action_mask()) if ok]
                    and np.array_equal(np.asarray(os_state.information_state_tensor(), dtype = np.float32),
                                       np.asarray(eas_state.compute_openspiel_infostate(), dtype = np.float32)))
            if not same:
                mismatches += 1
                break
            action = int(rng.choice(legal))
            os_state.apply_action(action)
            eas_state.next(action)
    return mismatches


class ExactEvaluator:
    """一个游戏的遍历器加上每名玩家全部信息集的网络输入（布尔信息状态张量与合法动作掩码）。"""

    def __init__(self, env: str):
        self.env = env
        self.traverser = EAS_GAMES[env][0]()
        uniform = self.traverser.construct_uniform_strategies()
        self.legal: tuple[np.ndarray, np.ndarray] = (uniform[0] > 0.0, uniform[1] > 0.0)
        self.infostates: tuple[np.ndarray, np.ndarray] = (self.traverser.compute_openspiel_infostates(0),
                                                           self.traverser.compute_openspiel_infostates(1))
        self.span = float(load_game(env).max_utility() - load_game(env).min_utility())

    def net_strategy(self, net: SmallMLP, player: int) -> np.ndarray:
        """与 TorchActor.probs 相同：float32 softmax，再在合法动作上归一化；非法动作的概率为精确的 0。"""
        legal = self.legal[player]
        infostates = self.infostates[player]
        out = np.zeros(legal.shape, dtype = np.float64)
        net.eval()
        with torch.inference_mode():
            for start in range(0, len(out), FORWARD_BATCH):
                end = min(start + FORWARD_BATCH, len(out))
                obs = torch.from_numpy(infostates[start:end].astype(np.float32))
                logits, _value = net(obs, torch.from_numpy(legal[start:end]))
                out[start:end] = torch.softmax(logits, dim = -1).numpy()
        out *= legal
        out /= out.sum(axis = 1, keepdims = True)
        return out

    def strategies(self, nets: list[SmallMLP], weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """单网络直接展开；整局混合（PSRO）用序列式加权平均，与"每局按权重抽一个成员"等价。"""
        if len(nets) == 1:
            return self.net_strategy(nets[0], 0), self.net_strategy(nets[0], 1)
        out = []
        for player in (0, 1):
            averager = self.traverser.new_averager(player, eas.AveragingStrategy.CUSTOM)
            for net, weight in zip(nets, weights):
                if weight > 0.0:
                    averager.push(self.net_strategy(net, player), float(weight))
            out.append(averager.running_avg())
        return out[0], out[1]

    def _check(self, player: int, strategy: np.ndarray) -> None:
        # exp-a-spiel 遇到非法策略会直接 abort 整个进程，所以先在这里报错
        legal = self.legal[player]
        if strategy.shape != legal.shape or not np.all(np.isfinite(strategy)) or np.any(strategy < 0.0):
            raise ValueError(f"player {player}: strategy has the wrong shape or invalid entries")
        if np.any(strategy[~legal] != 0.0) or np.max(np.abs(strategy.sum(axis = 1) - 1.0)) > ROW_SUM_TOL:
            raise ValueError(f"player {player}: strategy puts mass on illegal actions or rows do not sum to 1")

    def evaluate(self, strategies: tuple[np.ndarray, np.ndarray]) -> dict[str, float]:
        """exploitability 与 OpenSpiel 口径相同：两名玩家被最佳响应时对手增益的平均（NashConv / 2）。"""
        for player, strategy in enumerate(strategies):
            self._check(player, strategy)
        result = self.traverser.ev_and_exploitability(strategies[0], strategies[1])
        expl = 0.5 * (float(result.expl[0]) + float(result.expl[1]))
        return dict(exploitability = expl, norm_exploitability = expl / self.span, ev0 = float(result.ev0),
                    expl_vs_player0 = float(result.expl[0]), expl_vs_player1 = float(result.expl[1]))
