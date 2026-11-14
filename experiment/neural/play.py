"""生成训练轨迹与评估对局；每完成一局都记入预算账本。"""
from __future__ import annotations

from typing import Any

import numpy as np

from experiment.neural.budget import count_game
from experiment.neural.envs import Scorer, information_tensor, legal_mask
from experiment.neural.ppo import PPOLearner, Step

MAX_ACTIONS = 2000


def _apply_chance(state: Any, rng: np.random.Generator) -> None:
    actions, probs = zip(*state.chance_outcomes())
    state.apply_action(int(actions[int(rng.choice(len(actions), p = np.asarray(probs, dtype = np.float64)))]))


def play_episode(
    game: Any,
    learner: PPOLearner,
    opponent: Any,
    rng: np.random.Generator,
    iw: float,
    scorer: Scorer,
    mirror: bool,
) -> tuple[list[Step], float]:
    """
    learner 随机坐一个座位打一局，只保留它那一侧的轨迹，返回 (轨迹, learner 得分)。
    mirror = True 时对手也是 latest（opponent 为 None 表示直接用 learner 当前参数；
    否则 opponent 是周期冻结的镜像快照），对手实际决策处的近端 log-ratio 挂到 learner 的前一个决策步。
    外部冻结成员没有移动的策略锚，对手侧近端项为 0。
    """
    if hasattr(opponent, "reset_episode"):
        opponent.reset_episode()
    state = game.new_initial_state()
    n_actions = int(game.num_distinct_actions())
    learner_seat = int(rng.integers(2))
    records: list[dict[str, Any]] = []
    # learner 尚未出手时对手的 log-ratio 先暂存，并入它的第一个决策步
    carried_ratio = 0.0

    def learner_move() -> int:
        nonlocal carried_ratio
        obs = np.asarray(information_tensor(state, learner_seat), dtype = np.float32)
        mask = np.asarray(legal_mask(state, learner_seat, n_actions), dtype = np.bool_)
        action, logp, value = learner.act(obs, mask)
        records.append(dict(obs = obs, mask = mask, action = action, logp = logp, value = value,
                            opponent_log_ratio = carried_ratio))
        carried_ratio = 0.0
        return action

    def opponent_move(seat: int) -> int:
        nonlocal carried_ratio
        if not mirror:
            return int(opponent.act(state, seat, n_actions, rng))
        obs = np.asarray(information_tensor(state, seat), dtype = np.float32)
        mask = np.asarray(legal_mask(state, seat, n_actions), dtype = np.bool_)
        if opponent is None:
            action, logp, _value = learner.act(obs, mask)
        else:
            dist = opponent.probs(state, seat, n_actions)
            actions = list(dist)
            p = np.array([dist[a] for a in actions], dtype = np.float64)
            action = int(actions[int(rng.choice(len(actions), p = p / p.sum()))])
            logp = float(np.log(max(dist[action], 1e-12)))
        if learner.anchor is not None:
            ratio = logp - learner.anchor_logp(obs, mask, action)
            if records:
                records[-1]["opponent_log_ratio"] += ratio
            else:
                carried_ratio += ratio
        return action

    for _ in range(MAX_ACTIONS):
        if state.is_terminal():
            break
        if state.is_chance_node():
            _apply_chance(state, rng)
            continue
        player = int(state.current_player())
        state.apply_action(learner_move() if player == learner_seat else opponent_move(player))
    if not state.is_terminal():
        raise RuntimeError(f"episode exceeded {MAX_ACTIONS} actions")
    count_game()
    score = scorer(list(state.returns()), learner_seat)
    steps = [Step(**rec, reward = score if i == len(records) - 1 else 0.0, done = i == len(records) - 1, iw = float(iw))
             for i, rec in enumerate(records)]
    return steps, score


def play_match(game: Any, actor_a: Any, actor_b: Any, rng: np.random.Generator, episodes: int, scorer: Scorer) -> float:
    """actor_a 对 actor_b 的平均得分，座位轮流交换。"""
    n_actions = int(game.num_distinct_actions())
    total = 0.0
    for ep in range(episodes):
        for actor in (actor_a, actor_b):
            if hasattr(actor, "reset_episode"):
                actor.reset_episode()
        state = game.new_initial_state()
        seat_a = ep % 2
        for _ in range(MAX_ACTIONS):
            if state.is_terminal():
                break
            if state.is_chance_node():
                _apply_chance(state, rng)
                continue
            player = int(state.current_player())
            actor = actor_a if player == seat_a else actor_b
            state.apply_action(int(actor.act(state, player, n_actions, rng)))
        if not state.is_terminal():
            raise RuntimeError(f"match exceeded {MAX_ACTIONS} actions")
        count_game()
        total += scorer(list(state.returns()), seat_a)
    return total / max(episodes, 1)
