"""NFSP 对照：OpenSpiel 的 PyTorch 实现，部署对象是平均策略网络。每局双方各自学习，一局记一次预算。"""
from __future__ import annotations

import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from open_spiel.python import rl_environment
from open_spiel.python.pytorch import nfsp as nfsp_mod

from experiment.common.io_util import append_jsonl, log, write_json
from experiment.neural.budget import count_game, current_ledger, measured_run
from experiment.neural.config import RunConfig
from experiment.neural.envs import load_game, score_span
from experiment.neural.evals import exploitability_of_table, tabular_from_callable
from experiment.neural.schedule import EvalSchedule


class _AveragePolicy:
    """把各座位 agent 的平均策略包装成联合策略，供全树评估。"""

    def __init__(self, env: Any, agents: list[Any]):
        self.agents = agents
        n = env.game.num_players()
        self._obs: dict[str, Any] = {"info_state": [None] * n, "legal_actions": [None] * n}

    def action_probabilities(self, state: Any, player_id: Any = None) -> dict[int, float]:
        player = int(state.current_player()) if player_id is None else int(player_id)
        legal = state.legal_actions(player)
        self._obs["current_player"] = player
        self._obs["info_state"][player] = state.information_state_tensor(player)
        self._obs["legal_actions"][player] = legal
        step = rl_environment.TimeStep(observations = self._obs, rewards = None, discounts = None, step_type = None)
        with self.agents[player].temp_mode_as(nfsp_mod.MODE.AVERAGE_POLICY):
            probs = self.agents[player].step(step, is_evaluation = True).probs
        return {int(a): float(probs[a]) for a in legal}


def _exploitability(env: Any, agents: list[Any]) -> float:
    table = tabular_from_callable(env.game, _AveragePolicy(env, agents).action_probabilities)
    return exploitability_of_table(env.game, table)


@measured_run
def run_nfsp(cfg: RunConfig, out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents = True, exist_ok = True)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    game = load_game(cfg.env)
    span = score_span(game)
    env = rl_environment.Environment(game)
    info_size: Any = int(env.observation_spec()["info_state"][0])
    agents = [
        nfsp_mod.NFSP(
            player, info_size, int(game.num_distinct_actions()), hidden_layers_sizes = [cfg.hidden, cfg.hidden],
            reservoir_buffer_capacity = cfg.nfsp_reservoir_capacity, anticipatory_param = cfg.nfsp_anticipatory,
            batch_size = cfg.nfsp_batch_size, rl_learning_rate = cfg.nfsp_lr, sl_learning_rate = cfg.nfsp_lr,
            min_buffer_size_to_learn = cfg.nfsp_min_buffer, learn_every = cfg.nfsp_learn_every,
            optimizer_str = cfg.nfsp_optimizer, seed = cfg.seed * 1009 + player * 17, device = cfg.device,
            # 以下经 NFSP 原样转给内部的 DQN 最佳响应
            replay_buffer_capacity = cfg.nfsp_replay_capacity, update_target_network_every = cfg.nfsp_target_update_every,
            discount_factor = cfg.gamma, epsilon_start = cfg.nfsp_epsilon_start, epsilon_end = cfg.nfsp_epsilon_end,
            epsilon_decay_duration = cfg.nfsp_epsilon_decay_steps, epsilon_decay_schedule_str = cfg.nfsp_epsilon_schedule,
        )
        for player in range(game.num_players())
    ]
    ledger = current_ledger()
    schedule = EvalSchedule(cfg)
    tail_every = cfg.tail_eval_rollouts * cfg.games_per_rollout
    log_path = out_dir / "rollouts.jsonl"
    log_path.unlink(missing_ok = True)
    t0 = time.time()
    log(f"start NFSP {cfg.env} {cfg.label} seed={cfg.seed} budget={cfg.budget}")

    def evaluate() -> None:
        expl = _exploitability(env, agents)
        append_jsonl(log_path, dict(games = ledger.algorithmic_total, total_games = ledger.total, evaluation = True,
                                    exploitability = expl, norm_exploitability = expl / span))

    evaluate()
    while ledger.algorithmic_total < cfg.budget:
        before = ledger.algorithmic_total
        time_step = env.reset()
        while not time_step.last():
            player = time_step.observations["current_player"]
            time_step = env.step([agents[player].step(time_step).action])
        for agent in agents:
            agent.step(time_step)
        count_game()
        after = ledger.algorithmic_total
        if schedule.crossed(before, after) or (schedule.in_tail(after) and after % tail_every == 0):
            evaluate()
    summary = dict(games = ledger.algorithmic_total, total_games = ledger.total, elapsed_s = time.time() - t0)
    write_json(out_dir / "run_summary.json", summary)
    return summary
