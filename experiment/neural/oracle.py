"""exploiter（oracle）：对冻结对手训练 PPO 最佳响应，或全树精确 BR。所有方法共用同一协议。"""
from __future__ import annotations

from typing import Any

import numpy as np

from experiment.neural.config import RunConfig
from experiment.neural.envs import Scorer, obs_and_action_size
from experiment.neural.evals import exact_br_actor
from experiment.neural.mlp import SmallMLP
from experiment.neural.play import play_episode
from experiment.neural.policies import TorchActor, frozen_actor
from experiment.neural.ppo import PPOHyper, PPOLearner
from experiment.neural.schedule import exploiter_rollouts


def ppo_hyper(cfg: RunConfig, lr_scale: float = 1.0) -> PPOHyper:
    return PPOHyper(lr = cfg.lr * lr_scale, clip_eps = cfg.clip_eps, gae_lambda = cfg.gae_lambda, gamma = cfg.gamma,
                    epochs = cfg.epochs, batch_size = cfg.batch_size, vf_coef = cfg.vf_coef, max_grad_norm = cfg.max_grad_norm)


def train_ppo_br(
    cfg: RunConfig,
    game: Any,
    opponent: Any,
    rng: np.random.Generator,
    scorer: Scorer,
    warm_from: SmallMLP | None = None,
    rollouts: int | None = None,
) -> TorchActor:
    """
    对冻结对手训练 PPO 最佳响应：对手固定，所以不用重要性比，也不加近端项。
    - 初始化：warm_from 给出时继承它的主干与 value 头、重置策略头（整网克隆会带着低熵 logits 训不动）；否则从零开始。
    - 熵：继承时从 oracle_ent_warm、从零开始时从 oracle_ent_cold 起，在计划轮数内线性退到 oracle_ent_final（缺省为 ent_final）。
    """
    obs_dim, n_actions = obs_and_action_size(game)
    net = SmallMLP(obs_dim, n_actions, cfg.hidden)
    if warm_from is not None:
        net.load_state_dict(warm_from.state_dict())
        net.pi.reset_parameters()
    ent_start = cfg.oracle_ent_warm if warm_from is not None else cfg.oracle_ent_cold
    ent_end = cfg.oracle_ent_final if cfg.oracle_ent_final is not None else cfg.ent_final
    exploiter = PPOLearner(net, ppo_hyper(cfg, lr_scale = 2.0), cfg.device, ent_coef = ent_start)
    n_rollouts = rollouts if rollouts is not None else exploiter_rollouts(cfg)
    for r in range(n_rollouts):
        frac = r / max(n_rollouts - 1, 1)
        exploiter.ent_coef = (1.0 - frac) * ent_start + frac * ent_end
        steps = []
        for _ in range(cfg.games_per_rollout):
            episode, _score = play_episode(game, exploiter, opponent, rng, iw = 1.0, scorer = scorer, mirror = False)
            steps.extend(episode)
        exploiter.update(steps)
    return frozen_actor(exploiter.clone_net(), cfg.device)


def make_exploiter(cfg: RunConfig, game: Any, latest_net: SmallMLP, rng: np.random.Generator, scorer: Scorer) -> Any:
    latest = frozen_actor(latest_net, cfg.device)
    if cfg.oracle == "exact_br":
        return exact_br_actor(game, latest)
    return train_ppo_br(cfg, game, latest, rng, scorer, warm_from = latest_net)


def hoeffding_lower(mean: float, n: int, alpha: float = 0.05) -> float:
    """[0, 1] 得分均值的单侧 Hoeffding 下界。"""
    return mean - float(np.sqrt(np.log(1.0 / alpha) / (2.0 * max(n, 1))))
