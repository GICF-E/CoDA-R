"""PPO 系方法的唯一训练循环：自博弈、R-NaD-PPO、MMD-PPO，以及所有带对手池的方法（CoDA-R、PFSP、均匀 FSP 等）。

三个时钟：
- rollout：按池的采样分布抽对手打完一批对局，PPO 更新一次，对偶侧把本批败率并入估计。
- 外环（每 outer_rollouts 个 rollout）：策略锚 <- latest，可选地加入快照、做全池扫描、触发 oracle。
- 评估：按 EvalSchedule 做全树 exploitability；不可展开的环境只存 checkpoint，留给近似评估。
训练按算法性对局恰好停在预算 B 上；全树评估不消耗环境对局，只计墙钟。
"""
from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from experiment.common.io_util import append_jsonl, log, write_json
from experiment.core.estimator import LossEstimator
from experiment.core.pool import OpponentPool, SamplingConfig
from experiment.core.pool_rules import (
    CoDARRule,
    FixedRule,
    HedgeRule,
    MovingAnchorRule,
    PFSPRule,
    PoolRule,
    UniformRule,
)
from experiment.neural.budget import current_ledger, game_kind, measured_run
from experiment.neural.config import RunConfig
from experiment.neural.envs import FULL_TREE_INTRACTABLE, load_game, make_scorer, obs_and_action_size, score_span
from experiment.neural.evals import (
    EXACT_LOSS_ENVS,
    TableCache,
    exact_loss,
    exploitability_of_table,
    full_exploitability,
    to_tabular,
)
from experiment.neural.mixtures import OuterMixtures
from experiment.neural.mlp import SmallMLP
from experiment.neural.oracle import hoeffding_lower, make_exploiter, ppo_hyper
from experiment.neural.play import play_match
from experiment.neural.policies import TorchActor, frozen_actor
from experiment.neural.ppo import PPOLearner
from experiment.neural.rollout import collect_rollout
from experiment.neural.schedule import EvalSchedule, entropy_coef, learning_rate, oracle_cost, oracle_period


@dataclass
class TrainHooks:
    """实验在训练循环之外注入的对象；主实验的方法都不用。"""
    # 训练开始前放进池的外部成员 (ID, 策略)
    initial_members: list[tuple[str, Any]] = field(default_factory = list)
    # learner 的初始参数
    initial_state: dict[str, torch.Tensor] | None = None
    # 在这个 rollout 开始前把 learner 与策略锚重置回 initial_state（模拟退化）
    reset_at_rollout: int | None = None
    # 固定比例规则的成员份额
    fixed_shares: Mapping[str, float] = field(default_factory = dict)
    # 每个 rollout 更新之后对 latest 做的诊断（只读策略，不消耗环境对局），参数为 (rollout 序号, latest)，返回值并入该 rollout 的日志
    probe: Callable[[int, TorchActor], dict[str, Any]] | None = None


def build_rule(cfg: RunConfig, hooks: TrainHooks) -> PoolRule:
    if cfg.rule == "coda_r":
        return CoDARRule(cfg.coda_c)
    if cfg.rule == "moving_anchor":
        return MovingAnchorRule(cfg.coda_c)
    if cfg.rule == "pfsp":
        return PFSPRule(cfg.pfsp_exponent, cfg.self_share)
    if cfg.rule == "uniform":
        return UniformRule(cfg.self_share)
    if cfg.rule == "hedge":
        return HedgeRule(cfg.hedge_beta, cfg.self_share)
    if cfg.rule == "fixed":
        return FixedRule(hooks.fixed_shares)
    raise ValueError(f"rule {cfg.rule!r} has no pool")


def build_pool(cfg: RunConfig, hooks: TrainHooks) -> OpponentPool:
    sampling = SamplingConfig(eps = cfg.eps, mirror_floor = cfg.mirror_floor, detection = cfg.detection,
                              member_floor_games = cfg.member_floor_games, probe_share = cfg.probe_share,
                              base = cfg.sampling_base)
    return OpponentPool(build_rule(cfg, hooks), LossEstimator(cfg.estimator_n0, cfg.reset_k), sampling,
                        capacity = cfg.pool_capacity, protect_recent = cfg.protect_recent, freeze_field = cfg.freeze_field)


def _seed_everything(seed: int) -> np.random.Generator:
    np.random.seed(seed)
    torch.manual_seed(seed)
    return np.random.default_rng(seed)


class TrainingRun:
    def __init__(self, cfg: RunConfig, out_dir: Path, hooks: TrainHooks):
        self.cfg = cfg
        self.out_dir = out_dir
        self.hooks = hooks
        self.rng = _seed_everything(cfg.seed)
        self.game = load_game(cfg.env)
        self.scorer = make_scorer(self.game)
        self.span = score_span(self.game)
        self.exact_env = cfg.env not in FULL_TREE_INTRACTABLE
        obs_dim, n_actions = obs_and_action_size(self.game)
        kl_coef = cfg.eta if cfg.proximal == "kl" else 0.0
        self.learner = PPOLearner(SmallMLP(obs_dim, n_actions, cfg.hidden), ppo_hyper(cfg), cfg.device,
                                  ent_coef = cfg.ent_start, kl_coef = kl_coef)
        if hooks.initial_state is not None:
            self.learner.load_state(hooks.initial_state)
        if cfg.proximal != "none":
            self.learner.snapshot_anchor()
        self.ledger = current_ledger()
        self.schedule = EvalSchedule(cfg)
        self.oracle_period = oracle_period(cfg)
        self.cache = TableCache(self.game)
        self.pool = build_pool(cfg, hooks) if cfg.uses_pool else None
        self.mixtures = OuterMixtures(self.game, self.cache) if cfg.evaluate_mixtures and self.exact_env else None
        self.mirror_snapshot: TorchActor | None = None
        self.learner_games = 0
        self.n_snapshots = 0
        self.n_oracles = 0
        self.mixture_pending = False
        self.log_path = out_dir / "rollouts.jsonl"
        self.ckpt_dir = out_dir / "ckpts"
        self.oracle_path = out_dir / "oracles.jsonl"
        for path in (self.log_path, self.oracle_path):
            path.unlink(missing_ok = True)
        if self.pool is not None:
            if cfg.snapshots_keep > 0:
                self._add_snapshot()
            for member_id, actor in hooks.initial_members:
                self.pool.add(member_id, actor, "external")

    # -------- 小工具 --------
    def _latest(self) -> TorchActor:
        return frozen_actor(self.learner.clone_net(), self.cfg.device)

    def _members(self) -> list[tuple[str, Any]]:
        if self.pool is None:
            return []
        return [(m.id, m.policy) for m in self.pool.members]

    def _exact_losses(self, latest_table: Any) -> dict[str, float]:
        return {member_id: exact_loss(self.game, latest_table, self.cache.get(member_id, actor))
                for member_id, actor in self._members()}

    def _wants_exact_losses(self, roll: int) -> bool:
        if self.pool is None or self.cfg.env not in EXACT_LOSS_ENVS:
            return False
        if self.cfg.loss_estimator == "exact":
            return True
        every = self.cfg.exact_loss_every
        return every > 0 and (roll + 1) % every == 0

    # -------- 评估 --------
    def _evaluate(self, roll: int, latest_table: Any | None, exact: dict[str, float] | None = None) -> dict[str, Any]:
        rec: dict[str, Any] = dict(evaluation = True)
        if self.cfg.save_checkpoints or not self.exact_env:
            self.ckpt_dir.mkdir(parents = True, exist_ok = True)
            torch.save(self.learner.net.state_dict(), self.ckpt_dir / f"games_{self.ledger.algorithmic_total:010d}.pt")
        if not self.exact_env:
            rec.update(exploitability = float("nan"), norm_exploitability = float("nan"))
            return rec
        if latest_table is None:
            latest_table = to_tabular(self.game, self._latest())
        expl = exploitability_of_table(self.game, latest_table)
        rec.update(exploitability = expl, norm_exploitability = expl / self.span)
        if self.pool is not None and self.cfg.exact_loss_every >= 0 and self.cfg.env in EXACT_LOSS_ENVS:
            rec["pool_losses_true"] = list((exact or self._exact_losses(latest_table)).values())
        self.mixture_pending = True
        return rec

    # -------- 外环事件 --------
    def _add_snapshot(self) -> None:
        assert self.pool is not None
        self.pool.add(f"snap_{self.n_snapshots:05d}", self._latest(), "snapshot")
        self.n_snapshots += 1
        snapshots = [m.id for m in self.pool.members if m.kind == "snapshot"]
        for old in snapshots[:max(len(snapshots) - self.cfg.snapshots_keep, 0)]:
            self.pool.remove(old)

    def _sweep(self) -> None:
        """全池扫描：对每个成员打固定局数，只更新败率估计，不进梯度。"""
        assert self.pool is not None
        latest = self._latest()
        with game_kind("sweep"):
            for member in list(self.pool.members):
                score = play_match(self.game, latest, member.policy, self.rng, self.cfg.sweep_games, self.scorer)
                self.pool.add_observations(member.id, self.cfg.sweep_games, 1.0 - score)

    def _oracle_due(self, outer_index: int) -> bool:
        if self.oracle_period <= 0 or outer_index % self.oracle_period != 0:
            return False
        cost = oracle_cost(self.cfg) if self.cfg.oracle == "ppo" else self.cfg.oracle_match_games
        reserve = self.cfg.oracle_cooldown_rollouts * self.cfg.games_per_rollout
        return self.ledger.algorithmic_total + cost + reserve <= self.cfg.budget

    def _insert_oracle(self, roll: int) -> None:
        assert self.pool is not None
        with game_kind("oracle_train"):
            exploiter = make_exploiter(self.cfg, self.game, self.learner.clone_net(), self.rng, self.scorer)
        latest = self._latest()
        n = self.cfg.oracle_match_games
        with game_kind("oracle_match"):
            loss = 1.0 - play_match(self.game, latest, exploiter, self.rng, n, self.scorer)
        # exploiter 的领先给出 latest 归一化 exploitability 的 95% 单侧下界
        lower = max(hoeffding_lower(loss, n) - 0.5, 0.0)
        member_id = f"oracle_{self.n_oracles:04d}"
        evicted = self.pool.add(member_id, exploiter, "oracle", initial_loss = loss)
        self.n_oracles += 1
        rec = dict(roll = roll, games = self.ledger.algorithmic_total, member = member_id, evicted = evicted,
                   match_loss = loss, lower_bound_norm = lower, lower_bound = lower * self.span)
        if self.exact_env:
            rec["exploitability"] = full_exploitability(self.game, latest)
        append_jsonl(self.oracle_path, rec)

    def _outer_boundary(self, roll: int, latest_table: Any | None) -> dict[str, Any]:
        """latest_table 是本 rollout 更新之后已经展开过的 latest（评估或精确败率时），没有则在这里展开。"""
        rec: dict[str, Any] = {}
        if self.mixtures is not None and self.pool is not None:
            # 先在结账、入池与淘汰之前捕获 (latest, w)
            table = latest_table if latest_table is not None else to_tabular(self.game, self._latest())
            self.mixtures.update(table, self._members(), self.pool.output().field[:-1].tolist())
            if self.mixture_pending:
                rec.update(self.mixtures.evaluate())
        self.mixture_pending = False
        if self.cfg.proximal != "none":
            self.learner.snapshot_anchor()
        if self.cfg.mirror_refresh == "outer":
            self.mirror_snapshot = self._latest()
        # 预算用完后不再有后续训练，入池、扫描与 oracle 都没有意义
        if self.pool is None or self.ledger.algorithmic_total >= self.cfg.budget:
            return rec
        self.pool.end_outer_step()
        sweep_cost = self.cfg.sweep_games * len(self.pool.members)
        if self.cfg.detection == "sweep" and self.ledger.algorithmic_total + sweep_cost < self.cfg.budget:
            self._sweep()
        if self.cfg.snapshots_keep > 0:
            self._add_snapshot()
        if self._oracle_due((roll + 1) // self.cfg.outer_rollouts):
            self._insert_oracle(roll)
        self.cache.retain(set(self.pool.ids))
        return rec

    # -------- 主循环 --------
    def _rollout(self, roll: int) -> dict[str, Any]:
        cfg = self.cfg
        games_before = self.ledger.algorithmic_total
        if self.hooks.reset_at_rollout == roll and self.hooks.initial_state is not None:
            self.learner.load_state(self.hooks.initial_state)
            if cfg.proximal != "none":
                self.learner.snapshot_anchor()
        self.learner.ent_coef = entropy_coef(cfg, games_before)
        lr = learning_rate(cfg, games_before)
        self.learner.set_lr(lr)
        games = min(cfg.games_per_rollout, cfg.budget - games_before)
        t0 = time.perf_counter()
        data = collect_rollout(self.game, self.learner, self.pool, self.mirror_snapshot, self.rng, self.scorer, games)
        if cfg.proximal == "shaping":
            self.learner.apply_shaping(data.steps, cfg.eta, data.field_mass)
        metrics = self.learner.update(data.steps, data.field_mass)
        self.learner_games += games
        # rollout_s 只含采样与 PPO 更新，不含评估与外环事件
        rec: dict[str, Any] = dict(
            roll = roll, games = self.ledger.algorithmic_total, total_games = self.ledger.total,
            learner_games = self.learner_games, ent_coef = self.learner.ent_coef, lr = lr, **metrics,
            rollout_s = time.perf_counter() - t0, winrate = float(np.mean(data.scores)),
            proximal_proxy = self.learner.proximal_proxy(data.steps, cfg.eta, data.field_mass)
            if cfg.proximal != "none" else 0.0,
        )
        if self.hooks.probe is not None:
            rec.update(self.hooks.probe(roll, self._latest()))
        due = self.schedule.due(games_before, self.ledger.algorithmic_total, roll)
        latest_table = None
        exact: dict[str, float] | None = None
        if self.pool is not None:
            if self._wants_exact_losses(roll):
                latest_table = to_tabular(self.game, self._latest())
                exact = self._exact_losses(latest_table)
            rec.update(self.pool.end_rollout(exact if cfg.loss_estimator == "exact" else None))
            if exact is not None:
                rec["pool_losses_true"] = list(exact.values())
            if exact is not None or due:
                rec.update(self.pool.snapshot())
        if due:
            if latest_table is None and self.exact_env:
                latest_table = to_tabular(self.game, self._latest())
            rec.update(self._evaluate(roll, latest_table, exact))
        if (roll + 1) % cfg.outer_rollouts == 0 or self.ledger.algorithmic_total >= cfg.budget:
            rec.update(self._outer_boundary(roll, latest_table))
        return rec

    def run(self) -> dict[str, Any]:
        cfg = self.cfg
        t0 = time.time()
        log(f"start {cfg.env} {cfg.label} seed={cfg.seed} budget={cfg.budget}")
        append_jsonl(self.log_path, dict(roll = -1, games = 0, total_games = 0, learner_games = 0,
                                         **self._evaluate(-1, None)))
        roll = 0
        while self.ledger.algorithmic_total < cfg.budget:
            rec = self._rollout(roll)
            append_jsonl(self.log_path, rec)
            if rec.get("evaluation"):
                log(f"{cfg.env} {cfg.label} s{cfg.seed} games={rec['games']}/{cfg.budget} "
                    f"expl={rec.get('exploitability', float('nan')):.4g}")
            roll += 1
        torch.save(self.learner.net.state_dict(), self.out_dir / "final.pt")
        summary = dict(rollouts = roll, games = self.ledger.algorithmic_total, total_games = self.ledger.total,
                       learner_games = self.learner_games, breakdown = dict(self.ledger.counts),
                       n_oracles = self.n_oracles, oracle_period_outer = self.oracle_period,
                       elapsed_s = time.time() - t0)
        write_json(self.out_dir / "run_summary.json", summary)
        return summary


@measured_run
def run_training(cfg: RunConfig, out_dir: Path, hooks: TrainHooks | None = None) -> dict[str, Any]:
    out_dir.mkdir(parents = True, exist_ok = True)
    return TrainingRun(cfg, out_dir, hooks or TrainHooks()).run()
