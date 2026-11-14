"""一次神经训练运行的完整配置；所有字段都可 JSON 序列化，格子 ID 的哈希由它计算。"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
from typing import Any

ALGORITHMS = ("ppo", "psro", "nfsp")
PROXIMAL_FORMS = ("none", "shaping", "kl")
RULES = ("none", "coda_r", "moving_anchor", "pfsp", "uniform", "hedge", "fixed")
ORACLES = ("none", "ppo", "exact_br")
# 已有实验完成之后才加入的字段：取默认值时不写进 as_dict，已有格子的配置哈希（格子 ID）因此不变
LATE_FIELDS = ("lr_final_frac", "oracle_ent_final")


@dataclass(frozen = True)
class RunConfig:
    env: str
    label: str
    seed: int
    # 算法性对局预算 B：learner、exploiter 训练、入池评估、全池扫描与 PSRO 元矩阵对局都计入
    budget: int
    algorithm: str = "ppo"

    # -------- 网络与 PPO --------
    hidden: int = 128
    lr: float = 3e-4
    # learner 的学习率在整个预算上从 lr 线性降到 lr * lr_final_frac；1 = 恒定
    lr_final_frac: float = 1.0
    clip_eps: float = 0.2
    gae_lambda: float = 0.95
    gamma: float = 1.0
    epochs: int = 4
    batch_size: int = 256
    vf_coef: float = 0.5
    max_grad_norm: float = 1.0
    games_per_rollout: int = 512
    device: str = "cpu"

    # -------- 熵日程：g <= anneal_frac * B 时从 ent_start 线性降到 ent_final，之后保持 --------
    ent_start: float = 0.03
    ent_final: float = 0.01
    ent_anneal_frac: float = 0.8

    # -------- 原侧近端项 --------
    proximal: str = "shaping"
    eta: float = 0.1
    # 策略锚每 outer_rollouts 个 rollout 结账一次
    outer_rollouts: int = 32
    # rollout = 镜像对手为当前参数（隐式）；outer = 镜像对手为外环开始时的冻结快照（消融）
    mirror_refresh: str = "rollout"

    # -------- 对手池 --------
    rule: str = "none"
    coda_c: float = 6.0
    pfsp_exponent: float = 2.0
    # PFSP / 均匀 FSP / Hedge 的固定自博弈份额
    self_share: float = 0.5
    hedge_beta: float = 1.0
    pool_capacity: int = 8
    protect_recent: int = 2
    snapshots_keep: int = 2
    sampling_base: str = "dual"
    detection: str = "probe"
    sweep_games: int = 64
    eps: float = 0.05
    mirror_floor: float = 0.1
    member_floor_games: int = 8
    probe_share: float = 0.25
    estimator_n0: int = 64
    reset_k: float = 2.5
    # ema = 在线估计；exact = 每个 rollout 用全树精确败率覆盖（只作 e_k = 0 的上限）
    loss_estimator: str = "ema"
    freeze_field: bool = False

    # -------- oracle（exploiter）--------
    oracle: str = "none"
    # oracle 的触发周期由 PPO exploiter 的单次成本与该目标预算占比反推；精确 BR 沿用同一周期
    oracle_share: float = 0.12
    oracle_rollouts: int = 20
    oracle_match_games: int = 256
    oracle_ent_warm: float = 0.08
    oracle_ent_cold: float = 0.02
    # exploiter 的熵终值；None = 沿用 learner 的 ent_final
    oracle_ent_final: float | None = None
    # 最后这么多个 rollout 不再插入 oracle
    oracle_cooldown_rollouts: int = 48

    # -------- PSRO / NFSP --------
    meta_games: int = 256
    nfsp_anticipatory: float = 0.1
    # 最佳响应（DQN）与平均策略网络共用的学习率
    nfsp_lr: float = 0.01
    # NFSP 内部 DQN 最佳响应的全部参数都在这里显式给出，不依赖 OpenSpiel 的默认值
    nfsp_optimizer: str = "sgd"
    nfsp_batch_size: int = 128
    # 每个座位每这么多个决策步各学习一次（平均策略与 DQN 相同）
    nfsp_learn_every: int = 64
    nfsp_min_buffer: int = 1000
    nfsp_replay_capacity: int = 200_000
    nfsp_reservoir_capacity: int = 2 ** 17
    nfsp_target_update_every: int = 19_200
    # 探索率按座位的决策步从 start 衰减到 end，之后保持
    nfsp_epsilon_start: float = 0.06
    nfsp_epsilon_end: float = 0.001
    nfsp_epsilon_decay_steps: int = 20_000_000
    nfsp_epsilon_schedule: str = "linear"

    # -------- 评估 --------
    # 末段之前按对数间隔评估的点数；末段 [anneal_frac * B, B] 内每 tail_eval_rollouts 个 rollout 再评一次
    eval_points: int = 12
    tail_eval_rollouts: int = 8
    evaluate_mixtures: bool = False
    # 每这么多个 rollout 记录一次成员的精确败率（诊断，0 = 只在评估点记录，-1 = 不记录）
    exact_loss_every: int = 0
    save_checkpoints: bool = False

    def __post_init__(self) -> None:
        checks = [
            (self.algorithm, ALGORITHMS), (self.proximal, PROXIMAL_FORMS), (self.rule, RULES), (self.oracle, ORACLES),
            (self.mirror_refresh, ("rollout", "outer")), (self.loss_estimator, ("ema", "exact")),
            (self.nfsp_optimizer, ("sgd", "adam")), (self.nfsp_epsilon_schedule, ("linear", "exp")),
        ]
        for value, allowed in checks:
            if value not in allowed:
                raise ValueError(f"{value!r} is not one of {allowed}")
        # PSRO 用 oracle 字段表示 BR 的种类；PPO 方法的 oracle 必须有池可插
        if self.algorithm == "ppo" and self.rule == "none" and self.oracle != "none":
            raise ValueError("an oracle needs a pool to insert into")

    @property
    def uses_pool(self) -> bool:
        return self.algorithm == "ppo" and self.rule != "none"

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        defaults = {f.name: f.default for f in fields(self)}
        for name in LATE_FIELDS:
            if payload[name] == defaults[name]:
                del payload[name]
        return payload

    def shrunk(self) -> RunConfig:
        """冒烟版本：只验证路径能走通，数字没有意义；oracle 占比调到 1 使每个外环都触发 oracle。"""
        return replace(
            self, budget = 128, hidden = 8, games_per_rollout = 16, epochs = 1, batch_size = 16, outer_rollouts = 2,
            oracle_rollouts = 1, oracle_match_games = 8, oracle_cooldown_rollouts = 1, oracle_share = 1.0, meta_games = 8,
            sweep_games = 4, eval_points = 2, tail_eval_rollouts = 2, member_floor_games = 1,
        )


def config_from_dict(payload: dict[str, Any]) -> RunConfig:
    return RunConfig(**payload)
