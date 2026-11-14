"""机制实验。

M1 vuln-repair：R-NaD-PPO 训练出 base，取它的全树精确 BR 作攻击者 A（漏洞一定存在），再放 3 个随机网络作诱饵。
  各池规则从同一个 base 出发训练；压制 T 个 rollout 后把 learner 重置回 base（模拟退化），看哪种规则能及时把权重还给 A。
  T 取多档：有状态的规则（Hedge）在压制期间积累的历史越长，重新加权越慢（sleeper）；无状态规则不受 T 影响。
M2 open-pool：池里预先放入脚本 bot、CFR+ 早期策略、小网络旧 agent、随机网络与重复成员，池规模 N in {4, 16, 64}，
  看各规则把多少训练权重浪费在 latest 已经打赢的成员上，以及对最终 exploitability 的影响。
M3 hidden-hole：CFR+ 的近似纳什表格策略蒸馏进网络，在对手到达概率受控的信息集上换成最差动作，形成隐藏漏洞；
  自博弈要等对手（也是 latest）走到漏洞才能修复，池里的精确 BR 会主动走过去。
M4 forgetting：同一个 CFR+ 底座（不埋漏洞）作为起点，各方法按主协议（含 oracle）继续训练，熵全程固定在调参终值；
  看 exploitability 能否守住、漂移集中在哪个对手到达概率档，以及 latest 会不会输给自己的历史版本。
四个实验都先跑 prep 格子（底座、攻击者、外部成员），再跑依赖它们的分支格子；M1-M3 的分支沿用主协议，只关掉 oracle。
"""
from __future__ import annotations

import copy
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from open_spiel.python.algorithms import cfr

from experiment.common.io_util import load_json, write_json
from experiment.common.runner import CellSpec, RunOptions, prepare_output, require_complete, run_cells
from experiment.common.stats import bootstrap_iqm, history_forgetting
from experiment.experiments.cells import train_spec
from experiment.experiments.methods import build_config, title
from experiment.experiments.protocol import ABLATION_BUDGET, ABLATION_GAMES, OPEN_POOL_BUDGET, SMALL_SEEDS, SMOKE_SEEDS
from experiment.experiments.report import CellResult, group, load_results, write_grid_report
from experiment.experiments.tuning import load_selected
from experiment.neural.config import RunConfig, config_from_dict
from experiment.neural.envs import load_game, obs_and_action_size
from experiment.neural.evals import exact_br_actor, exact_loss, full_exploitability, tabular_actor_from_policy, to_tabular
from experiment.neural.holes import HoleProbe, Table, damaged, distill, infoset_profiles, select_holes
from experiment.neural.mlp import SmallMLP
from experiment.neural.policies import PreferenceBot, TabularActor, UniformActor, frozen_actor
from experiment.neural.retention import RetentionProbe
from experiment.neural.train import TrainHooks, run_training

# -------- 共用 --------
NO_POOL_LABEL = "R-NaD-PPO (no pool)"
LABELS = {"hedge": "Hedge", "moving_anchor": "Moving dual anchor", "none": NO_POOL_LABEL}


def _save_table(path: Path, actor: TabularActor) -> None:
    write_json(path, {key: {str(a): p for a, p in dist.items()} for key, dist in actor.table.items()})


def _load_table(path: Path) -> TabularActor:
    return TabularActor({key: {int(a): float(p) for a, p in dist.items()} for key, dist in load_json(path).items()})


def _random_net_actor(game: Any, hidden: int, seed: int) -> Any:
    obs_dim, n_actions = obs_and_action_size(game)
    torch.manual_seed(seed)
    return frozen_actor(SmallMLP(obs_dim, n_actions, hidden), "cpu")


def _load_net_actor(game: Any, hidden: int, path: Path) -> Any:
    obs_dim, n_actions = obs_and_action_size(game)
    net = SmallMLP(obs_dim, n_actions, hidden)
    net.load_state_dict(torch.load(path, map_location = "cpu"))
    return frozen_actor(net, "cpu")


def _cfr_table(game: Any, iterations: int) -> Table:
    """CFR+ 平均策略的表格：M3 与 M4 共用的近似纳什底座。"""
    solver = cfr.CFRPlusSolver(game)
    for _ in range(iterations):
        solver.evaluate_and_update_policy()
    return tabular_actor_from_policy(game, solver.average_policy()).table


def _distilled(game: Any, table: Table, seed: int, hidden: int, steps: int) -> tuple[SmallMLP, dict[str, float]]:
    """与 learner 同宽的网络，蒸馏到表格策略；seed 固定初始化。"""
    obs_dim, n_actions = obs_and_action_size(game)
    torch.manual_seed(seed)
    net = SmallMLP(obs_dim, n_actions, hidden)
    return net, distill(net, game, table, steps)


def _rule_config(rule: str, env: str, seed: int, budget: int, selected: dict[str, Any], label: str, **extra: Any) -> RunConfig:
    """调参选出的方法直接用自己的配置；移动锚沿用 CoDA-R 的配置只换规则；Hedge、固定比例与无池对照沿用 R-NaD-PPO 的 eta 与熵终值。"""
    if rule in ("coda_r", "pfsp", "ufsp"):
        return build_config(rule, env, seed, budget, selected[rule], label = label, **extra)
    if rule == "moving_anchor":
        return build_config("coda_r", env, seed, budget, selected["coda_r"], label = label, **{**extra, "rule": "moving_anchor"})
    rnad = selected["rnad"]
    common = dict(eta = rnad["eta"], ent_final = rnad["ent_final"], proximal = "shaping", **extra)
    if rule == "hedge":
        return RunConfig(env = env, label = label, seed = seed, budget = budget, rule = "hedge", **common)
    if rule == "fixed":
        return RunConfig(env = env, label = label, seed = seed, budget = budget, rule = "fixed", **common)
    # 无池对照仍挂一个份额全为 0 的池，只为记录 latest 对攻击者的精确败率；eps = 0 保证只抽自博弈
    return RunConfig(env = env, label = label, seed = seed, budget = budget, rule = "fixed",
                     **{**common, "eps": 0.0, "detection": "none"})


def _first_stable(values: list[float], start: int, threshold: float, run: int = 3) -> float:
    for i in range(start, len(values) - run + 1):
        if all(v <= threshold for v in values[i:i + run]):
            return float(i - start)
    return float("nan")


def _member_series(r: CellResult, member: str) -> tuple[list[float], list[float]]:
    """某个成员逐 rollout 的真实败率与训练系数（只取记录了精确败率的 rollout）。"""
    rows = [h for h in r.history if h.get("roll", -1) >= 0 and h.get("pool_losses_true") and member in h.get("pool_ids", [])]
    return ([h["pool_losses_true"][h["pool_ids"].index(member)] for h in rows],
            [h["pool_field"][h["pool_ids"].index(member)] for h in rows])


def field_share(r: CellResult, idle: Callable[[float], bool]) -> list[float]:
    """每个记录点上，真实败率满足 idle 的成员持有的训练场系数占整个训练场质量（含自博弈份额）的比例。"""
    shares = []
    for h in r.history:
        truth, field = h.get("pool_losses_true"), h.get("pool_field")
        if truth and field and "field_mass" in h:
            shares.append(float(sum(w for loss, w in zip(truth, field) if idle(loss))) / float(h["field_mass"]))
    return shares


def _metric_lines(groups: dict[Any, list[CellResult]], metrics: dict[str, Any], header: str) -> list[str]:
    keys: list[str] = []
    lines: list[str] = []
    for (env, label), cells in sorted(groups.items()):
        rows = [metrics[c.cell_id] for c in cells]
        keys = list(rows[0])
        lines.append(f"| {env} | {label} | {len(cells)} | " + " | ".join(
            f"{bootstrap_iqm([row[k] for row in rows])[0]:.4g}" for k in keys) + " |")
    return [header, "", "| Environment | Rule | n | " + " | ".join(keys) + " |",
            "|---|---|---:|" + "|".join("---:" for _ in keys) + "|", *lines, ""]


# -------- M1 vuln-repair --------
BASE_BUDGET = 200_000
SUPPRESSION_ROLLOUTS = (24, 48, 96, 192)
# 主图（攻击者轨迹）用的压制时长
DEFAULT_SUPPRESSION = 96
RECOVERY_ROLLOUTS = 96
FIXED_SHARE_A = 0.35
M1_RULES = ("coda_r", "moving_anchor", "pfsp", "ufsp", "hedge", "fixed", "none")
N_DECOYS = 3
ATTACKER = "attacker_A"
# 真实败率不超过 1/2 + IDLE_TOL 的成员算作没有在赢 latest（含与 latest 打平的快照）
IDLE_TOL = 0.01


def _m1_label(rule: str) -> str:
    return {**LABELS, "fixed": f"Fixed mix (A = {FIXED_SHARE_A})"}.get(rule) or title(rule)


def run_vuln_prep(payload: dict[str, Any], cell_dir: Path) -> dict[str, Any]:
    cfg = config_from_dict(payload["config"])
    summary = run_training(cfg, cell_dir)
    game = load_game(cfg.env)
    base = _load_net_actor(game, cfg.hidden, cell_dir / "final.pt")
    attacker = exact_br_actor(game, base)
    _save_table(cell_dir / "attacker_A.json", attacker)
    base_table = to_tabular(game, base)
    return dict(training = summary, base_exploitability = full_exploitability(game, base),
                base_loss_vs_A = exact_loss(game, base_table, to_tabular(game, attacker)))


def run_vuln_branch(payload: dict[str, Any], cell_dir: Path) -> dict[str, Any]:
    cfg = config_from_dict(payload["config"])
    prep_dir = cell_dir.parent / payload["prep"]
    game = load_game(cfg.env)
    members: list[tuple[str, Any]] = [(ATTACKER, _load_table(prep_dir / "attacker_A.json"))]
    members += [(f"decoy_{i}", _random_net_actor(game, cfg.hidden, cfg.seed * 10 + i)) for i in range(N_DECOYS)]
    hooks = TrainHooks(initial_members = members, initial_state = torch.load(prep_dir / "final.pt", map_location = "cpu"),
                       reset_at_rollout = payload["reset_at"], fixed_shares = {ATTACKER: FIXED_SHARE_A}
                       if payload["rule"] == "fixed" else {})
    return run_training(cfg, cell_dir, hooks)


def _vuln_specs(out_root: Path, smoke: bool) -> tuple[list[CellSpec], list[CellSpec]]:
    selected = load_selected(out_root, smoke)
    seeds = SMOKE_SEEDS if smoke else SMALL_SEEDS
    suppressions = (4,) if smoke else SUPPRESSION_ROLLOUTS
    recovery = 4 if smoke else RECOVERY_ROLLOUTS
    preps, branches = [], []
    for env in ABLATION_GAMES:
        for seed in seeds:
            base = build_config("rnad", env, seed, BASE_BUDGET, selected[env]["rnad"], label = "base")
            prep = train_spec(base, smoke, role = "prep")
            preps.append(prep)
            gpr = 16 if smoke else base.games_per_rollout
            for suppression in suppressions:
                for rule in M1_RULES:
                    cfg = _rule_config(rule, env, seed, (suppression + recovery) * gpr, selected[env],
                                       f"{_m1_label(rule)} (T = {suppression})", oracle = "none",
                                       exact_loss_every = 1, eval_points = 6, tail_eval_rollouts = 8)
                    branches.append(train_spec(cfg, smoke, prep = prep.cell_id, deployment = "single", rule = rule,
                                               reset_at = suppression, suppression = suppression))
    return preps, branches


def vuln_metrics(r: CellResult) -> dict[str, float]:
    """
    t_fix / t_fix_after_reset：A 的真实败率连续 3 个 rollout 不超过 0.52 所需的 rollout 数；
    reweight_delay：重置后 A 的训练系数回到它在重置前峰值的一半所需的 rollout 数（无状态规则约为 0）；
    idle_field：没有在赢 latest 的成员持有的训练场份额；decoy_field：诱饵持有的平均训练系数。
    """
    loss_a, field_a = _member_series(r, ATTACKER)
    decoy = [sum(w for m, w in zip(h["pool_ids"], h["pool_field"]) if m.startswith("decoy"))
             for h in r.history if h.get("roll", -1) >= 0 and h.get("pool_losses_true")]
    reset = int(r.meta["reset_at"])
    peak = max(field_a[:reset], default = 0.0)
    reweight = next((float(i - reset) for i in range(reset, len(field_a)) if peak > 0.0 and field_a[i] >= 0.5 * peak),
                    float("nan"))
    idle = field_share(r, lambda loss: loss <= 0.5 + IDLE_TOL)
    return dict(t_fix = _first_stable(loss_a, 0, 0.52), t_fix_after_reset = _first_stable(loss_a, reset, 0.52),
                reweight_delay = reweight, idle_field = float(np.mean(idle)) if idle else float("nan"),
                decoy_field = float(np.mean(decoy)), loss_a_after_reset = float(np.mean(loss_a[reset:])))


def run_vuln_repair(options: RunOptions) -> Path:
    out = prepare_output(options)
    preps, branches = _vuln_specs(out.parent, options.smoke)
    prep_records = run_cells(preps, run_vuln_prep, options)
    require_complete(preps, prep_records)
    records = run_cells(branches, run_vuln_branch, options)
    results = load_results(out, records)
    metrics = {r.cell_id: vuln_metrics(r) for r in results}
    lines = _metric_lines(group(results), metrics, "## M1 指标（seed IQM；rollout 计）")
    lines += [f"t_fix / t_fix_after_reset：A 的真实败率连续 3 个 rollout 不超过 0.52 所需的 rollout 数；"
              "reweight_delay：重置后 A 的训练系数回到重置前峰值的一半所需的 rollout 数；"
              f"idle_field：真实败率不超过 {0.5 + IDLE_TOL:g} 的成员（含打平的快照）占训练场质量的比例；"
              "decoy_field：诱饵持有的平均训练系数。", ""]
    notes = [f"base 为 R-NaD-PPO 训练 {BASE_BUDGET} 局；攻击者 A 为 base 的全树精确 BR；{N_DECOYS} 个随机网络诱饵；"
             "池沿用主协议（保留 2 个快照），只关掉 oracle。",
             f"压制 T in {list(SUPPRESSION_ROLLOUTS)} 个 rollout 后把 learner 重置回 base，之后再训练 {RECOVERY_ROLLOUTS} 个 rollout。",
             "移动锚沿用 CoDA-R 的配置只换规则；Hedge 与固定比例沿用 R-NaD-PPO 的 eta 与熵终值，自博弈份额 0.5，不调参。"]
    write_grid_report(out, "vuln-repair：漏洞注入与重开", notes, results, len(branches), extra_lines = lines)
    return out


# -------- M2 open-pool --------
POOL_SIZES = (4, 16, 64)
M2_RULES = ("coda_r", "pfsp", "ufsp")
LEGACY_BUDGET = 50_000
LEGACY_HIDDEN = 32
CFR_ITERATIONS = (3, 30)
BOTS = {
    "leduc_poker": {"bot_call": PreferenceBot([1]), "bot_raise": PreferenceBot([2, 1])},
    "goofspiel4": {"bot_highest": PreferenceBot(highest = True), "bot_lowest": PreferenceBot(lowest = True)},
}


def run_open_prep(payload: dict[str, Any], cell_dir: Path) -> dict[str, Any]:
    env = payload["env"]
    game = load_game(env)
    summaries = []
    for i, legacy in enumerate(payload["legacy"]):
        summaries.append(run_training(config_from_dict(legacy), cell_dir / f"legacy_{i}"))
    solver = cfr.CFRPlusSolver(game)
    for it in range(1, max(CFR_ITERATIONS) + 1):
        solver.evaluate_and_update_policy()
        if it in CFR_ITERATIONS:
            _save_table(cell_dir / f"cfr_{it}.json", tabular_actor_from_policy(game, solver.average_policy()))
    return dict(legacy = summaries)


def open_pool_members(env: str, prep_dir: Path, n: int, seed: int, legacy_hidden: list[int]) -> list[tuple[str, Any]]:
    """有用成员在前，不足 N 时交替补随机网络与有用成员的重复（同一个策略对象换一个 ID）。"""
    game = load_game(env)
    useful: list[tuple[str, Any]] = [(f"cfr_{it}", _load_table(prep_dir / f"cfr_{it}.json")) for it in CFR_ITERATIONS]
    useful += [(f"legacy_{i}", _load_net_actor(game, hidden, prep_dir / f"legacy_{i}" / "final.pt"))
               for i, hidden in enumerate(legacy_hidden)]
    useful += list(BOTS[env].items()) + [("uniform", UniformActor())]
    members = useful[:n]
    k = 0
    while len(members) < n:
        if k % 2 == 0:
            members.append((f"junk_random_{k}", _random_net_actor(game, 128, seed * 1000 + k)))
        else:
            name, actor = useful[(k // 2) % len(useful)]
            members.append((f"junk_dup_{k}_{name}", actor))
        k += 1
    return members


def run_open_branch(payload: dict[str, Any], cell_dir: Path) -> dict[str, Any]:
    cfg = config_from_dict(payload["config"])
    members = [] if payload["rule"] == "none" else open_pool_members(
        cfg.env, cell_dir.parent / payload["prep"], payload["pool_size"], cfg.seed, payload["legacy_hidden"])
    return run_training(cfg, cell_dir, TrainHooks(initial_members = members))


def _open_specs(out_root: Path, smoke: bool) -> tuple[list[CellSpec], list[CellSpec]]:
    selected = load_selected(out_root, smoke)
    seeds = SMOKE_SEEDS if smoke else SMALL_SEEDS
    preps, branches = [], []
    for env in ABLATION_GAMES:
        rnad = selected[env]["rnad"]
        for seed in seeds:
            legacy = [build_config("rnad", env, seed * 10 + i, LEGACY_BUDGET, rnad, label = f"legacy {i}",
                                   hidden = LEGACY_HIDDEN) for i in (1, 2)]
            legacy = [c.shrunk() if smoke else c for c in legacy]
            prep = CellSpec(f"{env}_open-prep_s{seed}", dict(env = env, seed = seed, legacy = [c.as_dict() for c in legacy]))
            preps.append(prep)
            # 冒烟会缩小网络宽度，加载时必须按 prep 实际使用的宽度重建网络
            meta = dict(prep = prep.cell_id, legacy_hidden = [c.hidden for c in legacy])
            ref = build_config("rnad", env, seed, OPEN_POOL_BUDGET, rnad, label = NO_POOL_LABEL)
            branches.append(train_spec(ref, smoke, rule = "none", pool_size = 0, deployment = "single", **meta))
            for n, rule in [(n, rule) for n in POOL_SIZES for rule in M2_RULES]:
                cfg = build_config(rule, env, seed, OPEN_POOL_BUDGET, selected[env][rule], label = f"{title(rule)} N={n}",
                                   oracle = "none", snapshots_keep = 0, pool_capacity = n, exact_loss_every = 16)
                branches.append(train_spec(cfg, smoke, rule = rule, pool_size = n, deployment = "single", **meta))
    return preps, branches


def open_pool_metrics(r: CellResult) -> dict[str, float]:
    """latest 已经打赢的成员（真实败率 < 1/2）持有的训练场系数，占整个训练场质量（含自博弈份额）的比例。"""
    shares = field_share(r, lambda loss: loss < 0.5)
    half = len(shares) // 2
    return dict(beaten_share = float(np.mean(shares)) if shares else float("nan"),
                beaten_share_second_half = float(np.mean(shares[half:])) if shares else float("nan"))


def run_open_pool(options: RunOptions) -> Path:
    out = prepare_output(options)
    preps, branches = _open_specs(out.parent, options.smoke)
    prep_records = run_cells(preps, run_open_prep, options)
    require_complete(preps, prep_records)
    records = run_cells(branches, run_open_branch, options)
    results = load_results(out, records)
    pooled = [r for r in results if r.meta["rule"] != "none"]
    metrics = {r.cell_id: open_pool_metrics(r) for r in pooled}
    lines = _metric_lines(group(pooled), metrics, "## M2 指标（seed IQM）")
    lines += ["beaten_share：latest 已经打赢的成员持有的训练场系数占整个训练场质量的比例（越低说明梯度浪费越少）。", ""]
    notes = [f"预算 {OPEN_POOL_BUDGET}；无 oracle、无快照，池里只有预先放入的外部成员；N in {list(POOL_SIZES)}。",
             "成员顺序：CFR+ 第 3 / 30 次迭代的平均策略、两个 2x32 网络的旧 agent、两个脚本 bot、均匀策略，之后交替补随机网络与重复成员。"]
    write_grid_report(out, "open-pool：开放池与无用成员", notes, results, len(branches), extra_lines = lines)
    return out


# -------- M3 hidden-hole --------
HOLE_CFR_ITERATIONS = 2000
# 对手动作到达概率的分档 [lo, hi)；Goofspiel(4) 的 CFR+ 策略几乎是确定性的，中间档没有候选信息集
HOLE_BANDS: dict[str, tuple[tuple[float, float], ...]] = {
    "leduc_poker": ((0.3, 1.01), (0.03, 0.1), (0.003, 0.01), (0.0, 0.001)),
    "goofspiel4": ((0.3, 1.01), (0.0, 0.001)),
}
HOLE_SIZE = 4
HOLE_OWN_MIN = 0.1
# 每次到达漏洞损失的归一化得分下限（Leduc 为 0.05 x 26 = 1.3 筹码）
HOLE_REGRET_MIN = 0.05
HOLE_ROLLOUTS = 256
HOLE_RULES = ("coda_r", "pfsp", "none")
DISTILL_STEPS = 3000
# 漏洞动作的平均概率连续 3 个 rollout 不超过它，算作修复
REPAIR_LEVEL = 0.5
HOLE_ATTACKER = "attacker_H"


def run_hole_prep(payload: dict[str, Any], cell_dir: Path) -> dict[str, Any]:
    """CFR+ 底座 -> 各档的漏洞 -> 蒸馏出底座与各档的受损网络 -> 每档一个受损网络的精确 BR 攻击者。"""
    game = load_game(payload["env"])
    table = _cfr_table(game, payload["iterations"])
    bands = [(float(lo), float(hi)) for lo, hi in payload["bands"]]
    holes = select_holes(infoset_profiles(game, table), table, tuple(bands), HOLE_SIZE, HOLE_OWN_MIN, HOLE_REGRET_MIN)
    base, base_fit = _distilled(game, table, payload["seed"], payload["hidden"], payload["distill_steps"])
    summary: dict[str, Any] = dict(base_max_tv = base_fit["max_tv"], bands = [],
                                   base_exploitability = full_exploitability(game, frozen_actor(copy.deepcopy(base), "cpu")))
    for b, chosen in enumerate(holes):
        net = copy.deepcopy(base)
        fit = distill(net, game, damaged(table, chosen), payload["distill_steps"])
        torch.save(net.state_dict(), cell_dir / f"damaged_{b}.pt")
        actor = frozen_actor(net, "cpu")
        attacker = exact_br_actor(game, actor)
        _save_table(cell_dir / f"attacker_{b}.json", attacker)
        lead = exact_loss(game, to_tabular(game, actor), to_tabular(game, attacker)) - 0.5
        summary["bands"].append(dict(band = list(bands[b]), holes = [h.as_dict() for h in chosen], max_tv = fit["max_tv"],
                                     damaged_exploitability = full_exploitability(game, actor), attacker_lead = lead))
    write_json(cell_dir / "holes.json", summary)
    return summary


def run_hole_branch(payload: dict[str, Any], cell_dir: Path) -> dict[str, Any]:
    cfg = config_from_dict(payload["config"])
    prep_dir = cell_dir.parent / payload["prep"]
    band = int(payload["band"])
    holes = load_json(prep_dir / "holes.json")["bands"][band]["holes"]
    hooks = TrainHooks(initial_members = [(HOLE_ATTACKER, _load_table(prep_dir / f"attacker_{band}.json"))],
                       initial_state = torch.load(prep_dir / f"damaged_{band}.pt", map_location = "cpu"),
                       probe = HoleProbe(load_game(cfg.env), holes))
    return run_training(cfg, cell_dir, hooks)


def _band_label(band: tuple[float, float]) -> str:
    return f"opp reach [{band[0]:g}, {min(band[1], 1.0):g})"


def _hole_specs(out_root: Path, smoke: bool) -> tuple[list[CellSpec], list[CellSpec]]:
    selected = load_selected(out_root, smoke)
    seeds = SMOKE_SEEDS if smoke else SMALL_SEEDS
    rollouts = 8 if smoke else HOLE_ROLLOUTS
    preps, branches = [], []
    for env in ABLATION_GAMES:
        for seed in seeds:
            reference = _rule_config("none", env, seed, 1, selected[env], NO_POOL_LABEL)
            branch_cfgs = {(b, rule): _rule_config(rule, env, seed, rollouts * reference.games_per_rollout, selected[env],
                                                   f"{_m1_label(rule)} ({_band_label(band)})", oracle = "none",
                                                   exact_loss_every = 1, eval_points = 6, tail_eval_rollouts = 8)
                           for b, band in enumerate(HOLE_BANDS[env]) for rule in HOLE_RULES}
            # 底座网络的宽度必须与分支 learner 一致（冒烟会缩小宽度）
            hidden = (reference.shrunk() if smoke else reference).hidden
            prep = CellSpec(f"{env}_hole-prep_s{seed}", dict(
                env = env, seed = seed, hidden = hidden, iterations = 4 if smoke else HOLE_CFR_ITERATIONS,
                distill_steps = 20 if smoke else DISTILL_STEPS, bands = [list(b) for b in HOLE_BANDS[env]]))
            preps.append(prep)
            for (b, rule), cfg in branch_cfgs.items():
                branches.append(train_spec(cfg, smoke, prep = prep.cell_id, deployment = "single", rule = rule, band = b))
    return preps, branches


def hole_metrics(r: CellResult) -> dict[str, float]:
    """t_repair：漏洞动作的平均概率连续 3 个 rollout 不超过 REPAIR_LEVEL 所需的 rollout 数，没有修复为 NaN。"""
    mass = [float(h["hole_mass"]) for h in r.history if h.get("roll", -1) >= 0 and "hole_mass" in h]
    _loss, field = _member_series(r, HOLE_ATTACKER)
    return dict(t_repair = _first_stable(mass, 0, REPAIR_LEVEL), initial_mass = mass[0] if mass else float("nan"),
                final_mass = mass[-1] if mass else float("nan"), attacker_field = float(np.mean(field)) if field else float("nan"))


def censored_median(values: list[float], cap: float) -> float:
    """未修复（NaN）记为 +inf 后的中位数；超过一半没有修复时返回 inf，报告里写成 "> cap"。"""
    xs = [v if np.isfinite(v) else np.inf for v in values]
    return float(np.median(xs)) if xs and np.median(xs) <= cap else float("inf")


def _hole_lines(results: list[CellResult], preps: list[tuple[str, dict[str, Any]]], rollouts: int) -> list[str]:
    lines = ["## M3 修复时间（漏洞动作的平均概率连续 3 个 rollout 不超过 "
             f"{REPAIR_LEVEL:g}；未修复按删失处理）", "",
             "| Environment | 档 | Rule | n | 修复时间中位数 | 修复数 | 末尾漏洞概率均值 | 攻击者系数均值 |", "|---|---|---|---:|---:|---:|---:|---:|"]
    by_key = group(results, lambda r: (r.cfg.env, int(r.meta["band"]), str(r.meta["rule"])))
    for (env, band, rule), cells in sorted(by_key.items()):
        rows = [hole_metrics(c) for c in cells]
        times = [row["t_repair"] for row in rows]
        median = censored_median(times, rollouts)
        median_text = f"{median:.4g}" if np.isfinite(median) else f"> {rollouts}"
        lines.append(f"| {env} | {_band_label(HOLE_BANDS[env][band])} | {_m1_label(rule)} | {len(cells)} | {median_text} | "
                     f"{sum(np.isfinite(t) for t in times)} | {np.mean([row['final_mass'] for row in rows]):.3g} | "
                     f"{np.nanmean([row['attacker_field'] for row in rows]):.3g} |")
    lines += ["", "## M3 漏洞与底座（prep，seed 均值）", "",
              "| Environment | 档 | 漏洞数 | 对手到达概率 | 每次到达损失 | 攻击者领先 | 受损网络 exploitability | 蒸馏最大总变差 |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
    stats: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for env, summary in preps:
        for b, band in enumerate(summary["bands"]):
            stats.setdefault((env, b), []).append(band)
    for (env, b), bands in sorted(stats.items()):
        holes = [h for band in bands for h in band["holes"]]
        lines.append(f"| {env} | {_band_label(HOLE_BANDS[env][b])} | {len(holes) / len(bands):.3g} | "
                     f"{np.mean([h['opponent_reach'] for h in holes]) if holes else float('nan'):.3g} | "
                     f"{np.mean([h['regret'] for h in holes]) if holes else float('nan'):.3g} | "
                     f"{np.mean([band['attacker_lead'] for band in bands]):.3g} | "
                     f"{np.mean([band['damaged_exploitability'] for band in bands]):.3g} | "
                     f"{np.mean([band['max_tv'] for band in bands]):.3g} |")
    return lines + [""]


def run_hidden_hole(options: RunOptions) -> Path:
    out = prepare_output(options)
    preps, branches = _hole_specs(out.parent, options.smoke)
    prep_records = run_cells(preps, run_hole_prep, options)
    require_complete(preps, prep_records)
    records = run_cells(branches, run_hole_branch, options)
    results = load_results(out, records)
    prep_summaries = [(str(spec.payload["env"]), load_json(out / spec.cell_id / "holes.json")) for spec in preps]
    notes = [f"底座为 CFR+ {HOLE_CFR_ITERATIONS} 次迭代的平均策略，蒸馏进与 learner 同宽的网络；每档最多 {HOLE_SIZE} 个漏洞信息集，"
             f"要求 latest 自己的到达概率 >= {HOLE_OWN_MIN:g}、每次到达至少损失 {HOLE_REGRET_MIN:g}（归一化得分）。",
             "池里只有受损网络的精确 BR 攻击者；池沿用主协议（保留 2 个快照），只关掉 oracle；无池对照只记录败率，不抽攻击者。",
             f"每个分支训练 {HOLE_ROLLOUTS} 个 rollout。"]
    lines = _hole_lines(results, prep_summaries, 8 if options.smoke else HOLE_ROLLOUTS)
    write_grid_report(out, "hidden-hole：自博弈到达概率受控的隐藏漏洞", notes, results, len(branches), extra_lines = lines)
    return out


# -------- M4 forgetting --------
RETENTION_METHODS = ("coda_r", "pfsp", "rnad", "mmd", "selfplay")
# CoDA-R 的对偶强度相对调参选择的倍数；这两档只作敏感性分析，不参与方法间比较
RETENTION_C_FACTORS = (0.5, 2.0)
# 对手动作到达概率的分档 [lo, hi)，与 M3 同一口径
RETENTION_BANDS: tuple[tuple[float, float], ...] = ((0.1, 1.01), (0.01, 0.1), (0.001, 0.01), (0.0, 0.001))
RETENTION_PROBE_EVERY = 16
RETENTION_ARCHIVE_EVERY = 128
# 精确败率没有采样噪声；越过 1/2 +- 这个量才算打败或被反超
RETENTION_MARGIN = 0.01


def _c_label(method: str, c_factor: float) -> str:
    return title(method) if c_factor == 1.0 else f"{title(method)} (c x {c_factor:g})"


def _retention_config(method: str, env: str, seed: int, params: dict[str, Any], c_factor: float) -> RunConfig:
    """调参选出的配置，熵全程固定在终值：从近似纳什出发时，退火的熵会把所有方法一起推离纳什，与遗忘混在一起。"""
    if c_factor != 1.0:
        params = {**params, "coda_c": float(params["coda_c"]) * c_factor}
    return build_config(method, env, seed, ABLATION_BUDGET, params, label = _c_label(method, c_factor),
                        ent_start = float(params["ent_final"]))


def run_retention_prep(payload: dict[str, Any], cell_dir: Path) -> dict[str, Any]:
    """CFR+ 底座 -> 蒸馏出起点网络；记录每个到达概率档里参与统计的信息集数。"""
    game = load_game(payload["env"])
    table = _cfr_table(game, payload["iterations"])
    base, fit = _distilled(game, table, payload["seed"], payload["hidden"], payload["distill_steps"])
    torch.save(base.state_dict(), cell_dir / "base.pt")
    _save_table(cell_dir / "cfr.json", TabularActor(table))
    actor = frozen_actor(base, "cpu")
    bands = tuple((float(lo), float(hi)) for lo, hi in payload["bands"])
    return dict(base_max_tv = fit["max_tv"], base_exploitability = full_exploitability(game, actor),
                cfr_exploitability = full_exploitability(game, TabularActor(table)),
                band_sizes = RetentionProbe(game, table, bands, HOLE_OWN_MIN, 1, 1, actor).band_sizes())


def run_retention_branch(payload: dict[str, Any], cell_dir: Path) -> dict[str, Any]:
    cfg = config_from_dict(payload["config"])
    prep_dir = cell_dir.parent / payload["prep"]
    game = load_game(cfg.env)
    bands = tuple((float(lo), float(hi)) for lo, hi in payload["bands"])
    probe = RetentionProbe(game, _load_table(prep_dir / "cfr.json").table, bands, HOLE_OWN_MIN, payload["probe_every"],
                           payload["archive_every"], _load_net_actor(game, cfg.hidden, prep_dir / "base.pt"))
    hooks = TrainHooks(initial_state = torch.load(prep_dir / "base.pt", map_location = "cpu"), probe = probe)
    return run_training(cfg, cell_dir, hooks)


def _retention_specs(out_root: Path, smoke: bool) -> tuple[list[CellSpec], list[CellSpec]]:
    selected = load_selected(out_root, smoke)
    seeds = SMOKE_SEEDS if smoke else SMALL_SEEDS
    bands = [list(b) for b in RETENTION_BANDS]
    # 冒烟的预算只有几个 rollout，探针与档案必须更密才会触发
    cadence = dict(probe_every = 2, archive_every = 4) if smoke else dict(probe_every = RETENTION_PROBE_EVERY,
                                                                           archive_every = RETENTION_ARCHIVE_EVERY)
    arms = [(method, 1.0) for method in RETENTION_METHODS] + [("coda_r", f) for f in RETENTION_C_FACTORS]
    preps, branches = [], []
    for env in ABLATION_GAMES:
        for seed in seeds:
            cfgs = [(method, factor, _retention_config(method, env, seed, selected[env][method], factor))
                    for method, factor in arms]
            # 起点网络的宽度必须与分支 learner 一致（冒烟会缩小宽度）
            hidden = (cfgs[0][2].shrunk() if smoke else cfgs[0][2]).hidden
            prep = CellSpec(f"{env}_retention-prep_s{seed}", dict(
                env = env, seed = seed, hidden = hidden, iterations = 4 if smoke else HOLE_CFR_ITERATIONS,
                distill_steps = 20 if smoke else DISTILL_STEPS, bands = bands))
            preps.append(prep)
            for method, factor, cfg in cfgs:
                branches.append(train_spec(cfg, smoke, prep = prep.cell_id, deployment = "single", method = method,
                                           c_factor = factor, bands = bands, **cadence))
    return preps, branches


def _finite_mean(values: list[float]) -> float:
    xs = [v for v in values if np.isfinite(v)]
    return float(np.mean(xs)) if xs else float("nan")


def retention_metrics(r: CellResult) -> dict[str, float]:
    """
    expl_start：起点网络的 exploitability；expl_rise / expl_rise_max：末段（anneal_frac * B 之后）探针的平均 / 全程最大
    exploitability 减去起点；worst_past、final_worst_past、forgotten：对历史档案的遗忘（common.stats.history_forgetting）；
    drift_loss 各档：末段每次到达比 CFR+ 底座少得的归一化得分（对手按底座行动）。
    """
    start = next(float(h["exploitability"]) for h in r.history if h.get("roll", -1) < 0 and "exploitability" in h)
    probes = [h for h in r.history if "probe_exploitability" in h]
    tail = [h for h in probes if h["games"] >= r.cfg.ent_anneal_frac * r.cfg.budget] or probes[-1:]
    expl = [float(h["probe_exploitability"]) for h in probes]
    out = dict(expl_start = start, expl_rise = _finite_mean([float(h["probe_exploitability"]) for h in tail]) - start,
               expl_rise_max = max(expl, default = float("nan")) - start,
               **history_forgetting([h["archive_losses"] for h in probes], RETENTION_MARGIN))
    for b, band in enumerate(RETENTION_BANDS):
        out[f"drift_loss {_band_label(band)}"] = _finite_mean([float(h["drift_loss"][b]) for h in tail])
    return out


def _retention_prep_lines(preps: list[CellSpec], records: dict[str, dict[str, Any]]) -> list[str]:
    lines = ["## M4 起点（prep，seed 均值）", "",
             "| Environment | CFR+ exploitability | 起点网络 exploitability | 蒸馏最大总变差 | "
             + " | ".join(f"{_band_label(b)} 信息集数" for b in RETENTION_BANDS) + " |",
             "|---|---:|---:|---:|" + "|".join("---:" for _ in RETENTION_BANDS) + "|"]
    for env in ABLATION_GAMES:
        rows = [records[s.cell_id]["summary"] for s in preps if s.payload["env"] == env and s.cell_id in records]
        if not rows:
            continue
        sizes = np.mean([row["band_sizes"] for row in rows], axis = 0)
        lines.append(f"| {env} | {np.mean([row['cfr_exploitability'] for row in rows]):.3g} | "
                     f"{np.mean([row['base_exploitability'] for row in rows]):.3g} | "
                     f"{np.mean([row['base_max_tv'] for row in rows]):.3g} | " + " | ".join(f"{n:.3g}" for n in sizes) + " |")
    return lines + [""]


def run_forgetting(options: RunOptions) -> Path:
    out = prepare_output(options)
    preps, branches = _retention_specs(out.parent, options.smoke)
    prep_records = run_cells(preps, run_retention_prep, options)
    require_complete(preps, prep_records)
    records = run_cells(branches, run_retention_branch, options)
    results = load_results(out, records)
    metrics = {r.cell_id: retention_metrics(r) for r in results}
    lines = _metric_lines(group(results), metrics, "## M4 指标（seed IQM）")
    lines += [f"expl_rise / expl_rise_max：末段平均 / 全程最大的 exploitability 减去起点（每 {RETENTION_PROBE_EVERY} 个 rollout "
              f"测一次）；worst_past：对历史档案的最坏精确败率减 1/2，按考试平均；forgotten：曾被打败（败率 <= "
              f"{0.5 - RETENTION_MARGIN:g}）后又打败 latest（>= {0.5 + RETENTION_MARGIN:g}）的历史版本占比；"
              "drift_loss：末段每次到达比 CFR+ 底座少得的归一化得分，按对手到达概率分档，只统计 latest 自己常走到的信息集。", ""]
    lines += _retention_prep_lines(preps, prep_records)
    notes = [f"起点为 CFR+ {HOLE_CFR_ITERATIONS} 次迭代的平均策略蒸馏出的网络（与 M3 的底座相同，不埋漏洞）；"
             f"预算 {ABLATION_BUDGET}，按主协议训练（带池的方法含 oracle），熵全程固定在各方法调参选出的终值。",
             f"历史档案从起点网络开始，每 {RETENTION_ARCHIVE_EVERY} 个 rollout 收进一个 latest，只用来考试，不进训练；"
             "有池与无池的方法用同一种档案。",
             f"CoDA-R 另跑 c x {list(RETENTION_C_FACTORS)} 两档作为敏感性分析。"]
    write_grid_report(out, "forgetting：从近似纳什出发的遗忘", notes, results, len(branches), extra_lines = lines)
    return out


# -------- 计划格子数 --------
def planned_vuln_cells() -> int:
    return len(ABLATION_GAMES) * len(SMALL_SEEDS) * (1 + len(SUPPRESSION_ROLLOUTS) * len(M1_RULES))


def planned_open_cells() -> int:
    return len(ABLATION_GAMES) * len(SMALL_SEEDS) * (2 + len(POOL_SIZES) * len(M2_RULES))


def planned_hole_cells() -> int:
    return len(SMALL_SEEDS) * sum(1 + len(HOLE_BANDS[env]) * len(HOLE_RULES) for env in ABLATION_GAMES)


def planned_forgetting_cells() -> int:
    return len(ABLATION_GAMES) * len(SMALL_SEEDS) * (1 + len(RETENTION_METHODS) + len(RETENTION_C_FACTORS))
