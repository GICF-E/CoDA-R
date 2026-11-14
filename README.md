# CoDA-R experiments

[中文说明](README_zh.md)

Source code for every experiment in the CoDA-R paper (Coupled Dual-Adversarial Regularized self-play): checks of the theory on finite matrix games, and the PPO implementation with its baselines on OpenSpiel. Algorithm definitions and proofs are in the paper (`paper/`). This directory is self-contained.

## The algorithm

CoDA-R trains one latest policy $x$ against a pool of frozen opponents $y_1,\dots,y_N$ plus a *mirror arm* (latest playing itself). The dual side sets the pool weights in closed form, $w(x)=\Pi_\Delta(e_m+c\,(\ell_1(x),\dots,\ell_N(x),\tfrac12))$, a Euclidean projection anchored at pure self-play $e_m$: only members that currently beat latest can hold weight, and tied members get exactly zero. The primal side takes a proximal best response against the field $(w_P,1)$ (self-play with coefficient 1 plus the pool weighted by $w$), with a proximal term of fixed strength $\eta$ toward the previous anchor. In the PPO implementation opponents are *sampled* from the dual weights $(w_P,w_m)$ with a floor on the mirror arm, and each game carries the importance ratio $\text{field}_i/r_i$, so the expected update is exactly the field $(w_P,1)$.

## Layout

```
experiment/
  cli.py, __main__.py     command line
  registry.py             every experiment, its group and dependencies
  common/                 cell runner (config hash, resume, workers), statistics, I/O, plotting
  core/                   simplex projection, pool rules, loss estimator, opponent pool
  theory/                 matrix-game families, exact VI step and certificates, pool-rule trajectories and adversarial schedules, T1-T12
  neural/                 OpenSpiel games, PPO, the single training loop, oracle, PSRO, NFSP, evaluation
  experiments/            protocol, method presets and tuning grids, tuning, main, ablations, diagnostics, mechanisms, scale, checks
  tests/                  unit and integration tests
analysis/
  cli.py, __main__.py     command line for figures and analysis tables
  data.py, style.py       output loading; fixed per-method colors and line styles
  plots.py, figures/      shared plotting helpers; one function per paper figure
  tables.py               analysis.md
paper/, ICLR/             LaTeX sources
```

The one design rule worth knowing: **every pool method differs only in its pool rule.** `core/pool_rules.py` maps loss estimates to field coefficients and dual weights (CoDA-R, moving anchor, PFSP, uniform FSP, Hedge, fixed mix). The pool, estimator, sampling correction, oracle and proximal term are shared, so CoDA-R versus PFSP isolates the dual rule.

| Object | File |
|---|---|
| Pool rules | `experiment/core/pool_rules.py` |
| Opponent pool: sampling distribution, importance ratios, probes, eviction | `experiment/core/pool.py` |
| Loss estimator (count-adaptive EMA with change-point reset) | `experiment/core/estimator.py` |
| Training loop for self-play, R-NaD-PPO, MMD-PPO and all pool methods | `experiment/neural/train.py` |
| Entropy, evaluation and oracle schedules | `experiment/neural/schedule.py` |
| Method presets and tuning grids | `experiment/experiments/methods.py` |
| Exact VI step and certificates on matrix games | `experiment/theory/solver.py` |

## Setup

Python 3.11 with the packages in `requirements.txt` (PyTorch, OpenSpiel, NumPy, SciPy, Matplotlib, dm-tree, pytest, pyright):

```powershell
conda create -n coda-r python=3.11
conda activate coda-r
pip install -r requirements.txt
```

The 2x128 MLPs run faster on CPU; parallelism comes from worker processes.

Exact evaluation of the scale games uses [exp-a-spiel](https://github.com/gabrfarina/exp-a-spiel) (upstream commit `9412882`). `third_party/exp-a-spiel` is a copy with Windows / MSVC portability patches, each marked with a `coda-r:` comment; no algorithm is changed. On Windows it needs Visual Studio Build Tools (Desktop development with C++), then:

```powershell
conda install -c conda-forge --no-deps "libboost-headers>=1.85"
pip install "cmake>=3.29" ninja pybind11 scikit-build-core
$env:CMAKE_GENERATOR = "Ninja"; $env:CMAKE_PREFIX_PATH = "$env:CONDA_PREFIX\Library"
cmd /c "call ""<VS install dir>\VC\Auxiliary\Build\vcvars64.bat"" && python -m pip install --no-build-isolation .\third_party\exp-a-spiel"
```

The `eas_coherence` check in `preflight` verifies step by step that exp-a-spiel and OpenSpiel agree on both scale games.

## Usage

Run every command from the `coda-r/` directory.

```powershell
python -m experiment --list                      # plan, cell counts and dependencies
python -m experiment --status                    # progress under experiment/outputs
python -m experiment all --smoke --workers 8     # tiny budgets end to end, output in .tmp/experiment-smoke
python -m experiment theory --workers 24
python -m experiment tune-a --workers 24 --resume
```

- `--workers N`: run cells in N processes (each pinned to one thread). Every cell gets its own child process; if it exits abnormally (native crash, killed by the OS) only that cell is retried, up to 3 attempts, after which it is recorded as failed without affecting other cells. Each cell re-imports the code when it starts, so do not edit the code while a run is in progress.
- `--resume`: skip cells that are already done. A cell ID contains a hash of its full configuration, so a changed configuration never reuses an old result; after a *code* change, rerun without `--resume`.
- `--max-cells N`: run at most N cells, then stop (continue with `--resume`).
- `--output-root`: root directory; each experiment writes to `<root>/<id>`.

Recommended order (also the order of `all`): `preflight`, `feasibility`, `theory`, `references`, `tune-a`, `tune-b`, `ablations`, `vuln-repair`, `open-pool`, `hidden-hole`, `forgetting`, `main`, `scale-train`, `scale-eval`, `scale-crossplay`, `scale-history`. Experiments read their dependencies from sibling directories under the same root.

## Protocol of the neural experiments

- **Games.** Leduc poker, Goofspiel(4) and Liar's Dice (one die each) with exact full-tree exploitability in OpenSpiel. For scale, 3x3 abrupt Dark Hex and abrupt Phantom Tic-Tac-Toe (about $2.7\times10^{10}$ nodes each, $2.7\times10^7$ / $2.3\times10^7$ information sets), evaluated after training with exact full-tree exploitability from exp-a-spiel at checkpoints $1/64, 1/16, 1/4, 0.8, 0.9, 1$ of the budget; `tail_mean` averages the last three.
- **Budget.** $B=4\times10^6$ algorithmic games per run for the main comparison: learner games (including probe games that carry no gradient), exploiter training, insertion matches, pool sweeps and PSRO meta-matrix games. Exact best responses use no games, so the exact-BR rows are oracle-strength upper bounds, not equal-cost comparisons.
- **Entropy.** The coefficient falls linearly from 0.03 to the tuned final value over the first $0.8B$ games and stays there, so the tail $[0.8B,B]$ is measured at one entropy level. Tuning and ablation runs use the same rule with their own budget. The learning rate is constant unless `lr_final_frac < 1` (linear decay over the whole budget); fields added after the existing runs (`lr_final_frac`, `oracle_ent_final`) are left out of the config hash at their defaults, so existing cell IDs stay valid.
- **Oracle.** The oracle period is derived from the cost of one PPO exploiter so that it uses about 12% of the budget; the exact-BR variant uses the same period. No oracle in the last 48 rollouts.
- **Metrics.** Evaluations are held forward on the budget axis. Primary: `tail_mean` over $[0.8B,B]$; secondary: `tail_worst`, `at_budget`; AUC and curves are descriptive. Groups report the IQM over seeds with a bootstrap 95% CI and per-seed paired differences. Single-network and mixture deployments are reported in separate tables.
- **Tuning.** Every tuned method has 9 configurations (3 final entropies x 3 values of one method parameter; NFSP uses its own two parameters). Stage 1 runs them at $B/4$ on seeds 900-902; stage 2 reruns the top 2 at the full $B$ on the same seeds and keeps the better one. Uniform FSP, PFSP and CoDA-R reuse the $\eta$ selected for R-NaD-PPO, so their primal side is identical. CoDA-R's dual strength $c$ is tuned per game ($\{12,24,48\}$ on Leduc, $\{3,6,15\}$ elsewhere) because Leduc's 26-chip payoff span shrinks loss-rate leads. The scale games are not tuned and reuse the selection of Goofspiel(4), which also pays $\pm1$.
- **Seeds.** Main: 100-109; ablations and mechanisms: 100-105; scale: 100-104. Tuning seeds never enter the reported tables.

## Experiments

| ID | Content |
|---|---|
| `theory-energy` ... `theory-inexact` (T1-T5) | energy inequality, pool changes, the four certificates, rates (including the last-iterate bound), error-tolerant versions |
| `theory-necessity` (T6) | explicit mirror, self-play share 0 / current $w_m$ / previous $w_m$ (rule B), frozen pool coefficients |
| `theory-anchor-witness` (T7) | {fixed, moving} dual anchor x {share 1, share $w_m$}, each from a constructed fixed point |
| `theory-anchor-basin` (T8) | how often the moving anchor gets stuck at a non-Nash point from random starts (four game families, 200 seeds per cell, with and without estimation noise) |
| `theory-sampling` (T9) | sampled games on matrix games with the real pool code: field vs dual vs half sampling base |
| `theory-pool-rules` (T10) | CoDA-R, moving anchor, PFSP, uniform, Hedge and plain proximal self-play: weight held by tied members and the sleeper delay |
| `theory-oracle-rate` (T11) | an (eps-)BR of the anchor joins the pool every step and $c$ is the largest value Theorem 9 allows: Theorem 9 checked at every step; convergence speed with and without the pool and under fixed-share rules |
| `theory-adversarial-pool` (T12) | an adversary builds the pool (periodic oracle, complement with respect to Nash, flooding with random members, pure strategies on the Nash support, lagged oracle); final exploitability of every rule, with the energy inequality checked at every CoDA-R step |
| `references` | CFR+ reference curves |
| `tune-a`, `tune-b` | two-stage tuning; writes `selected.json` |
| `main` | 12 methods x 3 games x 10 seeds |
| `ablations` | A1-A9 on CoDA-R: moving anchor, sampling base, zero-weight detection, exact losses, oracle, mirror refresh, frozen coefficients, proximal form, pool capacity |
| `vuln-repair` (M1) | exact-BR attacker plus decoys, two snapshots kept in the pool; the learner is reset after $T\in\{24,48,96,192\}$ rollouts of suppression, measuring how the re-weighting delay grows with $T$ (sleeper) and the field share of idle members |
| `open-pool` (M2) | scripted bots, CFR+ iterates, small legacy agents, random and duplicate members, $N\in\{4,16,64\}$ |
| `hidden-hole` (M3) | a CFR+ near-Nash policy distilled into the network, with the worst action planted at infosets whose opponent reach under self-play is controlled; repair time of R-NaD-PPO, PFSP and CoDA-R (pool: only the exact BR of the damaged network) |
| `forgetting` (M4) | same CFR+ near-Nash start (no planted hole); CoDA-R, PFSP, R-NaD-PPO, MMD-PPO and self-play continue under the main protocol with entropy fixed at the tuned final value; exploitability rise, drift by opponent-reach band, and losses to an exam-only archive of past versions; CoDA-R $c$ sensitivity at $\times0.5$ and $\times2$ |
| `scale-train`, `scale-eval`, `scale-crossplay`, `scale-history` | 6 methods x 5 seeds on both scale games; exact full-tree exploitability with exp-a-spiel (about 30 GB, one evaluation process at a time); round robin within each game; sampled matches of later checkpoints against earlier ones (M4 scale arm) |

## Outputs

Each experiment directory holds `cells.jsonl` (one status line per finished cell, with its full configuration), `summary.md` and usually `records.json` and figures. A cell directory holds `rollouts.jsonl` (per-rollout log; evaluation rows have `evaluation = true`), `run_summary.json`, `oracles.jsonl` (insertions with the exploiter lower bound) and, where relevant, checkpoints.

## Analysis and figures

`analysis/` only reads the experiment outputs. It builds the paper figures and `analysis.md` (degradation relative to the best checkpoint, all pre-registered pairs including Q against NFSP, paired differences on the scale games, and whether each tuned value sits on the edge of its grid). Every method has a fixed color and line style across all figures (`analysis/style.py`); figures whose experiment has no output are skipped.

```powershell
python -m analysis --list                           # available figures
python -m analysis                                  # everything, written to analysis/outputs
python -m analysis main_curves tables --dest paper/figures
```

## Type checking and tests

```powershell
python -m pyright experiment analysis
python -m pytest experiment/tests analysis/tests -q -p no:cacheprovider
python -m experiment all --smoke --workers 8
```
