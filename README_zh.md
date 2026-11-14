# CoDA-R 实验

[English](README.md)

本目录包含 CoDA-R（Coupled Dual-Adversarial Regularized self-play）论文的全部实验代码：有限矩阵博弈上的理论检验，以及 OpenSpiel 上的 PPO 实现与各基线。算法定义与证明见论文（`paper/`）。本目录自成一体。

## 算法

CoDA-R 训练一个 latest 策略 $x$，对手是冻结成员池 $y_1,\dots,y_N$ 加一个镜像臂（latest 自己）。对偶侧按闭式给出池权重 $w(x)=\Pi_\Delta(e_m+c(\ell_1(x),\dots,\ell_N(x),\tfrac12))$，锚固定在纯自博弈 $e_m$：只有正在赢 latest 的成员能持有权重，打平的成员权重精确为 0。原侧对训练场 $(w_P,1)$ 做近端最佳响应，即自博弈系数恒为 1、池按 $w$ 叠加，并带强度固定为 $\eta$ 的近端项。PPO 实现里对手按对偶权重 $(w_P,w_m)$ 抽取（镜像臂有采样下限），每局乘重要性比 $\text{field}_i/r_i$，期望更新恰好是训练场 $(w_P,1)$。

## 目录

```
experiment/
  cli.py, __main__.py     命令行
  registry.py             全部实验、分组与依赖
  common/                 格子运行器（配置哈希、续跑、并行）、统计、读写、作图
  core/                   单纯形投影、池规则、败率估计器、对手池
  theory/                 矩阵博弈族、精确 VI 一步与证书、池规则轨迹与对抗调度、T1-T12
  neural/                 OpenSpiel 环境、PPO、唯一的训练循环、oracle、PSRO、NFSP、评估
  experiments/            协议、方法预设与调参网格、调参、主实验、消融、诊断、机制实验、规模实验、检查
  tests/                  单元与集成测试
analysis/
  cli.py, __main__.py     出图与分析表的命令行
  data.py, style.py       读取实验输出；每个方法固定的颜色与线型
  plots.py, figures/      共用绘图工具；每张论文图一个函数
  tables.py               analysis.md
paper/, ICLR/             LaTeX 源文件
```

最需要知道的一条设计：**所有带池的方法只在池规则上不同**。`core/pool_rules.py` 把败率估计映射成训练场系数与对偶权重（CoDA-R、移动锚、PFSP、均匀 FSP、Hedge、固定比例）；池、估计器、采样修正、oracle 与近端项全部共用，所以 CoDA-R 对 PFSP 的比较只隔离了对偶规则。

| 对象 | 文件 |
|---|---|
| 池规则 | `experiment/core/pool_rules.py` |
| 对手池：采样分布、重要性比、探测、淘汰 | `experiment/core/pool.py` |
| 败率估计器（局数自适应 EMA 加变点重置） | `experiment/core/estimator.py` |
| 自博弈、R-NaD-PPO、MMD-PPO 与全部带池方法的训练循环 | `experiment/neural/train.py` |
| 熵、评估与 oracle 日程 | `experiment/neural/schedule.py` |
| 方法预设与调参网格 | `experiment/experiments/methods.py` |
| 矩阵博弈上的精确 VI 一步与证书 | `experiment/theory/solver.py` |

## 环境

Python 3.11，依赖见 `requirements.txt`：

```powershell
conda create -n coda-r python=3.11
conda activate coda-r
pip install -r requirements.txt
```

2x128 的 MLP 在 CPU 上更快，并行靠多个 worker 进程。

规模实验的精确评估用 [exp-a-spiel](https://github.com/gabrfarina/exp-a-spiel)（上游提交 `9412882`）。`third_party/exp-a-spiel` 是它加上 Windows / MSVC 适配补丁后的副本，补丁处都以 `coda-r:` 注释标出，不改动任何算法。在 Windows 上需要 Visual Studio Build Tools（C++ 桌面开发），然后：

```powershell
conda install -c conda-forge --no-deps "libboost-headers>=1.85"
pip install "cmake>=3.29" ninja pybind11 scikit-build-core
$env:CMAKE_GENERATOR = "Ninja"; $env:CMAKE_PREFIX_PATH = "$env:CONDA_PREFIX\Library"
cmd /c "call ""<VS 安装目录>\VC\Auxiliary\Build\vcvars64.bat"" && python -m pip install --no-build-isolation .\third_party\exp-a-spiel"
```

`preflight` 的 `eas_coherence` 检查 exp-a-spiel 与 OpenSpiel 在两个规模游戏上逐步一致。

## 使用

所有命令都在 `coda-r/` 目录下运行。

```powershell
python -m experiment --list                      # 计划、格子数与依赖
python -m experiment --status                    # experiment/outputs 下的进度
python -m experiment all --smoke --workers 8     # 极小预算跑通全流程，输出在 .tmp/experiment-smoke
python -m experiment theory --workers 24
python -m experiment tune-a --workers 24 --resume
```

- `--workers N`：用 N 个进程并行跑格子（每个进程单线程）。每个格子独占一个子进程，子进程异常退出（崩溃、被系统杀掉）只重跑这一个格子，最多 3 次，之后记为失败，不影响其它格子。每个格子启动时重新导入代码，所以运行期间不要修改代码。
- `--resume`：跳过已完成的格子。格子 ID 含完整配置的哈希，配置改动不会误用旧结果；改了代码之后应不带 `--resume` 重跑。
- `--max-cells N`：本次最多跑 N 个格子（之后用 `--resume` 继续）。
- `--output-root`：输出根目录，每个实验写到 `<root>/<id>`。

推荐顺序（也是 `all` 的顺序）：`preflight`、`feasibility`、`theory`、`references`、`tune-a`、`tune-b`、`ablations`、`vuln-repair`、`open-pool`、`hidden-hole`、`forgetting`、`main`、`scale-train`、`scale-eval`、`scale-crossplay`、`scale-history`。实验从同一根目录下的兄弟目录读取依赖。

## 神经实验协议

- **游戏**：Leduc、Goofspiel(4)、Liar's Dice（每人 1 颗骰子），OpenSpiel 全树精确 exploitability。规模实验用 3x3 abrupt Dark Hex 与 abrupt Phantom Tic-Tac-Toe（各约 $2.7\times10^{10}$ 个节点、$2.7\times10^7$ / $2.3\times10^7$ 个信息集），训练后由 exp-a-spiel 做全树精确 exploitability，checkpoint 取预算的 $1/64, 1/16, 1/4, 0.8, 0.9, 1$ 处，`tail_mean` 为后三个点的平均。
- **预算**：主实验每次运行 $B=4\times10^6$ 局算法性对局，包括 learner 对局（含不进梯度的探测局）、exploiter 训练、入池评估、全池扫描与 PSRO 元矩阵对局。精确 BR 不消耗环境对局，因此 exact-BR 各行是 oracle 强度上限，不是同成本比较。
- **熵**：前 $0.8B$ 局内从 0.03 线性降到调参选出的终值，之后保持不变，末段 $[0.8B,B]$ 在同一个熵终值下测得。调参与消融按各自的预算套同一规则。学习率默认恒定，`lr_final_frac < 1` 时在整个预算上线性下降；已有实验之后才加入的字段（`lr_final_frac`、`oracle_ent_final`）取默认值时不进配置哈希，已有格子 ID 保持有效。
- **oracle**：触发周期由一次 PPO exploiter 的成本反推，使它约占 12% 的预算；精确 BR 版本用同一周期；最后 48 个 rollout 不再触发。
- **指标**：评估点在预算轴上前值保持。主指标为 $[0.8B,B]$ 上的 `tail_mean`，次要指标 `tail_worst`、`at_budget`；AUC 与曲线只作描述。分组报告 seed 的 IQM 与 bootstrap 95% CI，以及逐 seed 配对差。单网络与混合部署分表。
- **调参**：每种调参方法 9 个配置（熵终值 3 个 x 一个方法参数 3 个；NFSP 用自己的两个参数）。阶段 1 在 seed 900-902 上用 $B/4$；阶段 2 把前 2 名在同样的 seed 上用完整 $B$ 复核，取更好的一个。均匀 FSP、PFSP、CoDA-R 沿用 R-NaD-PPO 选出的 $\eta$，原侧完全相同。CoDA-R 的对偶强度 $c$ 按游戏调（Leduc 为 $12,24,48$，其余为 $3,6,15$），因为 Leduc 的 26 筹码收益跨度会压小败率领先。规模游戏不调参，沿用收益同为 $\pm1$ 的 Goofspiel(4) 的选择。
- **seed**：主实验 100-109；消融与机制实验 100-105；规模实验 100-104。调参 seed 不进入任何报告表。

## 实验

| ID                                             | 内容                                                               |
| ---------------------------------------------- | ---------------------------------------------------------------- |
| `theory-energy` ... `theory-inexact`（T1-T5）    | 能量式、池变化、四条证书、速率（含 last-iterate 界）、带误差版本                          |
| `theory-necessity`（T6）                         | 显式镜像、自博弈份额 0 / 本步 $w_m$ / 上一步 $w_m$（规则 B）、冻结池系数                  |
| `theory-anchor-witness`（T7）                    | {固定, 移动} 对偶锚 x {份额 1, 份额 $w_m$}，各从一个构造的不动点出发                     |
| `theory-anchor-basin`（T8）                      | 移动锚从随机初始点出发卡在非纳什点的频率（四个博弈族，每格 200 个 seed，有无估计误差）                 |
| `theory-sampling`（T9）                          | 用真实的池代码在矩阵博弈上打采样对局：按场、按对偶权重、各半三种采样基                              |
| `theory-pool-rules`（T10）                       | CoDA-R、移动锚、PFSP、均匀、Hedge 与纯近端自博弈：打平成员持有的权重与 sleeper 延迟           |
| `theory-oracle-rate`（T11）                      | 每步把锚的（eps-）BR 放进池、$c$ 取定理 9 允许的最大值：逐步检查定理 9，比较有无池与固定份额规则的收敛速度 |
| `theory-adversarial-pool`（T12）                 | 对手构造池（周期 oracle、关于纳什的补、随机成员灌水、纳什支撑上的纯策略、滞后 oracle），各规则的最终 exploitability；CoDA-R 逐步检查能量式 |
| `references`                                   | CFR+ 参考线                                                         |
| `tune-a`、`tune-b`                              | 两阶段调参，写出 `selected.json`                                         |
| `main`                                         | 12 种方法 x 3 个游戏 x 10 个 seed                                       |
| `ablations`                                    | CoDA-R 消融 A1-A9：移动锚、采样基、零权重成员发现机制、精确败率、oracle、镜像刷新、冻结系数、近端形式、池容量 |
| `vuln-repair`（M1）                              | 精确 BR 攻击者加诱饵，池保留 2 个快照；压制 $T\in\{24,48,96,192\}$ 个 rollout 后重置 learner，测重新给权重的延迟随 $T$ 的变化（sleeper）与闲置成员的训练份额 |
| `open-pool`（M2）                                | 脚本 bot、CFR+ 早期策略、小网络旧 agent、随机与重复成员，$N\in4,16,64$                |
| `hidden-hole`（M3）                              | CFR+ 近似纳什蒸馏成网络，在自博弈对手到达概率受控的信息集上植入最差动作；比较 R-NaD-PPO、PFSP 与 CoDA-R（池里只有受损网络的精确 BR）的修复时间 |
| `forgetting`（M4）                               | 同一个 CFR+ 近似纳什起点（不埋漏洞）；CoDA-R、PFSP、R-NaD-PPO、MMD-PPO 与自博弈按主协议继续训练，熵全程固定在调参终值；测 exploitability 上升、按对手到达概率分档的漂移，以及对考试用历史档案的败率；CoDA-R 另跑 $c\times0.5$ 与 $\times2$ 作敏感性分析 |
| `scale-train`、`scale-eval`、`scale-crossplay`、`scale-history` | 两个规模游戏上 6 种方法 x 5 个 seed 的训练；exp-a-spiel 全树精确 exploitability（约 30 GB，一次一个评估进程）；同游戏内循环赛；更晚的 checkpoint 对更早的采样对局（M4 规模臂） |

## 输出

每个实验目录包含 `cells.jsonl`（每个完成格子一行状态，含完整配置）、`summary.md`，通常还有 `records.json` 与图。格子目录包含 `rollouts.jsonl`（逐 rollout 日志，评估行带 `evaluation = true`）、`run_summary.json`、`oracles.jsonl`（每次入池及 exploiter 下界），以及需要时的 checkpoint。

## 分析与出图

`analysis/` 只读取实验输出，生成论文图与 `analysis.md`（相对全程最低点的退化、全部预先确定的配对（含 Q 对 NFSP）、规模实验的逐 seed 配对差，以及各调参维度的选定值是否落在网格边界）。每个方法在所有图里的颜色与线型固定（`analysis/style.py`）；对应实验没有输出的图直接跳过。

```powershell
python -m analysis --list                           # 可用的图
python -m analysis                                  # 全部，写到 analysis/outputs
python -m analysis main_curves tables --dest paper/figures
```

## 类型检查与测试

```powershell
python -m pyright experiment analysis
python -m pytest experiment/tests analysis/tests -q -p no:cacheprovider
python -m experiment all --smoke --workers 8
```
