"""池成员败率的在线估计：局数自适应 EMA 加变点重置。

估计器只决定误差 e_k 的大小，不改变任何定理。只吃原始得分，塑形收益不得进入。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# 写入精确败率后视为无穷多局的可信度
EXACT_N_EFF = 1e9


@dataclass
class _Estimate:
    loss: float
    n_eff: float
    # 上一 rollout 样本均值高出估计且超出一个抽样标准差的部分：正在变强的成员的早期信号
    last_gain: float = 0.0


class LossEstimator:
    """
    每个成员一个估计。本 rollout 有 n 局、均值为 m 时：
    - 变点检验：|m - L| 超过 reset_k 个保守标准差 sqrt(0.25 / n + 0.25 / n_eff) 时丢弃历史，
      向中性先验 0.5 收缩后重新起步：L = (0.5 n0 + n m) / (n0 + n)。
    - 否则 EMA：beta = n / (n + n0)，有效窗口约 n + 2 n0 局，与 rollout 大小无关。
    """

    def __init__(self, n0: int = 64, reset_k: float = 2.5):
        if n0 <= 0:
            raise ValueError("n0 must be positive")
        self.n0 = int(n0)
        self.reset_k = float(reset_k)
        self._by_id: dict[str, _Estimate] = {}

    def add(self, member_id: str, initial_loss: float = 0.5) -> None:
        self._by_id[member_id] = _Estimate(loss = float(np.clip(initial_loss, 0.0, 1.0)), n_eff = float(self.n0))

    def remove(self, member_id: str) -> None:
        self._by_id.pop(member_id, None)

    def losses(self, ids: list[str]) -> np.ndarray:
        return np.array([self._by_id[i].loss for i in ids], dtype = np.float64)

    def n_eff(self, ids: list[str]) -> np.ndarray:
        return np.array([self._by_id[i].n_eff for i in ids], dtype = np.float64)

    def update(self, member_id: str, count: int, mean_loss: float) -> bool:
        """并入一批对局，返回是否触发了变点重置。"""
        est = self._by_id[member_id]
        if count <= 0:
            est.last_gain = 0.0
            return False
        noise = 0.5 / float(np.sqrt(count))
        est.last_gain = max(mean_loss - est.loss - noise, 0.0)
        sd = float(np.sqrt(0.25 / count + 0.25 / max(est.n_eff, 1.0)))
        if self.reset_k > 0.0 and abs(mean_loss - est.loss) > self.reset_k * sd:
            est.loss = (0.5 * self.n0 + count * mean_loss) / (self.n0 + count)
            est.n_eff = float(self.n0 + count)
            return True
        beta = count / (count + self.n0)
        est.loss = (1.0 - beta) * est.loss + beta * mean_loss
        est.n_eff = min(est.n_eff, float(self.n0)) + count
        return False

    def set_exact(self, member_id: str, loss: float) -> None:
        est = self._by_id[member_id]
        est.loss = float(np.clip(loss, 0.0, 1.0))
        est.n_eff = EXACT_N_EFF
        est.last_gain = 0.0

    def probe_priority(self, ids: list[str], z: float) -> np.ndarray:
        """
        探测优先级 = 上置信界超出 0.5 的幅度 + 上一轮显著正残差。
        前者盯住估计贴近阈值的成员，后者盯住估计很低但正在快速变强的成员。
        """
        ests = [self._by_id[i] for i in ids]
        upper = np.array([e.loss + z * 0.5 / np.sqrt(max(e.n_eff, 1.0)) for e in ests], dtype = np.float64)
        gain = np.array([e.last_gain for e in ests], dtype = np.float64)
        return np.maximum(upper - 0.5, 0.0) + gain
