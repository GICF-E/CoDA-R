"""共享的双层策略/价值网络，前向时屏蔽非法动作。"""
from __future__ import annotations

import torch
import torch.nn as nn


class SmallMLP(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 128):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.pi = nn.Linear(hidden, n_actions)
        self.v = nn.Linear(hidden, 1)

    def forward(
        self,
        obs: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.body(obs)
        logits = self.pi(h)
        valid = mask.bool()
        has_legal = valid.any(dim = -1, keepdim = True)
        logits = logits.masked_fill(~valid, -1e9)
        # 全非法行回退为有限 logits，避免空掩码造成分布数值异常。
        logits = torch.where(has_legal, logits, torch.zeros_like(logits))
        value = self.v(h).squeeze(-1)
        return logits, value
