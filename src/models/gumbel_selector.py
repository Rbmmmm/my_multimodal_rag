from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import gumbel_softmax

class GumbelModalSelector(nn.Module):
    """
    Gumbel-Softmax 模态选择器，支持可训练 / 冻结两种模式。
    输入: query embedding (B, D)
    输出: probs (B, K), y_onehot (B, K), logits (B, K)
    """
    def __init__(
        self,
        input_dim: int,
        num_choices: int = 2,
        hidden_dim: int = 0,
        tau: float = 1.0,
        trainable: bool = True,  # 新增：是否允许训练
    ):
        super().__init__()
        self.tau = float(tau)
        self.num_choices = int(num_choices)

        # 定义 MLP
        if hidden_dim and hidden_dim > 0:
            self.head = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, num_choices),
            )
        else:
            self.head = nn.Linear(input_dim, num_choices)

        # 控制是否训练
        self.set_trainable(trainable)

    def set_trainable(self, trainable: bool):
        """
        控制 selector 是否可训练
        """
        for p in self.parameters():
            p.requires_grad = trainable
        if not trainable:
            self.eval()
        else:
            self.train()

    def forward(self, x: torch.Tensor, hard: bool = True, bias: torch.Tensor | None = None):
        """
        x: (B, D); bias: (K,) 或 (B, K)，可为 None
        返回: probs, y_onehot, logits
        """
        device = self.head[0].weight.device if isinstance(self.head, nn.Sequential) else self.head.weight.device
        x = x.to(device).float()

        logits = self.head(x)
        if bias is not None:
            logits = logits + bias

        probs = F.softmax(logits, dim=-1)
        y = gumbel_softmax(logits, tau=self.tau, hard=hard, dim=-1)
        return probs, y, logits