# File: my_multimodal_rag/src/models/gumbel_selector.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class GumbelModalSelector(nn.Module):
    """
    二分类模态选择器（text=0, image=1）。
    训练：Gumbel-Softmax（可退火）；
    推理：one-hot 选择（argmax），同时返回概率与logits。
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 0,     # =0 表示用单层线性；>0 则用一层 MLP
        num_choices: int = 2,    # 固定两模态：text / image
        tau: float = 1.0,        # 训练期温度
        dropout: float = 0.0
    ):
        super().__init__()
        assert num_choices == 2, "Stage A: only support 2 choices (text=0, image=1)."

        if hidden_dim and hidden_dim > 0:
            self.classifier = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_choices)
            )
        else:
            self.classifier = nn.Linear(input_dim, num_choices)

        self.tau = tau
        self.num_choices = num_choices

        # 简单初始化（可选）
        if isinstance(self.classifier, nn.Linear):
            nn.init.xavier_uniform_(self.classifier.weight)
            nn.init.zeros_(self.classifier.bias)
        else:
            for m in self.classifier.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.zeros_(m.bias)

    @torch.no_grad()
    def infer(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        推理用：返回 (probs, logits, choice)
        probs: [B, 2]，softmax 概率
        logits: [B, 2]
        choice: [B]，{0=text, 1=image}
        """
        logits = self.classifier(x)                       # [B, 2]
        probs = torch.softmax(logits, dim=-1)            # [B, 2]
        choice = torch.argmax(probs, dim=-1)             # [B]
        return probs, logits, choice

    def set_temperature(self, tau: float):
        self.tau = float(tau)

    def forward(
        self,
        x: torch.Tensor,
        training: bool = True,
        hard: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        训练/调试通用前向：
        - training=True: 使用 Gumbel-Softmax 采样（hard=False 输出连续概率；hard=True 输出one-hot并用STE反传）
        - training=False: 与 infer 相同（one-hot by argmax）
        返回 (probs_or_y, logits, choice)
        """
        logits = self.classifier(x)  # [B, 2]

        if training:
            y = F.gumbel_softmax(logits, tau=self.tau, hard=hard, dim=-1)  # [B, 2]
            choice = torch.argmax(y, dim=-1)
            return y, logits, choice
        else:
            # 推理：用 softmax 概率 + argmax one-hot 选择
            probs = torch.softmax(logits, dim=-1)
            choice = torch.argmax(probs, dim=-1)
            return probs, logits, choice