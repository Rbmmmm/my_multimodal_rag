# 文件路径: my_multimodal_rag/src/models/gumbel_selector.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class GumbelModalSelector(nn.Module):
    """
    一个使用Gumbel-Softmax进行模态选择的决策器模块。
    """
    def __init__(self, input_dim: int, num_choices: int):
        """
        初始化模块。
        
        参数:
            input_dim (int): 输入嵌入向量的维度 (例如: 768)。
            num_choices (int): 可供选择的模态数量 (例如: 3，对应文本、图像、表格)。
        """
        super().__init__()
        # 定义内部的线性层，它负责从输入向量生成各个选项的“渴望分数”(logits)
        self.classifier = nn.Linear(input_dim, num_choices)
        
    def forward(self, x: torch.Tensor, temperature: float = 1.0, hard: bool = True) -> torch.Tensor:
        """
        定义前向传播过程。
        
        参数:
            x (torch.Tensor): 输入的查询嵌入张量，形状为 [batch_size, input_dim]。
            temperature (float): Gumbel-Softmax的温度参数。值越小，输出结果越接近one-hot。
            hard (bool): 如果为True，则前向传播的输出是one-hot向量。
                         在反向传播时，梯度都会被平滑地计算。
                         
        返回:
            torch.Tensor: 一个形状为 [batch_size, num_choices] 的独热向量，表示为每个查询选择的模态。
        """
        # 1. 通过分类器（线性层）计算得到每个选项的原始分数 (logits)
        logits = self.classifier(x)
        
        # 2. 应用Gumbel-Softmax技巧，得到最终的独热向量选择
        selection = F.gumbel_softmax(logits, tau=temperature, hard=hard)
        
        return selection