# from __future__ import annotations
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.nn.functional import gumbel_softmax

# class GumbelModalSelector(nn.Module):
#     """
#     Gumbel-Softmax 模态选择器，支持可训练 / 冻结两种模式。
#     输入: query embedding (B, D)
#     输出: probs (B, K), y_onehot (B, K), logits (B, K)
#     """
#     def __init__(
#         self,
#         input_dim: int,
#         num_choices: int = 2,
#         hidden_dim: int = 0,
#         tau: float = 1.0,
#         trainable: bool = True,  # 新增：是否允许训练
#     ):
#         super().__init__()
#         self.tau = float(tau)
#         self.num_choices = int(num_choices)

#         # 定义 MLP
#         if hidden_dim and hidden_dim > 0:
#             self.head = nn.Sequential(
#                 nn.Linear(input_dim, hidden_dim),
#                 nn.ReLU(inplace=True),
#                 nn.Linear(hidden_dim, num_choices),
#             )
#         else:
#             self.head = nn.Linear(input_dim, num_choices)

#         # 控制是否训练
#         self.set_trainable(trainable)

#     def set_trainable(self, trainable: bool):
#         """
#         控制 selector 是否可训练
#         """
#         for p in self.parameters():
#             p.requires_grad = trainable
#         if not trainable:
#             self.eval()
#         else:
#             self.train()

#     def forward(self, x: torch.Tensor, hard: bool = True, bias: torch.Tensor | None = None):
#         """
#         x: (B, D); bias: (K,) 或 (B, K)，可为 None
#         返回: probs, y_onehot, logits
#         """
#         device = self.head[0].weight.device if isinstance(self.head, nn.Sequential) else self.head.weight.device
#         x = x.to(device).float()

#         logits = self.head(x)
#         if bias is not None:
#             logits = logits + bias

#         probs = F.softmax(logits, dim=-1)
#         y = gumbel_softmax(logits, tau=self.tau, hard=hard, dim=-1)
#         return probs, y, logits

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple

class GumbelModalSelector(nn.Module):
    """
    Gumbel-Softmax 模态选择器。

    该模块接收一个查询嵌入 (query embedding)，并输出一个表示模态选择的 one-hot 向量。
    - 内部使用多层感知机 (MLP) 增加模型的表达能力。
    - 训练时，它利用 Gumbel-Softmax 技巧进行可微分的类别采样。
    - 推理时，它等效于一个标准的分类器，使用 argmax 来进行确定性的选择。
    """
    def __init__(
        self,
        input_dim: int,
        num_choices: int,
        hidden_dim: int = 256,
        tau: float = 1.0
    ):
        """
        初始化 GumbelModalSelector。

        参数:
            input_dim (int): 输入嵌入的维度。
            num_choices (int): 可选模态的数量。
            hidden_dim (int): MLP 中间隐藏层的维度。
            tau (float): Gumbel-Softmax 的温度参数，训练时使用。
        """
        super().__init__()
        self.tau = tau
        self.num_choices = num_choices

        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),  # Dropout 只在 .train() 模式下生效
            nn.Linear(hidden_dim, num_choices)
        )

    def forward(
        self,
        x: torch.Tensor,
        hard: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        模型的前向传播。

        参数:
            x (torch.Tensor): 输入的查询嵌入，形状为 (B, D)，其中 B 是批量大小，D 是嵌入维度。
            hard (bool): 是否使用 Gumbel-Softmax 的 hard 模式（返回 one-hot 向量）。

        返回:
            一个元组 (probs, y_onehot, logits):
            - probs (torch.Tensor): Softmax 后的概率分布，形状为 (B, K)，用于计算准确率或调试。
            - y_onehot (torch.Tensor): 最终的选择结果，形状为 (B, K)。
                                      训练时是 Gumbel-Softmax 的输出，推理时是 argmax 的 one-hot 结果。
            - logits (torch.Tensor): MLP 的原始输出，形状为 (B, K)，用于计算交叉熵损失。
        """
        logits = self.head(x)

        # 1. 计算用于分类和调试的概率
        probs = F.softmax(logits, dim=-1)

        # 2. 核心选择逻辑
        if not self.training:
            # 推理模式 (eval mode): 直接使用 argmax，结果是确定的
            # F.one_hot 会生成类似 [0, 1, 0] 的 one-hot 向量
            indices = torch.argmax(logits, dim=-1)
            y_onehot = F.one_hot(indices, num_classes=self.num_choices).float()
        else:
            # 训练模式 (train mode): 使用 Gumbel-Softmax 进行随机但可微的采样
            y_onehot = F.gumbel_softmax(logits, tau=self.tau, hard=hard, dim=-1)

        return probs, y_onehot, logits


def choose_modality(
    selector_model: GumbelModalSelector,
    query_embedding: torch.Tensor,
    modality_map: Dict[int, str]
) -> int:
    """
    使用训练好的 GumbelModalSelector 来选择模态。

    这是一个封装好的辅助函数，可以方便地在 Orchestrator 中调用。

    参数:
        selector_model (GumbelModalSelector): 一个已经加载了权重的 Gumbel 选择器模型实例。
        query_embedding (torch.Tensor): 单个查询的嵌入向量，形状为 (D,) 或 (1, D)。
        modality_map (Dict[int, str]): 一个用于打印日志的索引到模态名称的映射字典。

    返回:
        int: 被选中的模态的索引 (index)。
    """
    # 确保模型处于评估模式
    selector_model.eval()

    # 获取模型所在的设备
    device = next(selector_model.parameters()).device
    
    # 准备输入张量，确保有 batch 维度并移动到正确的设备
    x = query_embedding.clone().detach()
    if x.dim() == 1:
        x = x.unsqueeze(0)
    x = x.to(device)

    # 在不计算梯度的上下文中进行推理
    with torch.no_grad():
        probs, _, _ = selector_model(x)

    chosen_index = torch.argmax(probs[0]).item()

    # 打印详细的概率日志，便于调试
    prob_list = probs[0].cpu().tolist()
    prob_str = ", ".join([f"{modality_map.get(i, 'unk')}={p:.3f}" for i, p in enumerate(prob_list)])
    print(f"[ModalChooser] Probs: [{prob_str}] -> Chosen: {modality_map.get(chosen_index, 'unk').upper()}")

    return chosen_index


# ===============================================================
# 这是一个使用示例，您可以运行 `python gumbel_selector.py` 来测试
# ===============================================================
if __name__ == '__main__':
    # 1. 定义超参数
    INPUT_DIM = 128
    NUM_MODALITIES = 3
    HIDDEN_DIM = 64
    
    # 2. 定义模态映射
    MODALITY_MAP = {0: "text", 1: "image", 2: "chart"}

    # 3. 创建一个未训练的模型实例
    print("--- 测试一个未训练的模型 ---")
    dummy_selector = GumbelModalSelector(
        input_dim=INPUT_DIM,
        num_choices=NUM_MODALITIES,
        hidden_dim=HIDDEN_DIM
    )
    
    # 4. 创建一个假的查询嵌入
    dummy_query_embedding = torch.randn(INPUT_DIM)
    
    # 5. 调用选择函数
    print(f"输入一个形状为 {dummy_query_embedding.shape} 的随机嵌入:")
    selected_index = choose_modality(
        selector_model=dummy_selector,
        query_embedding=dummy_query_embedding,
        modality_map=MODALITY_MAP
    )
    print(f"函数返回的索引是: {selected_index}")

    # 6. 模拟加载了权重的模型 (这里我们只是再创建一个实例)
    print("\n--- 模拟一个加载了权重的模型 (行为应该和上面一样稳定) ---")
    # 假设这个模型已经 torch.load() 了权重
    # loaded_selector = ...
    # 在这个例子中，我们只是重新用同一个模型
    another_dummy_query = torch.tensor([0.1] * INPUT_DIM)
    print(f"输入一个形状为 {another_dummy_query.shape} 的固定嵌入:")
    
    # 多次调用，结果应该是相同的，因为 .eval() 和 torch.no_grad() 保证了确定性
    for i in range(3):
        print(f"第 {i+1} 次调用:")
        selected_index = choose_modality(dummy_selector, another_dummy_query, MODALITY_MAP)