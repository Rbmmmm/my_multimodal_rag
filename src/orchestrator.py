# File: src/orchestrator.py

import torch
from typing import Optional
from src.agents.seeker_agent import SeekerAgent
from src.agents.inspector_agent import InspectorAgent
from src.agents.synthesizer_agent import SynthesizerAgent
from src.models.gumbel_selector import GumbelModalSelector


class RAGOrchestrator:
    def __init__(
        self,
        seeker: SeekerAgent,
        inspector: InspectorAgent,
        synthesizer: SynthesizerAgent,
        gumbel_selector: Optional[GumbelModalSelector] = None,
        use_modal_selector: bool = True,
    ):
        self.seeker = seeker
        self.inspector = inspector
        self.synthesizer = synthesizer
        self.gumbel_selector = gumbel_selector
        self.use_modal_selector = use_modal_selector

        # 当前两模态：0=text, 1=image
        self.modality_map = {0: "text", 1: "image"}

    def run(self, query: str, query_embedding: torch.Tensor):
        """
        主流程：
          1) 模态选择（Gumbel + 关键词兜底）
          2) 检索 TopK=3
          3) Rerank + 置信度
          4) 若不足 → 扩 K 到 5 再检索
          5) 生成或兜底提示
        """
        # ---- 1) 准备 embedding ----
        if not isinstance(query_embedding, torch.Tensor):
            raise TypeError("query_embedding must be a torch.Tensor")
        q_emb = query_embedding if query_embedding.dim() == 2 else query_embedding.unsqueeze(0)  # [1, D]

        # 默认 text
        modality_index = 0
        reason = "fallback=text"

        # ---- 2) 模态决策 ----
        if self.use_modal_selector and self.gumbel_selector is not None:
            self.gumbel_selector.to(q_emb.device)
            self.gumbel_selector.eval()
            with torch.no_grad():
                # 统一使用 infer()（返回 probs/logits/choice）
                probs, logits, choice = self.gumbel_selector.infer(q_emb)  # [1,2], [1,2], [1]
                modality_index = int(choice.item())
                reason = f"gumbel_probs={probs.squeeze(0).tolist()}"

            # ---- 3) 规则兜底（避免误选 image）----
            lower_q = query.lower()
            image_triggers = ("图表", "图片", "图像", "figure", "chart", "plot", "diagram")
            if modality_index == 1 and probs[0, 1] < 0.6 and not any(t in lower_q for t in image_triggers):
                modality_index = 0
                reason += " | rule-fallback=text"

        modality_name = self.modality_map.get(modality_index, "unknown")
        print(f"[Orchestrator] Selected modality: {modality_index} ({modality_name}) | {reason}")

        # ---- 4) 初次检索 ----
        top_k = 3
        retrieved_nodes = self.seeker.run(query, modality=modality_index, top_k=top_k)

        # ---- 5) Rerank + 置信度 ----
        status, information, nodes, confidence = self.inspector.run(query, retrieved_nodes)

        # ---- 6) 动态扩展检索 ----
        if status == 'seeker':
            print("\n" + "=" * 20 + " 证据不足，进入二次检索 " + "=" * 20)
            top_k = 5
            print(f"动态扩展K值，新的TopK={top_k}")
            retrieved_nodes = self.seeker.run(
                query, modality=modality_index, top_k=top_k, feedback=information
            )
            status, information, nodes, confidence = self.inspector.run(
                query, retrieved_nodes
            )

        # ---- 7) 最终输出 ----
        if status == 'synthesizer':
            # Inspector 返回的是 NodeWithScore；生成器需要 BaseNode
            return self.synthesizer.generate(query, [n.node for n in nodes])
        else:
            return "经过多轮检索，仍未找到足够的信息来回答问题。"