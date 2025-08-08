# File: src/orchestrator.py

from __future__ import annotations
import torch
from typing import Optional, Callable, Dict, Any, Tuple

from src.agents.seeker_agent import SeekerAgent
from src.agents.inspector_agent import InspectorAgent
from src.agents.synthesizer_agent import SynthesizerAgent
from src.models.gumbel_selector import GumbelModalSelector


class RAGOrchestrator:
    """
    MM-RAG 编排：
      1) 模态选择（可强制 / Gumbel / 规则兜底）
      2) 检索 TopK=3
      3) 重排 + 置信度
      4) 证据不足 → TopK=5 再检索
      5) 生成
    """

    def __init__(
        self,
        seeker: Optional[SeekerAgent] = None,
        inspector: Optional[InspectorAgent] = None,
        synthesizer: Optional[SynthesizerAgent] = None,
        gumbel_selector: Optional[GumbelModalSelector] = None,
        use_modal_selector: bool = True,
        # factories 支持两类 key：
        #   - "seeker"/"inspector"/"synthesizer" -> 懒加载 agent
        #   - "text"/"image"/"chart"             -> 懒加载对应 retriever
        lazy_init_factories: Optional[Dict[str, Callable[[], Any]]] = None,
    ):
        self.seeker = seeker
        self.inspector = inspector
        self.synthesizer = synthesizer
        self.gumbel_selector = gumbel_selector
        self.use_modal_selector = use_modal_selector
        self.lazy_init_factories = lazy_init_factories or {}

        # 支持三模态：0=text, 1=image, 2=chart
        self.modality_map: Dict[int, str] = {0: "text", 1: "image", 2: "chart"}

    # ---------- 懒加载工具 ----------
    def _lazy_init_agent(self, name: str) -> None:
        if getattr(self, name, None) is None and name in self.lazy_init_factories:
            print(f"[Orchestrator] Lazy init agent: {name}")
            setattr(self, name, self.lazy_init_factories[name]())

    def _ensure_retriever(self, modality_index: int) -> None:
        """
        确保当前模态所需的 retriever 已创建（如果没有就用 factory 创建）。
        """
        modality = self.modality_map.get(modality_index, "text")
        attr = f"{modality}_retriever"

        # 先确保 seeker 存在
        self._lazy_init_agent("seeker")
        if self.seeker is None:
            raise RuntimeError("SeekerAgent is not initialized.")

        # 如果 retriever 缺失，用 factory 现建
        if getattr(self.seeker, attr, None) is None:
            if modality in self.lazy_init_factories:
                print(f"[Orchestrator] Lazy init retriever for modality: {modality}")
                retriever = self.lazy_init_factories[modality]()
                setattr(self.seeker, attr, retriever)
            else:
                # 这里软提示 + 更友好的报错信息
                have_keys = ", ".join(self.lazy_init_factories.keys()) or "(empty)"
                raise RuntimeError(
                    f"Retriever for modality '{modality}' is None, and no lazy factory provided.\n"
                    f"Expected key '{modality}' in lazy_init_factories. Current keys: {have_keys}\n"
                    f"Fix: 在 main.py 里为 '{modality}' 提供 factory，或在构造 SeekerAgent 时直接传入 {attr} 实例。"
                )

    # ---------- 选择模态 ----------
    def _choose_modality(
        self,
        query: str,
        query_embedding: torch.Tensor,
        force_modality: Optional[int] = None,
    ) -> Tuple[int, str]:
        if force_modality is not None:
            name = self.modality_map.get(force_modality, "unknown")
            return force_modality, f"forced={name} (debug)"

        modality_index = 0
        reason = "fallback=text"

        if (
            self.use_modal_selector
            and self.gumbel_selector is not None
            and isinstance(query_embedding, torch.Tensor)
        ):
            q = query_embedding
            if q.dim() == 1:
                q = q.unsqueeze(0)
            self.gumbel_selector.to(q.device)
            self.gumbel_selector.eval()
            with torch.no_grad():
                probs, logits, choice = self.gumbel_selector.infer(q)
                modality_index = int(choice.item())
                reason = f"gumbel_probs={probs.squeeze(0).tolist()}"

            # 简单规则兜底
            lower_q = query.lower()
            image_triggers = ("图表", "图片", "图像", "figure", "chart", "plot", "diagram", "screenshot")
            if modality_index == 1 and probs[0, 1] < 0.60 and not any(t in lower_q for t in image_triggers):
                modality_index = 0
                reason += " | rule-fallback=text"

            chart_triggers = ("chart", "table", "表格", "数据表", "柱状图", "折线图", "饼图", "layout", "2d layout")
            if probs.shape[-1] >= 3:
                if modality_index == 2 and probs[0, 2] < 0.60 and not any(t in lower_q for t in chart_triggers):
                    modality_index = 0
                    reason += " | rule-fallback=text"

        return modality_index, reason

    # ---------- 主流程 ----------
    def run(
        self,
        query: str,
        query_embedding: torch.Tensor,
        *,
        force_modality: Optional[int] = None,
    ):
        if not isinstance(query_embedding, torch.Tensor):
            raise TypeError("query_embedding must be a torch.Tensor")

        modality_index, reason = self._choose_modality(
            query=query, query_embedding=query_embedding, force_modality=force_modality
        )
        modality_name = self.modality_map.get(modality_index, "unknown")
        print(f"[Orchestrator] Selected modality: {modality_index} ({modality_name}) | {reason}")

        # 只在需要时初始化 agent / retriever
        self._lazy_init_agent("inspector")
        self._lazy_init_agent("synthesizer")
        self._ensure_retriever(modality_index)

        # 初次检索
        top_k = 3
        retrieved_nodes = self.seeker.run(query, modality=modality_index, top_k=top_k)

        # 重排 + 置信度
        status, information, nodes, confidence = self.inspector.run(query, retrieved_nodes)

        # 二次检索
        if status == "seeker":
            print("\n" + "=" * 20 + " 证据不足，进入二次检索 " + "=" * 20)
            top_k = 5
            print(f"动态扩展K值，新的TopK={top_k}")
            retrieved_nodes = self.seeker.run(
                query, modality=modality_index, top_k=top_k, feedback=information
            )
            status, information, nodes, confidence = self.inspector.run(query, retrieved_nodes)

        # 输出
        if status == "synthesizer":
            return self.synthesizer.generate(query, [n.node for n in nodes])
        else:
            return "经过多轮检索，仍未找到足够的信息来回答问题。"