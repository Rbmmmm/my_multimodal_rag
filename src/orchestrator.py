# File: src/orchestrator.py

from __future__ import annotations
import torch
from typing import Optional, Callable, Dict
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
        lazy_init_factories: Optional[Dict[str, Callable[[], object]]] = None,  # ← 关键：传入 factories
    ):
        self.seeker = seeker
        self.inspector = inspector
        self.synthesizer = synthesizer
        self.gumbel_selector = gumbel_selector
        self.use_modal_selector = use_modal_selector
        self.lazy_init_factories = lazy_init_factories or {}

        # 0=text, 1=image, 2=chart
        self.modality_map = {0: "text", 1: "image", 2: "chart"}

    def _ensure_retriever(self, modality_index: int):
        """
        懒加载并“回填”到 SeekerAgent 上。
        """
        name = self.modality_map.get(modality_index)
        if name is None:
            raise ValueError(f"Unknown modality index: {modality_index}")

        attr = f"{name}_retriever"
        retriever = getattr(self.seeker, attr, None)
        if retriever is not None:
            return retriever

        # 没有的话，用 factory 创建并回填
        factory = self.lazy_init_factories.get(name)
        if factory is None:
            raise RuntimeError(
                f"Retriever for modality '{name}' is None and no lazy factory provided."
            )
        print(f"[Orchestrator] Lazy init retriever for modality: {name}")
        retriever = factory()
        setattr(self.seeker, attr, retriever)  # ← 关键：回填给 SeekerAgent
        return retriever

    def _choose_modality(self, query: str, query_embedding: torch.Tensor, force_modality: Optional[int] = None):
        if force_modality is not None:
            name = self.modality_map.get(force_modality, "unknown")
            return force_modality, f"forced={name} (debug)"
        # …你之前的选择逻辑保持不变…
        return 0, "fallback=text"

    def run(self, query: str, query_embedding: torch.Tensor, *, force_modality: Optional[int] = None):
        if not isinstance(query_embedding, torch.Tensor):
            raise TypeError("query_embedding must be a torch.Tensor")

        modality_index, reason = self._choose_modality(query, query_embedding, force_modality=force_modality)
        modality_name = self.modality_map.get(modality_index, "unknown")
        print(f"[Orchestrator] Selected modality: {modality_index} ({modality_name}) | {reason}")

        # **在调用 Seeker 之前，确保 retriever 存在并已回填**
        self._ensure_retriever(modality_index)

        if modality_name == "image":
            top_k = 20
        else:
            top_k = 3
        
        # top_k = 3
            
        retrieved_nodes = self.seeker.run(query, modality=modality_index, top_k=top_k)

        status, information, nodes, confidence = self.inspector.run(query, retrieved_nodes)

        if status == "seeker":
            print("\n" + "=" * 20 + " 证据不足，进入二次检索 " + "=" * 20)
            # top_k = 5
            top_k = 30 if modality_name == "image" else 5
            print(f"动态扩展K值，新的TopK={top_k}")
            retrieved_nodes = self.seeker.run(query, modality=modality_index, top_k=top_k, feedback=information)
            status, information, nodes, confidence = self.inspector.run(query, retrieved_nodes)

        if status == "synthesizer":
            return self.synthesizer.generate(query, [n.node for n in nodes])
        else:
            return "经过多轮检索，仍未找到足够的信息来回答问题。"