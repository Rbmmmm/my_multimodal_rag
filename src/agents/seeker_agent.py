# File: src/agents/seeker_agent.py
from __future__ import annotations
from typing import Optional, Dict, List, Tuple, Any
from llama_index.core.schema import NodeWithScore

# 仅用于类型标注，实际运行时不会强制依赖
try:
    from src.retrievers.text_retriever import TextRetriever
    from src.retrievers.image_retriever import ImageRetriever
    from src.retrievers.chart_retriever import ChartRetriever
except Exception:
    TextRetriever = Any  # type: ignore
    ImageRetriever = Any  # type: ignore
    ChartRetriever = Any  # type: ignore


class SeekerAgent:
    """
    Unified Seeker Agent that dispatches coarse-grained retrieval across modalities.
    Modalities:
      0 -> text
      1 -> image
      2 -> chart
    支持在运行时通过 Orchestrator 懒加载回填各 retriever。
    """

    def __init__(
        self,
        text_retriever: Optional[TextRetriever] = None,
        image_retriever: Optional[ImageRetriever] = None,
        chart_retriever: Optional[ChartRetriever] = None,
    ):
        # 同名属性（便于 Orchestrator 用 setattr 回填）
        self.text_retriever = text_retriever
        self.image_retriever = image_retriever
        self.chart_retriever = chart_retriever

        # 维护索引 -> retriever 的映射
        self._retrievers_by_idx: Dict[int, Optional[object]] = {
            0: text_retriever,
            1: image_retriever,
            2: chart_retriever,
        }
        self._modality_map = {0: "text", 1: "image", 2: "chart"}

    # ---- 公共方法：给 Orchestrator 懒加载回填用 ----
    def set_retriever(self, name: str, retriever: object) -> None:
        """
        回填/替换指定模态的 retriever，并同步内部字典。
        name: "text" | "image" | "chart"
        """
        if name not in ("text", "image", "chart"):
            raise ValueError(f"Unknown retriever name: {name}")

        setattr(self, f"{name}_retriever", retriever)

        # 同步 _retrievers_by_idx
        idx = {v: k for k, v in self._modality_map.items()}[name]
        self._retrievers_by_idx[idx] = retriever

    # ---- 内部工具：根据模态号拿 retriever（优先读同名属性，确保回填后立刻生效）----
    def _get_retriever(self, modality: int) -> Tuple[Optional[object], str]:
        if modality not in self._modality_map:
            return None, "unknown"
        name = self._modality_map[modality]

        retriever = getattr(self, f"{name}_retriever", None)
        if retriever is None:
            retriever = self._retrievers_by_idx.get(modality)
        return retriever, name

    # ---- 主流程 ----
    def run(
        self,
        query: str,
        modality: int,
        top_k: int = 3,
        feedback: Optional[str] = None,
    ) -> List[NodeWithScore]:
        """
        Perform coarse retrieval based on the selected modality.

        :param query: Original user query.
        :param modality: 0=text, 1=image, 2=chart
        :param top_k: Number of results to retrieve.
        :param feedback: Optional feedback from the Inspector agent.
        :return: List[NodeWithScore]
        """
        retriever, modality_name = self._get_retriever(modality)

        if retriever is None:
            # 这里只提示；真正的懒加载应由 Orchestrator 保证在调用前完成
            print(f"⚠️ [{modality_name}] retriever is None, returning empty list.")
            return []

        # 二次检索 **不再** 拼接长反馈，避免噪声与长度问题
        current_query = query
        if feedback:
            print(f"[Seeker] Received feedback (omitted from query to reduce noise): {feedback}")
            # 如需轻微提示，可改为极短 focus 语句；默认直接用原 query
            # current_query = f"{query}\nFocus: prefer figure captions and section headers."

        print(f"\n[Seeker] Selected modality: [{modality_name}]")
        results: List[NodeWithScore] = retriever.retrieve(current_query, top_k=top_k)

        print(f"✅ Seeker retrieved {len(results)} initial results from [{modality_name}] retriever.")
        return results