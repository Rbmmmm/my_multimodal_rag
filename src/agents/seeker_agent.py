# File: my_multimodal_rag/src/agents/seeker_agent.py

from __future__ import annotations
from typing import Optional, Dict, List
from llama_index.core.schema import NodeWithScore

from src.retrievers.text_retriever import TextRetriever
from src.retrievers.image_retriever import ImageRetriever
from src.retrievers.chart_retriever import ChartRetriever


class SeekerAgent:
    """
    Unified Seeker Agent that dispatches coarse-grained retrieval across modalities.
    Modalities:
      0 -> text
      1 -> image
      2 -> chart  (以前叫 table，这里统一更名为 chart)
    """

    def __init__(
        self,
        text_retriever: TextRetriever,
        image_retriever: ImageRetriever,
        chart_retriever: ChartRetriever,
    ):
        self.retrievers: Dict[int, object] = {
            0: text_retriever,
            1: image_retriever,
            2: chart_retriever,
        }
        self.modality_map = {0: "text", 1: "image", 2: "chart"}

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
        if modality not in self.retrievers:
            raise ValueError(f"Unknown modality index: {modality}")

        retriever = self.retrievers[modality]
        modality_name = self.modality_map[modality]

        if retriever is None:
            print(f"⚠️ [{modality_name}] retriever is None, returning empty list.")
            return []

        current_query = query
        if feedback:
            print(f"[Seeker] Received feedback, refining query: {feedback}")
            current_query = f"{query}\n{feedback}"

        print(f"\n[Seeker] Selected modality: [{modality_name}]")
        results = retriever.retrieve(current_query, top_k=top_k)

        print(f"✅ Seeker retrieved {len(results)} initial results from [{modality_name}] retriever.")
        return results