# 文件路径: my_multimodal_rag/src/agents/seeker_agent.py (最终版)

from src.retrievers.text_retriever import TextRetriever
from src.retrievers.image_retriever import ImageRetriever
from src.retrievers.table_retriever import TableRetriever

class SeekerAgent:
    def __init__(self, text_retriever: TextRetriever, image_retriever: ImageRetriever, table_retriever: TableRetriever):
        """
        Unified Seeker Agent that manages coarse-grained retrieval across three modalities.
        """
        self.retrievers = {
            0: text_retriever,  # 0 = text
            1: image_retriever, # 1 = image
            2: table_retriever  # 2 = table
        }
        self.modality_map = {0: "text", 1: "image", 2: "table"}

    def run(self, query: str, modality: int, top_k: int = 3, feedback: str = None):
        """
        Perform coarse retrieval based on the modality index chosen by the Gumbel network.

        :param query: Original user query.
        :param modality: Modality index selected by the Gumbel network (0, 1, or 2).
        :param top_k: Number of results to retrieve.
        :param feedback: Optional feedback from the Inspector agent.
        :return: List of retrieved nodes (NodeWithScore objects).
        """
        if modality not in self.retrievers:
            raise ValueError(f"Unknown modality index: {modality}")

        current_query = query
        if feedback:
            print(f"[Seeker] Received feedback, refining query: {feedback}")
            current_query = f"{query}\nRefined search based on feedback: {feedback}"
        
        selected_retriever = self.retrievers[modality]
        modality_name = self.modality_map[modality]
        
        print(f"\n[Seeker] Selected modality: [{modality_name}]")
        retrieved_nodes = selected_retriever.retrieve(current_query, top_k=top_k)
        
        print(f"✅ Seeker retrieved {len(retrieved_nodes)} initial results from [{modality_name}] retriever.")
        return retrieved_nodes