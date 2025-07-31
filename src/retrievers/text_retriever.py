# 文件路径: my_multimodal_rag/src/retrievers/text_retriever.py

import os
import json
from typing import List
from llama_index.core.schema import TextNode, NodeWithScore
from ragatouille import RAGPretrainedModel  # Import RAGatouille

def nodefile2node(path: str) -> List[TextNode]:
    """
    Helper function to load and parse .node files into TextNode objects.
    """
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, list):
            return [TextNode.from_dict(d) for d in data]
        else:
            return [TextNode.from_dict(data)]
    except Exception as e:
        print(f"Warning: Failed to load or parse node file {os.path.basename(path)}: {e}")
        return []

class TextRetriever:
    """
    TextRetriever - ColBERT enhanced version
    
    Responsibilities:
    1. Load preprocessed text node data.
    2. Build a late-interaction index using RAGatouille and ColBERT.
    3. Provide an interface to retrieve top-k most relevant nodes based on query strings.
    """
    def __init__(self, node_dir: str, colbert_model_name: str = "colbert-ir/colbertv2.0"):
        """
        Initialize the ColBERT-based text retriever.
        :param node_dir: Directory containing preprocessed .node files.
        :param colbert_model_name: Name of the ColBERT model.
        """
        print(f"Loading nodes from directory '{node_dir}'...")
        self.nodes = self._load_nodes(node_dir)
        if not self.nodes:
            raise ValueError(f"No nodes loaded from '{node_dir}'. Please check the path and files.")
        print(f"✅ Successfully loaded {len(self.nodes)} nodes.")
        
        # --- ColBERT-specific initialization ---
        print(f"Loading ColBERT model: {colbert_model_name}...")
        self.rag_model = RAGPretrainedModel.from_pretrained(colbert_model_name)
        print("✅ ColBERT model loaded successfully.")

        print("Building ColBERT index for nodes. This may take a while...")
        # Create a mapping from document ID to the original node
        self.node_id_map = {str(i): node for i, node in enumerate(self.nodes)}
        
        # Prepare the content and document IDs for indexing
        documents_to_index = [node.get_content() for node in self.nodes]
        document_ids_to_index = list(self.node_id_map.keys())
        
        # Define the index path
        index_path = os.path.join(node_dir, ".colbert_index")
        
        # Build the index
        self.rag_model.index(
            collection=documents_to_index,
            document_ids=document_ids_to_index,
            index_name="vidorag_text_index",
            index_path=index_path,
            max_document_length=512,
            bsize=32
        )
        print(f"✅ ColBERT index built at: {index_path}")

    def _load_nodes(self, node_dir: str) -> List[TextNode]:
        """Private method to load all .node files from a directory."""
        all_nodes = []
        for filename in os.listdir(node_dir):
            if filename.endswith(".node"):
                file_path = os.path.join(node_dir, filename)
                all_nodes.extend(nodefile2node(file_path))
        return all_nodes

    def retrieve(self, query_str: str, top_k: int = 3) -> List[NodeWithScore]:
        """
        Perform ColBERT-based retrieval for a given text query.
        :param query_str: The user query.
        :param top_k: Number of top results to return.
        :return: List of NodeWithScore objects.
        """
        print(f"\nRunning ColBERT text retrieval (TopK={top_k})")
        results = self.rag_model.search(query=query_str, k=top_k)
        
        # Convert results into NodeWithScore format
        final_nodes = []
        for result in results:
            node_id = result['document_id']
            original_node = self.node_id_map.get(node_id)
            if original_node:
                final_nodes.append(NodeWithScore(node=original_node, score=result['score']))
        
        return final_nodes