# 文件: src/retrievers/text_retriever.py
# 最终、最稳妥的修正版本

import os
os.environ["COLBERT_LOAD_TORCH_EXTENSION_DISABLE"] = "True"
import json
from typing import List
from llama_index.core.schema import TextNode, NodeWithScore
from ragatouille import RAGPretrainedModel

def nodefile2node(path: str) -> List[TextNode]:
    # ... 此辅助函数无需改变 ...
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
    """
    def __init__(self, node_dir: str, colbert_model_name: str = "colbert-ir/colbertv2.0"):
        print(f"Loading nodes from directory '{node_dir}'...")
        self.nodes = self._load_nodes(node_dir)
        if not self.nodes:
            raise ValueError(f"No nodes loaded from '{node_dir}'. Please check the path and files.")
        print(f"✅ Successfully loaded {len(self.nodes)} nodes.")
        
        # --- ❗️❗️❗️ 关键修正 1: 使用 Node 自带的 ID 创建映射 ---
        # LlamaIndex 的 TextNode 自带一个 UUID 格式的 id_ 属性。
        # 我们用这个真实的 ID 来创建我们的查询映射。
        self.node_id_map = {node.id_: node for node in self.nodes}

        self.index_name = "vidorag_text_index"
        full_index_path = os.path.join(".ragatouille", "colbert", "indexes", self.index_name)

        if os.path.exists(full_index_path):
            print(f"✅ Found existing ColBERT index at '{full_index_path}'. Loading model and index from disk...")
            self.rag_model = RAGPretrainedModel.from_index(full_index_path)
            print("✅ Model and index loaded successfully.")
        else:
            print(f"ColBERT index not found at '{full_index_path}'. Building a new one...")
            self.rag_model = RAGPretrainedModel.from_pretrained(colbert_model_name)
            
            documents_to_index = [node.get_content() for node in self.nodes]
            # --- ❗️❗️❗️ 关键修正 2: 将 Node 自带的 ID 传递给索引器 ---
            # 这样可以确保索引内部存储的 ID 和我们的映射键完全一致。
            document_ids_to_index = [node.id_ for node in self.nodes]
            
            self.rag_model.index(
                collection=documents_to_index,
                document_ids=document_ids_to_index, # 明确提供 UUID 列表
                index_name=self.index_name,
                max_document_length=512,
                bsize=32,
                use_faiss=True
            )
            print(f"✅ ColBERT index built successfully.")

    def _load_nodes(self, node_dir: str) -> List[TextNode]:
        # ... 此方法无需改变 ...
        all_nodes = []
        for filename in os.listdir(node_dir):
            if filename.endswith(".node"):
                file_path = os.path.join(node_dir, filename)
                all_nodes.extend(nodefile2node(file_path))
        return all_nodes

    def retrieve(self, query_str: str, top_k: int = 3) -> List[NodeWithScore]:
        print(f"\nRunning ColBERT text retrieval (TopK={top_k})")
        
        results = self.rag_model.search(
            query=query_str,
            k=top_k,
            index_name=self.index_name
        )
        
        final_nodes = []
        if results:
            for result in results:
                # --- ❗️❗️❗️ 关键修正 3: 直接使用返回的字符串 ID 进行查询 ---
                # 因为返回的 ID ('d40ad664-...') 本身就是我们 map 中的键，所以不再需要 int() 转换。
                node_id = result['document_id']
                original_node = self.node_id_map.get(node_id)
                if original_node:
                    final_nodes.append(NodeWithScore(node=original_node, score=result['score']))
        
        return final_nodes