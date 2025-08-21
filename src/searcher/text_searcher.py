# File: src/searcher/text_searcher.py
# Final version with persistent storage logic for LlamaIndex (Bi-Encoder mode).

import os
import json
from tqdm import tqdm
from typing import List, Any, Optional

# --- LlamaIndex Imports (新增 StorageContext 和 load_index_from_storage) ---
from llama_index.core import VectorStoreIndex, Document, StorageContext, load_index_from_storage
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.schema import TextNode, NodeWithScore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# --- Ragatouille/ColBERT Imports ---
os.environ["COLBERT_LOAD_TORCH_EXTENSION_DISABLE"] = "True"
from ragatouille import RAGPretrainedModel

# --- 您的项目依赖 ---
from src.searcher.base_searcher import BaseSearcher 
from src.utils.format_converter import nodefile2node

class TextSearcher(BaseSearcher):
    """
    一个统一的文本检索器。Bi-Encoder 模式现在支持高效的磁盘持久化。
    """
    def __init__(self, 
                 dataset_name: str, 
                 node_dir_prefix: Optional[str] = None,
                 mode: str = 'bi_encoder',
                 bi_encoder_model: str = "BAAI/bge-m3", 
                 colbert_model: str = "colbert-ir/colbertv2.0"):

        print(f"Initializing TextSearcher for dataset '{dataset_name}' in '{mode}' mode...")
        self.mode = mode
        self.dataset_dir = os.path.join('./data', dataset_name)
        
        if self.mode == 'bi_encoder':
            if node_dir_prefix is None:
                if 'bge' in bi_encoder_model: node_dir_prefix = 'bge_ingestion'
                elif 'NV-Embed' in bi_encoder_model: node_dir_prefix = 'nv_ingestion'
                else: raise ValueError(f"Could not infer node_dir_prefix for Bi-Encoder model '{bi_encoder_model}'.")
            
            self.node_dir = os.path.join(self.dataset_dir, node_dir_prefix)
            self.index_persist_dir = os.path.join(".llama_index_storage", f"{dataset_name}_{node_dir_prefix}")
            
            embed_model_obj = HuggingFaceEmbedding(model_name=bi_encoder_model)
            self.query_engine = self._build_llama_index_engine(embed_model_obj)
            print("✅ Bi-Encoder engine (LlamaIndex) is ready.")

        elif self.mode == 'colbert':
            # (ColBERT 模式的逻辑保持不变，它已经具备持久化功能)
            if node_dir_prefix is None: raise ValueError("node_dir_prefix must be provided for ColBERT mode.")
            self.node_dir = os.path.join(self.dataset_dir, node_dir_prefix)
            self.nodes = self._load_all_nodes_for_colbert(self.node_dir)
            if not self.nodes: raise ValueError(f"No nodes loaded from '{self.node_dir}'.")
            
            self.node_id_map = {node.id_: node for node in self.nodes}
            self.index_name = "vidorag_text_index"
            full_index_path = os.path.join(".ragatouille", "colbert", "indexes", self.index_name)

            if os.path.exists(full_index_path):
                print(f"✅ Found existing ColBERT index. Loading from '{full_index_path}'...")
                self.rag_model = RAGPretrainedModel.from_index(full_index_path)
            else:
                print(f"ColBERT index not found. Building a new one...")
                self.rag_model = RAGPretrainedModel.from_pretrained(colbert_model)
                self.rag_model.index(
                    collection=[node.get_content() for node in self.nodes],
                    document_ids=[node.id_ for node in self.nodes],
                    index_name=self.index_name
                )
            print("✅ ColBERT engine (Ragatouille) is ready.")
        else:
            raise ValueError(f"Invalid mode '{self.mode}'. Choose 'colbert' or 'bi_encoder'.")

    def _load_nodes_generator_for_llama_index(self):
        files = [f for f in os.listdir(self.node_dir) if f.endswith('.node')]
        for file in tqdm(files, desc=f"Streaming text nodes from {os.path.basename(self.node_dir)}"):
            input_file = os.path.join(self.node_dir, file)
            try:
                for node in nodefile2node(input_file):
                    yield Document(id_=node.id_, text=node.text or "", metadata=node.metadata or {})
            except Exception as e:
                print(f"\n❌ Error processing file: {input_file}\n{e}\n")

    def _build_llama_index_engine(self, embed_model: Any) -> RetrieverQueryEngine:
        """
        [修正] 使用 LlamaIndex 构建或从磁盘加载持久化的向量索引。
        改为从内存列表构建，确保索引被正确填充。
        """
        if os.path.exists(self.index_persist_dir):
            print(f"✅ Found existing LlamaIndex. Loading from '{self.index_persist_dir}'...")
            storage_context = StorageContext.from_defaults(persist_dir=self.index_persist_dir)
            vector_index = load_index_from_storage(storage_context, embed_model=embed_model)
        else:
            print(f"LlamaIndex storage not found at '{self.index_persist_dir}'. Building a new one...")
            os.makedirs(self.index_persist_dir, exist_ok=True)
            
            # --- 核心修正：先将生成器内容完整加载到列表 ---
            print("Loading all nodes into a list before indexing...")
            nodes_generator = self._load_nodes_generator_for_llama_index()
            nodes_list = list(nodes_generator) # 将所有节点加载到内存
            print(f"✅ Loaded {len(nodes_list)} nodes into list.")
            
            if not nodes_list:
                raise ValueError("No nodes were loaded. Cannot build an empty index.")

            # --- 从列表构建索引 ---
            vector_index = VectorStoreIndex(
                nodes_list, # 使用内存列表
                embed_model=embed_model,
                show_progress=True
            )
            print("Persisting index to disk...")
            vector_index.storage_context.persist(persist_dir=self.index_persist_dir)
            print(f"✅ New LlamaIndex built and persisted to '{self.index_persist_dir}'.")

        vector_retriever = vector_index.as_retriever()
        return RetrieverQueryEngine.from_args(retriever=vector_retriever, llm=None)
    
    def _load_all_nodes_for_colbert(self, node_dir: str) -> List[TextNode]:
        all_nodes = []
        for filename in os.listdir(node_dir):
            if filename.endswith(".node"):
                file_path = os.path.join(node_dir, filename)
                all_nodes.extend(nodefile2node(file_path))
        return all_nodes

    def search(self, query: str, top_k: int) -> List[NodeWithScore]:
        print(f"\n--- Executing search in '{self.mode}' mode (TopK={top_k}) ---")
        if self.mode == 'bi_encoder':
            self.query_engine.retriever.similarity_top_k = top_k
            return self.query_engine.retrieve(query)
        elif self.mode == 'colbert':
            results = self.rag_model.search(query=query, k=top_k, index_name=self.index_name)
            final_nodes = []
            if results:
                for result in results:
                    original_node = self.node_id_map.get(result['document_id'])
                    if original_node:
                        final_nodes.append(NodeWithScore(node=original_node, score=result['score']))
            return final_nodes
        return []