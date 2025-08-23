# File: src/searcher/text_searcher.py

import os
import json
from tqdm import tqdm
from typing import List, Any, Optional

from llama_index.core import VectorStoreIndex, Document, StorageContext, load_index_from_storage
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.schema import TextNode, NodeWithScore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

os.environ["COLBERT_LOAD_TORCH_EXTENSION_DISABLE"] = "True"
from ragatouille import RAGPretrainedModel

from src.searcher.base_searcher import BaseSearcher 
from src.utils.format_converter import nodefile2node

class TextSearcher(BaseSearcher):

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
            print("[TextSearcher] ✅ Bi-Encoder engine (LlamaIndex) is ready.")

        elif self.mode == 'colbert':
            if node_dir_prefix is None: raise ValueError("node_dir_prefix must be provided for ColBERT mode.")
            self.node_dir = os.path.join(self.dataset_dir, node_dir_prefix)
            self.nodes = self._load_all_nodes_for_colbert(self.node_dir)
            if not self.nodes: raise ValueError(f"No nodes loaded from '{self.node_dir}'.")
            
            self.node_id_map = {node.id_: node for node in self.nodes}
            # [MODIFIED] 使索引名称与数据集关联，避免冲突
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
        
        # [NEW] 新增 hybrid 模式的初始化逻辑
        elif self.mode == 'hybrid':
            # --- 1. 初始化 Bi-Encoder ---
            print("--- Initializing Hybrid Mode: Bi-Encoder ---")
            if node_dir_prefix is None:
                if 'bge' in bi_encoder_model: node_dir_prefix = 'bge_ingestion'
                elif 'NV-Embed' in bi_encoder_model: node_dir_prefix = 'nv_ingestion'
                else: raise ValueError(f"Could not infer node_dir_prefix for Bi-Encoder model '{bi_encoder_model}'.")
            
            # 假设两种模式使用相同的 node_dir
            self.node_dir = os.path.join(self.dataset_dir, node_dir_prefix)
            self.index_persist_dir = os.path.join(".llama_index_storage", f"{dataset_name}_{node_dir_prefix}")
            
            embed_model_obj = HuggingFaceEmbedding(model_name=bi_encoder_model)
            self.query_engine = self._build_llama_index_engine(embed_model_obj)
            print("[TextSearcher] ✅ Bi-Encoder engine (LlamaIndex) is ready.")

            # --- 2. 初始化 ColBERT ---
            print("--- Initializing Hybrid Mode: ColBERT ---")
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
            print("[TextSearcher] ✅ Hybrid engine is ready.")

        else:
            # [MODIFIED] 在错误提示中加入 'hybrid'
            raise ValueError(f"Invalid mode '{self.mode}'. Choose 'colbert', 'bi_encoder', or 'hybrid'.")

    def _load_nodes_generator_for_llama_index(self):
        # ... (此函数无变化)
        files = [f for f in os.listdir(self.node_dir) if f.endswith('.node')]
        for file in tqdm(files, desc=f"Streaming text nodes from {os.path.basename(self.node_dir)}"):
            input_file = os.path.join(self.node_dir, file)
            try:
                for node in nodefile2node(input_file):
                    yield Document(id_=node.id_, text=node.text or "", metadata=node.metadata or {})
            except Exception as e:
                print(f"\n❌ Error processing file: {input_file}\n{e}\n")

    def _build_llama_index_engine(self, embed_model: Any) -> RetrieverQueryEngine:
        # ... (此函数无变化)
        if os.path.exists(self.index_persist_dir):
            print(f"✅ Found existing LlamaIndex. Loading from '{self.index_persist_dir}'...")
            storage_context = StorageContext.from_defaults(persist_dir=self.index_persist_dir)
            vector_index = load_index_from_storage(storage_context, embed_model=embed_model)
        else:
            print(f"LlamaIndex storage not found at '{self.index_persist_dir}'. Building a new one...")
            os.makedirs(self.index_persist_dir, exist_ok=True)
            
            print("Loading all nodes into a list before indexing...")
            nodes_generator = self._load_nodes_generator_for_llama_index()
            nodes_list = list(nodes_generator)
            print(f"✅ Loaded {len(nodes_list)} nodes into list.")
            
            if not nodes_list:
                raise ValueError("No nodes were loaded. Cannot build an empty index.")

            vector_index = VectorStoreIndex(
                nodes_list,
                embed_model=embed_model,
                show_progress=True
            )
            print("Persisting index to disk...")
            vector_index.storage_context.persist(persist_dir=self.index_persist_dir)
            print(f"✅ New LlamaIndex built and persisted to '{self.index_persist_dir}'.")

        vector_retriever = vector_index.as_retriever()
        return RetrieverQueryEngine.from_args(retriever=vector_retriever, llm=None)
    
    def _load_all_nodes_for_colbert(self, node_dir: str) -> List[TextNode]:
        # ... (此函数无变化)
        all_nodes = []
        for filename in os.listdir(node_dir):
            if filename.endswith(".node"):
                file_path = os.path.join(node_dir, filename)
                all_nodes.extend(nodefile2node(file_path))
        return all_nodes

    # [NEW] 新增私有方法，用于执行 Bi-Encoder 检索
    def _search_bi_encoder(self, query: str, top_k: int) -> List[NodeWithScore]:
        """Helper to run search with the bi-encoder engine."""
        self.query_engine.retriever.similarity_top_k = top_k
        return self.query_engine.retrieve(query)

    # [NEW] 新增私有方法，用于执行 ColBERT 检索
    def _search_colbert(self, query: str, top_k: int) -> List[NodeWithScore]:
        """Helper to run search with the ColBERT engine."""
        results = self.rag_model.search(query=query, k=top_k, index_name=self.index_name)
        final_nodes = []
        if results:
            for result in results:
                original_node = self.node_id_map.get(result['document_id'])
                if original_node:
                    final_nodes.append(NodeWithScore(node=original_node, score=result['score']))
        return final_nodes

    # [NEW] 新增 RRF 融合算法的实现
    def _reciprocal_rank_fusion(self, results_lists: List[List[NodeWithScore]], k: int = 60) -> List[NodeWithScore]:
        """Performs Reciprocal Rank Fusion on multiple lists of search results."""
        fused_scores = {}
        node_map = {}

        # 遍历每个检索器的结果列表
        for results in results_lists:
            # 遍历单个列表中的每个结果
            for rank, node_with_score in enumerate(results):
                node_id = node_with_score.node.id_
                if node_id not in node_map:
                    node_map[node_id] = node_with_score.node
                
                # 计算 RRF 分数
                if node_id not in fused_scores:
                    fused_scores[node_id] = 0
                # rank 是从0开始的，所以要+1
                fused_scores[node_id] += 1 / (k + rank + 1)

        # 按融合后的分数进行降序排序
        reranked_results = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)

        # 构建最终的 NodeWithScore 列表
        final_nodes = []
        for node_id, score in reranked_results:
            final_nodes.append(NodeWithScore(node=node_map[node_id], score=score))
        
        return final_nodes

    # [MODIFIED] 重构 search 方法以支持所有模式
    def search(self, query: str, top_k: int) -> List[NodeWithScore]:
        print(f"\n--- Executing search in '{self.mode}' mode (TopK={top_k}) ---")
        
        if self.mode == 'bi_encoder':
            return self._search_bi_encoder(query, top_k)
        
        elif self.mode == 'colbert':
            return self._search_colbert(query, top_k)
        
        elif self.mode == 'hybrid':
            # 为了给RRF提供更丰富的候选集，从每个检索器获取更多的结果
            k_hybrid = top_k * 2 
            
            print(f"--- [Hybrid] Retrieving from Bi-Encoder (TopK={k_hybrid}) ---")
            results_be = self._search_bi_encoder(query, k_hybrid)
            
            print(f"--- [Hybrid] Retrieving from ColBERT (TopK={k_hybrid}) ---")
            results_colbert = self._search_colbert(query, k_hybrid)
            
            print("--- [Hybrid] Fusing results using RRF ---")
            fused_results = self._reciprocal_rank_fusion([results_be, results_colbert])
            
            # 返回融合、排序后得分最高的 top_k 个结果
            return fused_results[:top_k]
            
        return []