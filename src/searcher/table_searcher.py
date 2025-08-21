# File: table_searcher.py
# A unified table searcher with switchable backends: true VL search vs. OCR-based search.

import os
import json
import glob
import time
import hashlib
from tqdm import tqdm
from typing import List, Any, Optional, Dict

import torch
import torch.nn.functional as F
import numpy as np

# --- LlamaIndex Imports (for data structures) ---
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# --- 假设您的项目中有这些基类和工具函数 ---
from src.searcher.base_searcher import BaseSearcher
from src.utils.format_converter import nodefile2node
from src.llms.vl_embedding import VL_Embedding

# def nodefile2node(path: str) -> List[TextNode]:
#     """Loads a .node file and parses it into a list of LlamaIndex TextNode objects."""
#     try:
#         with open(path, 'r', encoding='utf-8') as f: data = json.load(f)
#         return [TextNode.from_dict(d) for d in data] if isinstance(data, list) else [TextNode.from_dict(data)]
#     except Exception: return []

class TableSearcher(BaseSearcher):
    """
    一个统一的表格检索器，支持两种模式:
    1. 'vl_search' (默认): 真正的多模态检索，理解表格的结构和内容。
    2. 'ocr_search': 高效的、基于表格内文本的代理检索。
    """

    def __init__(self,
                 dataset_name: str,
                 mode: str = 'vl_search',
                 # VL Search Params
                 vl_node_dir_prefix: Optional[str] = None,
                 vl_model_name: str = 'vidore/colqwen2-v1.0', # Placeholder, might need a table-specific model
                 # OCR Search Params
                 ocr_node_dir_prefix: Optional[str] = None,
                 ocr_text_encoder: str = "BAAI/bge-m3"
                 ):
        
        print(f"Initializing TableSearcher for dataset '{dataset_name}' in '{mode}' mode...")
        self.mode = mode
        self.dataset_dir = os.path.join('./data', dataset_name)
        
        if self.mode == 'vl_search':
            self._init_vl_mode(vl_node_dir_prefix, vl_model_name)
        elif self.mode == 'ocr_search':
            self._init_ocr_mode(vl_node_dir_prefix, ocr_node_dir_prefix, ocr_text_encoder)
        else:
            raise ValueError(f"Invalid mode '{self.mode}'. Choose 'vl_search' or 'ocr_search'.")

    # --- VL Search Mode Methods ---
    def _init_vl_mode(self, node_dir_prefix, model_name):
        if node_dir_prefix is None:
            # This logic might need adjustment for table-specific models
            if 'colqwen' in model_name: node_dir_prefix = 'colqwen_ingestion' # Example name
            else: raise ValueError(f"Could not infer vl_node_dir_prefix for model {model_name}")
        
        self.node_dir = os.path.join(self.dataset_dir, node_dir_prefix)
        self.embed_model = VL_Embedding(model=model_name, mode='image') # Assuming VL model handles tables as images
        self.metadata_store = []
        self.embedding_store = []
        self._load_vl_index()
        print("✅ VL Search engine for tables is ready.")

    def _load_vl_index(self):
        print("Applying memory-optimized streaming load for VL Table Engine...")
        embedding_tensors = []
        for node in self._load_nodes_generator(self.node_dir):
            self.metadata_store.append(node.metadata or {})
            if node.embedding:
                embedding_tensors.append(torch.tensor(node.embedding).view(-1, 128).bfloat16())
        
        self.embedding_store = [t.to(self.embed_model.embed_model.device) for t in embedding_tensors]
        print(f"VL index loaded. Found {len(self.metadata_store)} table documents.")

    def _search_vl(self, query: str, top_k: int) -> List[NodeWithScore]:
        query_embedding = self.embed_model.embed_text(query)
        scores = self.embed_model.processor.score(query_embedding, self.embedding_store)
        
        k = min(top_k, scores[0].numel())
        values, indices = torch.topk(scores[0], k=k)
        
        recall_nodes = []
        for i in indices.cpu().numpy():
            meta = self.metadata_store[i]
            node = TextNode(
                id_=meta.get('id_', f'tbl_{i}'),
                text=meta.get('text', ''), # Often a summary or title of the table
                metadata=meta # Metadata should contain pointers to the original table
            )
            recall_nodes.append(node)
        return [NodeWithScore(node=n, score=s.item()) for n, s in zip(recall_nodes, values)]

    # --- OCR Search Mode Methods ---
    def _init_ocr_mode(self, main_node_dir, ocr_node_dir, text_encoder):
        if main_node_dir is None or ocr_node_dir is None:
            raise ValueError("main_node_dir_prefix and ocr_node_dir_prefix must be provided for OCR mode.")
        
        self.main_node_dir = os.path.join(self.dataset_dir, main_node_dir)
        self.ocr_dir = os.path.join(self.dataset_dir, ocr_node_dir)
        self.text_encoder = text_encoder
        self.cache_dir = ".cache/ocr_table_embeds"
        os.makedirs(self.cache_dir, exist_ok=True)
        
        self.stems = sorted([os.path.splitext(os.path.basename(p))[0] for p in glob.glob(os.path.join(self.main_node_dir, "*.node"))])
        print(f"OCR mode: Found {len(self.stems)} table pages.")
        
        self.texts = [self._load_ocr_text(stem) for stem in self.stems]
        self.embed = HuggingFaceEmbedding(model_name=self.text_encoder, device="cuda")
        self.doc_emb_ocr = self._load_or_build_ocr_cache()
        print("✅ OCR Search engine for tables is ready.")

    def _load_or_build_ocr_cache(self) -> np.ndarray:
        cache_file = os.path.join(self.cache_dir, f"{os.path.basename(self.dataset_dir)}_{self.text_encoder.replace('/','_')}.npy")
        if os.path.exists(cache_file):
            print(f"OCR mode: Loading table embeddings from cache: {cache_file}")
            return np.load(cache_file)
        
        print("OCR mode: Building table text embeddings...")
        embs = self.embed.get_text_embedding_batch(self.texts, show_progress=True)
        emb_np = F.normalize(torch.tensor(embs, dtype=torch.float32)).cpu().numpy()
        np.save(cache_file, emb_np)
        print("OCR mode: Table embeddings cached.")
        return emb_np

    def _load_ocr_text(self, stem: str) -> str:
        path = os.path.join(self.ocr_dir, f"{stem}.node")
        if not os.path.exists(path): return ""
        try:
            with open(path, "r", encoding="utf-8") as f: obj = json.load(f)
            if isinstance(obj, list):
                return " ".join([x.get("text", "") for x in obj if isinstance(x, dict)]).strip()
            return str(obj.get("text", "")).strip()
        except Exception: return ""
        
    def _search_ocr(self, query: str, top_k: int) -> List[NodeWithScore]:
        q_emb = F.normalize(torch.tensor(self.embed.get_text_embedding(query), dtype=torch.float32)).cpu().numpy()
        sims = self.doc_emb_ocr @ q_emb.T
        
        k = min(top_k, len(self.stems))
        idx = np.argpartition(-sims, k-1)[:k]
        sorted_idx = idx[np.argsort(-sims[idx])]
        
        results = []
        for i in sorted_idx:
            stem = self.stems[i]
            node = TextNode(
                id_=stem,
                text=self.texts[i] or f"[Table from {stem} without text]",
                metadata={"stem": stem, "source": "ocr_search"}
            )
            results.append(NodeWithScore(node=node, score=float(sims[i])))
        return results

    # --- Unified and Helper Methods ---
    def _load_nodes_generator(self, node_dir):
        """Generic node stream loader."""
        files = [f for f in os.listdir(node_dir) if f.endswith('.node')]
        for file in tqdm(files, desc=f"Streaming nodes from {os.path.basename(node_dir)}"):
            try:
                yield from nodefile2node(os.path.join(node_dir, file))
            except Exception as e:
                print(f"\n❌ Error processing file: {file}\n{e}\n")

    def search(self, query: str, top_k: int) -> List[NodeWithScore]:
        print(f"\n--- Executing table search in '{self.mode}' mode (TopK={top_k}) ---")
        if self.mode == 'vl_search':
            return self._search_vl(query, top_k)
        elif self.mode == 'ocr_search':
            return self._search_ocr(query, top_k)
        return []