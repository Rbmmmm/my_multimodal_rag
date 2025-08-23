# File: src/searcher/image_searcher.py
# Final version with corrected data caching to preserve image_path and all node info.

import os
import json
import glob
import torch
import numpy as np
from tqdm import tqdm
from typing import List, Any, Optional, Tuple

import torch.nn.functional as F
from llama_index.core.schema import NodeWithScore, ImageNode, TextNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from src.searcher.base_searcher import BaseSearcher
from src.utils.format_converter import nodefile2node
from src.llms.vl_embedding import VL_Embedding

class ImageSearcher(BaseSearcher):
    """
    一个统一的图像检索器，支持两种模式，并为两种模式都提供了高效的磁盘缓存。
    1. 'vl_search': 真正的视觉-语言（VL）多模态检索。
    2. 'ocr_search': 高效的、基于OCR文本的代理检索。
    """
    def __init__(self,
                 dataset_name: str,
                 mode: str = 'vl_search',
                 # VL Search Params
                 vl_node_dir_prefix: Optional[str] = None,
                 vl_model_name: str = 'vidore/colqwen2-v1.0',
                 # OCR Search Params
                 ocr_node_dir_prefix: Optional[str] = None,
                 ocr_text_encoder: str = "BAAI/bge-m3",
                 img_dir_prefix: str = 'img'):
        
        print(f"[ImageSearcher] 步骤 2.2: Initializing ImageSearcher for dataset '{dataset_name}' in '{mode}' mode...")
        self.mode = mode
        self.dataset_dir = os.path.join('./data', dataset_name)
        
        if self.mode == 'vl_search':
            if vl_node_dir_prefix is None:
                if 'colqwen' in vl_model_name: vl_node_dir_prefix = 'colqwen_ingestion'
                else: raise ValueError(f"Could not infer vl_node_dir_prefix for model {vl_model_name}")
            
            self.node_dir = os.path.join(self.dataset_dir, vl_node_dir_prefix)
            self.embed_model = VL_Embedding(model=vl_model_name, mode='image')
            
            # --- 持久化逻辑 ---
            self.cache_dir = os.path.join(".image_cache", f"{dataset_name}_{vl_node_dir_prefix}")
            os.makedirs(self.cache_dir, exist_ok=True)
            self._load_vl_index_from_cache()
            print("[ImageSearcher] 步骤 2.2: ✅ VL Search engine for images is ready.")

        elif self.mode == 'ocr_search':
            if ocr_node_dir_prefix is None or vl_node_dir_prefix is None:
                raise ValueError("Both ocr_node_dir_prefix and vl_node_dir_prefix are required for OCR mode.")
            
            self.ocr_dir = os.path.join(self.dataset_dir, ocr_node_dir_prefix)
            self.vl_node_dir = os.path.join(self.dataset_dir, vl_node_dir_prefix) # 用来获取完整的stems列表
            self.img_dir = os.path.join(self.dataset_dir, img_dir_prefix)
            
            self.embed_model_ocr = HuggingFaceEmbedding(model_name=ocr_text_encoder, device="cuda")
            self.stems = sorted([os.path.splitext(os.path.basename(p))[0] for p in glob.glob(os.path.join(self.vl_node_dir, "*.node"))])
            self.texts_ocr = [self._load_ocr_text(stem) for stem in self.stems]
            
            # --- 持久化逻辑 ---
            self.cache_dir_ocr = os.path.join(".cache/ocr_image_embeds", f"{dataset_name}_{ocr_node_dir_prefix}")
            os.makedirs(self.cache_dir_ocr, exist_ok=True)
            self.doc_emb_ocr = self._load_or_build_ocr_cache()
            print("✅ OCR Search engine for images is ready.")
        else:
            raise ValueError(f"Invalid mode '{self.mode}'. Choose 'vl_search' or 'ocr_search'.")

    # --- VL Search Mode Methods ---
    def _load_vl_index_from_cache(self):
        """[修正] 为 VL 模式构建或加载持久化的、包含完整节点信息的缓存。"""
        node_info_cache_path = os.path.join(self.cache_dir, "node_info_store.json")
        embed_cache_path = os.path.join(self.cache_dir, "embedding_store.pt")

        if os.path.exists(node_info_cache_path) and os.path.exists(embed_cache_path):
            print(f"[ImageSearcher] 步骤 2.2: ✅ Found existing VL cache. Loading from '{self.cache_dir}'...")
            with open(node_info_cache_path, 'r', encoding='utf-8') as f:
                self.node_info_store = json.load(f)
            self.embedding_store = torch.load(embed_cache_path)
            self.embedding_store = [t.to(self.embed_model.embed_model.device) for t in self.embedding_store]
            print(f"[ImageSearcher] 步骤 2.2: ✅ VL cache loaded. Found {len(self.node_info_store)} documents.")
        else:
            print(f"VL cache not found at '{self.cache_dir}'. Building a new one...")
            self.node_info_store = []
            embedding_tensors = []
            
            for node in self._load_nodes_generator(self.node_dir):
                # --- 修正: 缓存重建 ImageNode 所需的所有关键信息 ---
                node_info = {
                    "id_": node.id_,
                    "text": node.text or "",
                    "image_path": getattr(node, 'image_path', None) or node.metadata.get('file_path'),
                    "metadata": node.metadata or {}
                }
                self.node_info_store.append(node_info)

                if node.embedding:
                    embedding_tensors.append(torch.tensor(node.embedding).view(-1, 128).bfloat16())
            
            with open(node_info_cache_path, 'w', encoding='utf-8') as f:
                json.dump(self.node_info_store, f, indent=2)
            torch.save(embedding_tensors, embed_cache_path)
            print(f"✅ New VL cache built and persisted to '{self.cache_dir}'.")
            
            self.embedding_store = [t.to(self.embed_model.embed_model.device) for t in embedding_tensors]

    def _search_vl(self, query: str, top_k: int) -> List[NodeWithScore]:
        query_embedding = self.embed_model.embed_text(query)
        scores = self.embed_model.processor.score(query_embedding, self.embedding_store)
        
        k = min(top_k, scores[0].numel())
        values, indices = torch.topk(scores[0], k=k)
        
        recall_nodes = []
        for i in indices.cpu().numpy():
    
            info = self.node_info_store[i]
            
            node = ImageNode(
                id_=info.get('id_', f'img_{i}'),
                text=info.get('text', ''),
                image_path=info.get('image_path'), 
                metadata=info.get('metadata', {})
            )
            recall_nodes.append(node)
        return [NodeWithScore(node=n, score=s.item()) for n, s in zip(recall_nodes, values)]

    def _load_ocr_text(self, stem: str) -> str:
        path = os.path.join(self.ocr_dir, f"{stem}.node")
        if not os.path.exists(path): return ""
        try:
            with open(path, "r", encoding="utf-8") as f: obj = json.load(f)
            if isinstance(obj, list):
                return " ".join([x.get("text", "") for x in obj if isinstance(x, dict)]).strip()
            return str(obj.get("text", "")).strip()
        except Exception: return ""

    def _load_or_build_ocr_cache(self) -> np.ndarray:
        cache_file = os.path.join(self.cache_dir_ocr, "ocr_embeddings.npy")
        if os.path.exists(cache_file):
            print(f"✅ Found existing OCR cache. Loading from '{cache_file}'")
            return np.load(cache_file)
        
        print("OCR cache not found. Building a new one...")
        embs = self.embed_model_ocr.get_text_embedding_batch(self.texts_ocr, show_progress=True)
        emb_np = F.normalize(torch.tensor(embs, dtype=torch.float32)).cpu().numpy()
        np.save(cache_file, emb_np)
        print(f"✅ New OCR cache built and persisted to '{cache_file}'.")
        return emb_np
        
    def _search_ocr(self, query: str, top_k: int) -> List[NodeWithScore]:
        q_emb = F.normalize(torch.tensor(self.embed_model_ocr.get_text_embedding(query), dtype=torch.float32)).cpu().numpy()
        sims = self.doc_emb_ocr @ q_emb.T
        
        k = min(top_k, len(self.stems))
        idx = np.argpartition(-sims, k-1)[:k]
        sorted_idx = idx[np.argsort(-sims[idx])]
        
        results = []
        for i in sorted_idx:
            stem = self.stems[i]
            img_path = os.path.join(self.img_dir, f"{stem}.jpg")
            node = ImageNode(
                id_=stem,
                text=self.texts_ocr[i],
                image_path=img_path if os.path.exists(img_path) else None,
                metadata={"stem": stem, "source": "ocr_search", "file_path": img_path}
            )
            results.append(NodeWithScore(node=node, score=float(sims[i])))
        return results


    def _load_nodes_generator(self, node_dir):
        files = [f for f in os.listdir(node_dir) if f.endswith('.node')]
        for file in tqdm(files, desc=f"Streaming nodes from {os.path.basename(node_dir)}"):
            try:
                yield from nodefile2node(os.path.join(node_dir, file))
            except Exception as e:
                print(f"\n❌ Error processing file: {file}\n{e}\n")

    def search(self, query: str, top_k: int) -> List[NodeWithScore]:
        print(f"[ImageSearcher] 步骤 2.3: Executing image search in '{self.mode}' mode (TopK={top_k}) ---")
        if self.mode == 'vl_search':
            return self._search_vl(query, top_k)
        elif self.mode == 'ocr_search':
            return self._search_ocr(query, top_k)
        return []