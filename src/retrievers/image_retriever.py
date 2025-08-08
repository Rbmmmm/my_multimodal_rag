# File: src/retrievers/image_retriever.py
from __future__ import annotations
import os, json, glob
from typing import List, Dict, Tuple, Optional
import numpy as np
import torch
import torch.nn.functional as F
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


class ImageRetriever:
    """
    基于 OCR 文本的“图像页检索”：
    - 扫描 colqwen_ingestion 里的 *.node 获取所有页的 stem（doc_page）
    - 对应到 bge_ingestion/{stem}.node 读取 OCR 文本（若缺失则空串）
    - 预先用 BGE-m3 编码全部 OCR 文本为向量，归一化缓存
    - 查询时用同一模型编码 query，做余弦相似度，取 top-k
    - 返回 TextNode（text=OCR，metadata 标注来源与 image 路径）
    这样可稳定跑通 image 路径（效果由 OCR 质量决定），后续想接入 ColQwen patch-rerank 再加层重排即可。
    """

    def __init__(
        self,
        node_dir: str = "data/ViDoSeek/colqwen_ingestion",
        ocr_dir: str = "data/ViDoSeek/bge_ingestion",
        img_dir: Optional[str] = "data/ViDoSeek/img",
        text_encoder: str = "BAAI/bge-m3",
        device: Optional[str] = None,
        embed_batch_size: int = 64,
    ):
        self.node_dir = node_dir
        self.ocr_dir = ocr_dir
        self.img_dir = img_dir
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # 1) 收集所有图像页的 stem（不读 patch 向量，避免显存/内存压力）
        self.stems: List[str] = []
        for p in glob.glob(os.path.join(self.node_dir, "*.node")):
            stem = os.path.splitext(os.path.basename(p))[0]
            self.stems.append(stem)
        self.stems.sort()

        print(f"图像检索器: 已发现 {len(self.stems)} 个图像页 (dir={self.node_dir}).")

        # 2) 读取 OCR 文本
        self.texts: List[str] = [self._load_ocr_text(stem) for stem in self.stems]

        # 3) 准备 BGE-m3 文本向量器（与主流程一致）
        self.embed = HuggingFaceEmbedding(
            model_name=text_encoder,
            embed_batch_size=embed_batch_size,
            max_length=512,
            trust_remote_code=True,
            device=self.device,
        )

        # 4) 预编码所有页 OCR -> 向量 (N, D) 并归一化缓存为 float32
        print("图像检索器: 正在预编码 OCR 文本向量（一次性）...")
        arr: List[List[float]] = self.embed.get_text_embedding_batch(self.texts) if len(self.texts) > 0 else []
        if len(arr) == 0:
            self.doc_emb = np.zeros((0, 1), dtype=np.float32)
        else:
            m = torch.tensor(arr, dtype=torch.float32)
            m = F.normalize(m, dim=-1)
            self.doc_emb = m.cpu().numpy()
        print(f"图像检索器: 预编码完成，shape={self.doc_emb.shape}。")

    # -------- 工具函数 --------
    def _load_ocr_text(self, stem: str) -> str:
        """
        从 bge_ingestion/{stem}.node 读取 OCR 文本。
        兼容两种格式：
        - 列表：[{"text": "..."} ...]
        - 字典：{"text": "..."}
        缺失则返回空串。
        """
        ocr_path = os.path.join(self.ocr_dir, f"{stem}.node")
        if not os.path.exists(ocr_path):
            return ""
        try:
            with open(ocr_path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            if isinstance(obj, list):
                parts = [x.get("text", "") for x in obj if isinstance(x, dict)]
                return " ".join(parts).strip()
            if isinstance(obj, dict) and "text" in obj:
                return str(obj["text"]).strip()
        except Exception as e:
            print(f"⚠️ OCR 读取失败 {ocr_path}: {e}")
        return ""

    def _stem_to_image_path(self, stem: str) -> Optional[str]:
        """
        根据 stem 推测图像路径（若提供 img_dir，则尝试 {img_dir}/{stem}.jpg/.png）
        找不到就返回 None，仅放到 metadata 里。
        """
        if not self.img_dir:
            return None
        for ext in (".jpg", ".png", ".jpeg", ".webp"):
            p = os.path.join(self.img_dir, stem + ext)
            if os.path.exists(p):
                return p
        return None

    # -------- 对外接口 --------
    def retrieve(self, query: str, top_k: int = 3) -> List[NodeWithScore]:
        print(f"Running Image retrieval (TopK={top_k}) [OCR+BGE]")
        if self.doc_emb.shape[0] == 0:
            print("⚠️ 没有可用的 OCR 文本向量；返回空。")
            return []

        # 编码 query -> 归一化向量 (D,)
        q = torch.tensor(self.embed.get_text_embedding(query), dtype=torch.float32)
        q = F.normalize(q, dim=-1).cpu().numpy()

        # 余弦相似度 = dot(单位化向量)
        sims = (self.doc_emb @ q.reshape(-1))  # (N,)

        # 取 TopK
        k = int(min(top_k, sims.shape[0]))
        if k <= 0:
            return []
        top_idx = np.argpartition(-sims, kth=k-1)[:k]
        # 排序
        top_idx = top_idx[np.argsort(-sims[top_idx])]

        results: List[NodeWithScore] = []
        for idx in top_idx:
            stem = self.stems[idx]
            score = float(sims[idx])
            text = self.texts[idx] or f"[image:{stem}] (no OCR)"
            metadata = {
                "source": "image",
                "stem": stem,
                "ocr_file": os.path.join(self.ocr_dir, f"{stem}.node"),
                "image_path": self._stem_to_image_path(stem),
                "image_node_dir": self.node_dir,
            }
            node = TextNode(text=text, metadata=metadata)
            results.append(NodeWithScore(node=node, score=score))

        return results

    @staticmethod
    def to_text_view(node: NodeWithScore) -> str:
        # 统一给 Inspector 使用
        return node.get_content() or ""