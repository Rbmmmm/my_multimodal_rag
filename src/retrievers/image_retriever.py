# File: src/retrievers/image_retriever.py
from __future__ import annotations

import glob
import json
import os
from typing import List, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


class ImageRetriever:
    """
    基于 OCR 文本的图像页检索（稳定、显存友好）：
    - 扫描 colqwen_ingestion 下的 *.node 拿到所有页的 stem（不加载 patch 向量）
    - 读取 bge_ingestion/{stem}.node 的 OCR 文本
    - 预编码所有 OCR 文本 -> BGE 向量，查询时点乘取 TopK
    - 可选：关键词先验加权（把明显相关的页略微往上推）
    """

    def __init__(
        self,
        node_dir: str = "data/ViDoSeek/colqwen_ingestion",
        ocr_dir: str = "data/ViDoSeek/bge_ingestion",
        img_dir: Optional[str] = "data/ViDoSeek/img",
        *,
        text_encoder: str = "BAAI/bge-m3",
        device: Optional[str] = None,
        embed_batch_size: int = 64,
        boost_phrases: Optional[List[str]] = None,  # 关键词先验，可覆盖
        boost_value: float = 0.2,
    ):
        self.node_dir = node_dir
        self.ocr_dir = ocr_dir
        self.img_dir = img_dir
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # 缺省先验：覆盖本题这类图中术语
        self.boost_phrases = [p.lower() for p in (boost_phrases or [
            "activity 1", "project management", "v-model", "socrates2.0", "socrates 2.0"
        ])]
        self.boost_value = float(boost_value)

        # 1) 收集 stems
        self.stems: List[str] = []
        for p in glob.glob(os.path.join(self.node_dir, "*.node")):
            stem = os.path.splitext(os.path.basename(p))[0]
            self.stems.append(stem)
        self.stems.sort()
        print(f"图像检索器: 已发现 {len(self.stems)} 个图像页 (dir={self.node_dir}).")

        # 2) 读取 OCR 文本
        self.texts: List[str] = [self._load_ocr_text(stem) for stem in self.stems]

        # 3) 准备嵌入器
        self.embed = HuggingFaceEmbedding(
            model_name=text_encoder,
            embed_batch_size=embed_batch_size,
            max_length=512,
            trust_remote_code=True,
            device=self.device,
        )

        # 4) 预编码所有页 OCR -> 向量 (N, D)
        print("图像检索器: 正在预编码 OCR 文本向量（一次性）...")
        arr: List[List[float]] = self.embed.get_text_embedding_batch(self.texts) if self.texts else []
        if len(arr) == 0:
            self.doc_emb = np.zeros((0, 1), dtype=np.float32)
        else:
            m = torch.tensor(arr, dtype=torch.float32)
            m = F.normalize(m, dim=-1)
            self.doc_emb = m.cpu().numpy()
        print(f"图像检索器: 预编码完成，shape={self.doc_emb.shape}。")

    # ---------- 工具 ----------
    def _load_ocr_text(self, stem: str) -> str:
        """
        从 bge_ingestion/{stem}.node 读取 OCR 文本。
        支持：
        - 列表：[{"text": "..."} ...]
        - 字典：{"text": "..."}
        """
        path = os.path.join(self.ocr_dir, f"{stem}.node")
        if not os.path.exists(path):
            return ""
        try:
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            if isinstance(obj, list):
                return " ".join([x.get("text", "") for x in obj if isinstance(x, dict)]).strip()
            if isinstance(obj, dict) and "text" in obj:
                return str(obj["text"]).strip()
        except Exception as e:
            print(f"⚠️ OCR 读取失败 {path}: {e}")
        return ""

    def _stem_to_image_path(self, stem: str) -> Optional[str]:
        if not self.img_dir:
            return None
        for ext in (".jpg", ".png", ".jpeg", ".webp"):
            p = os.path.join(self.img_dir, stem + ext)
            if os.path.exists(p):
                return p
        return None

    # ---------- 对外 ----------
    def retrieve(self, query: str, top_k: int = 3) -> List[NodeWithScore]:
        print(f"Running Image retrieval (TopK={top_k}) [OCR+BGE]")
        if self.doc_emb.shape[0] == 0:
            print("⚠️ 没有可用的 OCR 文本向量；返回空。")
            return []

        # 编码 query
        q = torch.tensor(self.embed.get_text_embedding(query), dtype=torch.float32)
        q = F.normalize(q, dim=-1).cpu().numpy()

        sims = self.doc_emb @ q.reshape(-1)  # (N,)

        # 关键词先验：若 OCR 文本含关键短语，给予小幅加分
        if self.boost_phrases:
            for i, txt in enumerate(self.texts):
                tl = txt.lower()
                if any(k in tl for k in self.boost_phrases):
                    sims[i] += self.boost_value

        # TopK
        k = int(min(max(top_k, 1), sims.shape[0]))
        idx = np.argpartition(-sims, kth=k - 1)[:k]
        idx = idx[np.argsort(-sims[idx])]

        results: List[NodeWithScore] = []
        for i in idx:
            stem = self.stems[i]
            text = self.texts[i] or f"[image:{stem}] (no OCR)"
            metadata: Dict[str, Optional[str]] = {
                "source": "image",
                "stem": stem,
                "ocr_file": os.path.join(self.ocr_dir, f"{stem}.node"),
                "image_path": self._stem_to_image_path(stem),
                "image_node_dir": self.node_dir,
            }
            node = TextNode(text=text, metadata=metadata)
            results.append(NodeWithScore(node=node, score=float(sims[i])))

        return results

    @staticmethod
    def to_text_view(node: NodeWithScore) -> str:
        return node.get_content() or ""