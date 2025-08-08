from __future__ import annotations

import glob
import json
import os
import time
import hashlib
from typing import List, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


class ImageRetriever:
    """
    基于 OCR 文本的图像页检索（稳定、显存友好）+ 磁盘缓存：
    - 扫描 colqwen_ingestion 下的 *.node 拿到所有页的 stem（不加载 patch 向量）
    - 读取 bge_ingestion/{stem}.node 的 OCR 文本
    - 预编码所有 OCR 文本 -> 向量，查询时点乘取 TopK
    - 缓存统一用 float32（若命中旧的 fp16 缓存，会自动升回 float32 并写出 f32 缓存）
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
        cache_dir: str = ".cache/ocr_embeds",
        boost_phrases: Optional[List[str]] = None,
        boost_value: float = 0.2,
    ):
        self.node_dir = node_dir
        self.ocr_dir = ocr_dir
        self.img_dir = img_dir
        self.text_encoder = text_encoder
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.embed_batch_size = embed_batch_size
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

        # 关键词先验
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

        # 3) 嵌入器
        self.embed = HuggingFaceEmbedding(
            model_name=self.text_encoder,
            embed_batch_size=self.embed_batch_size,
            max_length=512,
            trust_remote_code=True,
            device=self.device,
        )

        # 4) 预编码或读缓存（优先 f32，其次旧的 fp16）
        self.doc_emb = self._load_or_build_cache()

    # ---------- 缓存 ----------
    def _cache_key(self) -> str:
        payload = json.dumps(
            {
                "node_dir": os.path.abspath(self.node_dir),
                "ocr_dir": os.path.abspath(self.ocr_dir),
                "text_encoder": self.text_encoder,
                "count": len(self.stems),
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()

    def _cache_paths(self) -> Dict[str, str]:
        key = self._cache_key()
        base = os.path.join(self.cache_dir, key)
        return {
            "f32": base + ".f32.npy",
            "fp16": base + ".fp16.npy",
            "stems": base + ".stems.json",
            "meta": base + ".meta.json",
        }

    def _load_or_build_cache(self) -> np.ndarray:
        paths = self._cache_paths()

        # 命中 f32 缓存
        if os.path.exists(paths["f32"]) and os.path.exists(paths["stems"]):
            try:
                with open(paths["stems"], "r", encoding="utf-8") as f:
                    cached_stems = json.load(f)
                if cached_stems == self.stems:
                    arr = np.load(paths["f32"], mmap_mode="r").astype(np.float32, copy=False)
                    print(f"图像检索器: 命中 OCR 向量缓存 -> {os.path.basename(paths['f32'])} | shape={arr.shape}")
                    return arr
                else:
                    print("图像检索器: 缓存 stem 不匹配，重建缓存。")
            except Exception as e:
                print(f"图像检索器: 加载 f32 缓存失败，尝试 fp16。原因: {e}")

        # 命中旧 fp16 缓存，升回 f32 并写回
        if os.path.exists(paths["fp16"]) and os.path.exists(paths["stems"]):
            try:
                with open(paths["stems"], "r", encoding="utf-8") as f:
                    cached_stems = json.load(f)
                if cached_stems == self.stems:
                    old = np.load(paths["fp16"], mmap_mode="r").astype(np.float16, copy=False)
                    arr = old.astype(np.float32, copy=False)
                    np.save(paths["f32"], arr)
                    print(f"图像检索器: 命中旧 fp16 缓存 -> {os.path.basename(paths['fp16'])} | shape={arr.shape}，已转为 f32 并写回缓存")
                    return arr
            except Exception as e:
                print(f"图像检索器: 加载 fp16 缓存失败，重建缓存。原因: {e}")

        # 真正构建
        print("图像检索器: 正在预编码 OCR 文本向量（一次性）...")
        t0 = time.time()
        if not self.texts:
            emb_np = np.zeros((0, 1), dtype=np.float32)
        else:
            embs: List[List[float]] = self.embed.get_text_embedding_batch(self.texts)
            m = torch.tensor(embs, dtype=torch.float32)
            m = F.normalize(m, dim=-1)
            emb_np = m.cpu().numpy().astype(np.float32, copy=False)
        dt = time.time() - t0
        print(f"图像检索器: 预编码完成，shape={emb_np.shape}，耗时 {dt:.2f}s。")

        # 落盘（只写 f32）
        try:
            np.save(paths["f32"], emb_np)
            with open(paths["stems"], "w", encoding="utf-8") as f:
                json.dump(self.stems, f, ensure_ascii=False)
            with open(paths["meta"], "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "text_encoder": self.text_encoder,
                        "dtype": "f32",
                        "N": int(emb_np.shape[0]),
                        "D": int(emb_np.shape[1] if emb_np.size else 0),
                        "created_at": time.time(),
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
            print(f"图像检索器: 已写入缓存至 {self.cache_dir}")
        except Exception as e:
            print(f"⚠️ 写入缓存失败（不影响检索）：{e}")

        return emb_np

    # ---------- 工具 ----------
    def _load_ocr_text(self, stem: str) -> str:
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
        print(f"Running Image retrieval (TopK={top_k}) [OCR+BGE+DiskCache]")
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
                if any(k in txt.lower() for k in self.boost_phrases):
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