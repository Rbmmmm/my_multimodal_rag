# File: my_multimodal_rag/src/retrievers/chart_retriever.py
from __future__ import annotations
import os, json, glob, re
from typing import List, Tuple, Dict
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from llama_index.core.schema import NodeWithScore, TextNode

class ChartRetriever:
    """
    基于 ColQwen2 的“图表页”检索。
    - 读取 colqwen_ingestion/*.node 的扁平 patch 向量 (T*128)
    - 仅在图表候选页上检索（通过正则/白名单清单）
    - 命中后用 bge_ingestion 的 OCR 文本生成 TextNode
    """
    def __init__(
        self,
        node_dir: str = "data/ViDoSeek/colqwen_ingestion",
        ocr_dir: str = "data/ViDoSeek/bge_ingestion",
        text_encoder: str = "openai/clip-vit-base-patch32",
        chart_list_path: str | None = None,      # 可选：图表页白名单清单(.txt/.json)
        chart_name_regex: str = r"(chart|figure|fig|plot|graph)_?\d*$",
        device: str | None = None,
    ):
        self.node_dir = node_dir
        self.ocr_dir = ocr_dir
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.pattern = re.compile(chart_name_regex, re.IGNORECASE)
        self.chart_whitelist: set[str] = set()

        # 加载可选白名单（强烈建议：准确性更高）
        if chart_list_path and os.path.exists(chart_list_path):
            with open(chart_list_path, "r", encoding="utf-8") as f:
                if chart_list_path.endswith(".json"):
                    arr = json.load(f)
                    self.chart_whitelist = set(arr) if isinstance(arr, list) else set(arr.get("stems", []))
                else:
                    self.chart_whitelist = set([ln.strip() for ln in f if ln.strip()])
            print(f"ChartRetriever: loaded whitelist {len(self.chart_whitelist)} stems from {chart_list_path}")

        # 文本塔
        self.tok = AutoTokenizer.from_pretrained(text_encoder, trust_remote_code=True)
        self.txt_model = AutoModel.from_pretrained(text_encoder, trust_remote_code=True, use_safetensors=True).to(self.device).eval()

        # 建索引（只记录路径；必要时过滤只保留图表页）
        self._index: Dict[str, str] = {}
        for p in glob.glob(os.path.join(self.node_dir, "*.node")):
            stem = os.path.splitext(os.path.basename(p))[0]  # e.g. foo_12
            if self._is_chart_stem(stem):
                self._index[stem] = p
        print(f"图表检索器: 已索引 {len(self._index)} 个候选图表页 (dir={self.node_dir}).")

    def _is_chart_stem(self, stem: str) -> bool:
        if self.chart_whitelist:
            return stem in self.chart_whitelist
        # 没有白名单时，用名字正则做个温和过滤；不匹配也可能是图表，但先降噪
        return bool(self.pattern.search(stem))

    @torch.no_grad()
    def _encode_query_128(self, query: str) -> torch.Tensor:
        x = self.tok(query, return_tensors="pt", truncation=True, max_length=256).to(self.device)
        if hasattr(self.txt_model, "get_text_features"):
            q = self.txt_model.get_text_features(**x)
        else:
            out = self.txt_model(**x)
            h = out.last_hidden_state.mean(dim=1)
            if hasattr(self.txt_model, "text_projection"):
                h = h @ self.txt_model.text_projection
            q = h
        return F.normalize(q.float(), dim=-1).squeeze(0)  # [128]

    def _load_img_tokens(self, node_path: str) -> torch.Tensor:
        with open(node_path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        emb = obj[0]["embedding"] if isinstance(obj, list) else obj["embedding"]
        t = torch.tensor(emb, dtype=torch.float32, device=self.device).view(-1, 128)
        return F.normalize(t, dim=-1)

    def _best_score(self, img_tokens: torch.Tensor, q128: torch.Tensor) -> float:
        return float((img_tokens @ q128).max().item())

    def _load_ocr_text(self, stem: str) -> str:
        ocr_path = os.path.join(self.ocr_dir, f"{stem}.node")
        if not os.path.exists(ocr_path):
            return ""
        try:
            with open(ocr_path, "r", encoding="utf-8") as f:
                arr = json.load(f)
            if isinstance(arr, list) and arr and "text" in arr[0]:
                return " ".join(x.get("text", "") for x in arr).strip()
            if isinstance(arr, dict) and "text" in arr:
                return str(arr["text"]).strip()
        except Exception as e:
            print(f"⚠️ OCR 读取失败 {ocr_path}: {e}")
        return ""

    def retrieve(self, query: str, top_k: int = 3) -> List[NodeWithScore]:
        print(f"Running Chart retrieval (TopK={top_k})")
        if not self._index:
            print("⚠️ 没有可用的图表节点；返回空。")
            return []

        q = self._encode_query_128(query)
        scores: List[Tuple[str, float]] = []

        for stem, node_path in self._index.items():
            try:
                toks = self._load_img_tokens(node_path)
                s = self._best_score(toks, q)
                scores.append((stem, s))
            except Exception as e:
                print(f"⚠️ 跳过 {node_path}: {e}")

        if not scores:
            return []

        scores.sort(key=lambda x: x[1], reverse=True)
        top = scores[:top_k]

        results: List[NodeWithScore] = []
        for stem, score in top:
            text = self._load_ocr_text(stem)
            md = {"source": "chart", "stem": stem, "ocr_dir": self.ocr_dir, "image_node_dir": self.node_dir}
            node = TextNode(text=text or f"[chart:{stem}] (no OCR)", metadata=md)
            results.append(NodeWithScore(node=node, score=score))
        return results