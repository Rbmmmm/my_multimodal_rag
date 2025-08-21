# File: src/agents/inspector_agent.py
# Final version, strictly adhering to the Vidorag Inspector's stateful logic and page_map mechanism.

from __future__ import annotations
import os
import math
import re
from typing import List, Tuple, Any, Optional, Dict

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from llama_index.core.schema import NodeWithScore, ImageNode

# --- 您需要自行解决的导入 ---
from src.agent.agent_prompt import inspector_prompt
from src.agent.map_dict import page_map_dict_normal
from src.utils.parse_tool import extract_json

class InspectorAgent:
    """
    一个严格遵循 Vidorag Inspector 逻辑的有状态智能体。
    它内部维护一个证据缓冲区，并使用 page_map 来约束 VLM，防止越界幻觉。
    它依然使用 Reranker 对外提供置信度评估。
    """
    def __init__(self, 
                 vlm: Any,
                 image_base_dir: str,
                 reranker_model_name: str = "BAAI/bge-reranker-large",
                 # Reranker 参数
                 window_tokens: int = 256,
                 window_stride: int = 128,
                 batch_size: int = 16):
        
        print("Initializing STATEFUL InspectorAgent (Vidorag-style)...")
        self.vlm = vlm
        self.image_base_dir = image_base_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # --- 初始化 Reranker (用于外部置信度评估) ---
        print(f"Loading Reranker for confidence scoring: {reranker_model_name}...")
        torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        self.reranker_tokenizer = AutoTokenizer.from_pretrained(reranker_model_name, use_fast=False)
        self.reranker_model = AutoModelForSequenceClassification.from_pretrained(
            reranker_model_name, torch_dtype=torch_dtype
        ).to(self.device).eval()
        self.window_tokens, self.window_stride, self.batch_size = window_tokens, window_stride, batch_size
        print("✅ Reranker loaded.")

        # --- 初始化 VLM 相关配置 (完全来自 Vidorag) ---
        self.page_map = page_map_dict_normal
        
        # --- 核心：内部状态缓冲区 ---
        self.buffer_nodes: List[NodeWithScore] = []
        self.node_map_by_path: Dict[str, NodeWithScore] = {} # 用于根据路径找回node

        print("✅ STATEFUL InspectorAgent is ready.")
    
    def clear_buffer(self):
        """在处理新查询时，由 Orchestrator 调用的重置方法。"""
        self.buffer_nodes = []
        self.node_map_by_path = {}
        print("[Inspector] Buffer cleared for new query.")

    def _get_reranker_confidence(self, query: str, nodes: List[NodeWithScore]) -> torch.Tensor:
        """阶段一：使用 Reranker 计算置信度分数。"""
        if not nodes: return torch.tensor(0.0, device=self.device)
        node_texts = [n.get_content(metadata_mode="all") or "" for n in nodes]
        pair_iter = ((query, win) for text in node_texts for win in self._windows(text))
        all_scores = []
        with torch.no_grad():
            for batch in self._batched(pair_iter, self.batch_size):
                inputs = self.reranker_tokenizer([p[0] for p in batch], [p[1] for p in batch], padding=True, truncation=True, max_length=512, return_tensors="pt").to(self.device)
                logits = self.reranker_model(**inputs).logits.squeeze(-1).float()
                all_scores.extend(logits.tolist())
        best_score_per_doc = [-math.inf] * len(nodes)
        score_idx = 0
        for i, text in enumerate(node_texts):
            num_windows = len(self._windows(text))
            if num_windows > 0:
                best_score_per_doc[i] = max(all_scores[score_idx : score_idx + num_windows])
                score_idx += num_windows
        top_logit = torch.tensor(max(best_score_per_doc) if best_score_per_doc else -math.inf, device=self.device)
        confidence = torch.sigmoid(top_logit)
        print(f"[Inspector-Reranker] Top Logit: {top_logit.item():.4f} -> Confidence: {confidence.item():.4f}")
        return confidence
    
    def _nodes_to_paths(self, nodes: List[NodeWithScore]) -> List[str]:
        """适配器：从 Node 列表提取图片路径，并更新内部的路径->Node映射。"""
        paths = []
        for node_with_score in nodes:
            node = node_with_score.node
            metadata = getattr(node, "metadata", {}) or {}
            image_path = None
            explicit_path = metadata.get("image_path") or metadata.get("file_path")
            if explicit_path and isinstance(explicit_path, str): image_path = explicit_path
            elif 'filename' in metadata and isinstance(metadata['filename'], str):
                base_filename = os.path.splitext(metadata['filename'])[0]
                image_path = os.path.join(self.image_base_dir, f"{base_filename}.jpg")

            if image_path and os.path.exists(image_path) and image_path not in self.node_map_by_path:
                paths.append(image_path)
                self.node_map_by_path[image_path] = node_with_score
        return paths

    def run(self, 
            query: str, 
            nodes: List[NodeWithScore]
           ) -> Tuple[str, Any, List[NodeWithScore], torch.Tensor]:
        
        # 阶段一：Reranker 评分依然先行，它是一个独立的评估
        confidence = self._get_reranker_confidence(query, nodes)
        
        # 阶段二：VLM 决策，严格遵循 Vidorag 的有状态逻辑
        new_image_paths = self._nodes_to_paths(nodes)
        
        # 状态管理
        if not self.buffer_nodes and not new_image_paths:
            return "answer", "No evidence found to inspect.", [], confidence
        elif not new_image_paths:
            return 'synthesizer', "Sufficient evidence collected from previous turn.", self.buffer_nodes, confidence
        else:
            # 核心的累积逻辑
            self.buffer_nodes.extend([self.node_map_by_path[p] for p in new_image_paths if p in self.node_map_by_path])
            
        buffer_image_paths = list(self.node_map_by_path.keys())
        
        # --- 核心修正：严格仿照 Vidorag 构造 Prompt ---
        num_candidates = len(buffer_image_paths)
        page_map_info = self.page_map.get(num_candidates, f"A total of {num_candidates} pages are provided, indexed 0 to {num_candidates-1}.")
        prompt = inspector_prompt.replace('{question}', query).replace('{page_map}', page_map_info)
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                print(f"\n[Inspector-VLM] Calling VLM with {len(buffer_image_paths)} image paths (Attempt {attempt + 1}/{max_retries})...")
                response_text = self.vlm.generate(query=prompt, image=buffer_image_paths)
                response_json = extract_json(response_text)
                
                reason = response_json.get('reason')
                if reason is None: raise ValueError("'reason' field is missing.")

                answer, ref = response_json.get('answer'), response_json.get('reference')
                info, choice = response_json.get('information'), response_json.get('choice')

                if answer is not None and ref is not None:
                    if any(page < 0 or page >= len(buffer_image_paths) for page in ref) or not ref: raise ValueError("Index out of bounds in 'reference'.")
                    final_paths = [buffer_image_paths[i] for i in ref]
                    final_nodes = [self.node_map_by_path[p] for p in final_paths]
                    return ('answer' if len(ref) == len(buffer_image_paths) else 'synthesizer'), answer, final_nodes, confidence
                
                elif info is not None and choice is not None:
                    if any(page < 0 or page >= len(buffer_image_paths) for page in choice): raise ValueError("Index out of bounds in 'choice'.")
                    chosen_paths = [buffer_image_paths[i] for i in choice]
                    self.buffer_nodes = [self.node_map_by_path[p] for p in chosen_paths]
                    self.node_map_by_path = {p: self.node_map_by_path[p] for p in chosen_paths}
                    return 'seeker', info, self.buffer_nodes, confidence

                raise ValueError("VLM response did not match expected format.")

            except Exception as e:
                print(f"❌ [Inspector-VLM] Error on attempt {attempt + 1}: {e}")
                if attempt == max_retries - 1:
                    return 'seeker', f"VLM failed: {e}", [], confidence
        
        return 'seeker', "VLM failed after max retries.", [], confidence
    
    # --- Reranker 的辅助函数 (保持不变) ---
    def _windows(self, text: str) -> List[str]:
        toks = self.reranker_tokenizer(text, truncation=False, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        if len(toks) <= self.window_tokens: return [text]
        spans: List[str] = []
        i = 0
        while i < len(toks):
            j = min(i + self.window_tokens, len(toks))
            spans.append(self.reranker_tokenizer.decode(toks[i:j], skip_special_tokens=True))
            if j >= len(toks): break
            i += self.window_stride
        return spans or [""]

    def _batched(self, iterable, bs: int):
        buf = [];
        for x in iterable:
            buf.append(x)
            if len(buf) == bs: yield buf; buf = []
        if buf: yield buf