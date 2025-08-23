# File: src/agents/inspector_agent.py

from __future__ import annotations
import os
import math
import re
from typing import List, Tuple, Any, Optional, Dict

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from llama_index.core.schema import NodeWithScore, ImageNode

from src.agent.agent_prompt import inspector_prompt
from src.agent.map_dict import page_map_dict_normal
from src.utils.parse_tool import extract_json

class InspectorAgent:

    def __init__(self, 
                 vlm: Any,
                 image_base_dir: str,
                 reranker_model_name: str = "BAAI/bge-reranker-large",
                 # Reranker 参数
                 window_tokens: int = 256,
                 window_stride: int = 128,
                 batch_size: int = 16):
        
        print("[1f] [Inspector] 初始化 InspectorAgent")
        self.vlm = vlm
        self.image_base_dir = image_base_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"[1f] [Inspector] Loading Reranker for confidence scoring: {reranker_model_name}...")
        torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        self.reranker_tokenizer = AutoTokenizer.from_pretrained(reranker_model_name, use_fast=False)
        self.reranker_model = AutoModelForSequenceClassification.from_pretrained(
            reranker_model_name, torch_dtype=torch_dtype
        ).to(self.device).eval()
        self.window_tokens, self.window_stride, self.batch_size = window_tokens, window_stride, batch_size
        print("[1f] [Inspector] ✅ Reranker loaded.")

        self.page_map = page_map_dict_normal
        self.buffer_nodes: List[NodeWithScore] = []
        self.buffer_images: List[str] = []
        print("[1f] [Inspector] ✅ STATEFUL InspectorAgent is ready.")
    
    def clear_buffer(self):
        """在处理新查询时，由 Orchestrator 调用的重置方法。"""
        self.buffer_nodes = []
        self.buffer_images = []
        print("[Inspector] Buffer cleared for new query.")

    def _get_reranker_confidence(self, query: str, nodes: List[NodeWithScore]) -> torch.Tensor:
        
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
        print(f"[Inspector-Reranker] 步骤 4.1: Top Logit: {top_logit.item():.4f} -> Confidence: {confidence.item():.4f}")
        return confidence

    def _is_image_centric(self, nodes: List[NodeWithScore]) -> bool:
        """检查节点是否主要为图像类型"""
        if not nodes:
            return False
        first_node_meta = getattr(nodes[0].node, "metadata", {}) or {}
        if first_node_meta.get("image_path") or first_node_meta.get("file_path"):
            return True
        if 'filename' in first_node_meta and isinstance(first_node_meta['filename'], str):
            return False 
        return True 

    def run(self, 
            query: str, 
            nodes: List[NodeWithScore],
            image_paths: List[str]
           ) -> Tuple[str, Any, List[str], torch.Tensor]:
        
        if self._is_image_centric(nodes):
            print("[Inspector] 步骤 4.1: Image-centric nodes detected. Start Reranker.")
            confidence = torch.tensor(0.95, device=self.device) 
        else:
            confidence = self._get_reranker_confidence(query, nodes)
        
        # if not self.buffer_nodes and not image_paths:
        #       return "seeker", "No evidence found to inspect.", [], confidence
        # elif not image_paths:
        #     return 'synthesizer', "Sufficient evidence collected.", self.buffer_nodes, confidence
        # else:
        #     self.buffer_nodes = nodes
        #     self.buffer_images = image_paths
        
        if len(self.buffer_images) == 0 and len(image_paths) == 0:
            return None, None, None
        elif len(image_paths) == 0:
            return 'synthesizer', None, self.buffer_images
        elif len(image_paths) != 0:
            self.buffer_images.extend(image_paths)
            self.buffer_nodes.extend(nodes)
            
        input_images = self.buffer_images
        prompt = inspector_prompt.replace('{question}',query).replace('{page_map}',self.page_map[len(self.buffer_images)])
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                print(f"[Inspector] 步骤 4.2: Calling VLM with {len(input_images)} image paths (Attempt {attempt + 1}/{max_retries})...")
                response_text = self.vlm.generate(query=prompt, image=input_images)
                response_json = extract_json(response_text)
                
                reason = response_json.get('reason')
                answer, ref = response_json.get('answer'), response_json.get('reference')
                info, choice = response_json.get('information'), response_json.get('choice')

                if reason is None:
                    raise Exception('answer no reason')
                elif answer is not None and ref is not None:
                    # if any([page >= len(self.buffer_images) for page in ref]) or len(ref)==0:
                    #     raise Exception('ref error')
                    if len(ref) == len(self.buffer_images):
                        print("[Inspector] 所有节点都有用, 直接回答")
                        return 'answer', answer, self.buffer_images, self.buffer_nodes, confidence
                    else:
                        ref_images = [self.buffer_images[page] for page in ref]
                        ref_nodes = [self.buffer_nodes[page] for page in ref]
                        print("[Inspector] 选择的节点为:", ref_images, ", 移交给 synthesizer.")
                        return 'synthesizer', answer, ref_images, ref_nodes, confidence
                elif info is not None and choice is not None:
                    if any([page >= len(self.buffer_images) for page in choice]):
                        raise Exception('choice error')
                    self.buffer_images = [self.buffer_images[page] for page in choice]
                    self.buffer_nodes = [self.buffer_nodes[page] for page in choice]
                    print("[Inspector] 证据不足，重新交给 seeker. 但目前可用的节点为:", self.buffer_images)
                    return 'seeker', info, self.buffer_images, self.buffer_nodes, confidence

            except Exception as e:
                print(f"❌ [Inspector-VLM] Error on attempt {attempt + 1}: {e}")
                if attempt == max_retries - 1:
                    return 'seeker', f"VLM failed: {e}", [], [], confidence
        
        return 'seeker', "VLM failed after max retries.", [], [], confidence
    
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
  