# File: src/agents/seeker_agent.py
# Final version, strictly adhering to the Vidorag prompt construction logic.

from __future__ import annotations
import os
import json
from typing import List, Tuple, Any

from PIL import Image
from llama_index.core.schema import NodeWithScore, ImageNode, TextNode

# --- 您需要自行解决的导入 ---
from src.agent.agent_prompt import seeker_prompt
from src.agent.map_dict import page_map_dict_normal
from src.utils.parse_tool import extract_json

class SeekerAgent:
    """
    Seeker Agent - VLM-Powered Evidence Selector.
    This version's prompt engineering logic strictly adheres to the Vidorag reference.
    """
    def __init__(self, vlm: Any, image_base_dir: str):
        self.vlm = vlm
        self.image_base_dir = image_base_dir
        self.page_map = page_map_dict_normal # Vidorag style page map
        
        # Vidorag State Attributes
        self.query: str | None = None
        self.buffer_nodes: List[NodeWithScore] = []
        self.buffer_images: List[str] = []

    def run(self, 
            query: str | None = None, 
            candidate_nodes: List[NodeWithScore] | None = None,
            image_paths: List[str] | None = None,
            feedback: str | None = None
           ) -> Tuple[List[NodeWithScore], List[str], str, str]:

        if query is not None and image_paths is not None:
            self.buffer_nodes = candidate_nodes
            self.buffer_images = image_paths
            self.query = query
            prompt = seeker_prompt.replace('{question}', self.query).replace('{page_map}', self.page_map[len(self.buffer_images)])

        elif feedback is not None:
            additional_information = self.query + '\n\n## Additional Information\n' + feedback
            prompt = seeker_prompt.replace('{question}', additional_information).replace('{page_map}', self.page_map[len(self.buffer_images)])        

        # --- Vidorag 核心 VLM 调用逻辑 ---
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                print(f"[Seeker] 步骤 3.1: 用 {len(self.buffer_images)} 个图片提供给 VLM (Attempt {attempt + 1}/{max_retries})...")
                response_text = self.vlm.generate(query=prompt, image=self.buffer_images)
                
                response_json = extract_json(response_text)
                reason, summary, choice = (response_json.get(k) for k in ['reason', 'summary', 'choice'])

                if reason is None or summary is None or not isinstance(choice, list):
                    raise ValueError("JSON format error: missing fields.")
                # 注意：Vidorag的buffer是 image_path 列表，我们的buffer是 Node 列表，所以校验长度时用 self.buffer_nodes
                if any(page < 0 or page >= len(self.buffer_nodes) for page in choice):
                    raise ValueError(f"Index out of bounds for buffer size {len(self.buffer_nodes)}.")

                # Vidorag 的 buffer 在这里会更新，但我们的 Orchestrator 架构不需要 Agent 维护状态
                # self.buffer_images = [image for image in self.buffer_images if image not in selected_images]

                selected_nodes = [self.buffer_nodes[i] for i in choice]
                selected_images = [self.buffer_images[i] for i in choice]
                self.buffer_images = [img for img in self.buffer_images if img not in selected_images]
                self.buffer_nodes = [node for node in self.buffer_nodes if node not in selected_nodes]
                
                print(f"[Seeker] 步骤 3.2: ✅ VLM 成功选取了 {len(selected_nodes)} 个节点.")
                return selected_nodes, selected_images, summary, reason

            except Exception as e:
                print(f"[Seeker] 步骤 3.2: ❌ Error on attempt {attempt + 1}: {e}")
                if attempt == max_retries - 1:
                    return self.buffer_nodes, self.buffer_images, "Failed to process with VLM.", str(e)
        
        return self.buffer_nodes, self.buffer_images, "An unexpected error occurred.", "No selection was made."