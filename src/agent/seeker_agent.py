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
        self.page_map_template = page_map_dict_normal # Vidorag style page map
        
        # Vidorag State Attributes
        self.query: str | None = None
        self.buffer_nodes: List[NodeWithScore] = []

    def run(self, 
            query: str | None = None, 
            candidate_nodes: List[NodeWithScore] | None = None,
            feedback: str | None = None
           ) -> Tuple[List[NodeWithScore], str, str]:

        if candidate_nodes is not None:
            self.buffer_nodes = candidate_nodes

        if not self.buffer_nodes:
            return [], "No candidates provided to Seeker.", "No action taken."

        # --- [新] 严格仿照 Vidorag 的 prompt 构造逻辑 ---

        # 1. 准备 VLM 需要的图片路径列表
        image_paths_for_vlm = []
        node_map = {} # 用于最后根据路径找回 Node 对象
        for node_with_score in self.buffer_nodes:
            node = node_with_score.node
            metadata = getattr(node, "metadata", {}) or {}
            
            image_path = None
            explicit_path = metadata.get("image_path") or metadata.get("file_path")
            if explicit_path and isinstance(explicit_path, str): image_path = explicit_path
            elif 'filename' in metadata and isinstance(metadata['filename'], str):
                base_filename = os.path.splitext(metadata['filename'])[0]
                potential_path = os.path.join(self.image_base_dir, f"{base_filename}.jpg")
                image_path = potential_path

            if image_path and os.path.exists(image_path):
                image_paths_for_vlm.append(image_path)
                node_map[image_path] = node_with_score
        
        # 2. 构造 prompt
        prompt = ""
        num_candidates = len(self.buffer_nodes)
        page_map_info = self.page_map_template.get(num_candidates, f"A total of {num_candidates} pages are provided, indexed 0 to {num_candidates-1}.")

        if query is not None:
            self.query = query
            prompt = seeker_prompt.replace('{question}', self.query).replace('{page_map}', page_map_info)
        
        elif feedback is not None:
            if self.query is None: raise ValueError("Cannot run with feedback without a prior query.")
            additional_information = self.query + '\n\n## Additional Information\n' + feedback
            prompt = seeker_prompt.replace('{question}', additional_information).replace('{page_map}', page_map_info)
        
        # --- Vidorag 核心 VLM 调用逻辑 ---
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                print(f"\n[Seeker] Calling VLM with {len(image_paths_for_vlm)} image paths (Attempt {attempt + 1}/{max_retries})...")
                response_text = self.vlm.generate(query=prompt, image=image_paths_for_vlm)
                
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
                
                print(f"✅ [Seeker] VLM successfully selected {len(selected_nodes)} nodes.")
                return selected_nodes, summary, reason

            except Exception as e:
                print(f"❌ [Seeker] Error on attempt {attempt + 1}: {e}")
                if attempt == max_retries - 1:
                    return self.buffer_nodes, "Failed to process with VLM.", str(e)
        
        return self.buffer_nodes, "An unexpected error occurred.", "No selection was made."