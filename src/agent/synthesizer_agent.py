# File: src/agents/synthesizer_agent.py
# Final corrected version that passes image paths (str) instead of Image objects to the VLM.

from __future__ import annotations
import os
import json
from typing import List, Tuple, Any

from PIL import Image
from llama_index.core.schema import NodeWithScore, ImageNode, TextNode

# --- 您需要自行解决的导入 ---
from src.agent.agent_prompt import answer_prompt
from src.agent.map_dict import page_map_dict_normal
from src.utils.parse_tool import extract_json
# from llms.vlm_interface import VLM # 您的VLM模型接口

class SynthesizerAgent:
    """
    Synthesizer Agent - VLM-Powered Final Answer Generator.
    This implementation mirrors Vidorag's logic and passes the correct data types to the VLM.
    """
    def __init__(self, vlm: Any, image_base_dir: str):
        """
        Initializes the SynthesizerAgent.
        Args:
            vlm: An instance of your Visual Language Model wrapper.
            image_base_dir (str): The base directory where raw images are stored. Needed for path derivation.
        """
        self.vlm = vlm
        self.image_base_dir = image_base_dir
        # --- Vidorag logic for multi-image handling ---
        self.synthesizer_multi_image = True 
        self.page_map = page_map_dict_normal
        print("✅ SynthesizerAgent is ready.")

    def run(self, 
            query: str, 
            evidence_nodes: List[NodeWithScore], 
            candidate_answer: str | None = None
           ) -> Tuple[str, str]:
        """
        Generates a final answer using the VLM based on the provided evidence.
        """
        if not evidence_nodes:
            return "No evidence provided.", "I cannot answer the question based on the information I found."

        # --- [修正] 智能提取文本和图片路径 ---
        ref_image_paths = []
        text_evidence = []
        for node_with_score in evidence_nodes:
            node = node_with_score.node
            metadata = getattr(node, "metadata", {}) or {}
            
            # 统一逻辑：从任何节点类型中尝试寻找图片路径
            image_path = None
            explicit_path = metadata.get("image_path") or metadata.get("file_path")
            if explicit_path and isinstance(explicit_path, str):
                image_path = explicit_path
            elif 'filename' in metadata and isinstance(metadata['filename'], str):
                base_filename = os.path.splitext(metadata['filename'])[0]
                potential_path = os.path.join(self.image_base_dir, f"{base_filename}.jpg")
                image_path = potential_path

            if image_path and os.path.exists(image_path):
                ref_image_paths.append(image_path)
            
            if node.text:
                text_evidence.append(node.text)

        # --- Vidorag Synthesizer.run 的核心逻辑 ---
        context_info = ""
        if candidate_answer:
            context_info += candidate_answer
        if text_evidence:
            # 去重并保持顺序
            unique_texts = list(dict.fromkeys(text_evidence))
            context_info += "\n\n## Reference Text\n" + "\n\n".join(unique_texts)
        
        augmented_query = query + '\n\n## Related Information\n' + context_info if context_info else query

        prompt = answer_prompt.replace('{question}', augmented_query).replace('{page_map}', self.page_map.get(len(ref_image_paths), ""))
        
        # 准备图像输入 (注意：这里现在是路径列表)
        input_images_for_vlm = ref_image_paths
        
        # VLM 调用与重试循环
        max_retries = 3
        for attempt in range(max_retries):
            try:
                print(f"\n[Synthesizer] Calling VLM with {len(input_images_for_vlm)} image paths to generate final answer...")
                final_answer_response = self.vlm.generate(query=prompt, image=input_images_for_vlm)
                
                final_answer_response_json = extract_json(final_answer_response)
                reason = final_answer_response_json.get('reason')
                answer = final_answer_response_json.get('answer')

                if reason is None or answer is None:
                    raise ValueError("'reason' or 'answer' field is missing in the final response.")
                
                print("✅ [Synthesizer] Successfully generated final answer.")
                return reason, answer

            except Exception as e:
                print(f"❌ [Synthesizer] Error on attempt {attempt + 1}: {e}")
                # 为了调试，打印原始回复
                if 'final_answer_response' in locals():
                    print("Raw response:", final_answer_response)

                if attempt == max_retries - 1:
                    return "Error during final answer synthesis.", f"Failed to generate a valid response after {max_retries} attempts. Error: {e}"

        # 理论上不会执行到这里
        return "Synthesis failed.", "An unexpected error occurred after the retry loop."  