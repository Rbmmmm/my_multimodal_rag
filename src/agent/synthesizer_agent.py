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
        print("[1f] [Synthesizer] ✅ SynthesizerAgent is ready.")

    def run(self, 
            query: str, 
            candidate_answer: str | None = None,
            ref_images : List[str] | None = None
           ) -> Tuple[str, str]:
        
        if candidate_answer is not None:
            query = query + '\n\n## Related Information\n' + candidate_answer
        prompt = answer_prompt.replace('{question}',query).replace('{page_map}',self.page_map[len(ref_images)])

        input_images = ref_images

        while True:
            final_answer_response = self.vlm.generate(query=prompt,image=input_images)
            print("[synthesizer] 给出的答案为:\n", final_answer_response)
            try:
                final_answer_response_json = extract_json(final_answer_response)
                reason = final_answer_response_json.get('reason',None)
                answer = final_answer_response_json.get('answer',None)
                if reason is None or answer is None :
                    raise Exception('Synthesizer time out')
                return reason, answer
            except Exception as e:
                print(e)
                print(final_answer_response)
                print("answer")
                
                
