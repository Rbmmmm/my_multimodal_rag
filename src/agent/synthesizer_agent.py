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
        
    def extract_json(text: str) -> dict:
        # 1. 预处理字符串，修复无效的转义符
        cleaned_text = text.replace('\\(', '(').replace('\\)', ')')
        
        # 2. 清理LLM可能返回的Markdown代码块标记
        if cleaned_text.strip().startswith("```json"):
            cleaned_text = cleaned_text.strip()[7:-3].strip()
        elif cleaned_text.strip().startswith("```"):
            cleaned_text = cleaned_text.strip()[3:-3].strip()

        # 3. 使用清理后的文本进行解析
        return json.loads(cleaned_text)

    def run(self, 
            query: str, 
            candidate_answer: str | None = None,
            ref_images : List[str] | None = None
           ) -> Tuple[str, str]:
        
        if candidate_answer is not None:
            query = query + '\n\n## Related Information\n' + candidate_answer
        prompt = answer_prompt.replace('{question}',query).replace('{page_map}',self.page_map[len(ref_images)])

        input_images = ref_images

        # while True:
        #     final_answer_response = self.vlm.generate(query=prompt,image=input_images)
        #     print("[synthesizer] 给出的答案为:\n", final_answer_response)
        #     try:
        #         final_answer_response_json = extract_json(final_answer_response)
        #         reason = final_answer_response_json.get('reason',None)
        #         answer = final_answer_response_json.get('answer',None)
        #         if reason is None or answer is None :
        #             raise Exception('Synthesizer time out')
        #         return reason, answer
        #     except Exception as e:
        #         print(e)
        #         print(final_answer_response)
        #         print("answer")
        
            # --- START: 使用带重试限制的循环替换 while True ---
        max_retries = 5  # 最多重试3次
        last_error = None
        
        for attempt in range(max_retries):
            try:
                # 调用VLM获取原始文本
                final_answer_response = self.vlm.generate(query=prompt, image=input_images)
                print(f"[Synthesizer] VLM Response (Attempt {attempt + 1}/{max_retries}):\n{final_answer_response}")

                # 使用我们健壮的 extract_json 函数
                final_answer_response_json = extract_json(final_answer_response)

                reason = final_answer_response_json.get('reason')
                answer = final_answer_response_json.get('answer')

                # 如果成功提取到 reason 和 answer，就直接返回并结束函数
                if reason is not None and answer is not None:
                    return reason, str(answer) # 成功退出
                else:
                    # JSON有效但缺少关键字段，也视为一种错误
                    last_error = ValueError("Parsed JSON is missing 'reason' or 'answer' key.")
                    print(f"Attempt {attempt + 1} failed: {last_error}")
                    continue # 继续下一次重试

            except Exception as e:
                # 捕获所有异常（包括JSON解析失败）
                last_error = e
                print(f"❌ [Synthesizer] Attempt {attempt + 1} failed with error: {e}")
                # 继续下一次重试
                continue
                
                
 