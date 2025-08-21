import os
import torch
import base64
from PIL import Image
from pathlib import Path
from io import BytesIO

# ✅ Gemini
import google.generativeai as genai

# 导入 DashScope (用于通义千问API)
import dashscope

os.environ['GEMINI_API_KEY'] = 'AIzaSyAqYzObls24w0pGO0WjhMicery6R22nfn0'

def _encode_image(image_path):
    if isinstance(image_path, Image.Image):
        buffered = BytesIO()
        image_path.save(buffered, format="JPEG")
        img_data = buffered.getvalue()
        return base64.b64encode(img_data).decode("utf-8")
    else:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")


class Qwen_VL_2_5:
    def __init__(self, model_name="Qwen/Qwen2.5-VL-7B-Instruct"):
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto",
            # attn_implementation="flash_attention_2"
            attn_implementation="sdpa"  # 修改到这里，使用 PyTorch 内置的、无需额外安装的实现
        )
        self.processor = AutoProcessor.from_pretrained(model_name)

    def generate(self, query, images):
        from qwen_vl_utils import process_vision_info
        if not isinstance(query, str):
            raise ValueError("Query must be a string")

        content = [{"type": "image", "image": img} for img in images]
        content.append({"type": "text", "text": query})
        messages = [{"role": "user", "content": content}]

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to(self.model.device)

        output_ids = self.model.generate(**inputs, max_new_tokens=1024)
        trimmed_ids = [
            out[len(inp):] for inp, out in zip(inputs.input_ids, output_ids)
        ]
        outputs = self.processor.batch_decode(
            trimmed_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return outputs[0]


class LLM:
    def __init__(self, model_name):
        self.model_name = model_name
        # ✅ (新增!) 增加了对 'qwen-vl-max' 的处理分支
        if 'qwen-vl-max' in self.model_name:
            api_key = os.getenv("DASHSCOPE_API_KEY")
            if not api_key:
                raise EnvironmentError("❌ 错误: 环境变量 DASHSCOPE_API_KEY 未设置。")
            dashscope.api_key = api_key
            # self.model 在此情况下就是模型名称字符串，供API调用
            self.model = self.model_name
            print(f"✅ DashScope API for '{self.model_name}' configured.")

        elif "Qwen2.5-VL" in model_name:
            self.model = Qwen_VL_2_5(model_name)

        elif model_name.startswith("gpt"):
            from openai import OpenAI
            self.model = OpenAI()

        elif model_name.startswith("gemini"):
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise EnvironmentError("❌ GEMINI_API_KEY not set in environment variables.")
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(model_name)

        else:
            raise ValueError(f"Unsupported model: {model_name}")

    def generate(self, query='', image=None):
        image = image or []
        if isinstance(image, str):
            image = [image]

        # ✅ (新增!) 增加了调用 DashScope API 的逻辑
        if 'qwen-vl-max' in self.model_name:
            messages = [{'role': 'user', 'content': []}]
            
            # 组织图片内容 (DashScope 需要 'file://' 格式的本地路径)
            for img_path in image:
                local_image_path = f'file://{Path(img_path).resolve()}'
                messages[0]['content'].append({'image': local_image_path})
            
            # 组织文字内容
            messages[0]['content'].append({'text': query})
            
            try:
                response = dashscope.MultiModalConversation.call(model=self.model, messages=messages)

                if response.status_code == 200:
                # API返回的内容可能是一个列表，例如 [{'text': '...'}]
                # 我们需要从中提取出真正的文本内容
                    content = response.output.choices[0].message.content
                    if isinstance(content, list) and len(content) > 0 and 'text' in content[0]:
                        return content[0]['text']
                    else:
                    # 如果格式不是预期的列表，则按原样返回（以防万一）
                        return str(content) 

                else:
                    return f"API错误: 代码 {response.code}, 信息: {response.message}"
            except Exception as e:
                raise RuntimeError(f"DashScope API 调用失败: {e}")
            
        # ✅ Qwen2.5
        elif "Qwen2.5-VL" in self.model_name:
            return self.model.generate(query, image)

        # ✅ GPT (base64 编码 + chat.completions)
        elif self.model_name.startswith("gpt"):
            content = [{"type": "text", "text": query}]
            for img_path in image:
                base64_img = _encode_image(img_path)
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_img}"
                    }
                })
            completion = self.model.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": content}]
            )
            return completion.choices[0].message.content

        # ✅ Gemini (直接传 PIL Image)
        elif self.model_name.startswith("gemini"):
            try:
                if image:
                    pil_img = Image.open(image[0]).convert("RGB")
                    print("🔵 Sending image to Gemini ...")
                    response = self.model.generate_content([query, pil_img])
                    print("🟢 Gemini response received.")

                else:
                    response = self.model.generate_content(query)
                return response.text
            except Exception as e:
                raise RuntimeError(f"Gemini API call failed: {e}")

        else:
            raise ValueError("Unsupported model in generate()")


# ✅ 示例用法
if __name__ == "__main__":
    llm = LLM("gemini-1.5-pro-latest")
    response = llm.generate(query="Describe this image in 3 words.", image=["your_image.jpg"])
    print(response)
