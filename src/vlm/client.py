# File: src/vlm/client.py
import os
from typing import List, Dict, Any, Optional
import requests

class VLMClient:
    def __init__(self, api_base: str, api_key: str, model: str):
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.model = model

    def multi_image_json(self, prompt: str, image_paths: List[str], timeout: int = 60) -> Dict[str, Any]:
        """
        发送多图 + 文本，要求模型严格返回 JSON。
        这里接口按你们的 VLM 服务来改；先给个占位。
        """
        # TODO: 替换为你们真实的 API 协议
        files = [("images", open(p, "rb")) for p in image_paths]
        headers = {"Authorization": f"Bearer {self.api_key}"}
        data = {"model": self.model, "prompt": prompt}
        r = requests.post(f"{self.api_base}/v1/multi-image-json", headers=headers, data=data, files=files, timeout=timeout)
        r.raise_for_status()
        return r.json()