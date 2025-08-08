import os
import base64
import mimetypes
from typing import List, Optional, Dict, Any

# 优先尝试官方 SDK；没有则退回 HTTP
try:
    import dashscope  # type: ignore
    _HAS_DASHSCOPE = True
except Exception:
    _HAS_DASHSCOPE = False

import requests


def _is_http_url(path: str) -> bool:
    return path.startswith("http://") or path.startswith("https://")


def _to_file_uri(path: str) -> str:
    # dashscope 多模态 SDK 支持 file:// 协议本地图片
    if path.startswith("file://") or _is_http_url(path):
        return path
    return f"file://{os.path.abspath(path)}"


class QwenVLM:
    """
    极简 Qwen-VL 客户端封装：
      - 读取环境变量 DASHSCOPE_API_KEY
      - ask(question, image_paths) -> str
      - 支持多图输入（最多 6 张做个兜底保护）
      - 优先使用 dashscope SDK；若不可用，fallback 到 HTTP REST
    """

    def __init__(self, model: str = "qwen-vl-max", timeout: int = 60):
        self.model = model
        self.timeout = timeout
        self.api_key = os.getenv("DASHSCOPE_API_KEY", "").strip()
        if not self.api_key:
            raise RuntimeError(
                "DASHSCOPE_API_KEY is not set. Please export it in your environment."
            )
        # 配置 SDK 的 key（若存在 SDK）
        if _HAS_DASHSCOPE:
            dashscope.api_key = self.api_key  # type: ignore

    # ----------------------
    # 对外主方法
    # ----------------------
    def ask(self, question: str, image_paths: List[str]) -> str:
        if not image_paths:
            # 没图就让上层回退到文本模型
            return ""

        # 限制一下图片数量，避免过多图导致 token 或时延爆炸
        images = image_paths[:6]

        # 统一构建 Qwen-VL 消息结构
        content: List[Dict[str, Any]] = []
        for p in images:
            # SDK 支持 URL 或 file:// 路径
            content.append({"image": _to_file_uri(p)})

        content.append({"text": self._build_prompt(question)})

        if _HAS_DASHSCOPE:
            return self._call_with_sdk(content)
        else:
            return self._call_with_http(content)

    # ----------------------
    # Prompt 轻度规范
    # ----------------------
    def _build_prompt(self, question: str) -> str:
        return (
            "You are a precise visual reader. Answer with a SHORT phrase only.\n"
            "If the answer is a number, date or a short label, output ONLY that token.\n\n"
            f"Question: {question}"
        )

    # ----------------------
    # SDK 路径
    # ----------------------
    def _call_with_sdk(self, content: List[Dict[str, Any]]) -> str:
        # 兼容 dashscope 的多模态会话接口
        try:
            # 新版 SDK：client.chat.completions；老版：MultiModalConversation
            # 我们优先尝试新版，失败再回退老版
            try:
                from dashscope import Chat  # type: ignore
                rsp = Chat.multi_modal_conversation(  # type: ignore
                    model=self.model,
                    messages=[{"role": "user", "content": content}],
                    timeout=self.timeout,
                )
                return self._extract_sdk_text(rsp)
            except Exception:
                from dashscope import MultiModalConversation  # type: ignore
                rsp = MultiModalConversation.call(  # type: ignore
                    model=self.model,
                    messages=[{"role": "user", "content": content}],
                    timeout=self.timeout,
                )
                return self._extract_sdk_text(rsp)
        except Exception as e:
            # SDK 路径失败，兜底走 HTTP
            print(f"[QwenVLM] dashscope SDK path failed: {e}. Fallback to HTTP.")
            return self._call_with_http(content)

    @staticmethod
    def _extract_sdk_text(resp: Any) -> str:
        """
        兼容不同 SDK 返回结构，尽量抽出第一条文本。
        常见路径：resp.output.choices[0].message.content[0]['text']
        """
        try:
            output = getattr(resp, "output", None) or {}
            choices = output.get("choices") or []
            if choices:
                msg = choices[0].get("message", {})
                cont = msg.get("content") or []
                # content 是一个列表：[{ "text": "..." }, {"image": "..."} ...]
                for seg in cont:
                    if isinstance(seg, dict) and "text" in seg:
                        return str(seg["text"]).strip()
        except Exception:
            pass
        # 再试 raw dict
        try:
            if isinstance(resp, dict):
                choices = (((resp.get("output") or {}).get("choices")) or [])
                if choices:
                    cont = ((choices[0].get("message") or {}).get("content")) or []
                    for seg in cont:
                        if isinstance(seg, dict) and "text" in seg:
                            return str(seg["text"]).strip()
        except Exception:
            pass
        return ""

    # ----------------------
    # HTTP 路径
    # ----------------------
    def _call_with_http(self, content: List[Dict[str, Any]]) -> str:
        """
        直连 DashScope HTTP 接口。兼容 URL / file://；对于本地文件也支持直接上传（必要时）。
        参考文档：阿里云 DashScope 多模态生成接口
        """
        url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "input": {
                "messages": [
                    {"role": "user", "content": content}
                ]
            }
        }

        # HTTP JSON 调用（图片为 URL 或 file:// 的情况）
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
            r.raise_for_status()
            data = r.json()
            # 解析与 SDK 一致的路径
            return self._extract_sdk_text(data)
        except Exception as e:
            print(f"[QwenVLM] HTTP JSON path failed: {e}")
            # 作为兜底：如果是纯本地路径且不支持 file://，可以尝试 Base64 直传（有的版本不支持，这里只作保底）
            try:
                b64_content = self._content_with_base64_images(content)
                payload_b64 = {
                    "model": self.model,
                    "input": {"messages": [{"role": "user", "content": b64_content}]}
                }
                r2 = requests.post(url, headers=headers, json=payload_b64, timeout=self.timeout)
                r2.raise_for_status()
                data2 = r2.json()
                return self._extract_sdk_text(data2)
            except Exception as ee:
                print(f"[QwenVLM] HTTP base64 fallback failed: {ee}")
                return ""

    def _content_with_base64_images(self, content: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        将 file:// 或普通本地路径转换为 base64:image/<mime>;base64,<data> 结构。
        对于 URL 或者已是 file:// 的不一定需要，但作为兜底都尝试转一下。
        """
        out: List[Dict[str, Any]] = []
        for seg in content:
            if "image" in seg:
                src = seg["image"]
                if _is_http_url(src):
                    # URL 直接保留
                    out.append({"image": src})
                else:
                    # 去掉 file:// 前缀
                    path = src.replace("file://", "")
                    try:
                        with open(path, "rb") as f:
                            raw = f.read()
                        mime = mimetypes.guess_type(path)[0] or "image/png"
                        b64 = base64.b64encode(raw).decode("utf-8")
                        out.append({"image": f"data:{mime};base64,{b64}"})
                    except Exception:
                        # 失败就尽量原样放回，避免彻底丢图
                        out.append({"image": src})
            elif "text" in seg:
                out.append(seg)
        return out