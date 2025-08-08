import os
import re
import torch
from typing import List, Optional
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from llama_index.core.schema import BaseNode


class SynthesizerAgent:
    """
    面向 T5/FLAN 的 Seq2Seq 生成器（稳定版） + 可选的 Qwen-VL 支路
    修复点：
      1) （新增）若存在图片证据且注入了 VLM 客户端，优先用 VLM 直接抽取短答案；
         拿不到再回退到原有 T5 文本生成。
      2) 不使用 chat_template
      3) 严格 token 级限长（<= 512）
      4) Top-1 证据优先保留，其它片段小额度补充
      5) 动态压缩上下文，直到总 token 数满足上限
    """

    def __init__(self, model_name: str, vlm_client: Optional[object] = None, use_vlm: bool = True):
        print(f"Synthesizer: Loading generation model: {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.tokenizer.model_max_length = 10000

        # 选择 dtype
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            torch_dtype = torch.bfloat16
        elif torch.cuda.is_available():
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.float32

        # 模型与设备
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map="auto",
            trust_remote_code=True,
        )
        self.device = next(self.model.parameters()).device
        print("✅ Generation model loaded successfully.")

        # 超参（可按需调整）
        self.max_input_tokens = 512      # 编码器最大输入
        self.max_new_tokens = 256        # 生成长度
        self.max_nodes = 3               # 使用的证据片段数量
        # 片段预算：优先保留 Top-1，其他给较小额度
        self.first_max_tokens = 360
        self.rest_max_tokens = 90
        self.min_rest_tokens = 40        # 动态压缩时的片段最小额度
        self.compress_step = 30          # 每次压缩步长（tokens）

        # VLM
        self.vlm_client = vlm_client
        self.use_vlm = bool(use_vlm)

    # --- 工具：把文本截到指定 token 数 ---
    def _truncate_by_tokens(self, text: str, max_tokens: int) -> str:
        if not isinstance(text, str):
            text = str(text)
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        if len(ids) <= max_tokens:
            return text
        ids = ids[:max_tokens]
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    def _pack_snippets(self, query: str, raw_snippets: List[str]) -> str:
        """
        动态把片段打包进 prompt；若超限就按策略压缩/丢弃尾部片段。
        """
        # 先按预算分别截断
        clipped = []
        for i, s in enumerate(raw_snippets):
            budget = self.first_max_tokens if i == 0 else self.rest_max_tokens
            clipped.append(self._truncate_by_tokens(s, budget))

        def build_prompt(snips: List[str]) -> str:
            context_str = "\n\n".join(snips)
            # 更偏“抽取式”的提示，降低“轻易说不知道”的倾向
            return (
                "Answer the question using ONLY the provided context. "
                "If possible, extract the exact phrase from the context.\n\n"
                "Context:\n"
                f"{context_str}\n\n"
                f"Question: {query}\n"
                "Answer (concise):"
            )

        prompt = build_prompt(clipped)
        ids = self.tokenizer.encode(prompt, add_special_tokens=True)

        # 若仍超限，则从尾部开始压缩/移除
        while len(ids) > self.max_input_tokens and len(clipped) > 0:
            # 先尝试压缩最后一个片段
            last = clipped[-1]
            # 当前最后片段的 token 长度
            last_ids = self.tokenizer.encode(last, add_special_tokens=False)
            if len(last_ids) > self.min_rest_tokens:
                # 压缩一步
                new_budget = max(self.min_rest_tokens, len(last_ids) - self.compress_step)
                clipped[-1] = self._truncate_by_tokens(last, new_budget)
            else:
                # 最小额度也不够，则移除该片段
                clipped.pop()

            prompt = build_prompt(clipped)
            ids = self.tokenizer.encode(prompt, add_special_tokens=True)

        return prompt

    @staticmethod
    def _collect_texts(nodes: List[BaseNode], limit: int = 3) -> List[str]:
        texts: List[str] = []
        for n in nodes[:limit]:
            try:
                c = n.get_content()
            except Exception:
                c = ""
            if not isinstance(c, str):
                c = str(c or "")
            c = c.strip()
            if c:
                texts.append(c)
        return texts

    @staticmethod
    def _collect_image_paths(nodes: List[BaseNode], limit: int = 4) -> List[str]:
        """
        从节点元数据里抽取 image_path（由 ImageRetriever 提供）。
        仅返回真实存在/可访问的本地路径或 URL（这里不过滤 URL，VLM 端可支持 URL 时直接使用）。
        """
        paths: List[str] = []
        for n in nodes:
            md = getattr(n, "metadata", {}) or {}
            p = md.get("image_path")
            if isinstance(p, str) and p:
                paths.append(p)
            if len(paths) >= limit:
                break
        return paths

    @staticmethod
    def _post_clean(ans: str) -> str:
        """简单清洗 VLM 短答案"""
        if not ans:
            return ans
        s = re.sub(r"\s+", " ", ans).strip()
        # 去掉多余的尾部标点
        s = s.rstrip(" .,:;")
        # 控长度，避免 VLM 啰嗦
        return s[:200]

    def _try_vlm(self, query: str, nodes: List[BaseNode]) -> Optional[str]:
        """
        若有图片证据且配置了 VLM 客户端，则让 VLM 直接抽取短答案。
        返回 None 表示放弃（交给 T5 处理）。
        """
        if not self.use_vlm or self.vlm_client is None:
            return None

        image_paths = self._collect_image_paths(nodes, limit=6)
        if not image_paths:
            return None

        try:
            # 你的 QwenVLM 实现应提供 ask(question, image_paths) -> str
            vlm_answer = self.vlm_client.ask(question=query, image_paths=image_paths)
            vlm_answer = self._post_clean(vlm_answer or "")
            if vlm_answer:
                print("[Synthesizer][VLM] Using Qwen-VL candidate answer:", vlm_answer)
                return vlm_answer
        except Exception as e:
            print(f"[Synthesizer][VLM] error: {e}")

        return None

    def generate(self, query: str, relevant_nodes: List[BaseNode]) -> str:
        if not relevant_nodes:
            return "Insufficient information to generate an answer."

        # （新增）先走 VLM 支路（仅当有图片证据）
        vlm_ans = self._try_vlm(query, relevant_nodes)
        if vlm_ans:
            return vlm_ans

        # 取前若干证据（Top-1 最重要）
        raw_snippets: List[str] = []
        for node in relevant_nodes[: self.max_nodes]:
            try:
                content = node.get_content()
            except Exception:
                content = ""
            content = content if isinstance(content, str) else str(content)
            if content.strip():
                raw_snippets.append(content)

        if not raw_snippets:
            return "Insufficient information to generate an answer."

        # 动态打包，确保 <= 512 tokens
        prompt = self._pack_snippets(query, raw_snippets)

        print("\n[Synthesizer] Generating final answer based on high-quality evidence...")

        # 编码 + 严格截断
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_tokens,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # 生成
        eos_id = self.tokenizer.eos_token_id
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            num_beams=1,
            eos_token_id=eos_id if eos_id is not None else None,
        )

        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        return text