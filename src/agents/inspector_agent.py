# File: my_multimodal_rag/src/agents/inspector_agent.py

import torch
from typing import List, Tuple, Any
from llama_index.core.schema import NodeWithScore
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class InspectorAgent:
    """
    使用纯 Transformers 的统一重排器（Cross-Encoder）。
    默认模型：BAAI/bge-reranker-large
    """

    def __init__(self, reranker_model_name: str = "BAAI/bge-reranker-large"):
        print(f"Inspector: Loading unified reranker with Transformers: {reranker_model_name} ...")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 选择合适 dtype（优先 bf16，其次 fp16，最后 fp32）
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            torch_dtype = torch.bfloat16
        elif torch.cuda.is_available():
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.float32

        # 1) Tokenizer
        self.reranker_tokenizer = AutoTokenizer.from_pretrained(
            reranker_model_name,
            trust_remote_code=True,
            use_fast=False  # 某些模型的快版分词器在 pair 模式下不稳定
        )

        # 2) Model（不滥用 device_map，加载后手动挪到目标设备）
        self.reranker_model = AutoModelForSequenceClassification.from_pretrained(
            reranker_model_name,
            torch_dtype=torch_dtype,
            trust_remote_code=True
        )
        self.reranker_model.to(self.device)
        self.reranker_model.eval()

        print(f"✅ Unified reranker loaded. device={self.device}, dtype={torch_dtype}")

    # ---- 内部工具：把 Node 内容安全地转成文本 ----
    @staticmethod
    def _to_text_view(node: NodeWithScore) -> str:
        """
        将节点内容转换为可重排的文本。
        若是图像节点，需确保在建索引时已填入 caption/OCR 文本。
        """
        try:
            content = node.get_content()
        except Exception:
            content = ""
        if isinstance(content, str):
            return content
        return str(content) if content is not None else ""

    def run(
        self,
        query: str,
        nodes: List[NodeWithScore],
        confidence_threshold: float = 0.7
    ) -> Tuple[str, Any, List[NodeWithScore], torch.Tensor]:
        """
        对候选 nodes 进行重排，计算置信度，并决定是否进入生成阶段。
        返回:
          status: 'synthesizer' 或 'seeker'
          feedback: 文本反馈（当需回退检索时）
          nodes: 排序后的节点，节点.score 为 reranker 的 logit 分数
          confidence: torch.Tensor(标量)，对 top-1 logit 做 sigmoid 后的置信度
        """
        if not nodes:
            return 'seeker', "Initial retrieval found no results.", [], torch.tensor(0.0, device=self.device)

        # 1) 准备文本对 (query, node_text)
        node_texts = [self._to_text_view(n) for n in nodes]
        print(f"\n[Inspector] Performing reranking with {self.reranker_model.config._name_or_path} ...")

        with torch.no_grad():
            # 成对编码：text=[query]*N, text_pair=node_texts
            inputs = self.reranker_tokenizer(
                [query] * len(node_texts),
                node_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            outputs = self.reranker_model(**inputs)
            # bge-reranker-large 输出 [N, 1]，压成 [N]
            rerank_scores_logits = outputs.logits.squeeze(-1)  # [N]

        # 2) 写回分数并排序（降序）
        for i in range(len(nodes)):
            nodes[i].score = float(rerank_scores_logits[i].item())
        nodes.sort(key=lambda x: x.score, reverse=True)
        print("✅ Reranking completed.")

        # 3) 计算置信度（对 top-1 logit 做 sigmoid）
        if len(nodes) == 0:
            return 'seeker', "Reranking resulted in zero nodes.", [], torch.tensor(0.0, device=self.device)

        top_score_logit = torch.max(rerank_scores_logits)           # 标量 tensor
        confidence_score = torch.sigmoid(top_score_logit)           # 标量 tensor

        print("[Inspector] Evaluating confidence from top logit ...")
        print(f"  [Debug] Top Logit: {top_score_logit.item():.4f} -> Sigmoid: {confidence_score.item():.4f}")
        print(f"✅ Confidence evaluation completed. Top confidence: {confidence_score.item():.4f}")

        # 4) 决策
        if confidence_score.item() > confidence_threshold:
            print("Decision: Evidence is sufficient. Proceeding to Synthesizer.")
            return 'synthesizer', "Evidence is sufficient.", nodes, confidence_score
        else:
            print("Decision: Evidence is insufficient. Sending feedback to Seeker.")
            feedback = (
                "The top reranked document was deemed not relevant enough "
                f"(confidence: {confidence_score.item():.2f}). "
                f"We need documents that more directly answer the question: '{query}'"
            )
            return 'seeker', feedback, nodes, confidence_score