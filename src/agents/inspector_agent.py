# File: my_multimodal_rag/src/agents/inspector_agent.py (Final version, using pure Transformers)

import torch
from typing import List, Tuple
from llama_index.core.schema import NodeWithScore
# 不再需要从 sentence_transformers 导入 CrossEncoder
# from sentence_transformers import CrossEncoder

# 直接从 transformers 导入核心组件
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

class InspectorAgent:
    def __init__(self, 
                 reranker_model_name: str = 'BAAI/bge-reranker-large'):
        
        print(f"Inspector: Loading unified reranker model directly with Transformers: {reranker_model_name} ...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # --- 核心修改：直接使用 transformers 加载 ---
        # 1. 加载分词器 (Tokenizer)
        self.reranker_tokenizer = AutoTokenizer.from_pretrained(
            reranker_model_name, 
            trust_remote_code=True,
            use_fast=False  
        )
        
        # 2. 加载模型
        self.reranker_model = AutoModelForSequenceClassification.from_pretrained(
            reranker_model_name,
            torch_dtype=torch.bfloat16, # 使用 bfloat16 提高效率
            device_map=self.device,
            trust_remote_code=True
        )
        # 将模型设置为评估模式
        self.reranker_model.eval()
        
        print("✅ Unified reranker model loaded successfully using Transformers.")

    def run(self, query: str, nodes: List[NodeWithScore], confidence_threshold: float = 0.7) -> Tuple[str, any, List[NodeWithScore], torch.Tensor]:
        if not nodes:
            return 'seeker', "Initial retrieval found no results.", [], torch.tensor(0.0, device=self.device)

        # 1. 手动实现重排逻辑
        node_contents = [node.get_content() for node in nodes]
        sentence_pairs = [(query, content) for content in node_contents]
        print(f"\n[Inspector] Performing reranking with {self.reranker_model.config._name_or_path}...")
        
        # 使用 torch.no_grad() 以节省显存并加速
        with torch.no_grad():
            # 使用分词器处理所有句子对
            inputs = self.reranker_tokenizer(
                sentence_pairs,
                padding=True,
                truncation=True,
                return_tensors='pt',
                max_length=512
            ).to(self.device)
            
            # 将处理好的输入传递给模型，获取原始分数 (logits)
            outputs = self.reranker_model(**inputs)
            rerank_scores_logits = outputs.logits.squeeze(-1) # 形状从 [N, 1] 变为 [N]
        
        # 将 rerank scores (logits) 存回 nodes
        for i in range(len(nodes)):
            nodes[i].score = rerank_scores_logits[i].item()
        
        nodes.sort(key=lambda x: x.score, reverse=True)
        print("✅ Reranking completed.")

        # 2. 从最高分的文档中计算可微的置信度分数
        if not nodes:
             return 'seeker', "Reranking resulted in zero nodes.", [], torch.tensor(0.0, device=self.device)
        
        top_score_logit = rerank_scores_logits.max() # 直接从 tensor 中获取最大值
        
        confidence_score = torch.sigmoid(top_score_logit)
        
        print(f"[Inspector] Evaluating confidence using reranker's top score...")
        print(f"  [Debug] Top Logit: {top_score_logit.item():.4f} -> Sigmoid Confidence: {confidence_score.item():.4f}")
        print(f"✅ Confidence evaluation completed. Top confidence: {confidence_score.item():.4f}")

        # 3. 根据置信度进行决策
        if confidence_score.item() > confidence_threshold:
            print("Decision: Evidence is sufficient. Proceeding to Synthesizer.")
            return 'synthesizer', "Evidence is sufficient.", nodes, confidence_score
        else:
            print("Decision: Evidence is insufficient. Sending feedback to Seeker.")
            feedback = f"The top reranked document was deemed not relevant enough (confidence: {confidence_score.item():.2f}). We need documents that more directly answer the question: '{query}'"
            return 'seeker', feedback, nodes, confidence_score