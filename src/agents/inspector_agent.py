# File: my_multimodal_rag/src/agents/inspector_agent.py (SmolLM3B - differentiable version)

import torch
import torch.nn.functional as F
import json
from typing import List, Tuple
from llama_index.core.schema import NodeWithScore
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import CrossEncoder

class InspectorAgent:
    def __init__(self, 
                 reranker_model_name: str = 'BAAI/bge-reranker-large',
                 eval_model_name: str = 'HuggingFaceTB/SmolLM3-3B'):
        
        print(f"Inspector: Loading reranker model: {reranker_model_name} ...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.reranker_model = CrossEncoder(reranker_model_name, max_length=512, device=self.device)
        print("✅ Reranker model loaded successfully.")

        print(f"Inspector: Loading lightweight evaluation model: {eval_model_name} ...")
        self.eval_tokenizer = AutoTokenizer.from_pretrained(eval_model_name)
        self.eval_model = AutoModelForCausalLM.from_pretrained(
            eval_model_name, torch_dtype=torch.bfloat16, device_map="auto"
        )
        print("✅ Lightweight evaluation model loaded successfully.")

        # Pre-fetch token IDs for "yes" and "no"
        self.yes_token_id = self.eval_tokenizer.encode('yes', add_special_tokens=False)[-1]
        self.no_token_id = self.eval_tokenizer.encode('no', add_special_tokens=False)[-1]
        print(f"✅ Token IDs acquired: 'yes' -> {self.yes_token_id}, 'no' -> {self.no_token_id}")

    def _get_confidence_score(self, query: str, node: NodeWithScore) -> torch.Tensor:
        """
        Evaluate a single node using the lightweight model and return a differentiable confidence score ∈ [0, 1].
        """
        context = node.get_content()
        prompt = f"""
        Does the following context contain the answer to the question? Answer with only "yes" or "no".

        Context:
        "{context[:1500]}"

        Question:
        "{query}"

        Answer (yes or no):
        """
        inputs = self.eval_tokenizer(prompt, return_tensors="pt").to(self.eval_model.device)

        with torch.no_grad():
            outputs = self.eval_model(**inputs)

        last_token_logits = outputs.logits[0, -1, :]
        yes_logit = last_token_logits[self.yes_token_id]
        no_logit = last_token_logits[self.no_token_id]

        probabilities = F.softmax(torch.stack([no_logit, yes_logit]), dim=0)
        confidence_score = probabilities[1]

        print(f"  [Debug] Logits -> yes: {yes_logit:.2f}, no: {no_logit:.2f} | Probabilities -> yes: {confidence_score:.4f}, no: {probabilities[0]:.4f}")
        return confidence_score

    def run(self, query: str, nodes: List[NodeWithScore], confidence_threshold: float = 0.7) -> Tuple[str, any, List[NodeWithScore], torch.Tensor]:
        if not nodes:
            return 'seeker', "Initial retrieval found no results.", [], torch.tensor(0.0)

        node_contents = [node.get_content() for node in nodes]
        sentence_pairs = [(query, content) for content in node_contents]
        print("\n[Inspector] Performing reranking ...")
        rerank_scores = self.reranker_model.predict(sentence_pairs)

        for i in range(len(nodes)):
            nodes[i].score = rerank_scores[i].item()
        nodes.sort(key=lambda x: x.score, reverse=True)
        print("✅ Reranking completed.")

        top_node = nodes[0]
        print("[Inspector] Evaluating confidence using lightweight model ...")
        confidence_score = self._get_confidence_score(query, top_node)
        print(f"✅ Confidence evaluation completed. Top confidence: {confidence_score.item():.4f}")

        if confidence_score.item() > confidence_threshold:
            print("Decision: Evidence is sufficient. Proceeding to Synthesizer.")
            return 'synthesizer', "Evidence is sufficient.", nodes, confidence_score
        else:
            print("Decision: Evidence is insufficient. Sending feedback to Seeker.")
            feedback = f"The top retrieved document was deemed not directly relevant. We need documents that more directly answer the question: '{query}'"
            return 'seeker', feedback, nodes, confidence_score