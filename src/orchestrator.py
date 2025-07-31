# src/orchestrator.py (最终版)

from src.agents.seeker_agent import SeekerAgent
from src.agents.inspector_agent import InspectorAgent
from src.agents.synthesizer_agent import SynthesizerAgent
from src.models.gumbel_selector import GumbelModalSelector # 引入Gumbel

class RAGOrchestrator:
    def __init__(self, gumbel_selector: GumbelModalSelector, seeker: SeekerAgent, inspector: InspectorAgent, synthesizer: SynthesizerAgent):
        self.gumbel_selector = gumbel_selector
        self.seeker = seeker
        self.inspector = inspector
        self.synthesizer = synthesizer

    def run(self, query: str, query_embedding):
        # 1. 模态决策
        # 注意：Gumbel选择器需要被移动到和嵌入向量相同的设备上
        self.gumbel_selector.to(query_embedding.device) 
        modality_one_hot = self.gumbel_selector(query_embedding)
        modality_index = modality_one_hot.argmax().item()

        # 2. 动态检索策略
        top_k = 3
        retrieved_nodes = self.seeker.run(query, modality=modality_index, top_k=top_k)

        # 3. 检验
        status, information, nodes, confidence = self.inspector.run(query, retrieved_nodes)

        # 4. 迭代式细化
        if status == 'seeker':
            print("\n" + "="*20 + " 证据不足，进入二次检索 " + "="*20)
            top_k = 5 # 动态扩展K值
            print(f"动态扩展K值，新的TopK={top_k}")
            retrieved_nodes = self.seeker.run(query, modality=modality_index, top_k=top_k, feedback=information)
            status, information, nodes, confidence = self.inspector.run(query, retrieved_nodes)

        # 5. 最终决策
        if status == 'synthesizer':
            answer = self.synthesizer.generate(query, [n.node for n in nodes])
            return answer
        else:
            return "经过多轮检索，仍未找到足够的信息来回答问题。"