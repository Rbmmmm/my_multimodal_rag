# File: src/orchestrator.py
# Final version with state management for stateful Agents.

from __future__ import annotations
import torch
from typing import Optional, Dict, Any, List

# --- 导入我们新设计的 Agent 和 SearchEngine ---
from src.searcher import SearchEngine
from src.agent.seeker_agent import SeekerAgent
from src.agent.inspector_agent import InspectorAgent
from src.agent.synthesizer_agent import SynthesizerAgent
from src.models.gumbel_selector import GumbelModalSelector

class RAGOrchestrator:
    """
    RAGOrchestrator 驱动 Seeker, Inspector, 和 Synthesizer Agent 
    完成一个迭代式的、多模态的推理流程。
    [修正] 此版本负责管理有状态 Agent (Seeker, Inspector) 的生命周期。
    """
    def __init__(
        self,
        search_engine: SearchEngine,
        seeker: SeekerAgent,
        inspector: InspectorAgent,
        synthesizer: SynthesizerAgent,
        gumbel_selector: Optional[GumbelModalSelector] = None,
        use_modal_selector: bool = True,
    ):
        self.search_engine = search_engine
        self.seeker = seeker
        self.inspector = inspector
        self.synthesizer = synthesizer
        self.gumbel_selector = gumbel_selector
        self.use_modal_selector = use_modal_selector
        self.modality_map = {0: "text", 1: "image", 2: "table"}

    def _choose_modality(self, query_embedding: torch.Tensor) -> int:
        # (此方法无需修改)
        if not (self.use_modal_selector and self.gumbel_selector is not None):
            print("[Orchestrator] No modal selector, defaulting to 'text' (0).")
            return 0
        x = query_embedding.unsqueeze(0) if query_embedding.dim() == 1 else query_embedding
        probs, _, _ = self.gumbel_selector(x, hard=True)
        prob_list = probs[0].detach().cpu().tolist()
        prob_str = ", ".join([f"{self.modality_map.get(i, 'unk')}={p:.3f}" for i, p in enumerate(prob_list)])
        print(f"[Orchestrator] Selector softmax probs: {prob_str}")
        return int(torch.argmax(probs[0]).item())

    def run(self, 
            query: str, 
            query_embedding: torch.Tensor, 
            max_iterations: int = 2,
            initial_top_k: int = 10
           ) -> str:
        
        print("\n" + "="*20 + " RAG Pipeline Started " + "="*20)
        
        final_answer: str = "After several attempts, I could not find sufficient information to answer the question."

        # --- 修正 1: 在处理新查询前，重置所有有状态 Agent 的缓冲区 ---
        # (您需要在 SeekerAgent 中也添加一个类似的 clear_buffer 方法)
        if hasattr(self.seeker, 'clear_buffer'):
            self.seeker.clear_buffer()
        if hasattr(self.inspector, 'clear_buffer'):
            self.inspector.clear_buffer()
        
        # --- 初始变量设置 ---
        feedback: Optional[str] = None
        current_nodes: List = [] # 用于在迭代中传递节点

        for i in range(max_iterations):
            print(f"\n--- Iteration {i + 1}/{max_iterations} ---")

            # 在第一次迭代时，我们需要进行模态选择和初始检索
            if i == 0:
                # 步骤 1: 模态选择
                modality_index = self._choose_modality(query_embedding)
                modality_name = self.modality_map.get(modality_index, "unknown")
                print(f"[Orchestrator] Step 1: Modality selected -> {modality_name.upper()}")

                # 步骤 2: 粗粒度检索
                print(f"[Orchestrator] Step 2: Retrieving Top-{initial_top_k} candidates from SearchEngine...")
                retrieved_nodes = self.search_engine.search(
                    query=query, 
                    modality=modality_name, 
                    top_k=initial_top_k
                )
                if not retrieved_nodes:
                    print("[Orchestrator] SearchEngine returned no results. Ending pipeline.")
                    break
                print(f"[Orchestrator] Retrieved {len(retrieved_nodes)} nodes.")
                current_nodes = retrieved_nodes
            
            # 步骤 3: Seeker 精细化寻证
            print("[Orchestrator] Step 3: Handing over to SeekerAgent for VLM-based filtering...")
            # --- 修正 2: 根据是否是第一次迭代，选择不同的调用方式 ---
            if i == 0:
                # 第一次调用，传入 query 和检索到的节点
                selected_nodes, summary, reason = self.seeker.run(
                    query=query,
                    candidate_nodes=current_nodes
                )
            else:
                # 后续调用（接收到 feedback），只传入 feedback
                selected_nodes, summary, reason = self.seeker.run(
                    feedback=feedback
                )
            
            print(f"[Orchestrator] Seeker selected {len(selected_nodes)} nodes. Reason: {reason}")
            current_nodes = selected_nodes # 更新当前正在处理的节点列表

            # 步骤 4: Inspector 检验与决策
            print("[Orchestrator] Step 4: Handing over to InspectorAgent for final validation...")
            status, information, final_nodes, confidence = self.inspector.run(
                query=query,
                nodes=current_nodes
            )
            print(f"[Orchestrator] Inspector decision: '{status}'. Confidence: {confidence.item():.4f}")

            # 步骤 5: 路由决策
            if status == "synthesizer" or status == "answer":
                print("[Orchestrator] Decision: Evidence is sufficient. Proceeding to Synthesizer.")
                candidate_answer = information if isinstance(information, str) else summary
                
                _, final_answer = self.synthesizer.run(
                    query=query,
                    evidence_nodes=final_nodes,
                    candidate_answer=candidate_answer
                )
                break # 成功，跳出循环
            
            elif status == "seeker":
                print(f"[Orchestrator] Decision: Evidence insufficient. Preparing for next iteration with feedback: {information}")
                feedback = information # 将 Inspector 的反馈用于下一次迭代
                # 继续下一次循环
            else:
                print(f"[Orchestrator] Unknown or final status '{status}'. Ending pipeline.")
                final_answer = information if isinstance(information, str) else "The pipeline ended with an unhandled state."
                break
        
        print("\n" + "="*20 + " RAG Pipeline Finished " + "="*20)
        return final_answer