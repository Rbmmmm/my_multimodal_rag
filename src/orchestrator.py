# File: src/orchestrator.py
# Final version with state management for stateful Agents.

# 可以手动强制选择模态

from __future__ import annotations
import torch
import os
from typing import Optional, Dict, Any, List

from src.searcher import SearchEngine
from src.agent.seeker_agent import SeekerAgent
from src.agent.inspector_agent import InspectorAgent
from src.agent.synthesizer_agent import SynthesizerAgent
from src.models.gumbel_selector import GumbelModalSelector

class RAGOrchestrator:
    """
    RAGOrchestrator 驱动 Seeker, Inspector, 和 Synthesizer Agent 
    完成一个迭代式的、多模态的推理流程。
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

    # 调用 gumbel selctor 来选择模态
    def _choose_modality(self, query_embedding: torch.Tensor) -> int:
        
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
        
        print("\n" + "="*20 + " RAG 流程开始 " + "="*20)
        
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
            print(f"\n--- Orchestrator  第 {i + 1}/{max_iterations} 轮运行 ---")

            # 在第一次迭代时，我们需要进行模态选择和初始检索
            if i == 0:
                # 步骤 1: 模态选择
                # modality_index = self._choose_modality(query_embedding)
                modality_index = 1
                modality_name = self.modality_map.get(modality_index, "unknown")
                print(f"[Orchestrator] 步骤 1: 模态选择 -> {modality_name.upper()}")

                # 步骤 2: 粗粒度检索
                print(f"[Orchestrator] 步骤 2: 为 SearchEngine 检索出 Top-{initial_top_k} 个候选节点...")
                retrieved_nodes = self.search_engine.search(
                    query=query, 
                    modality=modality_name, 
                    top_k=initial_top_k
                )
                if not retrieved_nodes:
                    print("[Orchestrator] 步骤 2.1: SearchEngine 没有返回任何结果. 结束.")
                    break
                print(f"[Orchestrator] 步骤 2.2: 检索出 {len(retrieved_nodes)} 个节点.")
                
                # 使用 Search Engine 检索出的节点
                current_nodes = retrieved_nodes
                # 找出节点对应的图片路径
   
                current_images = []
                node_map = {} # 用于最后根据路径找回 Node 对象
                for node_with_score in current_nodes:
                    node = node_with_score.node
                    metadata = getattr(node, "metadata", {}) or {}
                
                    image_path = None
                    explicit_path = metadata.get("image_path") or metadata.get("file_path") # 只有 ImageSearch 返回的 ImageNode 才有这两个属性
                
                    if explicit_path and isinstance(explicit_path, str): # ImageSearch
                        image_path = explicit_path
                        image_path = image_path.replace("\\", "/")
                    elif 'filename' in metadata and isinstance(metadata['filename'], str): # TextSearch
                        base_filename = os.path.splitext(metadata['filename'])[0]
                        potential_path = os.path.join("data/ViDoSeek/img", f"{base_filename}.jpg")
                        image_path = potential_path
                    
                    if image_path and os.path.exists(image_path):
                        current_images.append(image_path)
                        # 做一个用 image_path 来找到原始 Node 的映射字典
                        node_map[image_path] = node_with_score
            
            # 步骤 3: Seeker 精细化寻证
            print("\n[Orchestrator] 步骤 3: 把节点交给 Seeker Agent 进行筛选...")
            # --- 修正 2: 根据是否是第一次迭代，选择不同的调用方式 ---
            if i == 0:
                # 第一次调用，传入 query 和检索到的节点
                # seeker agent 已在 run.py 实例化
                selected_nodes, selected_images, summary, reason = self.seeker.run(
                    query=query,
                    candidate_nodes=current_nodes,
                    image_paths = current_images
                )
            else:
                # 后续调用（接收到 feedback），只传入 feedback
                selected_nodes, selected_images, summary, reason = self.seeker.run(
                    feedback=feedback
                )
            
            print(f"[Orchestrator] 步骤 3.3: Seeker 选择了 {len(selected_nodes)} 个节点. \n Reason: {reason}")
            current_nodes = selected_nodes # 更新当前正在处理的节点列表
            current_images = selected_images
            # 步骤 4: Inspector 检验与决策
            print("\n[Orchestrator] 步骤 4: 交给 Inspector 来做检验...")
            status, information, images, confidence = self.inspector.run(
                query=query,
                nodes=current_nodes,
                image_paths=current_images
            )
            print(f"[Orchestrator] Inspector decision: '{status}'. Confidence: {confidence.item():.4f}")

            # 步骤 5: 路由决策
            if status == "answer":
                print("[Orchestrator] Decision: Evidence is sufficient. Answer Directly.")
                final_answer = information
                break
            
            elif status == "synthesizer":
                print("[Orchestrator] Decision: Evidence is sufficient. Proceeding to Synthesizer.")
                candidate_answer = information if isinstance(information, str) else summary
                
                reason, final_answer = self.synthesizer.run(
                    query=query,
                    candidate_answer = candidate_answer,
                    ref_images= images 
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