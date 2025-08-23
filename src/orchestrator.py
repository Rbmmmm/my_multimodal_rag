# File: src/orchestrator.py

# 可以手动强制选择模态

from __future__ import annotations
import torch
import os
from typing import Optional, Dict, Any, List

from src.searcher import SearchEngine
from src.agent.seeker_agent import SeekerAgent
from src.agent.inspector_agent import InspectorAgent
from src.agent.synthesizer_agent import SynthesizerAgent
from src.models.gumbel_selector import GumbelModalSelector, choose_modality

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
        self.modality_map = {0: "text", 1: "image", 2: "image"}
    
    def _choose_modality(self, query_embedding: torch.Tensor) -> int:
        if not self.use_modal_selector:
            print("[Orchestrator] Modal selector not in use, defaulting to 'text' (0).")
            return 0
            
        # 直接调用封装好的函数
        return choose_modality(
            selector_model=self.gumbel_selector,
            query_embedding=query_embedding,
            modality_map=self.modality_map
        )

    def run(self, 
            query: str, 
            query_embedding: torch.Tensor, 
            max_iterations: int = 2,
            initial_top_k: int = 10,
            setted_modality_index : int = None
           ) -> str:
        
        print("\n" + "="*20 + " RAG 流程开始 " + "="*20)
        
        default_failure_message = "After several attempts, I could not find sufficient information to answer the question."
        final_answer: str = "After several attempts, I could not find sufficient information to answer the question."

        # --- 修正 1: 在处理新查询前，重置所有有状态 Agent 的缓冲区 ---
        # (您需要在 SeekerAgent 中也添加一个类似的 clear_buffer 方法)
        if hasattr(self.seeker, 'clear_buffer'):
            self.seeker.clear_buffer()
        if hasattr(self.inspector, 'clear_buffer'):
            self.inspector.clear_buffer()
        
        # --- 初始变量设置 ---
        feedback: Optional[str] = None
        last_confidence: Optional[torch.Tensor] = None
        current_nodes: List = [] # 用于在迭代中传递节点
        
        # 用初始 top_k 尝试
        for i in range(max_iterations):
            
            print(f"\n--- Orchestrator  第 {i+1}/{max_iterations} 轮运行 ---")
            
            
            
            if i == 0:    
                # 步骤 1: 模态选择
                if isinstance(setted_modality_index, int):
                    modality_index = setted_modality_index
                else:
                    modality_index = self._choose_modality(query_embedding)
                # modality_index = 0
                modality_name = self.modality_map.get(modality_index, "unknown")
                print(f"[Orchestrator] 步骤 1: 模态选择 -> {modality_name.upper()}")

                # 步骤 2: 粗粒度检索
                print(f"\n[Orchestrator] 步骤 2: 开始调用 SearchEngine 检索 Top-{initial_top_k} 个候选节点...")
                retrieved_nodes = self.search_engine.search(
                    query=query, 
                    modality=modality_name, 
                    top_k=initial_top_k
                )   
                if not retrieved_nodes:
                    print("[Orchestrator] 步骤 2.1: SearchEngine 没有返回任何结果. 结束.")
                    return "Not retrieved any nodes"
                
                print(f"[Orchestrator] 步骤 2.3: ✅ 检索出 {len(retrieved_nodes)} 个节点.")
                        
                # 使用 Search Engine 检索出的节点
                current_nodes = retrieved_nodes
            
                # 找出节点对应的图片路径
                current_images = []
                
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

                print("[Orchestrator] 步骤 2.4: 检索出的节点为: ", current_images)



            # 步骤 3: Seeker 精细化寻证
            print("\n[Orchestrator] 步骤 3: 把节点交给 Seeker Agent 进行筛选...")
            # 根据是否是第一次迭代，选择不同的调用方式 ---
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
            
            if len(selected_nodes) != len(selected_images):
                print("Seeker返回的 nodes 和 images 数量不一样")
            
            print(f"[Orchestrator] 步骤 3.3: Seeker 选择了 {len(selected_nodes)} 个节点. \n[Orchestrator] 步骤 3.4: Reason: {reason}")
            
            current_images = selected_images
            current_nodes = selected_nodes
        
            print("[Orchestrator] 步骤 3.5: 当前的节点为:", current_images)
            
            # 步骤 4: Inspector 检验与决策
            print("\n[Orchestrator] 步骤 4: 交给 Inspector 来做检验...")
            status, information, images, nodes, confidence = self.inspector.run(
                query=query,
                nodes=current_nodes,
                image_paths=current_images
            )
            print(f"[Orchestrator] Inspector decision: '{status}'. Confidence: {confidence.item():.4f}")
            last_confidence = confidence
            
            # 步骤 5: 路由决策
            if status == "answer":
                print("[Orchestrator] Decision: Evidence is sufficient. Answer Directly.")
                final_answer = information
                print("\n" + "="*20 + " RAG Pipeline Finished " + "="*20)
                return final_answer
            
            elif status == "synthesizer":
                print("[Orchestrator] Decision: Evidence is sufficient. Proceeding to Synthesizer.")
                candidate_answer = information if isinstance(information, str) else summary
                
                reason, final_answer = self.synthesizer.run(
                    query=query,
                    candidate_answer = candidate_answer,
                    ref_images= images 
                )
                
                print("\n" + "="*20 + " RAG Pipeline Finished " + "="*20)
                return final_answer
            
            elif status == "seeker":
                print(f"[Orchestrator] Decision: Evidence insufficient. Preparing for next iteration with feedback: {information}")
                feedback = information # 将 Inspector 的反馈用于下一次迭代
                # 继续下一次循环
            else:
                print(f"[Orchestrator] Unknown or final status '{status}'. Ending pipeline.")
                final_answer = information if isinstance(information, str) else "The pipeline ended with an unhandled state."
                print("\n" + "="*20 + " RAG Pipeline Finished " + "="*20)
                return final_answer
            
        # 调整 k 值
        for i in range(max_iterations):
            # top_k = calculate_func(initial_top_k) 占位，后续用论文公式实现
            top_k = 10
            
            print(f"\n[Orchestrator] 扩展 Top-K 从 {initial_top_k} 到 {top_k} 并重新检索...")
            
            print(f"\n--- Orchestrator  第 {i+1}/{max_iterations} 轮运行 ---")
            
            if i == 0:    
                # 步骤 2: 粗粒度检索
                print(f"\n[Orchestrator] 步骤 2: 为 SearchEngine 检索出 Top-{top_k} 个候选节点...")
                retrieved_nodes = self.search_engine.search(
                    query=query, 
                    modality=modality_name, 
                    top_k=top_k
                )   
                if not retrieved_nodes:
                    print("[Orchestrator] 步骤 2.1: SearchEngine 没有返回任何结果. 结束.")
                    return "Not retrieved any nodes"
                
                print(f"[Orchestrator] 步骤 2.3: ✅ 检索出 {len(retrieved_nodes)} 个节点.")
                        
                # 使用 Search Engine 检索出的节点
                current_nodes = retrieved_nodes
            
                # 找出节点对应的图片路径
                current_images = []
                
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
                
            if hasattr(self.seeker, 'clear_buffer'):
                self.seeker.clear_buffer()
            if hasattr(self.inspector, 'clear_buffer'):
                self.inspector.clear_buffer()
            
            print(f"[Orchestrator] 步骤 3.3: Seeker 选择了 {len(selected_nodes)} 个节点. \n[Orchestrator] 步骤 3.4: Reason: {reason}")
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
            last_confidence = confidence
            
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