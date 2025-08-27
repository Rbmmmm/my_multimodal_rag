# File: eval.py (版本 3.1 - 补全初始化代码的最终版)

import os
import torch
import json
from tqdm import tqdm
import numpy as np

# --------------------------------------------------------------------------
# 步骤 0: 导入您项目中的模块和新的评估工具
# --------------------------------------------------------------------------
# --- 确保您的项目组件可以被正确导入 ---
from src.searcher.text_searcher import TextSearcher
from src.searcher.image_searcher import ImageSearcher
from src.searcher.table_searcher import TableSearcher
from src.searcher.SearchEngine import SearchEngine
from src.agent.seeker_agent import SeekerAgent
from src.agent.inspector_agent import InspectorAgent
from src.agent.synthesizer_agent import SynthesizerAgent
from src.models.gumbel_selector import GumbelModalSelector
from src.orchestrator import RAGOrchestrator
from src.llms.llm import LLM
from src.utils.embedding_utils import QueryEmbedder, get_query_embedding # <--- 确保 get_query_embedding 已导入
from llama_index.core import Settings

# --- 导入您提供的、模块化的评估器 ---
from src.llms.evaluator import Evaluator
from src.utils.overall_evaluator import eval_search, eval_search_type_wise

Settings.llm = None

# ==========================================================================
# 步骤 1: 在这里配置您的评估任务
# ==========================================================================
CONFIG = {
    "DATASET_PATH": "data/ViDoSeek/rag_dataset.json",
    "OUTPUT_DIR": "eval_results/run_modular_test_final",
    "START_INDEX": 0,
    "NUM_TO_TEST": 20,
    "TOP_K": 5,
}

# --------------------------------------------------------------------------
# 步骤 2: 主评估函数
# --------------------------------------------------------------------------
def main():
    """主函数，加载组件，循环处理样本，最后调用外部评估器进行汇总。"""
    # ==========================================================================
    # 2.1: 初始化所有系统组件
    # ==========================================================================
    print("="*30)
    print("步骤 1: 初始化所有系统组件")

    # --- START: 补全的初始化代码 ---
    # 这部分是之前版本中被省略的，现在已完整添加
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if "sk" in os.environ["DASHSCOPE_API_KEY"]:
        print("✅ vlm API Key 已成功设置.")

    # 初始化嵌入模型和VLM
    print("\n[Init] 初始化嵌入模型和 VLM...")
    query_embedder = QueryEmbedder(model_name="BAAI/bge-m3", device=device)
    vlm = LLM("qwen-vl-max")
    print("✅ 嵌入模型和 VLM 初始化成功.")

    # 创建检索器工厂
    print("\n[Init] 创建检索器工厂...")
    retriever_factories = {
        "text": lambda: TextSearcher(dataset_name='ViDoSeek', mode='hybrid', node_dir_prefix='bge_ingestion'),
        "image": lambda: ImageSearcher(dataset_name='ViDoSeek', mode='vl_search', vl_node_dir_prefix='colqwen_ingestion'),
        "table": lambda: TableSearcher(dataset_name='ViDoSeek', mode='vl_search', vl_node_dir_prefix='colqwen_ingestion')
    }
    search_engine = SearchEngine(retriever_factories=retriever_factories)
    print("✅ 检索器工厂创建成功.")

    # 初始化Gumbel模态选择器
    print("\n[Init] 初始化 Gumbel 模态选择器...")
    gumbel_selector = GumbelModalSelector(input_dim=query_embedder.out_dim, num_choices=3).to(device).eval()
    ckpt_path = "checkpoints/modal_selector_best.pt"
    if os.path.exists(ckpt_path):
        gumbel_selector.load_state_dict(torch.load(ckpt_path, map_location=device))
        print(f"✅ 成功从 {ckpt_path} 加载 Gumbel Selector 权重.")
    else:
        print("ℹ️ 未找到 Gumbel Selector 检查点，使用随机初始化.")

    # 初始化Agents
    print("\n[Init] 初始化所有 Agents...")
    image_base_dir = "data/ViDoSeek/img" 
    seeker_agent = SeekerAgent(vlm=vlm, image_base_dir=image_base_dir)
    inspector_agent = InspectorAgent(vlm=vlm, image_base_dir=image_base_dir, reranker_model_name="BAAI/bge-reranker-large")
    synthesizer_agent = SynthesizerAgent(vlm=vlm, image_base_dir=image_base_dir)
    print("✅ 所有 Agents 初始化成功.")

    # 初始化Orchestrator
    print("\n[Init] 初始化 Orchestrator...")
    orchestrator = RAGOrchestrator(
        search_engine=search_engine,
        seeker=seeker_agent,
        inspector=inspector_agent,
        synthesizer=synthesizer_agent,
        gumbel_selector=gumbel_selector
    )
    print("✅ Orchestrator 初始化成功.")
    
    # --- END: 补全的初始化代码 ---
    
    # 初始化您提供的 Evaluator
    evaluator = Evaluator()
    
    print("\n✅ 所有组件已就位.")
    print("="*30)

    # ==========================================================================
    # 2.2: 批量运行并收集结果
    # ==========================================================================
    print("\n步骤 2: 开始批量运行并收集结果")
    
    with open(CONFIG["DATASET_PATH"], "r", encoding="utf-8") as f:
        examples = json.load(f)["examples"]

    all_results = []
    start_index = CONFIG["START_INDEX"]
    end_index = min(start_index + CONFIG["NUM_TO_TEST"], len(examples))
    
    for i in tqdm(range(start_index, end_index), desc="处理样本"):
        sample = examples[i]
        query = sample.get("query")
        if not query: continue

        # --- 运行完整的RAG流程 ---
        query_embedding = get_query_embedding(query_embedder, query)
        
        # 为了评估检索，我们需要先独立获取检索结果
        modality_index = orchestrator._choose_modality(query_embedding)
        modality_name = orchestrator.modality_map.get(modality_index, "unknown")
        retrieved_nodes = search_engine.search(
            query=query, modality=modality_name, top_k=CONFIG["TOP_K"]
        )
        
        # 运行完整的Orchestrator来获取最终答案
        final_answer = orchestrator.run(
            query=query,
            query_embedding=query_embedding,
            initial_top_k=CONFIG["TOP_K"]
        )

        # --- 使用您的 Evaluator 进行答案评分 ---
        eval_result = evaluator.evaluate(
            query=query,
            reference_answer=sample.get("reference_answer", ""),
            generated_answer=final_answer
        )
        
        # --- 将结果附加到原始样本中，以便后续评估 ---
        sample['recall_results'] = {
            "source_nodes": [node.to_dict() for node in retrieved_nodes]
        }
        sample['eval_result'] = eval_result
        
        all_results.append(sample)

    # ==========================================================================
    # 2.3: 调用外部评估器进行汇总计算
    # ==========================================================================
    print("\n" + "="*30)
    print("所有样本处理完毕，调用评估器进行汇总...")

    overall_summary = eval_search(all_results)
    type_wise_summary = eval_search_type_wise(all_results)
    
    print("\n[整体评估摘要]:")
    print(json.dumps(overall_summary, indent=4, ensure_ascii=False))
    
    print("\n[分类别评估摘要]:")
    print(json.dumps(type_wise_summary, indent=4, ensure_ascii=False))

    # ==========================================================================
    # 2.4: 保存所有结果文件
    # ==========================================================================
    output_dir = CONFIG["OUTPUT_DIR"]
    os.makedirs(output_dir, exist_ok=True)
    
    details_path = os.path.join(output_dir, "results_details.jsonl")
    with open(details_path, "w", encoding="utf-8") as f:
        for res in all_results:
            f.write(json.dumps(res, ensure_ascii=False) + "\n")
    print(f"\n✅ 详细结果已保存至: {details_path}")

    summary_path = os.path.join(output_dir, "results_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(overall_summary, f, indent=4, ensure_ascii=False)
    print(f"✅ 整体摘要已保存至: {summary_path}")

    type_wise_path = os.path.join(output_dir, "results_type_wise.json")
    with open(type_wise_path, "w", encoding="utf-8") as f:
        json.dump(type_wise_summary, f, indent=4, ensure_ascii=False)
    print(f"✅ 分类别摘要已保存至: {type_wise_path}")


if __name__ == '__main__':
    main()