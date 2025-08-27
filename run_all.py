# # File: run_all.py
# # 作用: 遍历整个数据集，运行RAG流程，并将生成的结果保存到文件。
# # 特点:
# # 1. 代码完整，可直接运行。
# # 2. 对每个样本进行异常处理，确保程序不因单个样本失败而中断。
# # 3. 自动记录处理失败的样本及错误信息到 _errors.jsonl 文件。
# # 4. 支持断点续跑，自动跳过已成功处理的样本。

# import os
# import torch
# import json
# from tqdm import tqdm
# from llama_index.core import Settings
# import traceback # 用于获取更详细的错误信息

# # --- 导入您的项目组件 ---
# from src.searcher.text_searcher import TextSearcher
# from src.searcher.image_searcher import ImageSearcher
# from src.searcher.table_searcher import TableSearcher
# from src.searcher.SearchEngine import SearchEngine
# from src.agent.seeker_agent import SeekerAgent
# from src.agent.inspector_agent import InspectorAgent
# from src.agent.synthesizer_agent import SynthesizerAgent
# from src.models.gumbel_selector import GumbelModalSelector
# from src.orchestrator import RAGOrchestrator
# from src.llms.llm import LLM
# from src.utils.embedding_utils import QueryEmbedder, get_query_embedding

# Settings.llm = None

# # --- 配置区 ---
# CONFIG = {
#     "DATASET_PATH": "data/ViDoSeek/rag_dataset.json",
#     "OUTPUT_FILE": "generation_results/run_output.jsonl", # 所有成功结果将保存在这里
#     "TOP_K": 5
# }

# def main():
#     # ==========================================================================
#     # 步骤 1: 初始化所有系统组件 (完整版)
#     # ==========================================================================
#     print("="*30)
#     print("步骤 1: 初始化所有系统组件")
    
#     # --- 1a. 设置 API Key 和设备 ---
#     print("\n[1a] 设置 vlm API Key...")
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     if "sk" in os.environ.get("DASHSCOPE_API_KEY", ""):
#         print("[1a] ✅ vlm API Key 已成功设置.")
#     else:
#         print("[1a] ⚠️  警告: 未找到 DASHSCOPE_API_KEY 环境变量。")

#     # --- 1b. 初始化核心模型 ---
#     print("\n[1b] 初始化 query 嵌入模型和 vlm 模型...")
#     query_embedder = QueryEmbedder(model_name="BAAI/bge-m3", device=device)
#     vlm = LLM("qwen-vl-max")
#     print("[1b] ✅ 初始化 query 嵌入模型和 vlm 模型 已成功初始化.")

#     # --- 1c. 创建检索器“工厂” (Retriever Factories) ---
#     print("\n[1c] 创建检索器工厂...")
#     retriever_factories = {
#         "text": lambda: TextSearcher(
#             dataset_name='ViDoSeek',
#             mode='hybrid',
#             node_dir_prefix='bge_ingestion' 
#         ),
#         "image": lambda: ImageSearcher(
#             dataset_name='ViDoSeek',
#             mode='vl_search',
#             vl_node_dir_prefix='colqwen_ingestion' 
#         ),
#         "table": lambda: TableSearcher(
#             dataset_name='ViDoSeek',
#             mode='vl_search',
#             vl_node_dir_prefix='colqwen_ingestion' 
#         )
#     }
#     print("[1c] ✅ 检索器工厂完成...")

#     # --- 1d. 初始化支持懒加载的 SearchEngine ---
#     print("\n[1d] 初始化支持懒加载的 SearchEngine...")
#     search_engine = SearchEngine(retriever_factories=retriever_factories)
#     print("[1d] ✅ 懒加载 SearchEngine 已成功初始化.")

#     # --- 1e. 初始化 Gumbel 模态选择器 ---
#     print("\n[1e] 初始化 Gumbel 模态选择器...")
#     gumbel_selector = GumbelModalSelector(
#         input_dim=query_embedder.out_dim,
#         num_choices=3 # 0=text, 1=image, 2=table
#     ).to(device).eval()

#     ckpt_path = "checkpoints/modal_selector_best.pt"
#     if os.path.exists(ckpt_path):
#         gumbel_selector.load_state_dict(torch.load(ckpt_path, map_location=device))
#         print(f"[1e] ✅ 成功从 {ckpt_path} 加载 Gumbel Selector 权重.")
#     else:
#         print("[1e] ℹ️ 未找到 Gumbel Selector 检查点，使用随机初始化.")

#     # --- 1f. 初始化 seeker, inspector, synthesizer agents ---
#     print("\n[1f] 初始化 agents...")
#     image_base_dir = "data/ViDoSeek/img" 
#     seeker_agent = SeekerAgent(vlm=vlm, image_base_dir=image_base_dir)
#     inspector_agent = InspectorAgent(vlm=vlm, image_base_dir=image_base_dir, reranker_model_name="BAAI/bge-reranker-large")
#     synthesizer_agent = SynthesizerAgent(vlm=vlm, image_base_dir=image_base_dir)
#     print("[1f] ✅ 所有 agents 已被成功初始化.")

#     # --- 1g. 组装总指挥 (Orchestrator) ---
#     print("\n[1g] 初始化 orchestrator...")
#     orchestrator = RAGOrchestrator(
#         search_engine=search_engine,
#         seeker=seeker_agent,
#         inspector=inspector_agent,
#         synthesizer=synthesizer_agent,
#         gumbel_selector=gumbel_selector
#     )
#     print("[1g] ✅ Orchestrator 已成功初始化.")
    
#     print("\n✅ 步骤一结束：所有组件已就位.")
#     print("="*30)

#     # ==========================================================================
#     # 步骤 2: 批量生成结果
#     # ==========================================================================
#     print("\n步骤 2: 开始批量生成结果")

#     with open(CONFIG["DATASET_PATH"], "r", encoding="utf-8") as f:
#         examples = json.load(f)["examples"]
    
#     # --- 断点续跑逻辑 ---
#     completed_uids = set()
#     output_dir = os.path.dirname(CONFIG["OUTPUT_FILE"])
#     if output_dir:
#         os.makedirs(output_dir, exist_ok=True)
        
#     if os.path.exists(CONFIG["OUTPUT_FILE"]):
#         print(f"ℹ️  发现已存在的结果文件，将进行断点续跑...")
#         with open(CONFIG["OUTPUT_FILE"], "r", encoding="utf-8") as f:
#             for line in f:
#                 try:
#                     completed_uids.add(json.loads(line)["uid"])
#                 except (json.JSONDecodeError, KeyError):
#                     continue
#         print(f"✅ 已成功处理 {len(completed_uids)} 个样本，将跳过它们。")

#     # --- 循环处理所有样本 ---
#     error_file_path = CONFIG["OUTPUT_FILE"].replace(".jsonl", "_errors.jsonl")
#     for sample in tqdm(examples, desc="正在生成结果"):
#         uid = sample.get("uid")
#         query = sample.get("query")
        
#         if not query or (uid and uid in completed_uids):
#             continue

#         # --- 对每个样本的处理都包裹在 try...except 块中 ---
#         try:
#             query_embedding = get_query_embedding(query_embedder, query)
            
#             # 1. 独立获取检索结果
#             modality_index = orchestrator._choose_modality(query_embedding)
#             modality_name = orchestrator.modality_map.get(modality_index, "unknown")
#             retrieved_nodes = search_engine.search(
#                 query=query, modality=modality_name, top_k=CONFIG["TOP_K"]
#             )
            
#             # 2. 运行Orchestrator获取答案 (传入已检索的节点以避免重复检索)
#             # 提示: 为了获得性能提升，建议修改您的Orchestrator以接受pre_retrieved_nodes参数
#             final_answer = orchestrator.run(
#                 query=query,
#                 query_embedding=query_embedding,
#                 initial_top_k=CONFIG["TOP_K"],
#                 pre_retrieved_nodes=retrieved_nodes # 取消此行注释以启用优化
#             )

#             # --- 准备要保存的数据 ---
#             result_data = {
#                 "uid": uid,
#                 "query": query,
#                 "reference_answer": sample.get("reference_answer"),
#                 "meta_info": sample.get("meta_info"),
#                 "generated_answer": final_answer,
#                 "retrieved_nodes": [node.to_dict() for node in retrieved_nodes]
#             }

#             # --- 将成功的结果实时写入文件 ---
#             with open(CONFIG["OUTPUT_FILE"], "a", encoding="utf-8") as f:
#                 f.write(json.dumps(result_data, ensure_ascii=False) + "\n")

#         except Exception as e:
#             # --- 如果发生异常，记录错误并继续 ---
#             print(f"\n❌ 处理样本 {uid} 时发生严重错误: {e}")
#             error_details = {
#                 "uid": uid,
#                 "query": query,
#                 "error_message": str(e),
#                 "traceback": traceback.format_exc() # 记录完整的错误堆栈信息
#             }
#             # 写入独立的错误日志文件
#             with open(error_file_path, "a", encoding="utf-8") as f:
#                 f.write(json.dumps(error_details, ensure_ascii=False) + "\n")
#             continue # 继续下一个样本
    
#     print("\n🎉 所有样本生成完毕！")
#     print(f"✅ 成功结果已保存至: {CONFIG['OUTPUT_FILE']}")
#     if os.path.exists(error_file_path):
#         print(f"⚠️  部分样本处理失败，错误详情已记录在: {error_file_path}")

# if __name__ == '__main__':
#     main()

# File: run_all.py (最终版)
# 作用: 遍历整个数据集，运行RAG流程，并将生成的结果保存到文件。
# 特点:
# 1. 代码完整，可直接运行。
# 2. 对每个样本进行异常处理，确保程序不因单个样本失败而中断。
# 3. 自动记录处理失败的样本及错误信息到 _errors.jsonl 文件。
# 4. 支持最稳健的断点续跑：自动跳过所有已处理过的样本（无论成功或失败）。

import os
import torch
import json
from tqdm import tqdm
from llama_index.core import Settings
import traceback # 用于获取更详细的错误信息

# --- 导入您的项目组件 ---
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
from src.utils.embedding_utils import QueryEmbedder, get_query_embedding

Settings.llm = None

# --- 配置区 ---
CONFIG = {
    "DATASET_PATH": "data/ViDoSeek/rag_dataset.json",
    "OUTPUT_FILE": "generation_results/run_output.jsonl", # 所有成功结果将保存在这里
    "TOP_K": 5
}

def main():
    # ==========================================================================
    # 步骤 1: 初始化所有系统组件 (完整版)
    # ==========================================================================
    print("="*30)
    print("步骤 1: 初始化所有系统组件")
    
    # --- 1a. 设置 API Key 和设备 ---
    print("\n[1a] 设置 vlm API Key...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if "sk" in os.environ.get("DASHSCOPE_API_KEY", ""):
        print("[1a] ✅ vlm API Key 已成功设置.")
    else:
        print("[1a] ⚠️  警告: 未找到 DASHSCOPE_API_KEY 环境变量。")

    # --- 1b. 初始化核心模型 ---
    print("\n[1b] 初始化 query 嵌入模型和 vlm 模型...")
    query_embedder = QueryEmbedder(model_name="BAAI/bge-m3", device=device)
    vlm = LLM("qwen-vl-max")
    print("[1b] ✅ 初始化 query 嵌入模型和 vlm 模型 已成功初始化.")

    # --- 1c. 创建检索器“工厂” (Retriever Factories) ---
    print("\n[1c] 创建检索器工厂...")
    retriever_factories = {
        "text": lambda: TextSearcher(
            dataset_name='ViDoSeek',
            mode='hybrid',
            node_dir_prefix='bge_ingestion' 
        ),
        "image": lambda: ImageSearcher(
            dataset_name='ViDoSeek',
            mode='vl_search',
            vl_node_dir_prefix='colqwen_ingestion' 
        ),
        "table": lambda: TableSearcher(
            dataset_name='ViDoSeek',
            mode='vl_search',
            vl_node_dir_prefix='colqwen_ingestion' 
        )
    }
    print("[1c] ✅ 检索器工厂完成...")

    # --- 1d. 初始化支持懒加载的 SearchEngine ---
    print("\n[1d] 初始化支持懒加载的 SearchEngine...")
    search_engine = SearchEngine(retriever_factories=retriever_factories)
    print("[1d] ✅ 懒加载 SearchEngine 已成功初始化.")

    # --- 1e. 初始化 Gumbel 模态选择器 ---
    print("\n[1e] 初始化 Gumbel 模态选择器...")
    gumbel_selector = GumbelModalSelector(
        input_dim=query_embedder.out_dim,
        num_choices=3 # 0=text, 1=image, 2=table
    ).to(device).eval()

    ckpt_path = "checkpoints/modal_selector_best.pt"
    if os.path.exists(ckpt_path):
        gumbel_selector.load_state_dict(torch.load(ckpt_path, map_location=device))
        print(f"[1e] ✅ 成功从 {ckpt_path} 加载 Gumbel Selector 权重.")
    else:
        print("[1e] ℹ️ 未找到 Gumbel Selector 检查点，使用随机初始化.")

    # --- 1f. 初始化 seeker, inspector, synthesizer agents ---
    print("\n[1f] 初始化 agents...")
    image_base_dir = "data/ViDoSeek/img" 
    seeker_agent = SeekerAgent(vlm=vlm, image_base_dir=image_base_dir)
    inspector_agent = InspectorAgent(vlm=vlm, image_base_dir=image_base_dir, reranker_model_name="BAAI/bge-reranker-large")
    synthesizer_agent = SynthesizerAgent(vlm=vlm, image_base_dir=image_base_dir)
    print("[1f] ✅ 所有 agents 已被成功初始化.")

    # --- 1g. 组装总指挥 (Orchestrator) ---
    print("\n[1g] 初始化 orchestrator...")
    orchestrator = RAGOrchestrator(
        search_engine=search_engine,
        seeker=seeker_agent,
        inspector=inspector_agent,
        synthesizer=synthesizer_agent,
        gumbel_selector=gumbel_selector
    )
    print("[1g] ✅ Orchestrator 已成功初始化.")
    
    print("\n✅ 步骤一结束：所有组件已就位.")
    print("="*30)

    # ==========================================================================
    # 步骤 2: 批量生成结果
    # ==========================================================================
    print("\n步骤 2: 开始批量生成结果")

    with open(CONFIG["DATASET_PATH"], "r", encoding="utf-8") as f:
        examples = json.load(f)["examples"]
    
    # --- 断点续跑逻辑 (最终版：同时跳过成功和失败的样本) ---
    uids_to_skip = set()
    output_dir = os.path.dirname(CONFIG["OUTPUT_FILE"])
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    success_file_path = CONFIG["OUTPUT_FILE"]
    error_file_path = success_file_path.replace(".jsonl", "_errors.jsonl")

    # 1. 读取已成功的UID
    success_count = 0
    if os.path.exists(success_file_path):
        with open(success_file_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    # 使用.add()的返回值来计数是无效的，需要单独计数
                    if json.loads(line)["uid"] not in uids_to_skip:
                        uids_to_skip.add(json.loads(line)["uid"])
                        success_count += 1
                except (json.JSONDecodeError, KeyError):
                    continue
    if success_count > 0:
        print(f"ℹ️  发现成功日志，将跳过 {success_count} 个已成功的样本。")

    # 2. 读取已失败的UID
    error_count = 0
    if os.path.exists(error_file_path):
        with open(error_file_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    if json.loads(line)["uid"] not in uids_to_skip:
                        uids_to_skip.add(json.loads(line)["uid"])
                        error_count += 1
                except (json.JSONDecodeError, KeyError):
                    continue
    if error_count > 0:
        print(f"ℹ️  发现错误日志，将额外跳过 {error_count} 个已知会出错的样本。")
    # --- 逻辑修改结束 ---

    # --- 循环处理所有样本 ---
    for sample in tqdm(examples, desc="正在生成结果"):
        uid = sample.get("uid")
        query = sample.get("query")
        
        # --- 更新判断条件：跳过所有已处理过的样本 ---
        if not query or (uid and uid in uids_to_skip):
            continue

        # --- 对每个样本的处理都包裹在 try...except 块中 ---
        try:
            query_embedding = get_query_embedding(query_embedder, query)
            
            # 1. 独立获取检索结果
            modality_index = orchestrator._choose_modality(query_embedding)
            modality_name = orchestrator.modality_map.get(modality_index, "unknown")
            retrieved_nodes = search_engine.search(
                query=query, modality=modality_name, top_k=CONFIG["TOP_K"]
            )
            
            # 2. 运行Orchestrator获取答案
            final_answer = orchestrator.run(
                query=query,
                query_embedding=query_embedding,
                initial_top_k=CONFIG["TOP_K"],
                # pre_retrieved_nodes=retrieved_nodes # 如需优化性能，请修改Orchestrator并取消此行注释
            )

            # --- 准备要保存的数据 ---
            result_data = {
                "uid": uid,
                "query": query,
                "reference_answer": sample.get("reference_answer"),
                "meta_info": sample.get("meta_info"),
                "generated_answer": final_answer,
                "retrieved_nodes": [node.to_dict() for node in retrieved_nodes]
            }

            # --- 将成功的结果实时写入文件 ---
            with open(CONFIG["OUTPUT_FILE"], "a", encoding="utf-8") as f:
                f.write(json.dumps(result_data, ensure_ascii=False) + "\n")

        except Exception as e:
            # --- 如果发生异常，记录错误并继续 ---
            print(f"\n❌ 处理样本 {uid} 时发生严重错误: {e}")
            error_details = {
                "uid": uid,
                "query": query,
                "error_message": str(e),
                "traceback": traceback.format_exc()
            }
            # 写入独立的错误日志文件
            with open(error_file_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(error_details, ensure_ascii=False) + "\n")
            continue # 继续下一个样本
    
    print("\n🎉 所有样本生成完毕！")
    print(f"✅ 成功结果已保存至: {CONFIG['OUTPUT_FILE']}")
    if os.path.exists(error_file_path):
        print(f"⚠️  部分样本处理失败，错误详情已记录在: {error_file_path}")

if __name__ == '__main__':
    main()