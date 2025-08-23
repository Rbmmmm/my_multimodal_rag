# File: run.py

import os
import torch

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
from src.utils.embedding_utils import QueryEmbedder      
from llama_index.core import Settings
import json

Settings.llm = None

def main():
    """
    主函数，用于设置和运行 RAG 流程。
    """
    # ==========================================================================
    # 步骤 1: 初始化所有系统组件
    # ==========================================================================
    print("="*30)
    print("步骤 1: 初始化所有系统组件")
    
    # --- 1a. 设置 API Key 和设备 ---
    print("\n[1a] 设置 vlm API Key...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if "sk" in os.environ["DASHSCOPE_API_KEY"]:
        print("[1a] ✅ vlm API Key 已成功设置.")

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

    # 加载训练好的选择器权重
    ckpt_path = "checkpoints/modal_selector_best.pt"
    if os.path.exists(ckpt_path):
        gumbel_selector.load_state_dict(torch.load(ckpt_path, map_location=device))
        print(f"[1e] ✅ 成功从 {ckpt_path} 加载 Gumbel Selector 权重.")
    else:
        print("[1e] ℹ️ No Gumbel Selector checkpoint found, using random initialization.")

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
    # 步骤 2: 批量测试
    # ==========================================================================
    
    print("\n步骤 2: 开始批量测试")
    
    # --- 2a. 设置参数 ---
    print("\n[2a] 设置测试参数...")
    DATASET_PATH = "data/ViDoSeek/rag_dataset.json"
    START_INDEX = 25  # 从第几个样本开始测试
    NUM_TO_TEST = 10  # 希望测试的样本数量
    print("[2a] ✅ 设置完成.")

    # --- 2b. 加载测试样本 ---
    print(f"\n[2b] 从 {DATASET_PATH} 加载测试样本...")
    try:
        with open(DATASET_PATH, "r", encoding="utf-8") as f:
            examples = json.load(f)["examples"]
        print(f"[2b] ✅ Dataset loaded. Found {len(examples)} total examples.")
    except FileNotFoundError:
        print(f"❌ ERROR: Dataset file not found at {DATASET_PATH}. Cannot proceed.")
        return

    # --- 2c. 循环执行测试 ---
    end_index = min(START_INDEX + NUM_TO_TEST, len(examples))
    cnt = 0
    for i in range(START_INDEX, end_index):
        cnt += 1
        sample = examples[i]
        query = sample.get("query")
        reference_answer = sample.get("reference_answer", "(This sample has no reference answer)")
        
        modality_index = None
        
        ### Test
        # modality = sample.get("meta_info")['source_type']
        
        # if modality == "text":
        #     modality_index = 0
        # elif modality == "2d_layout" or modality == "chart":
        #     modality_index = 1
        # elif modality == "table":
        #     modality_index = 2
        ### Test

        if not query:
            print(f"\n--- Skipping test case {i+1} (Index: {i}) because it has no query. ---")
            continue

        print("\n" + "#"*60)
        print(f"# Running Test Case {cnt} / {NUM_TO_TEST} (Dataset Index: {i})")
        print("#"*60)
        
        print(f"\n[User Query]: {query}")
        print("-" * 50)

        # --- 执行 ---
        # 1. 获取查询嵌入
        # (假设 query_embedder 已在初始化步骤中创建)
        from src.utils.embedding_utils import get_query_embedding
        query_embedding = get_query_embedding(query_embedder, query)
        
        # 2. 调用 Orchestrator 一键运行
        # 可以手动设置 top_k
        final_answer = orchestrator.run(
            query=query,
            query_embedding=query_embedding,
            initial_top_k=5,
            setted_modality_index = modality_index
        )

        # --- 步骤 3: 输出结果 (包含对比) ---
        print("\n" + "="*30)
        print(f"🎉 Pipeline Execution Finished for Case {i+1} 🎉")
        print(f"\n[Initial Query]: {query}")
        print("-" * 40)
        print(f"[Model's Final Answer]:\n{final_answer}")
        print("-" * 40)
        print(f"[Reference Answer]:\n{reference_answer}")
        print("="*30)

if __name__ == '__main__':
    main()