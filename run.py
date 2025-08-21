# File: run.py
# The final, clean entry point to initialize and run the entire RAG system.

import os
import torch

# --- 导入您项目中的所有核心模块 ---
from src.searcher.text_searcher import TextSearcher
from src.searcher.image_searcher import ImageSearcher
from src.searcher.table_searcher import TableSearcher
from src.searcher.SearchEngine import SearchEngine
from src.agent.seeker_agent import SeekerAgent
from src.agent.inspector_agent import InspectorAgent
from src.agent.synthesizer_agent import SynthesizerAgent
from src.models.gumbel_selector import GumbelModalSelector
from src.orchestrator import RAGOrchestrator
from src.llms.llm import LLM               # 假设您有一个VLM客户端的封装
from src.utils.embedding_utils import QueryEmbedder      # 假设您有一个查询嵌入器的封装

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
    print("Step 1: Initializing System Components...")
    
    # --- 1a. 设置 API Key 和设备 ---
    # !! 重要 !!: 请将您的 API Key 填入此处
    os.environ['DASHSCOPE_API_KEY'] = "sk-fdb11107afd1435398e9d40958af5e42"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if 'xxxx' in os.environ['DASHSCOPE_API_KEY']:
        print("⚠️ WARNING: DASHSCOPE_API_KEY is not set. VLM calls will fail.")

    # --- 1b. 初始化核心模型 ---
    print("\nInitializing core models...")
    query_embedder = QueryEmbedder(model_name="BAAI/bge-m3", device=device)
    vlm = LLM("qwen-vl-max")

    # --- 1c. 创建检索器“工厂” (Retriever Factories) ---
    print("\n[1c] Defining retriever factories for lazy loading...")
    retriever_factories = {
        "text": lambda: TextSearcher(
            dataset_name='ViDoSeek',
            mode='bi_encoder',
            # --- 修正：为 ColBERT 模式明确提供包含 .node 文件的目录前缀 ---
            # ColBERT 需要读取这些原始文本节点来建立自己的索引
            node_dir_prefix='bge_ingestion' 
        ),
        "image": lambda: ImageSearcher(
            dataset_name='ViDoSeek',
            mode='vl_search',
            # 图像检索器也可能需要指定其节点目录
            vl_node_dir_prefix='colqwen_ingestion' 
        ),
        "table": lambda: TableSearcher(
            dataset_name='ViDoSeek',
            mode='vl_search',
            # 表格检索器同理
            vl_node_dir_prefix='colqwen_ingestion' # 假设表格和图像用同一套多模态节点
        )
    }

    # --- 1d. 初始化支持懒加载的 SearchEngine ---
    # 注意：这里只传入了工厂，没有进行任何实际的模型加载！
    search_engine = SearchEngine(retriever_factories=retriever_factories)
    print("✅ LAZY SearchEngine initialized.")

    # --- 1e. 初始化 Gumbel 模态选择器 ---
    gumbel_selector = GumbelModalSelector(
        input_dim=query_embedder.out_dim,
        num_choices=3, # 0=text, 1=image, 2=table
        trainable=False
    ).to(device).eval()

    # (可选) 加载训练好的选择器权重
    ckpt_path = "checkpoints/modal_selector.pt"
    if os.path.exists(ckpt_path):
        gumbel_selector.load_state_dict(torch.load(ckpt_path, map_location=device))
        print(f"✅ Loaded Gumbel Selector weights from {ckpt_path}")
    else:
        print("ℹ️ No Gumbel Selector checkpoint found, using random initialization.")

    print("\nInitializing Agents...")
    image_base_dir = "data/ViDoSeek/img" # 定义图片所在的根目录
    seeker_agent = SeekerAgent(vlm=vlm, image_base_dir=image_base_dir)
    inspector_agent = InspectorAgent(vlm=vlm, image_base_dir=image_base_dir, reranker_model_name="BAAI/bge-reranker-large")
    synthesizer_agent = SynthesizerAgent(vlm=vlm, image_base_dir=image_base_dir)
    print("✅ All Agents initialized.")

    # --- 1f. 组装总指挥 (Orchestrator) ---
    orchestrator = RAGOrchestrator(
        search_engine=search_engine,
        seeker=seeker_agent,          # <-- 使用已经创建好的实例
        inspector=inspector_agent,    # <-- 使用已经创建好的实例
        synthesizer=synthesizer_agent,  # <-- 使用已经创建好的实例
        gumbel_selector=gumbel_selector
    )
    
    print("\n✅ All components initialized. Orchestrator is ready.")
    print("="*30)

    # --- 2a. 配置批量测试 ---
    DATASET_PATH = "data/ViDoSeek/rag_dataset.json"
    START_INDEX = 10  # 从第几个样本开始测试
    NUM_TO_TEST = 5  # 希望测试的样本数量

    # --- 2b. 加载测试样本 ---
    print(f"\n[Batch Test] Loading dataset from {DATASET_PATH}...")
    try:
        with open(DATASET_PATH, "r", encoding="utf-8") as f:
            examples = json.load(f)["examples"]
        print(f"✅ Dataset loaded. Found {len(examples)} total examples.")
    except FileNotFoundError:
        print(f"❌ ERROR: Dataset file not found at {DATASET_PATH}. Cannot proceed.")
        return

    # --- 2c. 循环执行测试 ---
    end_index = min(START_INDEX + NUM_TO_TEST, len(examples))
    for i in range(START_INDEX, end_index):
        sample = examples[i]
        query = sample.get("query")
        reference_answer = sample.get("reference_answer", "(This sample has no reference answer)")

        if not query:
            print(f"\n--- Skipping test case {i+1} (Index: {i}) because it has no query. ---")
            continue

        print("\n" + "#"*60)
        print(f"# Running Test Case {i+1} / {NUM_TO_TEST} (Dataset Index: {i})")
        print("#"*60)
        
        print(f"\n[User Query]: {query}")
        print("-" * 50)

        # --- 执行 ---
        # 1. 获取查询嵌入
        # (假设 query_embedder 已在初始化步骤中创建)
        from src.utils.embedding_utils import get_query_embedding
        query_embedding = get_query_embedding(query_embedder, query)
        
        # 2. 调用 Orchestrator 一键运行
        final_answer = orchestrator.run(
            query=query,
            query_embedding=query_embedding,
            initial_top_k=10
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