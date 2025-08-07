# 文件路径: my_multimodal_rag/main.py (强制使用TextRetriever的调试版)

import json
import os
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from src.retrievers.text_retriever import TextRetriever
from src.retrievers.image_retriever import ImageRetriever
from src.retrievers.table_retriever import TableRetriever
from src.agents.seeker_agent import SeekerAgent
from src.agents.inspector_agent import InspectorAgent
from src.agents.synthesizer_agent import SynthesizerAgent
from src.models.gumbel_selector import GumbelModalSelector
from src.orchestrator import RAGOrchestrator
from src.utils.embedding_utils import get_text_embedding # 确保这个文件存在

def main():
    print("🚀 初始化RAG多智能体系统...")
    
    # --- 1. 初始化核心组件 ---
    print("\n[1] 初始化嵌入模型...")
    # 这个LlamaIndex包装器主要给TextRetriever的索引器使用
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-m3", device="cuda")
    
    print("\n[2] 初始化各路径检索器...")
    text_retriever = TextRetriever(node_dir="data/ViDoSeek/bge_ingestion")
    # 保持占位符的初始化，但我们下面不会把它传给 Seeker
    image_retriever = ImageRetriever() 
    table_retriever = TableRetriever()
    
    # --- 2. 组装智能体 ---
    print("\n[3] 组装智能体...")
    gumbel_selector = GumbelModalSelector(input_dim=1024, num_choices=3) # bge-m3 的维度是 1024
    
    # --- 【调试修改】 ---
    # 在这里，我们只将 text_retriever 传递给 SeekerAgent，
    # 其他的设置为 None，强制它只能使用文本检索。
    seeker = SeekerAgent(
        text_retriever=text_retriever,
        image_retriever=None,
        table_retriever=None
    )
    
    inspector = InspectorAgent()
    synthesizer = SynthesizerAgent(model_name="google/flan-t5-large")
    
    # --- 3. 实例化总指挥 ---
    orchestrator = RAGOrchestrator(gumbel_selector, seeker, inspector, synthesizer)
    
    # --- 4. 加载测试样本并运行 ---
    print("\n[4] 加载测试样本...")
    dataset_path = "data/ViDoSeek/rag_dataset.json"
    with open(dataset_path, 'r', encoding='utf-8') as f:
        sample = json.load(f)['examples'][1]
    
    query = sample['query']
    
    print("\n" + "="*60)
    print(f"🚀 开始执行任务，查询: '{query}'")
    print("="*60)

    # 【关键修正】调用get_text_embedding时不应传递model_name参数
    query_embedding = get_text_embedding(query)

    final_answer = orchestrator.run(query, query_embedding)

    # ... (展示最终结果)
    print("\n" + "="*60)
    print("🎉 RAG 流程执行完毕 🎉")
    print(f"\n[用户问题]: {query}")
    print("-" * 30)
    print(f"[模型生成的最终答案]:\n{final_answer}")
    print("-" * 30)
    print(f"[数据中的参考答案]:\n{sample['reference_answer']}")
    print("="*60)

if __name__ == '__main__':
    main()