# File: my_multimodal_rag/main.py

import json
import os
import torch

from src.retrievers.text_retriever import TextRetriever
from src.retrievers.image_retriever import ImageRetriever
from src.retrievers.table_retriever import TableRetriever

from src.agents.seeker_agent import SeekerAgent
from src.agents.inspector_agent import InspectorAgent
from src.agents.synthesizer_agent import SynthesizerAgent

from src.models.gumbel_selector import GumbelModalSelector
from src.orchestrator import RAGOrchestrator
from src.utils.embedding_utils import QueryEmbedder, get_query_embedding


def main():
    print("🚀 Initializing Multi-Agent RAG System")

    # -------------------------------
    # 1) 设备与查询嵌入模型
    # -------------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[1] 初始化查询嵌入器... (device={device})")
    # 先用 BGE m3（1024d）跑通；后续切 Qwen3 只需改 model_name 与 selector input_dim
    q_embedder = QueryEmbedder(model_name="BAAI/bge-m3", device=device)

    # -------------------------------
    # 2) 初始化检索器
    # -------------------------------
    print("\n[2] 初始化各路径检索器...")
    text_retriever = TextRetriever(node_dir="data/ViDoSeek/bge_ingestion")
    image_retriever = ImageRetriever(node_dir="data/ViDoSeek/colqwen_ingestion")  # 若构造签名不同可去掉参数
    table_retriever = TableRetriever()  # 现阶段未用

    # -------------------------------
    # 3) 组装智能体与选择器
    # -------------------------------
    print("\n[3] 组装智能体与模态选择器...")
    selector = GumbelModalSelector(
        input_dim=q_embedder.out_dim,  # 与 QueryEmbedder 对齐（BGE m3 为 1024）
        hidden_dim=0,                  # 更稳可设 256
        num_choices=2,                 # 阶段B：text/image
        tau=1.0
    )

    # 自动加载已训练权重（可选）
    ckpt_path = "checkpoints/modal_selector_text_image.pt"
    if os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, map_location="cpu")
        selector.load_state_dict(state, strict=False)
        selector.eval()
        print(f"✅ Loaded selector weights from {ckpt_path}")
    else:
        print("ℹ️ No selector checkpoint found, using randomly initialized selector.")

    seeker = SeekerAgent(
        text_retriever=text_retriever,
        image_retriever=image_retriever,
        table_retriever=table_retriever
    )
    inspector = InspectorAgent()
    synthesizer = SynthesizerAgent(model_name="google/flan-t5-large")

    orchestrator = RAGOrchestrator(
        seeker=seeker,
        inspector=inspector,
        synthesizer=synthesizer,
        gumbel_selector=selector,
        use_modal_selector=True  # 一键开/关模态选择器
    )

    # -------------------------------
    # 4) 加载样例 & 计算 query embedding
    # -------------------------------
    print("\n[4] 加载测试样本...")
    dataset_path = "data/ViDoSeek/rag_dataset.json"
    with open(dataset_path, "r", encoding="utf-8") as f:
        examples = json.load(f)["examples"]

    # 你可以换索引测试不同 query
    sample = examples[1]
    query = sample["query"]

    print("\n" + "=" * 60)
    print(f"🚀 开始执行任务，查询: '{query}'")
    print("=" * 60)

    # 统一的 Query Embedding（形状：[1, D]）
    query_embedding = get_query_embedding(q_embedder, query)

    # -------------------------------
    # 5) 运行编排器
    # -------------------------------
    final_answer = orchestrator.run(query, query_embedding)

    # -------------------------------
    # 6) 打印结果
    # -------------------------------
    print("\n" + "=" * 60)
    print("🎉 RAG 流程执行完毕 🎉")
    print(f"\n[用户问题]: {query}")
    print("-" * 30)
    print(f"[模型生成的最终答案]:\n{final_answer}")
    print("-" * 30)
    print(f"[数据中的参考答案]:\n{sample.get('reference_answer', '(no reference)')}")
    print("=" * 60)


if __name__ == "__main__":
    main()