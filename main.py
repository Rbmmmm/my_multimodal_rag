# File: my_multimodal_rag/main.py

import json
import os
import torch

from src.retrievers.text_retriever import TextRetriever
from src.retrievers.image_retriever import ImageRetriever
from src.retrievers.chart_retriever import ChartRetriever

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
    q_embedder = QueryEmbedder(model_name="BAAI/bge-m3", device=device)

    # -------------------------------
    # 2) 加载样例
    # -------------------------------
    print("\n[2] 加载测试样本...")
    dataset_path = "data/ViDoSeek/rag_dataset.json"
    with open(dataset_path, "r", encoding="utf-8") as f:
        examples = json.load(f)["examples"]

    sample_idx = 3
    sample = examples[sample_idx]
    query = sample["query"]

    # 统一的 Query Embedding
    query_embedding = get_query_embedding(q_embedder, query)

    # 可选：强制某一路（0=text, 1=image, 2=chart）；不强制用 None
    force_modality = 1  # ← 现在先压测 image

    # -------------------------------
    # 3) 检索器按需初始化（只立即建被强制那一路，其余交给懒加载）
    # -------------------------------
    print("\n[3] 初始化检索器（按需）...")

    # 统一用工厂，防止重复实例化
    factories = {
        "text":  lambda: TextRetriever(node_dir="data/ViDoSeek/bge_ingestion"),
        "image": lambda: ImageRetriever(
            node_dir="data/ViDoSeek/colqwen_ingestion",
            ocr_dir="data/ViDoSeek/bge_ingestion",
            text_encoder="BAAI/bge-m3",
        ),
        "chart": lambda: ChartRetriever(node_dir="data/ViDoSeek/bge_ingestion"),
    }

    # 仅当被强制时才立即实例化；否则留给 orchestrator 懒加载
    text_retriever  = factories["text"]()  if force_modality == 0 else None
    image_retriever = factories["image"]() if force_modality == 1 else None
    chart_retriever = factories["chart"]() if force_modality == 2 else None

    # -------------------------------
    # 4) 组装智能体与模态选择器
    # -------------------------------
    print("\n[4] 组装智能体与模态选择器...")
    selector = GumbelModalSelector(
        input_dim=q_embedder.out_dim,  # 1024
        hidden_dim=0,
        num_choices=2,                 # 现在只 text / image
        tau=1.0
    )

    ckpt_path = "checkpoints/modal_selector_text_image.pt"
    if os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, map_localtion="cpu")  # 兼容不同环境
        selector.load_state_dict(state, strict=False)
        selector.eval()
        print(f"✅ Loaded selector weights from {ckpt_path}")
    else:
        print("ℹ️ No selector checkpoint found, using randomly initialized selector.")

    seeker = SeekerAgent(
        text_retriever=text_retriever,
        image_retriever=image_retriever,
        chart_retriever=chart_retriever,
    )
    inspector = InspectorAgent(  # 内部已含滑窗 + 启发式
        # 默认使用 BAAI/bge-reranker-large
        # 可按需覆盖 window_tokens / stride / batch_size
    )
    synthesizer = SynthesizerAgent(model_name="google/flan-t5-large")

    orchestrator = RAGOrchestrator(
        seeker=seeker,
        inspector=inspector,
        synthesizer=synthesizer,
        gumbel_selector=selector,
        use_modal_selector=True,
        # 关键：把工厂传下去，真正用到再初始化（只建一次）
        lazy_init_factories=factories,
    )

    print("\n" + "=" * 60)
    print(f"🚀 开始执行任务，查询: '{query}'")
    print("=" * 60)

    # -------------------------------
    # 5) 运行编排器
    # -------------------------------
    final_answer = orchestrator.run(query, query_embedding, force_modality=force_modality)

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