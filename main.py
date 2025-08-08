import os
import json
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

# ====== 可选：Qwen-VL 客户端 ======
_qwen_vlm = None
try:
    # 你需要提供 src/vlm/qwen_vlm.py，包含 QwenVLM 类（见之前说明）
    from src.vlm.qwen_vlm import QwenVLM  # noqa
    if os.getenv("DASHSCOPE_API_KEY"):
        _qwen_vlm = QwenVLM(model="qwen-vl-max")  # 会自动读取 DASHSCOPE_API_KEY
        print("✅ Qwen-VL client initialized.")
    else:
        print("ℹ️ DASHSCOPE_API_KEY not set. VLM path disabled (will fallback to text-only).")
except Exception as e:
    print(f"ℹ️ Qwen-VL client not available: {e} (will fallback to text-only).")


def main():
    print("🚀 Initializing Multi-Agent RAG System")

    # 1) 设备与查询嵌入模型
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[1] 初始化查询嵌入器... (device={device})")
    q_embedder = QueryEmbedder(model_name="BAAI/bge-m3", device=device)

    # 2) 加载样例
    print("\n[2] 加载测试样本...")
    dataset_path = "data/ViDoSeek/rag_dataset.json"
    with open(dataset_path, "r", encoding="utf-8") as f:
        examples = json.load(f)["examples"]

    # 按你自己的测试节奏来；这里跑前 3 条
    test_num = 3

    for i in range(test_num):
        sample_idx = i
        sample = examples[sample_idx]
        query = sample["query"]

        # 统一的 Query Embedding
        query_embedding = get_query_embedding(q_embedder, query)

        # 强制模态（0=text, 1=image, 2=chart），为了可复现实验
        src_type = sample.get("meta_info", {}).get("source_type")
        if src_type == "text":
            force_modality = 0
        elif src_type == "chart":
            force_modality = 1  # 图表目前仍走 image-OCR 线路
        else:
            force_modality = 1  # image

        # 3) 检索器懒加载工厂（不提前实例化，避免重复初始化与预编码）
        print("\n[3] 初始化检索器（按需）...")
        factories = {
            "text": lambda: TextRetriever(node_dir="data/ViDoSeek/bge_ingestion"),
            "image": lambda: ImageRetriever(
                node_dir="data/ViDoSeek/colqwen_ingestion",
                ocr_dir="data/ViDoSeek/bge_ingestion",
                text_encoder="BAAI/bge-m3",
                cache_dir=".cache/ocr_embeds",   # ✅ 保留缓存目录
            ),
            "chart": lambda: ChartRetriever(node_dir="data/ViDoSeek/bge_ingestion"),
        }

        # 不提前建，交给 orchestrator 需要时再建
        text_retriever = None
        image_retriever = None
        chart_retriever = None

        # 4) 组装智能体与模态选择器
        print("\n[4] 组装智能体与模态选择器...")
        selector = GumbelModalSelector(
            input_dim=q_embedder.out_dim,  # 1024
            hidden_dim=0,
            num_choices=2,  # 目前只 text / image
            tau=1.0,
        )

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
            chart_retriever=chart_retriever,
        )
        inspector = InspectorAgent(
            reranker_model_name="BAAI/bge-reranker-large",
            heuristic_enable=True,
            default_conf_threshold=0.15,
        )
        # 将 VLM 客户端注入 Synthesizer（仅在有图片证据时会被调用）
        synthesizer = SynthesizerAgent(
            model_name="google/flan-t5-large",
            vlm_client=_qwen_vlm,
            use_vlm=True,
        )

        orchestrator = RAGOrchestrator(
            seeker=seeker,
            inspector=inspector,
            synthesizer=synthesizer,
            gumbel_selector=selector,
            use_modal_selector=True,
            # 关键：永远把三种 factory 都传下去，真正用到哪个再懒加载
            lazy_init_factories=factories,
        )

        print("\n" + "=" * 60)
        print(f"🚀 开始执行任务，查询: '{query}'")
        print("=" * 60)

        # 5) 运行编排器
        final_answer = orchestrator.run(query, query_embedding, force_modality=force_modality)

        # 6) 打印结果
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