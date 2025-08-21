# File: my_multimodal_rag/scripts/build_selector_labels.py

import csv, json
from typing import List
from pathlib import Path

from src.searcher.text_retriever import TextRetriever
from src.searcher.image_retriever import ImageRetriever
from src.agent.inspector_agent import InspectorAgent

def top1_rerank_score(inspector: InspectorAgent, query: str, nodes) -> float:
    if not nodes:
        return float("-inf")
    # 直接用 inspector 的 reranker（复用其 tokenizer/model）
    # 这里走它的 run() 会改排序且返回状态；但我们只要 top1 分 -> 可复用其内部逻辑或写个轻量 api
    # 简化：把 run 结果取最高分（注意 run 里会计算 sigmoid(conf)；我们更关心 logits 排序）
    status, fb, ranked_nodes, conf = inspector.run(query, nodes, confidence_threshold=0.0)
    return ranked_nodes[0].score if ranked_nodes else float("-inf")

def main():
    data = json.load(open("data/ViDoSeek/rag_dataset.json", "r", encoding="utf-8"))
    queries: List[str] = [ex["query"] for ex in data["examples"]]

    text_ret = TextRetriever(node_dir="data/ViDoSeek/bge_ingestion")
    img_ret = ImageRetriever(node_dir="data/ViDoSeek/colqwen_ingestion")
    inspector = InspectorAgent()  # 统一文本 reranker

    out = Path("data/ViDoSeek/selector_labels.csv")
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["query", "label", "score_text", "score_image", "note"])

        for q in queries:
            text_nodes = text_ret.retrieve(q, top_k=5)
            img_nodes = img_ret.retrieve(q, top_k=5)

            s_text = top1_rerank_score(inspector, q, text_nodes)
            s_img  = top1_rerank_score(inspector, q, img_nodes)

            # 若 image 路暂时全空，就启发式兜底
            note = ""
            if s_img == float("-inf"):
                lower = q.lower()
                trig = any(t in lower for t in ("图表","图片","图像","figure","chart","diagram","plot"))
                label = 1 if trig else 0
                note = "heuristic"
                s_img = -1e9
            else:
                label = 0 if s_text >= s_img else 1

            w.writerow([q, label, f"{s_text:.4f}", f"{s_img:.4f}", note])

    print(f"✅ labels saved to: {out}")

if __name__ == "__main__":
    main()