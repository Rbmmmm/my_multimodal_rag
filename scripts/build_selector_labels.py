# File: my_multimodal_rag/scripts/build_selector_labels.py
import json
import csv
from pathlib import Path
import re

DATASET = "data/ViDoSeek/rag_dataset.json"
OUT_CSV = "data/ViDoSeek/selector_labels.csv"

# 简单触发词（你可以按数据再补充）
IMAGE_TRIGGERS = [
    "图片", "图像", "图表", "示意图", "曲线", "直方图", "柱状图", "饼图",
    "figure", "image", "diagram", "chart", "plot", "table", "screenshot"
]

def looks_image_query(q: str) -> bool:
    ql = q.lower()
    if any(t in q for t in IMAGE_TRIGGERS):
        return True
    # 简单启发：包含 "see the figure", "in the image", "as shown"
    if re.search(r"\b(figure|image|diagram|chart|plot|screenshot)\b", ql):
        return True
    return False

def main():
    with open(DATASET, "r", encoding="utf-8") as f:
        data = json.load(f)

    examples = data.get("examples", [])
    out_path = Path(OUT_CSV)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for ex in examples:
        q = ex["query"]
        # 规则：触发词 → image(1)，否则 text(0)
        label = 1 if looks_image_query(q) else 0
        rows.append({"query": q, "label": label})

    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["query", "label"])
        w.writeheader()
        w.writerows(rows)

    print(f"✅ wrote {len(rows)} rows to {out_path}")

if __name__ == "__main__":
    main()