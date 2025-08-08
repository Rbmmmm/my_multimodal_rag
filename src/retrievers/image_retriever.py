# File: my_multimodal_rag/src/retrievers/image_retriever.py
from typing import List
from llama_index.core.schema import NodeWithScore

class ImageRetriever:
    """
    占位或真实实现皆可。
    关键：retrieve() 返回 NodeWithScore；节点内容能被 inspector 的 reranker 使用。
    """
    def __init__(self, node_dir: str | None = None):
        self.node_dir = node_dir
        print("图像检索器 (ImageRetriever): 准备使用 ColQwen/UniME 图像索引。")

    def retrieve(self, query: str, top_k: int = 3) -> List[NodeWithScore]:
        """
        TODO: 替换为实际的图像检索逻辑（文本query -> 图像向量空间相似度）。
        目前先返回空或 mock，保证流程可跑。
        """
        print(f"Running Image retrieval (TopK={top_k}) [placeholder]")
        return []

    @staticmethod
    def to_text_view(node: NodeWithScore) -> str:
        """
        把图像节点转成可重排的文本视图：例如 caption 或 OCR 片段。
        真实实现里，你应当把 node 的元数据里的 'caption'/'ocr' 取出来。
        """
        content = node.get_content()
        if isinstance(content, str) and content.strip():
            return content
        # fallback：若没有文本，返回空字符串（reranker会给低分）
        return ""