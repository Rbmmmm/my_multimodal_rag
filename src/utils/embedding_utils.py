# 文件路径: my_multimodal_rag/src/utils/embedding_utils.py

import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F


class QueryEmbedder:
    """
    通用的查询嵌入器封装，便于后续切换不同模型（BGE/Qwen3等）。
    """
    def __init__(self, model_name: str = "BAAI/bge-m3", device: str = None):
        print(f"[Embedding] 正在加载查询嵌入模型: {model_name} ...")
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # 加载分词器和模型
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True, use_safetensors=True).to(self.device)
        self.model.eval()

        # 推一次，获取输出维度
        with torch.no_grad():
            test_inp = self.tokenizer("test", return_tensors="pt").to(self.device)
            out = self.model(**test_inp)[0][:, 0]
            self.out_dim = out.shape[-1]

        print(f"[Embedding] 模型已加载到 {self.device} | 输出维度: {self.out_dim}")

    def encode(self, text: str) -> torch.Tensor:
        """
        将输入文本编码为向量（形状：[D]）。
        """
        encoded_input = self.tokenizer(text, padding=True, truncation=True, return_tensors='pt').to(self.device)
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        # BGE 系列取 [CLS] token
        sentence_embedding = model_output[0][:, 0]
        sentence_embedding = F.normalize(sentence_embedding, p=2, dim=1)
        return sentence_embedding.squeeze(0)  # 返回 [D]


def get_query_embedding(embedder: QueryEmbedder, query: str) -> torch.Tensor:
    """
    给定 embedder 和 query，返回形状 [1, D] 的张量。
    """
    vec = embedder.encode(query)  # [D]
    return vec.unsqueeze(0)  # [1, D]