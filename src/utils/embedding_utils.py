# 文件路径: my_multimodal_rag/src/utils/embedding_utils.py

import torch
from transformers import AutoTokenizer, AutoModel

# --------------------------------------------------------------------------
# 在这里，我们定义了要使用的模型，参考vidorag，我们选择BAAI/bge-m3
MODEL_NAME = "BAAI/bge-m3"
print(f"正在加载嵌入模型: {MODEL_NAME}...")
print("（首次运行需要下载模型，请耐心等待...）")

# 尝试将模型加载到GPU（如果可用），否则使用CPU
try:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    embedding_model = AutoModel.from_pretrained(MODEL_NAME).to(device)
    embedding_model.eval() # 将模型设置为评估模式
    print(f"嵌入模型成功加载到: {device}")
except Exception as e:
    print(f"加载嵌入模型失败，请检查网络连接或模型名称是否正确。错误: {e}")
    # 如果加载失败，则退出或使用模拟函数
    embedding_model = None
# --------------------------------------------------------------------------


def get_text_embedding(text: str) -> torch.Tensor:
    """
    将单句文本转换为一个真实的嵌入向量。
    """
    if embedding_model is None:
        print("错误：嵌入模型未成功加载，返回一个随机向量。")
        return torch.randn(1, 1024) # BGE-M3 的维度是 1024

    # 1. 使用分词器对文本进行编码
    encoded_input = tokenizer(text, padding=True, truncation=True, return_tensors='pt').to(device)
    
    # 2. 将编码后的数据输入模型进行计算 (关闭梯度计算以节省资源)
    with torch.no_grad():
        model_output = embedding_model(**encoded_input)
        
    # 3. 从模型输出中提取句子嵌入向量
    #    BGE模型推荐的做法是取[CLS]token的向量并进行归一化
    sentence_embedding = model_output[0][:, 0] # 取[CLS] token
    sentence_embedding = torch.nn.functional.normalize(sentence_embedding, p=2, dim=1)
    
    return sentence_embedding