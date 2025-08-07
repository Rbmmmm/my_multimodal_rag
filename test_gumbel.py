# 文件路径: my_multimodal_rag/test_real_query.py

import torch

try:
    from src.models.gumbel_selector import GumbelModalSelector
except ImportError:
    print("错误: 无法找到 'GumbelModalSelector'。请确保 'src/models/gumbel_selector.py' 文件存在且路径正确。")
    exit()

try:
    from src.utils.embedding_utils import get_text_embedding, MODEL_NAME
except ImportError:
    print("错误: 无法找到 'get_text_embedding'。请确保 'src/utils/embedding_utils.py' 文件存在且路径正确。")
    exit()


def run_real_query_test():
    """使用一个真实的查询和真实的嵌入模型来详细演示工作流程。"""
    print("\n" + "="*60)
    print("🚀 开始使用真实Query和真实嵌入模型进行Gumbel网络测试")
    print("="*60)
    
    # Gumbel选择器的输入维度需要和嵌入模型的输出维度一致
    embedding_dimension = 1024 # BGE-M3 的输出维度是 1024
    modal_options = ["文本检索路径", "图像检索路径", "表格检索路径"]
    
    # --- 这是已修正的部分 ---
    # 1. 定义我们希望使用的设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"将在此设备上运行Gumbel选择器: {device}")

    # 2. 创建模型实例
    selector = GumbelModalSelector(input_dim=embedding_dimension, num_choices=len(modal_options))
    
    # 3. 将整个模型移动到指定的设备！
    selector.to(device)

    # 4. 将模型设置为评估模式
    selector.eval()
    # --- 修正结束 ---

    real_query = "Apply for Nordic Swan Ecolabel license, what is recommended as a web browser according to the Nordic Ecolabelling Portal instructions?"
    print(f"\n[待处理查询]: '{real_query}'")
    print("-" * 60)

    # --- 步骤 1: 将文本查询转换为真实的嵌入向量 ---
    print(f"[步骤 1/4]: 文本查询 -> 真实嵌入向量 (使用 {MODEL_NAME})")
    try:
        # get_text_embedding 函数内部已经处理了设备，所以返回的向量在GPU上
        query_embedding = get_text_embedding(real_query)
        print(f"✅ 成功生成查询向量，形状: {query_embedding.shape}")
    except Exception as e:
        print(f"❌ 生成查询向量失败: {e}")
        return
    print("-" * 60)

    # --- 步骤 2: 获取原始打分 (Logits) ---
    print("[步骤 2/4]: 嵌入向量 -> 原始分数 (Logits)")
    with torch.no_grad():
        # 现在模型和数据都在同一个设备上，这行代码将能正常运行
        logits = selector.classifier(query_embedding)
    print(f"✅ 模型生成的原始分数: {logits.numpy(force=True).flatten()}") # 使用 force=True 从GPU安全地移到CPU进行打印
    print("-" * 60)
    
    # --- 步骤 3: Gumbel-Softmax 决策 ---
    print("[步骤 3/4]: Logits -> Gumbel-Softmax -> 独热向量决策")
    with torch.no_grad():
        selection_one_hot = selector(query_embedding, temperature=1.0)
    print(f"✅ 生成的独热向量决策: {selection_one_hot.numpy(force=True).flatten()}")
    print("-" * 60)

    # --- 步骤 4: 解析最终结果 ---
    print("[步骤 4/4]: 独热向量 -> 人类可读的结果")
    choice_index = selection_one_hot.argmax().item()
    final_decision = modal_options[choice_index]
    
    print("\n" + "="*25)
    print("🎉 最终决策结果 🎉")
    print(f"对于查询: '{real_query}'")
    print(f"Gumbel网络决策的最终结果是: 【{final_decision}】")
    print("="*25)

if __name__ == '__main__':
    run_real_query_test()