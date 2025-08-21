# File: search_engine.py

import os,json
import torch
import torch.nn as nn
import numpy as np
from sklearn.mixture import GaussianMixture
from typing import List, Dict, Any, Callable

# 导入具体的检索器实现
# from src.searcher.text_searcher import TextSearcher
# from src.searcher.image_searcher import ImageSearcher
# from src.searcher.table_searcher import TableSearcher

# --- GMM 后处理函数 (从 Vidorag 借鉴) ---
def gmm_filter(results: List[Dict[str, Any]], max_valid_length: int = 10, min_valid_length: int = 5) -> List[Dict[str, Any]]:
    if not results or len(results) < 2:
        return results

    scores = [res['score'] for res in results if res['score'] is not None]
    if len(scores) <= min_valid_length:
        return results[:max_valid_length]

    scores_arr = np.array(scores).reshape(-1, 1)
    gmm = GaussianMixture(n_components=2, random_state=0).fit(scores_arr)
    
    # 假设分数高的那一簇是更相关的
    higher_mean_cluster_label = gmm.means_.argmax()
    labels = gmm.predict(scores_arr)
    
    filtered_results = [res for res, label in zip(results, labels) if label == higher_mean_cluster_label]
    
    return filtered_results[:max_valid_length]


# --- 自适应检索器的占位实现 (为第二阶段预留) ---
class AdaptiveRetriever(nn.Module):
    """
    可训练的自适应检索器 (STE K值选择)。
    这是第二阶段要实现的核心模块。
    """
    def __init__(self, base_searcher: Any):
        super().__init__()
        self.base_searcher = base_searcher
        
        # TODO: 定义信心预测器 f_conf, 例如一个小型MLP
        # self.confidence_predictor = nn.Sequential(...)

    def forward(self, query: Any, initial_k: int):
        print("Warning: AdaptiveRetriever is a placeholder and will use fixed K for now.")
        # 第一阶段：直接调用基础检索器，使用固定K值
        return self.base_searcher.search(query, initial_k)
        
        # 第二阶段的完整逻辑:
        # 1. results = self.base_searcher.search(query, initial_k)
        # 2. confidence = self.confidence_predictor(query, results)
        # 3. if confidence < threshold:
        # 4.     new_k = calculate_new_k(initial_k, confidence)
        # 5.     results = self.base_searcher.search(query, new_k)
        # 6. return results


# --- 核心调度器 ---
class SearchEngine:
    """
    一个支持懒加载（Lazy Loading）的总调度器。
    它在接收到第一个特定模态的查询时，才会去初始化对应的检索器。
    """
    def __init__(self, retriever_factories: Dict[str, Callable[[], Any]]):
        """
        初始化时只接收“如何创建检索器”的指令（工厂），而不是立即创建它们。

        Args:
            retriever_factories: 一个字典，key是模态名('text', 'image', 'table')，
                                 value是创建对应检索器的无参数函数 (lambda)。
        """
        print("Initializing the LAZY SearchEngine...")
        if not isinstance(retriever_factories, dict):
            raise TypeError("retriever_factories must be a dictionary of callable factories.")
            
        self.retriever_factories = retriever_factories
        self.active_retrievers: Dict[str, Any] = {} # 用于缓存已实例化的检索器
        print("✅ LAZY SearchEngine is ready. Retrievers will be loaded on first use.")


    def _get_retriever(self, modality: str) -> Any:
        """
        按需获取或创建检索器实例。这是懒加载的核心。
        """
        # 1. 检查是否已经创建过这个检索器（是否在缓存中）
        if modality in self.active_retrievers:
            print(f"[SearchEngine] Using cached '{modality}' retriever.")
            return self.active_retrievers[modality]

        # 2. 如果没有，就从工厂字典里找到对应的创建函数
        factory = self.retriever_factories.get(modality)
        if factory is None:
            raise ValueError(f"No retriever factory found for modality: '{modality}'")

        # 3. 调用工厂函数，真正地创建实例（这是最耗时的一步）
        print(f"============================================================")
        print(f" R LAZY LOADING retriever for '{modality.upper()}' NOW... ")
        print(f"============================================================")
        retriever = factory()
        
        # 4. 将创建好的实例存入缓存，以便下次直接使用
        self.active_retrievers[modality] = retriever
        return retriever


    def search(self, query: str, modality: str, top_k: int) -> List[Any]:
        """
        执行搜索。它会自动处理检索器的懒加载。
        """
        # 在真正搜索前，才去获取（或创建）相应的检索器
        try:
            retriever = self._get_retriever(modality)
        except Exception as e:
            print(f"❌ Failed to get or create retriever for modality '{modality}': {e}")
            return []
        
        # 调用检索器的 search 方法
        # (确保您所有的检索器都有一个统一的.search()或.retrieve()方法)
        if hasattr(retriever, 'search'):
            return retriever.search(query, top_k)
        elif hasattr(retriever, 'retrieve'):
            return retriever.retrieve(query, top_k)
        else:
            raise NotImplementedError(f"The retriever for '{modality}' does not have a 'search' or 'retrieve' method.")


# --- 使用示例 ---
if __name__ == '__main__':
    # 为了让这个示例能跑起来，我们需要创建一些假的索引文件
    print("--- Creating dummy data for demonstration ---")
    DUMMY_DATASET = 'placeholder_dataset'
    os.makedirs(os.path.join('data', DUMMY_DATASET), exist_ok=True)
    
    # 假向量 (5个文档, 384维度，这是all-MiniLM-L6-v2的维度)
    dummy_embeddings = torch.randn(5, 384)
    torch.save(dummy_embeddings, os.path.join('data', DUMMY_DATASET, 'embeddings.pt'))
    
    # 假元数据
    dummy_metadata = [
        {'id': 'doc1', 'text': 'The sky is blue.'},
        {'id': 'doc2', 'text': 'The cat sleeps on the mat.'},
        {'id': 'doc3', 'text': 'Blueberries are a type of fruit.'},
        {'id': 'doc4', 'text': 'PyTorch is a deep learning framework.'},
        {'id': 'doc5', 'text': 'A cat is a small domesticated carnivorous mammal.'}
    ]
    with open(os.path.join('data', DUMMY_DATASET, 'metadata.jsonl'), 'w') as f:
        for item in dummy_metadata:
            f.write(json.dumps(item) + '\n')
    print("--- Dummy data created. ---")

    # 1. 初始化搜索引擎
    engine = SearchEngine(dataset_name_1=DUMMY_DATASET)
    
    # 2. 执行一个文本搜索
    my_query = "What is a cat?"
    print(f"\nPerforming search for: '{my_query}'")
    
    # 2a. 不使用GMM
    search_results = engine.search(query=my_query, modality='text', top_k=3, use_gmm=False)
    print("\n--- Results (Top 3, without GMM) ---")
    for res in search_results:
        print(f"ID: {res['id']}, Score: {res['score']:.4f}, Content: {res['content']}")
        
    # 2b. 使用GMM
    # 为了让GMM有意义，我们检索更多结果
    search_results_for_gmm = engine.search(query=my_query, modality='text', top_k=5, use_gmm=True)
    print("\n--- Results (Top 5, with GMM filter) ---")
    for res in search_results_for_gmm:
        print(f"ID: {res['id']}, Score: {res['score']:.4f}, Content: {res['content']}")