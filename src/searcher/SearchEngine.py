# File: search_engine.py

import os,json
import torch
import torch.nn as nn
import numpy as np
from sklearn.mixture import GaussianMixture
from typing import List, Dict, Any, Callable

def gmm_filter(results: List[Dict[str, Any]], max_valid_length: int = 10, min_valid_length: int = 5) -> List[Dict[str, Any]]:
    if not results or len(results) < 2:
        return results

    scores = [res['score'] for res in results if res['score'] is not None]
    if len(scores) <= min_valid_length:
        return results[:max_valid_length]

    scores_arr = np.array(scores).reshape(-1, 1)
    gmm = GaussianMixture(n_components=2, random_state=0).fit(scores_arr)
    
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
        print("[1d] 初始化懒加载 SearchEngine...")
        if not isinstance(retriever_factories, dict):
            raise TypeError("retriever_factories must be a dictionary of callable factories.")
            
        self.retriever_factories = retriever_factories
        self.active_retrievers: Dict[str, Any] = {} # 用于缓存已实例化的检索器
        print("[1d] ✅ 懒加载 SearchEngine 已完成. 检索器会在第一次使用时加载.")


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


        print(f"[SeachEngine] 步骤 2.1: 加载 '{modality.upper()}' 模态检索器... ")

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

if __name__ == '__main__':
    print("hello")