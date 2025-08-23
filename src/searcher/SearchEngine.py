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
        按需获取或创建检索器实例。
        """
        
        print(f"[SeachEngine] 步骤 2.1: 加载 '{modality.upper()}' 模态检索器... ")
        
        if modality in self.active_retrievers:
            print(f"[SearchEngine] 步骤 2.2: 使用已经缓存过的 '{modality}' retriever.")
            return self.active_retrievers[modality]

        factory = self.retriever_factories.get(modality)
        if factory is None:
            raise ValueError(f"No retriever factory found for modality: '{modality}'")

        retriever = factory()
        
        self.active_retrievers[modality] = retriever
        return retriever


    def search(self, query: str, modality: str, top_k: int) -> List[Any]:
        """
        执行搜索。它会自动处理检索器的懒加载。
        """

        try:
            retriever = self._get_retriever(modality)
        except Exception as e:
            print(f"❌ Failed to get or create retriever for modality '{modality}': {e}")
            return []

        if hasattr(retriever, 'search'):
            return retriever.search(query, top_k)
        elif hasattr(retriever, 'retrieve'):
            return retriever.retrieve(query, top_k)
        else:
            raise NotImplementedError(f"The retriever for '{modality}' does not have a 'search' or 'retrieve' method.")

if __name__ == '__main__':
    print("hello")