# File: base_searcher.py

from abc import ABC, abstractmethod
from typing import List, Dict, Any

class BaseSearcher(ABC):
    """
    抽象基类，用于定义所有检索器的标准接口。
    """
    @abstractmethod
    def search(self, query: Any, top_k: int) -> List[Dict[str, Any]]:
        """
        根据查询，从索引中检索最相关的 top_k 个结果。

        Args:
            query (Any): 查询内容（可以是字符串、图像等）。
            top_k (int): 需要返回的最相关结果的数量。

        Returns:
            List[Dict[str, Any]]: 一个包含检索结果的列表。
                                  每个结果是一个字典，至少应包含 'id', 'content', 和 'score'。
        """
        pass