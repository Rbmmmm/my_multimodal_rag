# 2. search engine 需要的嵌入组件

import asyncio
from typing import Any, List, Optional, Union
import json

import torch
import torch.nn.functional as F # PyTorch 里的函数式 API（常用操作：F.normalize、F.cosine_similarity、F.cross_entropy 等）
from colpali_engine.models import (
    ColPali,
    ColPaliProcessor,
    ColQwen2,
    ColQwen2Processor,
) # 多模态 embedding 模型，用于把图文映射到同一向量空间

from llama_index.core.base.embeddings.base import Embedding # Embedding：llama-index 里所有 embedding 模型的基类
from llama_index.core.bridge.pydantic import Field # 来自 Pydantic（数据验证库），llama-index 内部用它来定义配置字段
from llama_index.core.callbacks import CallbackManager # 回调管理器，用来追踪 embedding 调用过程
from llama_index.core.embeddings import MultiModalEmbedding # 多模态 embedding 抽象类
from PIL import Image # Pillow 库，用来加载和处理图片（打开 .png/.jpg，resize，转 tensor），负责读入图片
from transformers import AutoModel, AutoTokenizer, AutoImageProcessor # AutoModel → 自动加载模型（文本、图像、多模态） / AutoModel → 自动加载模型（文本、图像、多模态） / AutoImageProcessor → 自动加载图像预处理器（处理图片输入）
from huggingface_hub import hf_hub_download #  HuggingFace Hub 下载模型或文件


def weighted_mean_pooling(hidden, attention_mask): # 带权平均池化，把序列表示（hidden states）压缩成一个向量表示
    attention_mask_ = attention_mask * attention_mask.cumsum(dim=1)
    s = torch.sum(hidden * attention_mask_.unsqueeze(-1).float(), dim=1)
    d = attention_mask_.sum(dim=1, keepdim=True).float()
    reps = s / d
    return reps # 返回每个序列的向量表示。这通常就是文本的最终 embedding，可以用来做相似度搜索。
# 把一句话里每个词的向量，合并成一个整体的“句子向量”，而且让后面的词更重要

class VL_Embedding(MultiModalEmbedding): # 继承自 MultiModalEmbedding
    model: str = Field(description="The Multi-model to use.") # 代表你要用的多模态模型名字，这是用户在配置文件里可以指定的参数
    dimensions: Optional[int] = Field(
        default=1024,
        description="The number of dimensions the resulting output embeddings should have.",
    ) # 最终 embedding 向量的维度。默认是 1024 维
    mode: str = Field(
        default="text",
        description="The mode of the model, either 'text' or 'image'.",
    ) # 表示当前使用的 模式。默认是 "text"（只做文本 embedding），也可以是 "image"（只做图像 embedding），这样可以灵活选择是 文本检索 还是 图像检索。
    embed_model: Union[ColQwen2, AutoModel, None] = Field(default=None) # 真正的底层模型（加载好的模型对象）
    processor: Optional[ColQwen2Processor] = Field(default=None) # 对应 输入处理器（Preprocessor）。把原始输入（文本/图片）变成模型需要的格式
    tokenizer: Optional[AutoTokenizer] = Field(default=None) # 专门处理文本的 分词器。把一句话（“ViDoRAG is great”）变成 token id [101, 4567, 89, ...]，交给模型。只对文本模式有用。

    def __init__(
        self,
        model: str = "vidore/colqwen2-v1.0",
        **kwargs: Any,
    ) -> None:
        super().__init__(model=model, **kwargs) # 把这些参数交给父类（MultiModalEmbedding/Pydantic 基类）去存储和校验。

        if "openbmb" in model: # 如果 model 字符串里包含 "openbmb"（比如某些 OpenBMB 家族模型），就按 Hugging Face 方式加载
            self.tokenizer = AutoTokenizer.from_pretrained( # 下载并构造分词器
                model, trust_remote_code=True
            )
            self.embed_model = ( # 下载并构造模型；用 bfloat16 省显存、提速推理；device_map="cuda:1" 指定把权重放到 第 1 号 GPU
                AutoModel.from_pretrained(
                    model,
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True,
                    device_map="cuda:1",
                )
                .cuda()
                .eval() # 切换到推理模式（关闭 Dropout等）
            )
        
        elif "vidore" in model and "qwen" in model:
            print("⏳ Applying final 'injection' fix for processor...")
            
            # Step 1: Load and fix the image processor's config dictionary
            config_path = hf_hub_download(repo_id=model, filename="preprocessor_config.json") # 从 Hugging Face Hub 上下载这个模型仓库里的预处理配置文件。
            with open(config_path, 'r', encoding='utf-8') as f:
                preprocessor_dict = json.load(f) # 把 JSON 文件读成一个 Python 字典 preprocessor_dict
            if 'size' in preprocessor_dict and 'max_pixels' in preprocessor_dict['size']:
                original_size_config = preprocessor_dict['size']
                preprocessor_dict['size'] = {
                    "shortest_edge": original_size_config.get('shortest_edge', 1024),
                    "longest_edge": original_size_config.get('longest_edge', 1024)
                } # 检查并修正其中的 size 字段，把不兼容的 max_pixels 转换成 shortest_edge/longest_edge
            
            # Step 2: Create *only* the image processor component from the fixed config
            image_processor = AutoImageProcessor.from_pretrained(model, **preprocessor_dict) # 从 HuggingFace 仓库加载该模型的 图像预处理器（AutoImageProcessor）
            print("✅ Fixed image processor component created.")

            # Step 3: Load the full processor, INJECTING our fixed component.
            print("⏳ Loading full processor with injected component...")
            self.processor = ColQwen2Processor.from_pretrained(
                model,
                image_processor=image_processor, # Inject our fixed part
                trust_remote_code=True
            ) # 加载完整的处理器 ColQwen2Processor
            print("✅ Full processor loaded successfully.")
            
            # Step 4: Load the main model
            print("⏳ Loading main model...")
            self.embed_model = ColQwen2.from_pretrained(
                model,
                torch_dtype=torch.bfloat16,
                device_map="cuda",
                # processor=self.processor, # <-- THIS LINE IS REMOVED
            ).eval()
            print("✅ Model loaded successfully.")
            
        elif "vidore" in model and "pali" in model:
            self.embed_model = ColPali.from_pretrained(
                model,
                torch_dtype=torch.bfloat16,
                device_map="cuda",
            ).eval()
            self.processor = ColPaliProcessor.from_pretrained(model)

    # ... (The rest of the file can remain the same) ...

    @classmethod # 	这是 Python 的 类方法装饰器。被它修饰的函数，第一个参数不是 self（实例对象），而是 cls（当前类本身）。这样方法既可以通过类本身调用，也可以通过实例对象调用
    def class_name(cls) -> str:
        return "VL_Embedding"

    def embed_img(self, img_path): # 根据不同的底层模型（Vidore / OpenBMB），把图片路径输入 → 读取 → 预处理 → 喂进模型 → 输出 embedding 向量
        if isinstance(img_path, str):
            img_path = [img_path]
        if "vidore" in self.model:
            images = [Image.open(img).convert("RGB") for img in img_path]
            batch_images = self.processor.process_images(images).to(
                self.embed_model.device
            )
            with torch.no_grad():
                image_embeddings = self.embed_model(**batch_images)
        elif "openbmb" in self.model:
            images = [Image.open(img).convert("RGB") for img in img_path]
            inputs = {
                "text": [""] * len(images),
                "image": images,
                "tokenizer": self.tokenizer,
            }
            with torch.no_grad():
                outputs = self.embed_model(**inputs)
                attention_mask = outputs.attention_mask
                hidden = outputs.last_hidden_state
                reps = weighted_mean_pooling(hidden, attention_mask)
                image_embeddings = (
                    F.normalize(reps, p=2, dim=1).detach().cpu().numpy()
                )
        return image_embeddings

    def embed_text(self, text): # 处理文字嵌入向量
        if isinstance(text, str):
            text = [text]
        if "colqwen" in self.model:
            batch_queries = self.processor.process_queries(text).to(
                self.embed_model.device
            )
            with torch.no_grad():
                query_embeddings = self.embed_model(**batch_queries)
        elif "colpali" in self.model:
            batch_queries = self.processor.process_queries(text).to(
                self.embed_model.device
            )
            with torch.no_grad():
                query_embeddings = self.embed_model(**batch_queries)
        elif "openbmb" in self.model:
            INSTRUCTION = (
                "Represent this query for retrieving relevant documents: "
            )
            queries = [INSTRUCTION + query for query in text]
            inputs = {
                "text": queries,
                "image": [None] * len(queries),
                "tokenizer": self.tokenizer,
            }
            with torch.no_grad():
                outputs = self.embed_model(**inputs)
                attention_mask = outputs.attention_mask
                hidden = outputs.last_hidden_state
                reps = weighted_mean_pooling(hidden, attention_mask)
                query_embeddings = (
                    F.normalize(reps, p=2, dim=1).detach().cpu().tolist()
                )
        return query_embeddings 

    def _get_query_embedding(self, query: str) -> List[float]:
        """Get query embedding."""
        return self.embed_text(query)[0]

    def _get_text_embedding(self, text: str) -> List[float]:
        """Get text embedding."""
        return self.embed_text(text)[0]

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get text embeddings."""
        embeddings_list: List[List[float]] = []
        for text in texts:
            embeddings = self.embed_text(text)
            embeddings = embeddings[0]
            embeddings_list.append(embeddings)
        return embeddings_list

    def _aget_query_embedding(self, query: str) -> List[float]:
        """Get query embedding."""
        return self.embed_text(query)[0]

    def _aget_text_embedding(self, text: str) -> List[float]:
        """Get text embedding."""
        return self.embed_text(text)[0]

    def _get_image_embedding(self, img_file_path) -> Embedding:
        return self.embed_img(img_file_path)

    def _aget_image_embedding(self, img_file_path) -> Embedding:
        return self.embed_img(img_file_path)

    def __call__(self, nodes, **kwargs): # 可以让类被当作函数一样调用  
        if "vidore" in self.model:
            if self.mode == "image":
                embeddings = self.embed_img(
                    [node.metadata["file_path"] for node in nodes]
                )
                embeddings = embeddings.view(embeddings.size(0), -1).tolist()
            else:
                embeddings = self.embed_text([node.text for node in nodes])
                embeddings = embeddings.view(embeddings.size(0), -1).tolist()

            for node, embedding in zip(nodes, embeddings): # 遍历 nodes 和它们的 embedding。给每个 node 添加一个属性 .embedding → 存储该节点的向量。
                node.embedding = embedding 

        elif "openbmb" in self.model:
            if self.mode == "image":
                embeddings = self.embed_img(
                    [node.metadata["file_path"] for node in nodes]
                )
                embeddings = embeddings.tolist()
            else:
                embeddings = self.embed_text([node.text for node in nodes])

            for node, embedding in zip(nodes, embeddings):
                node.embedding = embedding

        return nodes

    def score(self, image_embeddings, text_embeddings): # 计算图像向量（image_embeddings）和文本向量（text_embeddings）之间的相似度/匹配分数
        if "vidore" in self.model:
            score = self.processor.score_multi_vector(
                image_embeddings, text_embeddings
            )
        elif "openbmb" in self.model:
            score = text_embeddings @ image_embeddings.T
        return score


if __name__ == "__main__":
    colpali = VL_Embedding("vidore/colqwen2-v1.0")
    image_embeddings = colpali.embed_img(
        "./data/ExampleDataset/img/00a76e3a9a36255616e2dc14a6eb5dde598b321f_1.jpg"
    )
    text_embeddings = colpali.embed_text("Hello, world!")
    score = colpali.processor.score_multi_vector(
        image_embeddings, text_embeddings
    )
    print(image_embeddings.shape)
    print(text_embeddings.shape)
    print(score)