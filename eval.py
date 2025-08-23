import os
import json
import torch
import argparse
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# ===============================================================
# 1. 导入您项目的组件和原始 eval 脚本的依赖
# ===============================================================
# 导入您的项目组件
from src.searcher.text_searcher import TextSearcher
from src.searcher.image_searcher import ImageSearcher
from src.searcher.table_searcher import TableSearcher
from src.searcher.SearchEngine import SearchEngine
from src.agent.seeker_agent import SeekerAgent
from src.agent.inspector_agent import InspectorAgent
from src.agent.synthesizer_agent import SynthesizerAgent
from src.models.gumbel_selector import GumbelModalSelector
from src.orchestrator import RAGOrchestrator
from src.llms.llm import LLM
from src.utils.embedding_utils import QueryEmbedder, get_query_embedding

# 导入原始 eval 脚本的依赖
from src.llms.evaluator import Evaluator # 假设您已将 ViDoRAG 的 evaluator.py 放在 llms/ 目录下
from src.utils.overall_evaluator import eval_search, eval_search_type_wise # 假设评估函数在 utils/ 目录下

from llama_index.core import Settings
Settings.llm = None


class MMRAG:
    """
    评估框架主类，结构和逻辑与您提供的 ViDoRAG eval.py 保持一致。
    仅修改了初始化和 RAG 调用部分以适配您的项目。
    """
    def __init__(self, args):
        # --- 基础配置 ---
        self.args = args
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dataset_dir = os.path.join('./data', args.dataset)
        self.results_dir = os.path.join(self.dataset_dir, "results_final") # 使用新目录以区分
        os.makedirs(self.results_dir, exist_ok=True)

        # ==========================================================================
        # 步骤 1: 初始化所有系统组件 (适配您的项目)
        # ==========================================================================
        print("="*30)
        print("步骤 1: 初始化所有系统组件")

        # --- 1a. 初始化模型 ---
        print("\n[1a] 初始化嵌入和 VLM 模型...")
        self.query_embedder = QueryEmbedder(model_name=args.embed_model_name, device=self.device)
        self.vlm = LLM(args.generate_vlm)
        self.evaluator = Evaluator() # 使用同一个 VLM 进行评估
        print("[1a] ✅ 模型初始化完成.")

        # --- 1b. 初始化 SearchEngine ---
        print("\n[1b] 初始化 SearchEngine...")
        retriever_factories = {
            "text": lambda: TextSearcher(dataset_name=args.dataset, mode='bi_encoder'),
            "image": lambda: ImageSearcher(dataset_name=args.dataset, mode='vl_search'),
            "table": lambda: TableSearcher(dataset_name=args.dataset, mode='vl_search')
        }
        self.search_engine = SearchEngine(retriever_factories=retriever_factories)
        print("[1b] ✅ SearchEngine 初始化完成.")

        # --- 1c. 初始化 Gumbel 选择器 ---
        print("\n[1c] 初始化 Gumbel 模态选择器...")
        gumbel_selector = GumbelModalSelector(
            input_dim=self.query_embedder.out_dim, num_choices=3, hidden_dim=256
        ).to(self.device)
        if os.path.exists(args.selector_ckpt):
            gumbel_selector.load_state_dict(torch.load(args.selector_ckpt, map_location=self.device))
            print(f"[1c] ✅ Gumbel Selector 权重从 {args.selector_ckpt} 加载成功.")
        else:
            print(f"[1c] ℹ️ 未找到 Gumbel Selector 权重，将使用随机初始化.")
        gumbel_selector.eval()

        # --- 1d. 初始化 Agents 和 Orchestrator ---
        print("\n[1d] 初始化 Agents 和 Orchestrator...")
        image_base_dir = f"data/{args.dataset}/img"
        seeker_agent = SeekerAgent(vlm=self.vlm, image_base_dir=image_base_dir)
        inspector_agent = InspectorAgent(vlm=self.vlm, image_base_dir=image_base_dir, reranker_model_name="BAAI/bge-reranker-large")
        synthesizer_agent = SynthesizerAgent(vlm=self.vlm, image_base_dir=image_base_dir)
        
        self.orchestrator = RAGOrchestrator(
            search_engine=self.search_engine, seeker=seeker_agent, inspector=inspector_agent,
            synthesizer=synthesizer_agent, gumbel_selector=gumbel_selector
        )
        print("[1d] ✅ Agents 和 Orchestrator 初始化完成.")
        
        # ==========================================================================
        # 步骤 2: 根据实验类型设置评估函数和输出文件名
        # ==========================================================================
        print("\n步骤 2: 配置实验类型")
        if args.experiment_type == 'retrieval_infer':
            self.eval_func = self.retrieval_infer
            self.output_file_name = f'retrieval_{args.embed_model_name}.jsonl'
        elif args.experiment_type == 'vidorag':
            self.eval_func = self.vidorag_infer
            self.output_file_name = f'vidorag_{args.generate_vlm}.jsonl'
        else:
            raise ValueError(f"不支持的实验类型: {args.experiment_type}")

        self.output_file_path = os.path.join(self.results_dir, self.output_file_name.replace("/", "-"))
        print(f"✅ 实验类型: '{args.experiment_type}' | 输出文件: {self.output_file_path}")
        print("="*30)

    def retrieval_infer(self, sample):
        query = sample['query']
        # 注意：这里的 search 逻辑可能需要根据您的 SearchEngine 实现进行微调
        # 这里假设 search 直接返回 LlamaIndex 的 NodeWithScore 列表
        retrieved_nodes = self.search_engine.search(query, modality='text', top_k=self.args.topk) # 默认用 text 检索
        
        # 将检索结果转换为可序列化的字典格式
        sample['recall_results'] = {
            "source_nodes": [node.to_dict() for node in retrieved_nodes]
        }
        return sample

    def vidorag_infer(self, sample):
        query = sample['query']
        print(f"\nProcessing query: {query}")
        try:
            query_embedding = get_query_embedding(self.query_embedder, query)
            answer = self.orchestrator.run(
                query=query,
                query_embedding=query_embedding,
                initial_top_k=self.args.topk
            )
            # 由于 Orchestrator 内部封装了检索，我们在这里无法直接获取召回的节点
            # 评估将主要关注最终答案的质量
            sample['recall_results'] = {"source_nodes": []} # 留空或填充伪数据
        except Exception as e:
            print(f"处理查询时发生错误: {e}")
            return None
        
        # 使用 Evaluator 评估答案
        sample['eval_result'] = self.evaluator.evaluate(query, sample['reference_answer'], str(answer))
        sample['response'] = answer
        return sample

    def eval_dataset(self):
        rag_dataset_path = os.path.join(self.dataset_dir, self.args.query_file)
        with open(rag_dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)['examples']
        
        # 断点续评逻辑
        if os.path.exists(self.output_file_path):
            print("ℹ️ 发现已存在的评估结果文件，将进行断点续评...")
            with open(self.output_file_path, "r", encoding="utf-8") as f:
                completed_uids = {json.loads(line)["uid"] for line in f}
            data = [item for item in data if item['uid'] not in completed_uids]
            print(f"✅ 已完成 {len(completed_uids)} 个样本，剩余 {len(data)} 个待评估。")

        # 单线程或多线程处理
        if self.args.workers_num == 1:
            for item in tqdm(data, desc="Processing Samples"):
                result = self.eval_func(item)
                if result:
                    with open(self.output_file_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(result, ensure_ascii=False) + "\n")
        else:
            with ThreadPoolExecutor(max_workers=self.args.workers_num) as executor:
                futures = [executor.submit(self.eval_func, item) for item in data]
                for future in tqdm(as_completed(futures), total=len(futures), desc="Processing Samples"):
                    result = future.result()
                    if result:
                        with open(self.output_file_path, "a", encoding="utf-8") as f:
                            f.write(json.dumps(result, ensure_ascii=False) + "\n")
    
    def eval_overall(self):
        print("\n📊 计算整体评估指标...")
        data = [json.loads(line) for line in open(self.output_file_path, "r", encoding="utf-8")]
        results = eval_search(data) # 依赖您的评估函数
        with open(self.output_file_path.replace(".jsonl", "_eval.json"), "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"✅ 整体评估指标已保存。")
    
    def eval_overall_type_wise(self):
        print("\n📊 计算分类别评估指标...")
        data = [json.loads(line) for line in open(self.output_file_path, "r", encoding="utf-8")]
        results = eval_search_type_wise(data) # 依赖您的评估函数
        with open(self.output_file_path.replace(".jsonl", "_eval_type_wise.json"), "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"✅ 分类别评估指标已保存。")

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='ViDoSeek', help="The name of dataset")
    parser.add_argument("--query_file", type=str, default='rag_dataset.json', help="The name of anno_file")
    parser.add_argument("--experiment_type", type=str, default='vidorag', help="The type of experiment ('vidorag' or 'retrieval_infer')")
    parser.add_argument("--embed_model_name", type=str, default='BAAI/bge-m3', help="The name of embedding model")
    parser.add_argument("--workers_num", type=int, default=1, help="The number of workers")
    parser.add_argument("--topk", type=int, default=10, help="The number of topk")
    parser.add_argument("--generate_vlm", type=str, default='qwen-vl-max', help="The name of VLM model")
    parser.add_argument("--selector_ckpt", type=str, default="checkpoints/modal_selector_best.pt", help="Path to Gumbel Selector checkpoint.")
    return parser.parse_args()

if __name__ == "__main__":
    args = arg_parse()
    mmrag = MMRAG(args)
    mmrag.eval_dataset()
    mmrag.eval_overall()
    mmrag.eval_overall_type_wise()