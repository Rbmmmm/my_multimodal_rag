# File: evaluate_from_results.py (V2 - 支持断点续评的最终版)
# 作用: 读取生成好的结果文件，进行自动化评分和指标计算。
# 特点: 边评边存，如果中断可以从断点处继续，不会丢失已完成的评分。

import json
import os
from tqdm import tqdm

# --- 导入您的评估工具 ---
from src.llms.evaluator import Evaluator
from src.utils.overall_evaluator import eval_search, eval_search_type_wise

# --- 配置区 ---
CONFIG = {
    "GENERATION_RESULTS_FILE": "generation_results/run_output.jsonl",
    "OUTPUT_DIR": "eval_results/final_summary" 
}

def main():
    print("="*30)
    print("步骤 1: 初始化评估器 (Evaluator)")
    evaluator = Evaluator()
    print("✅ 评估器初始化成功.")
    print("="*30)

    # --- 加载已生成的结果 ---
    print(f"\n步骤 2: 从 {CONFIG['GENERATION_RESULTS_FILE']} 加载结果...")
    generated_results = []
    with open(CONFIG['GENERATION_RESULTS_FILE'], 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                if "error" not in data:
                    generated_results.append(data)
            except json.JSONDecodeError:
                continue
    print(f"✅ 成功加载 {len(generated_results)} 条待评分结果。")

    # --- 断点续评逻辑 ---
    output_dir = CONFIG["OUTPUT_DIR"]
    os.makedirs(output_dir, exist_ok=True)
    details_path = os.path.join(output_dir, "evaluated_details.jsonl")
    
    uids_already_evaluated = set()
    if os.path.exists(details_path):
        print(f"ℹ️  发现已存在的评估明细文件，将进行断点续评...")
        with open(details_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    uids_already_evaluated.add(json.loads(line)['uid'])
                except (json.JSONDecodeError, KeyError):
                    continue
        print(f"✅ 已评估完成 {len(uids_already_evaluated)} 个样本，将跳过它们。")

    # --- 逐条进行答案评分 ---
    print("\n步骤 3: 开始对每个答案进行评分 (LLM-as-a-Judge)...")
    
    # 我们仍然需要一个列表来收集本次运行的结果，以便最后计算总分
    all_evaluated_results = [] 
    
    for result in tqdm(generated_results, desc="正在评分"):
        uid = result.get('uid')
        # 跳过已评估的样本
        if uid and uid in uids_already_evaluated:
            continue

        eval_result = evaluator.evaluate(
            query=result["query"],
            reference_answer=result["reference_answer"],
            generated_answer=result["generated_answer"]
        )
        
        result['eval_result'] = eval_result
        result['recall_results'] = { "source_nodes": result["retrieved_nodes"] }
        
        # 将本次新评估的结果加入列表
        all_evaluated_results.append(result)
        
        # 【核心改动】每评完一个，立即以追加模式('a')写入文件
        with open(details_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    # --- 加载包含所有（历史+本次）结果的文件，以进行最终统计 ---
    print("\n✅ 所有新样本评分完成。正在加载完整数据进行统计...")
    full_results_for_summary = []
    with open(details_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                full_results_for_summary.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    # --- 进行最终的汇总统计 ---
    print("\n步骤 4: 计算检索指标和平均分...")
    if full_results_for_summary:
        overall_summary = eval_search(full_results_for_summary)
        type_wise_summary = eval_search_type_wise(full_results_for_summary)
        
        print("\n[整体评估摘要]:")
        print(json.dumps(overall_summary, indent=4, ensure_ascii=False))

        # --- 保存最终的报告 ---
        print("\n步骤 5: 保存最终评估报告...")
        summary_path = os.path.join(output_dir, "summary_overall.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(overall_summary, f, indent=4, ensure_ascii=False)
        print(f"✅ 整体摘要已保存至: {summary_path}")

        type_wise_path = os.path.join(output_dir, "summary_type_wise.json")
        with open(type_wise_path, "w", encoding="utf-8") as f:
            json.dump(type_wise_summary, f, indent=4, ensure_ascii=False)
        print(f"✅ 分类别摘要已保存至: {type_wise_path}")
    else:
        print("ℹ️  没有可用于汇总统计的结果。")


if __name__ == '__main__':
    main()