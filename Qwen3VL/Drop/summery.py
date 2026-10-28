import re
from pathlib import Path

import pandas as pd

import torch 
import os

DATASET_ORDER = [
    "ImageNet-R",
    "ArxivQA",
    "VizWiz",
    "IconQA",
    "CLEVR-Math",
    "Flickr30k",
]
TASKS = [f"task{i}" for i in range(1, 7)]


def parse_result_text(result_file: Path):
    """Read accuracy value from Result.text.

    Returns float (e.g. 90.7) when found, otherwise None.
    """
    if not result_file.exists():
        return None

    text = result_file.read_text(encoding="utf-8", errors="ignore").strip()
    if not text:
        return None

    # Match classification format: Accuracy: 90.70%
    m = re.search(r"Accuracy\s*:\s*([0-9]+(?:\.[0-9]+)?)%", text)
    if m:
        return float(m.group(1))

    # Match captioning format (e.g. Flickr30k): Average: 55.93
    m = re.search(r"Average\s*:\s*([0-9]+(?:\.[0-9]+)?)", text)
    if m:
        return float(m.group(1))

    return None


def build_summary(base_dir: Path) -> pd.DataFrame:
    # rows: task1~task6, cols: dataset order user requested
    df = pd.DataFrame(index=TASKS, columns=DATASET_ORDER)

    for dataset in DATASET_ORDER:
        for i, task in enumerate(TASKS, start=1):
            result_file = base_dir / dataset / f"Drop-task{i}" / "Result.text"
            acc = parse_result_text(result_file)
            if acc is not None:
                df.loc[task, dataset] = acc

    return df


def main():
    project_root = Path(__file__).resolve().parent
    base_dir = project_root / "results" / "UCIT" / "each_dataset"

    summary_df = build_summary(base_dir)

    output_csv = base_dir / "Drop_UCIT_summary.csv"

    # Keep empty cells empty in excel/csv
    summary_df.to_csv(output_csv, index=True, index_label="Task")

    print(f"Saved: {output_csv}")

def compute():
    """Compute continual-learning metrics from Drop_UCIT_summary.csv.

    Metrics:
    - MFT: Mean Finetune Accuracy (mean diagonal)
    - MFN: Mean Final Accuracy (mean of last row over all datasets)
    - MAA: Mean Average Accuracy (mean over each step's average seen-task accuracy)
    - BWT: Backward Transfer (mean(final - diagonal) over old tasks)
    """
    project_root = Path(__file__).resolve().parent
    base_dir = project_root / "results" / "UCIT" / "each_dataset"
    summary_csv = base_dir / "Drop_UCIT_summary.csv"

    if not summary_csv.exists():
        raise FileNotFoundError(f"Summary file not found: {summary_csv}")

    # index: task1~task6, columns: DATASET_ORDER
    df = pd.read_csv(summary_csv, index_col=0)

    # Ensure numeric values (empty cells -> NaN)
    df = df.apply(pd.to_numeric, errors="coerce")

    num_tasks = min(len(TASKS), len(DATASET_ORDER), len(df.index), len(df.columns))

    # Build square task-dataset matrix by the predefined order
    mat = df.reindex(index=TASKS, columns=DATASET_ORDER).iloc[:num_tasks, :num_tasks]

    # Diagonal: performance right after finetuning each task
    diag_vals = [mat.iloc[i, i] for i in range(num_tasks)]
    diag_series = pd.Series(diag_vals, dtype="float64")

    # MFT: mean diagonal
    mft = diag_series.mean(skipna=True)

    # MFN: mean performance at final step (last row) on all tasks/datasets
    final_row = mat.iloc[num_tasks - 1, :]
    mfn = final_row.mean(skipna=True)

    # MAA: average over steps of the average seen-task accuracy at each step
    step_means = []
    for t in range(num_tasks):
        seen = mat.iloc[t, : t + 1]
        step_means.append(seen.mean(skipna=True))
    maa = pd.Series(step_means, dtype="float64").mean(skipna=True)

    # BWT: final performance minus immediate post-training performance (excluding last task)
    if num_tasks > 1:
        bwt_terms = []
        for j in range(num_tasks - 1):
            a_final_j = mat.iloc[num_tasks - 1, j]
            a_diag_j = mat.iloc[j, j]
            bwt_terms.append(a_final_j - a_diag_j)
        bwt = pd.Series(bwt_terms, dtype="float64").mean(skipna=True)
    else:
        bwt = float("nan")

    metrics = {
        "MFT": None if pd.isna(mft) else float(mft),
        "MFN": None if pd.isna(mfn) else float(mfn),
        "MAA": None if pd.isna(maa) else float(maa),
        "BWT": None if pd.isna(bwt) else float(bwt),
    }

    print("Computed metrics from:", summary_csv)
    for k, v in metrics.items():
        print(f"{k}: {v}")

    return metrics


def compute_from_csv(csv_path="/data/taosen/code/MCITlib/LLaVA/Drop/results/UCIT/each_dataset/Drop_UCIT_summary.csv"):
    """
    从指定的CSV文件读取持续学习结果并计算详细指标

    Args:
        csv_path: CSV文件路径，格式如下：
                 Task,ImageNet-R,ArxivQA,VizWiz,IconQA,CLEVR-Math,Flickr30k
                 task1,90.7,,,,
                 task2,90.27,94.47,,,
                 task3,88.97,94.67,62.91,,
                 ...

    Returns:
        包含各种指标的字典
    """
    import numpy as np

    # 读取文件内容并手动解析（因为格式混合了逗号和制表符）
    with open(csv_path, 'r') as f:
        lines = f.readlines()

    # 解析表头（逗号分隔）
    header = lines[0].strip().split(',')
    dataset_names = [h.strip() for h in header[1:]]  # 跳过"Task"列

    # 解析数据行（制表符分隔）
    data = []
    task_names = []
    for line in lines[1:]:
        parts = line.strip().split('\t')
        task_names.append(parts[0])
        # 将空字符串转为NaN
        row = []
        for val in parts[1:]:
            if val.strip() == '':
                row.append(np.nan)
            else:
                try:
                    row.append(float(val))
                except:
                    row.append(np.nan)
        data.append(row)

    # 创建DataFrame
    df = pd.DataFrame(data, index=task_names, columns=dataset_names)

    # 获取任务和数据集信息
    num_tasks = len(task_names)
    num_datasets = len(dataset_names)

    print("=" * 80)
    print(f"持续学习指标计算 - {csv_path}")
    print("=" * 80)
    print(f"任务数: {num_tasks}")
    print(f"数据集: {dataset_names}")
    print()

    # 构建性能矩阵 R[i,j] = 在学习完task i后，在dataset j上的性能
    R = df.values

    print("性能矩阵 R[i,j] (学完task i后在dataset j上的性能):")
    print(f"{'Task':<10}", end="")
    for dataset in dataset_names:
        print(f"{dataset:<15}", end="")
    print()
    for i, task in enumerate(task_names):
        print(f"{task:<10}", end="")
        for j in range(num_datasets):
            if not np.isnan(R[i, j]):
                print(f"{R[i, j]:<15.2f}", end="")
            else:
                print(f"{'N/A':<15}", end="")
        print()
    print()

    # 1. MFT: Mean Finetune Accuracy (对角线平均)
    diag_vals = [R[i, i] for i in range(min(num_tasks, num_datasets))]
    mft = np.nanmean(diag_vals)

    # 2. MFN: Mean Final Accuracy (最后一行平均)
    final_row = R[-1, :]
    mfn = np.nanmean(final_row)

    # 3. MAA: Mean Average Accuracy (每步的平均准确率的平均)
    step_means = []
    for t in range(num_tasks):
        seen = R[t, :t+1]
        step_means.append(np.nanmean(seen))
    maa = np.nanmean(step_means)

    # 4. BWT: Backward Transfer (后向迁移)
    if num_tasks > 1:
        bwt_terms = []
        for j in range(num_tasks - 1):
            final_perf = R[-1, j]
            initial_perf = R[j, j]
            if not np.isnan(final_perf) and not np.isnan(initial_perf):
                bwt_terms.append(final_perf - initial_perf)
        bwt = np.mean(bwt_terms) if bwt_terms else np.nan
    else:
        bwt = np.nan

    # 5. Forgetting: 平均遗忘率
    forgetting_per_task = []
    for j in range(num_datasets - 1):  # 不包括最后一个任务
        if j < num_tasks:
            # 找到该任务在所有后续时刻的最大性能
            max_perf = np.nanmax(R[j:, j])
            final_perf = R[-1, j]
            if not np.isnan(max_perf) and not np.isnan(final_perf):
                forgetting = max_perf - final_perf
                forgetting_per_task.append(forgetting)

    avg_forgetting = np.mean(forgetting_per_task) if forgetting_per_task else 0.0

    # 6. 每个任务的最终准确率
    task_accuracies = {}
    for j, dataset in enumerate(dataset_names):
        final_acc = R[-1, j]
        if not np.isnan(final_acc):
            task_accuracies[dataset] = final_acc

    # 打印结果
    print("=" * 80)
    print("指标计算结果:")
    print("=" * 80)
    print(f"1. MFT (Mean Finetune Accuracy):         {mft:.2f}%")
    print(f"2. MFN (Mean Final Accuracy):            {mfn:.2f}%")
    print(f"3. MAA (Mean Average Accuracy):          {maa:.2f}%")
    print(f"4. BWT (Backward Transfer):              {bwt:+.2f}%")
    print(f"5. Forgetting (Average Forgetting):      {avg_forgetting:.2f}%")
    print()

    print("每个任务的最终准确率:")
    for dataset, acc in task_accuracies.items():
        print(f"  {dataset:<15}: {acc:.2f}%")
    print()

    if forgetting_per_task:
        print("每个任务的遗忘情况:")
        for j, dataset in enumerate(dataset_names[:len(forgetting_per_task)]):
            print(f"  {dataset:<15}: {forgetting_per_task[j]:+.2f}%")
        print()

    print("=" * 80)

    return {
        'MFT': mft,
        'MFN': mfn,
        'MAA': maa,
        'BWT': bwt,
        'Forgetting': avg_forgetting,
        'task_accuracies': task_accuracies,
        'forgetting_per_task': dict(zip(dataset_names[:len(forgetting_per_task)], forgetting_per_task)),
        'performance_matrix': R,
    }


def drop_pruner_summary(base_dir="/data/taosen/code/MCITlib/checkpoint/UCIT/LLaVA-1.5/Drop", total_tasks=6):
    print("=" * 80)
    print(f"LoRA-Drop: 细粒度具体共享层统计 (Task 1 to {total_tasks})")
    print("=" * 80)
    
    summary_results = {}
    target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
    
    # 检查 Task 1 是否有剪枝文件
    task1_file = os.path.join(base_dir, "Task1_llava_lora", "lora_drop_pruner.pt")
    if os.path.exists(task1_file):
        try:
            state = torch.load(task1_file, map_location="cpu", weights_only=False)
            if 'layer_lora_type' in state:
                layer_types = state['layer_lora_type']
                total_modules = len(layer_types)
                independent_count = sum(1 for t in layer_types.values() if t == 'independent')
                dropped_count = total_modules - independent_count
                reduction_ratio = dropped_count / total_modules if total_modules > 0 else 0.0

                print(f"Task 1 | 总体剪枝/共享模块数: {dropped_count:3d}/{total_modules} | 缩减率: {reduction_ratio:5.2%}")

                # 打印具体的共享层级详情
                module_total_counts = {mod: 0 for mod in target_modules}
                shared_layer_ids = {mod: [] for mod in target_modules}

                for module_name, lora_type in layer_types.items():
                    for target in target_modules:
                        if target in str(module_name):
                            module_total_counts[target] += 1
                            if lora_type != 'independent':
                                match = re.search(r'layers\.(\d+)', str(module_name))
                                if match:
                                    shared_layer_ids[target].append(int(match.group(1)))
                            break

                for target in target_modules:
                    total = module_total_counts[target]
                    if total > 0:
                        shared_ids = shared_layer_ids[target]
                        shared_count = len(shared_ids)
                        if shared_count > 0:
                            sorted_ids = sorted(list(set(shared_ids)))
                            ids_str = ", ".join([str(idx) for idx in sorted_ids])
                            print(f"  -> [{target:<9}] 共享: {shared_count:2d}/{total:2d} | 发生共享的 Layer ID: [{ids_str}]")
                        else:
                            print(f"  -> [{target:<9}] 共享:  0/{total:2d} | (全部保持独立)")
                print("-" * 80)
            else:
                print("Task 1 | [Error] 字典结构异常\n")
                print("-" * 80)
        except Exception as e:
            print(f"Task 1 | [Error] 读取时出错: {e}\n")
            print("-" * 80)
    else:
        print("Task 1 | 初始基石任务，未找到剪枝文件 (预期行为)\n")
        print("-" * 80)

    for task_id in range(2, total_tasks + 1):
        file_path = os.path.join(base_dir, f"Task{task_id}_llava_lora", "lora_drop_pruner.pt")
        
        if not os.path.exists(file_path):
            print(f"[Warning] 找不到 Task {task_id} 的剪枝文件: {file_path}")
            continue
            
        try:
            state = torch.load(file_path, map_location="cpu", weights_only=False)
            
            if 'layer_lora_type' in state:
                layer_types = state['layer_lora_type']
                total_modules = len(layer_types)
                independent_count = 0
                
                # 用于记录每个模块的具体被共享层号
                module_total_counts = {mod: 0 for mod in target_modules}
                shared_layer_ids = {mod: [] for mod in target_modules}
                
                for module_name, lora_type in layer_types.items():
                    if lora_type == 'independent':
                        independent_count += 1
                        
                    for target in target_modules:
                        if target in str(module_name):
                            module_total_counts[target] += 1
                            
                            # 如果该模块被共享/剪枝
                            if lora_type != 'independent':
                                # 使用正则提取 "layers.数字" 中的数字
                                match = re.search(r'layers\.(\d+)', str(module_name))
                                if match:
                                    layer_id = int(match.group(1))
                                    shared_layer_ids[target].append(layer_id)
                                else:
                                    # 如果正则没匹配到标准层名，保留最后一段作为标识
                                    shared_layer_ids[target].append(str(module_name).split('.')[-1])
                            break 
                
                dropped_count = total_modules - independent_count
                reduction_ratio = dropped_count / total_modules if total_modules > 0 else 0.0
                
                print(f"Task {task_id} | 总体剪枝/共享模块数: {dropped_count:3d}/{total_modules} | 缩减率: {reduction_ratio:5.2%}")
                
                # 打印具体的共享层级详情
                for target in target_modules:
                    total = module_total_counts[target]
                    if total > 0:
                        shared_ids = shared_layer_ids[target]
                        shared_count = len(shared_ids)
                        
                        # 仅当有模块被共享时，才打印具体的 Layer ID
                        if shared_count > 0:
                            # 确保按层号顺序从小到大排列
                            try:
                                sorted_ids = sorted(list(set(shared_ids)))
                                ids_str = ", ".join([str(idx) for idx in sorted_ids])
                            except TypeError:
                                ids_str = ", ".join(list(set(shared_ids)))
                                
                            print(f"  -> [{target:<9}] 共享: {shared_count:2d}/{total:2d} | 发生共享的 Layer ID: [{ids_str}]")
                        else:
                            print(f"  -> [{target:<9}] 共享:  0/{total:2d} | (全部保持独立)")
                
                print("-" * 80)
                
            else:
                print(f"[Error] Task {task_id} 字典结构异常。\n")
                
        except Exception as e:
            print(f"[Error] 读取 Task {task_id} 时出错: {e}\n")


if __name__ == "__main__":
    # main() #得到汇总CSV表
    compute() #计算持续学习指标
    # drop_pruner_summary() # 剪枝结果详情

    # compute_from_csv()