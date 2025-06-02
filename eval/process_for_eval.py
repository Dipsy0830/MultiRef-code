import json
import importlib
import sys
import os
os.environ["http_proxy"] = "http://127.0.0.1:7890"
os.environ["https_proxy"] = "http://127.0.0.1:7890"

from regex import F

# 添加当前目录到路径以确保导入正确
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from eval.eval_condition import ImageConditionEvaluator  # 已经通过类实现评估函数
import time
import traceback
import concurrent.futures
import json
import os
import time
from typing import List, Dict, Union, Any
from tqdm import tqdm
import traceback
import concurrent.futures
import threading

import json
import os
import time
from typing import List, Dict, Union, Any
import tqdm



def process_entry(entry: Dict[str, Any], evaluator):
    """
    Process a single JSON entry and compute evaluations for it
    
    Parameters:
    -----------
    entry : Dict[str, Any]
        Single JSON entry with specific structure
    evaluator : ImageConditionEvaluator
        Evaluator instance
        
    Returns:
    --------
    Dict[str, Any]
        Computed evaluation metrics
    """
    try:
        # Get the generated image path from results field
        if "results" in entry and isinstance(entry["results"], dict):
            gen_image_path = entry["results"].get("image_path", "")
        else:
            return {"error": f"Invalid results field: {entry.get('results')}"}
            
        if not gen_image_path or not os.path.exists(gen_image_path):
            return {"error": f"Generated image not found: {gen_image_path}"}
        
        # Get original image path
        original_image_path = entry.get("original_image_path", "")
        if not original_image_path or not os.path.exists(original_image_path):
            return {"error": f"Original image not found: {original_image_path}"}
            
        condition_types = []
        if "instructions" in entry and isinstance(entry["instructions"], dict):
            if "conditions" in entry["instructions"]:
                condition_types = entry["instructions"]["conditions"]
                print(f"Found condition types to evaluate: {condition_types}")

        if not condition_types:
            return {"warning": "No condition types specified in instructions.conditions"}

        # Build reference paths dictionary
        reference_paths = {}
        
        # Get conditions from the entry
        if "conditions" in entry and isinstance(entry["conditions"], dict):
            conditions = entry["conditions"]
            
            # Add paths from conditions dictionary based on condition_types
            for condition in condition_types:
                if condition == "caption" and "caption" in conditions:
                    reference_paths["caption"] = conditions["caption"]
                elif condition == "depth" and "depth_path" in conditions and os.path.exists(conditions["depth_path"]):
                    reference_paths["depth"] = conditions["depth_path"]
                elif condition == "sketch" and "sketch_path" in conditions and os.path.exists(conditions["sketch_path"]):
                    reference_paths["sketch"] = conditions["sketch_path"]
                elif condition == "canny" and "canny_path" in conditions and os.path.exists(conditions["canny_path"]):
                    reference_paths["canny"] = conditions["canny_path"]
                elif condition == "style" and "style_path" in conditions:
                    style_path = conditions["style_path"]
                    if isinstance(style_path, list) and len(style_path) > 0:
                        reference_paths["style"] = style_path[0]
                    else:
                        reference_paths["style"] = style_path
                elif condition == "semantic_map" and "semantic_map_path" in conditions and os.path.exists(conditions["semantic_map_path"]):
                    reference_paths["semantic_map"] = conditions["semantic_map_path"]
                elif condition == "bbox" and "bbox_path" in conditions and os.path.exists(conditions["bbox_path"]):
                    reference_paths["bbox"] = conditions["bbox_path"]
                elif condition == "mask" and "mask_path" in conditions and os.path.exists(conditions["mask_path"]):
                    reference_paths["mask"] = conditions["mask_path"]
                elif condition == "pose" and "pose_path" in conditions and os.path.exists(conditions["pose_path"]):
                    reference_paths["pose"] = conditions["pose_path"]
                elif condition == "subject" and "subject_paths" in conditions:
                    subject_paths = conditions["subject_paths"]
                    if isinstance(subject_paths, list) and len(subject_paths) > 0:
                        # Take the first subject path for evaluation
                        reference_paths["subject"] = subject_paths[0]
                    elif isinstance(subject_paths, str) and os.path.exists(subject_paths):
                        reference_paths["subject"] = subject_paths
        
        # Run evaluations
        if not reference_paths:
            return {"warning": "No valid conditions found for evaluation"}
            
        #print(f"Running evaluations for {gen_image_path} with reference paths: {reference_paths}")
        results = evaluator.evaluate_all(gen_image_path, reference_paths,conditions_to_evaluate=['fid', 'aesthetics', 'mask', 'caption', 'sketch', 'subject', 'depth', 'canny', 'bbox', 'semantic_map', 'style',"pose"])
        
        # Add original image path to results
        results["original_image_path"] = original_image_path
        
        return results
    
    except Exception as e:
        import traceback
        traceback_str = traceback.format_exc()
        return {"error": f"Error processing entry: {str(e)}\n{traceback_str}"}


def batch_evaluation_parallel(json_data: Union[str, List, Dict], 
                             evaluator, 
                             output_file: str, 
                             save_interval: int = 1,
                             resume: bool = True,
                             max_workers: int = 4,
                             gpu_exclusive: bool = True) -> List:
    """
    Process multiple JSON entries in parallel, update each entry's evaluations field,
    and save results periodically.
    
    Parameters:
    -----------
    json_data : str or list or dict
        JSON string or list of entries or single entry dict
    evaluator : ImageConditionEvaluator
        Evaluator instance
    output_file : str
        Path to save the results
    save_interval : int
        Number of entries to process before saving
    resume : bool
        Whether to resume from previous run if output file exists
    max_workers : int
        Maximum number of parallel workers
    gpu_exclusive : bool
        If True, only one task at a time will use the GPU (recommended for most evaluations)
        
    Returns:
    --------
    list
        Updated JSON entries
    """
    # Parse input data
    if isinstance(json_data, str):
        try:
            entries = json.loads(json_data)
            if not isinstance(entries, list):
                entries = [entries]
        except json.JSONDecodeError:
            return [{"error": "Invalid JSON format"}]
    else:
        if isinstance(json_data, dict):
            entries = [json_data]
        else:
            entries = json_data
    
    # 创建一个列表来标记哪些条目需要处理
    # True表示需要处理，False表示已成功处理
    needs_processing = [True] * len(entries)
    
    # 尝试加载现有结果（如果resume=True）
    successfully_processed = 0
    
    def has_error_in_keys(evaluations_dict):
        """检查评估字典中的键是否包含'error'或'warning'"""
        for key in evaluations_dict.keys():
            if 'error' in key.lower() or 'warning' in key.lower():
                return True
        return False
    
    if resume and os.path.exists(output_file):
        try:
            with open(output_file, 'r') as f:
                existing_results = json.load(f)
                
            # 确保长度匹配
            if len(existing_results) > len(entries):
                existing_results = existing_results[:len(entries)]
            elif len(existing_results) < len(entries):
                # 保持现有结果，只处理新增的条目
                entries[:len(existing_results)] = existing_results
            else:
                # 长度相同，直接使用现有结果
                entries = existing_results
            
            # 确定哪些条目已成功处理，哪些需要重新处理
            for i in range(len(existing_results)):
                if "evaluations" in existing_results[i]:
                    evaluations = existing_results[i]["evaluations"]
                    # 从 instructions 中获取所需的评估键，若不存在则使用空列表
                    required_keys = []
                    if "instructions" in existing_results[i] and "conditions" in existing_results[i]["instructions"]:
                        required_keys = existing_results[i]["instructions"]["conditions"]
                    # 检查评估中是否有错误相关的键，或是否缺少 instructions 中指定的所有条件键
                    if has_error_in_keys(evaluations) or not all(key in evaluations for key in required_keys):
                        needs_processing[i] = True  # 需要重新处理
                    else:
                        needs_processing[i] = False  # 已成功处理
                        successfully_processed += 1
                else:
                    needs_processing[i] = True
            
            print(f"从上次运行恢复。{successfully_processed}个条目已成功处理，{sum(needs_processing[:len(existing_results)])}个条目需要重新处理。")
            
        except Exception as e:
            print(f"从上次运行恢复时出错: {str(e)}")
            traceback.print_exc()
            # 如果恢复失败，继续使用原始条目
    # 要处理的条目总数
    total = len(entries)
    to_process = sum(needs_processing)
    print(f"总条目数: {total}，需要处理的条目数: {to_process}")
    
    # 如果没有需要处理的条目，直接返回
    if to_process == 0:
        print("所有条目已处理完毕，无需进一步处理。")
        return entries
    
    # 创建GPU访问锁（如果需要）
    gpu_lock = threading.Lock() if gpu_exclusive else None
    
    # 创建文件写入锁
    file_lock = threading.Lock()
    
    # 用于跟踪进度
    completed = [0]
    progress_bar = tqdm.tqdm(total=to_process, desc="处理条目", unit="条目")
    
    # 上次保存结果的时间
    last_save_time = [time.time()]
    
    def process_entry_wrapper(idx):
        """处理一个条目并处理结果，包括保存和进度更新"""
        # 跳过不需要处理的条目
        if not needs_processing[idx]:
            return None
        
        entry = entries[idx]
        start_time = time.time()
        
        # 使用GPU锁（如果需要）
        if gpu_lock:
            with gpu_lock:
                computed_evaluations = process_entry(entry, evaluator)
        else:
            computed_evaluations = process_entry(entry, evaluator)
        
        # 一致地构建评估结果
        ordered_evaluations = {
            "original_image": entry.get('original_image_path', ''),
            "generated_image": entry.get('results', {}).get('image_path', ''),
            "processing_time_seconds": time.time() - start_time
        }
        ordered_evaluations.update(computed_evaluations)
        
        # 更新条目
        entry["evaluations"] = ordered_evaluations
        
        # 更新进度并在需要时保存
        with file_lock:
            completed[0] += 1
            progress_bar.update(1)
            
            # 定期保存
            current_time = time.time()
            should_save = (completed[0] % save_interval == 0 or 
                           completed[0] == to_process or 
                           current_time - last_save_time[0] > 300)  # 每5分钟保存一次
            
            if should_save:
                with open(output_file, 'w') as f:
                    json.dump(entries, f, indent=4)
                last_save_time[0] = current_time
                print(f"\n处理了{completed[0]}/{to_process}个条目后保存了结果")
        
        return idx
    
    # 并行处理条目
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交需要处理的任务
            futures = []
            
            # 按原始顺序提交任务
            for i in range(total):
                if needs_processing[i]:
                    futures.append(executor.submit(process_entry_wrapper, i))
            
            # 等待所有任务完成
            concurrent.futures.wait(futures)
    except Exception as e:
        print(f"线程池错误: {str(e)}")
        traceback.print_exc()
    finally:
        progress_bar.close()
        
        # 最终保存
        with open(output_file, 'w') as f:
            json.dump(entries, f, indent=4)
    
    return entries


def main():
    """
    Main function demonstrating incremental batch evaluation
    """
    import argparse
    
    # 添加命令行参数解析
    parser = argparse.ArgumentParser(description='Run image condition evaluation')
    #parser.add_argument('--gpu', type=int, default=1, help='GPU ID to use')
    parser.add_argument('--input', type=str, required=True, help='Input JSON file path')
    parser.add_argument('--output', type=str, required=True, help='Output JSON file path')
    parser.add_argument('--workers', type=int, default=10, help='Number of parallel workers')
    args = parser.parse_args()

    # 设置 GPU
    #os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    
    # 从文件中加载 JSON 数据
    with open(args.input, 'r') as f:
        json_data = json.load(f)[:]
    
    evaluator = ImageConditionEvaluator()
    
    # Run parallel batch evaluation with gpu_exclusive=False
    updated_json = batch_evaluation_parallel(
        json_data, 
        evaluator, 
        args.output, 
        save_interval=5,
        resume=True,
        max_workers=args.workers,
        gpu_exclusive=True  # 设置为False，因为模型加载已经有锁保护
    )
    
    print(f"评估完成。结果已保存至 {args.output}")


if __name__ == "__main__":
    main()
    
    
    
    
    


