import pandas as pd
import random

#%%
def validate_entry(entry,condition_groups,criteria=None):
    """验证entry并返回合并后的验证结果"""
    validation_results = {}
    
    
    
    # 合并验证规则（新增条件分组）
    # "Semantic-Map Alignment": 2,
    # "Semantic-Map Quality": 2,
    # "Sketch Alignment": 5,
    # "Sketch Quality": 3,
    # "Canny Alignment": 5,
    # "Canny Quality": 5,
    # "Bounding-Box Accuracy": 5,
    # "Subject Alignment": 5,
    # "Depth Alignment": 5,
    # "Depth Quality": 3,
    # "Mask Alignment": 5,
    # "Caption Alignment": 5

    
    condition_include = {
        'semantic_map': ['Semantic-Map Quality', 'Semantic-Map Alignment'],
        'sketch': ['Sketch Quality', 'Sketch Alignment'],
        'canny': ['Canny Quality', 'Canny Alignment'],
        'mask': ['Mask Alignment'],
        'caption': ['Caption Alignment'],
        'bbox': ['Mask Alignment'],
        # 'subject': ['Subject Alignment'],
        'subject': [],
        # 'depth': ['Depth Quality', 'Depth Alignment'],
        'depth': [],
        # 'pose': ['Pose Alignment']
        # 'style':['Style Alignment']
    }
    condition_groups = {key: value for key, value in condition_include.items() if key in condition_groups}
    print(f"condition_groups: {condition_groups}")
    # if criteria is None:
    #     criteria = {
    #         'Sketch Quality': ['High', 'Average'],
    #         'Sketch Alignment': ['Align','Partially Align'],
    #         'Canny Quality': ['High', 'Average'],
    #         'Mask Alignment': ['Align', 'Partially Align'],
    #         'Semantic-Map Quality': ['High', 'Average'],
    #         'Semantic-Map Alignment': ['Align','Partially Align'],
    #         'Caption Alignment': ['Align','Partially Align'],
    #     }
    
    judge_dict = entry.get('judge', {})
    # 按分组进行验证
    for condition, judge_keys in condition_groups.items():
        # 只有当所有judge key都满足条件时，才返回True
        validation_results[condition] = all(
            judge_dict[judge_key] in criteria.get(judge_key,'')
            for judge_key in judge_keys
        )
        

    return validation_results



#%%


def select_random_compatible_conditions(valid_condition_list, n_conditions=3, matrix_file="condition-conpa.csv"):
    """
    Select a random subset of compatible conditions from the given list.
    
    Args:
        valid_condition_list (list): List of valid condition names
        n_conditions (int): Number of conditions to select
        matrix_file (str): Path to the compatibility matrix CSV
        
    Returns:
        list: A list of randomly selected compatible conditions, or empty list if no compatible set found
    """
    # Load compatibility matrix
    try:
        compatibility_matrix = pd.read_csv(matrix_file, index_col=0)
    except Exception as e:
        print(f"Error loading compatibility matrix: {e}")
        return []
    
    # 检查valid_condition_list中的所有条件是否都在compatibility_matrix中
    missing_conditions = [cond for cond in valid_condition_list if cond not in compatibility_matrix.index]
    if missing_conditions:
        print(f"Warning: The following conditions in valid_condition_list are not present in the compatibility matrix: {missing_conditions}")
        return []
    
    # Validate conditions
    valid_conditions = [cond for cond in valid_condition_list if cond in compatibility_matrix.index]
    if len(valid_conditions) < n_conditions:
        print(f"Warning: Not enough valid conditions. Requested {n_conditions}, but only {len(valid_conditions)} valid.")
        n_conditions = min(n_conditions, len(valid_conditions))
    
    if not valid_conditions:
        return []
    
    # Function to check compatibility of a set of conditions
    def is_compatible_with_selection(new_cond, current_selection):
        required = []
        # 收集所有需要添加的依赖条件
        for cond in current_selection:
            # 新条件对现有条件的依赖
            if compatibility_matrix.loc[new_cond, cond] == '_':
                required.append(cond)
            # 现有条件对新条件的依赖
            if compatibility_matrix.loc[cond, new_cond] == '_':
                required.append(new_cond)
        
        # 检查所有依赖条件是否都已存在
        missing = [cond for cond in required if cond not in current_selection]
        if missing:
            return False
        
        # 检查双向兼容性
        for cond in current_selection:
            if compatibility_matrix.loc[new_cond, cond] == 'x' or compatibility_matrix.loc[cond, new_cond] == 'x':
                return False
        return True

    # 新增回溯函数
    def build_compatible_set(current_selection, remaining_conditions, target_size):
        if len(current_selection) == target_size:
            return current_selection
        if not remaining_conditions:
            return None
            
        random.shuffle(remaining_conditions)
        for cond in remaining_conditions:
            # 自动处理依赖条件
            dependencies = []
            for other_cond in compatibility_matrix.index:
                if compatibility_matrix.loc[cond, other_cond] == '-' and other_cond in valid_conditions:
                    dependencies.append(other_cond)
            
            # 检查依赖条件是否满足
            if all(dep in current_selection or dep in remaining_conditions for dep in dependencies):
                new_selection = current_selection.copy()
                new_remaining = remaining_conditions.copy()
                
                # 添加依赖条件
                for dep in dependencies:
                    if dep not in new_selection and dep in new_remaining:
                        new_selection.append(dep)
                        new_remaining.remove(dep)
                
                # 添加当前条件
                if cond in new_remaining:
                    new_selection.append(cond)
                    new_remaining.remove(cond)
                
                # 检查总数限制
                if len(new_selection) > target_size:
                    continue
                    
                if is_compatible_with_selection(cond, new_selection):
                    result = build_compatible_set(new_selection, new_remaining, target_size)
                    if result:
                        return result
        return None

    # 修改主循环逻辑，允许返回最大可能集合
    # 替换原有的随机尝试逻辑
    valid_conditions = sorted(valid_conditions, key=lambda x: random.random())  # 更高效的随机排序
    
    best_result = []
    # 尝试不同起始条件
    for start_cond in valid_conditions:
        remaining = [c for c in valid_conditions if c != start_cond]
        # 尝试构建最大可能集合
        result = build_compatible_set([start_cond], remaining, n_conditions)
        if result and len(result) == n_conditions:
            return result
        # 记录找到的最大集合
        if result and len(result) > len(best_result):
            best_result = result
    
    # 如果找不到目标数量，返回找到的最大集合（如果允许部分返回）
    # 或者 return best_result if best_result else None
    return best_result if best_result else []

# Example usage
if __name__ == "__main__":
    # Example conditions
    example_conditions = ['sketch', 'canny', 'mask', 'caption']
    
    # 初始化条件出现次数字典
    condition_counts = {cond: 0 for cond in example_conditions}
    nx=10000
    # 生成1000次随机选择
    for _ in range(nx):
        # Select 5 random compatible conditions
        selected = select_random_compatible_conditions(example_conditions, n_conditions=2,matrix_file="/media/sata4/Contextaware/image2condition_model/conditions/condition-conpa.csv")
        # 更新条件出现次数
        for cond in selected:
            condition_counts[cond] += 1
    
    # 打印条件出现的比例
    for cond, count in condition_counts.items():
        print(f"{cond}: {count / nx:.2%}")