import os
import json
import tempfile
from diversity_enhance import *
import random
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from conditions.select_new_condition import select_random_compatible_conditions,validate_entry


SELECT_INPUT_FILE_PATH = " "
ENHANCED_PROMPTS_OUTPUT_FILE_PATH = ' '


# define path
PERSONA_FILE_PATH = './data/persona.json'
CONDITION_MAPPING_FILE_PATH = './data/condition_mapping.json'
CONDITION_CONPA_FILE_PATH = './conditions/condition-conpa.csv'

weights = [0.3,0.7]  
N_CONDITIONS=[2,3]

criteria = {
    'Sketch Quality': [3,4,5],
    'Sketch Alignment': [3,4,5],
    'Canny Quality': [3,4,5],
    'Canny Alignment': [3,4,5],
    'Mask Alignment':  [3,4,5],
    'Semantic-Map Quality': [3,4,5],
    'Semantic-Map Alignment': [3,4,5],
    # "Style Alignment": [3,4,5],
    # 'Bounding-Box Accuracy': [3,4,5],
    'Subject Alignment': [3,4,5],
    # 'Depth Alignment': [3,4,5],
    # 'Depth Quality': [3,4,5],
    'Caption Alignment': [3,4,5],
    'Pose Alignment': [3,4,5]
}
def load_config():
    with open('./config.json', 'r') as f:
        config = json.load(f)
    return config
CONFIG= load_config()

unique_combinations = set()

def match_condition(a:str, b:str):
    # Previous implementation remains the same
    if a.endswith("_path"):
        a = a[:-5]
    elif a.endswith("_paths"):
        a = a[:-6]
    if b.endswith("_path"):
        b = b[:-5]
    elif b.endswith("_paths"):
        b = b[:-6]
    return a == b   

def generate_basic_instruction(condition_list, conditions_dict, condition_mapping, start_condition=None):
    # Previous implementation remains the same
    image_mapping = {}

    def get_random_path(paths):
        if isinstance(paths, list) and paths:
            print(f"path_value: {paths}")
            return random.choice(paths)
        return paths 

    condition_texts = []

    for condition in condition_list:
        if condition == 'caption':
            continue
        
        if condition in condition_mapping:
            
            condition_info = condition_mapping[condition]
            print(condition_info)
            chosen_template = random.choice(condition_info['template'])
            condition_texts.append(chosen_template)

            path_key = condition_info.get('path_key')
            if path_key and path_key in conditions_dict:
                path_value = get_random_path(conditions_dict[path_key])
                if path_value:
                    image_mapping[condition_info['image_fill_key']] = path_value
                else:
                    warnings.warn(f"No valid path found for condition '{condition}'")

    if 'caption' in condition_list:
        caption_text = conditions_dict.get('caption', '<caption>')
        condition_texts.append(f"following the caption: {caption_text}.")
    
    if not start_condition:
        start_condition = "Generate an image"

    final_text = f"{start_condition} {' '.join(condition_texts)}"

    return {
        "text": final_text,
        "images": image_mapping
    }

def parallel_save_incremental_prompts(entry_list, num_responses, output_file_path, max_workers=None,manual_valid_conditions=None):
    """
    Parallel processing with incremental saving
    
    :param entry_list: List of image entries to process
    :param num_responses: Number of responses to generate per entry
    :param output_file_path: Path to save the output JSON file
    :param max_workers: Maximum number of worker threads (None uses system default)
    """
    # Load necessary files
    with open(PERSONA_FILE_PATH, 'r') as f:
        persona = json.load(f)

    with open(CONDITION_MAPPING_FILE_PATH, 'r') as f:
        condition_mapping_dict = json.load(f)

    def process_single_entry(entry):
        """Process a single entry"""
        try:
            conditions_dict = {k.replace('single_mask', 'mask'): entry.get(k, "" if not k.endswith('paths') else []) 
                              for k in entry.keys() 
                              if (k.endswith('path') or k.endswith('paths') or k == 'caption') and not k.startswith('original_')}
            # Initialize dictionary for the current image entry
            image_all = {
                "original_image_path": entry.get("original_image_path", ""),
                "conditions": conditions_dict,
                "instructions": [],
                "judge": entry.get("judge", {}),
            }
            
            condition_groups = [
                key[:-5] if key.endswith("_path") else 
                key[:-6] if key.endswith("_paths") else 
                key 
                for key in conditions_dict.keys()
            ]
            print(f"condition_groups: {condition_groups}")
            # Validate conditions
            condition_validation = validate_entry(entry,condition_groups,criteria)
            
            valid_conditions = {cond for cond, valid in condition_validation.items() if valid}
            #valid_conditions= set(condition_groups)
            
            #caption+subject
            #caption+style+semantic_map+sketch+depth
            #manual_valid_conditions={'style','depth','semantic_map','sketch'}
            
            # 合并外部传入的manual_valid_conditions
            if manual_valid_conditions:
                valid_conditions = valid_conditions.union(manual_valid_conditions)
            # valid_conditions={'pose','style','caption'}
            # 支持手动添加条件
            # 原有的随机选择条件逻辑
            # Select random compatible conditions with retry and fallback
            max_retries = 5
            retry_count = 0
            selected_conditions = []
            # 添加权重选择
            n_conditions = random.choices(N_CONDITIONS, weights=weights, k=1)[0]
            
            print(f"valid_conditions: {valid_conditions}")
            
            # 添加最大总尝试次数限制，防止死循环
            max_total_attempts = max_retries * 5  # 最多尝试3轮
            total_attempts = 0
            
            while retry_count < max_retries and total_attempts < max_total_attempts:
                selected_conditions = select_random_compatible_conditions(
                    valid_condition_list=valid_conditions,
                    n_conditions=n_conditions,
                    matrix_file=CONDITION_CONPA_FILE_PATH
                )
                
                # Check if enough conditions are available
                if len(selected_conditions) >= n_conditions:
                    # Convert to tuple (immutable) and add to unique_combinations
                    print(f"selected_conditions: {selected_conditions}")
                    unique_combinations.add(tuple(sorted(selected_conditions)))
                    break
                    
                retry_count += 1
                total_attempts += 1
                
                if retry_count < max_retries:
                    pass
                    print(f"Retrying condition selection (attempt {retry_count + 1}/{max_retries})")
                else:
                    # After 3 failed attempts, reduce n_conditions by 1
                    n_conditions = max(2, n_conditions - 1)
                    retry_count = 0  # Reset retry count for new n_conditions
                    print(f"Reducing n_conditions to {n_conditions} after {max_retries} failed attempts")
            
            # 如果达到最大总尝试次数仍未成功，直接返回
            if total_attempts >= max_total_attempts:
                print(f"Failed to select conditions after {max_total_attempts} total attempts")
                return None
            
            
            # Generate instructions and enhance prompts
            basic_instructions = [
                generate_basic_instruction(selected_conditions, conditions_dict, condition_mapping_dict)
                for _ in range(num_responses)
            ]
            
            # Parallel generation of enhance prompts
            enhance_prompts = []
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(
                        get_response_with_retry,
                        instruction['text'],
                        persona,
                        max_retries=3,
                        config=CONFIG
                    )
                    for instruction in basic_instructions
                ]
                
                for future in as_completed(futures):
                    try:
                        enhance_prompts.append(future.result())
                    except Exception as e:
                        print(f"Error generating enhance prompt: {e}")
            
            # Construct instructions
            image_all['instructions'] = {
                "conditions": selected_conditions,
                "image_path_mapping": basic_instructions[0]['images'],
                'original_prompts': basic_instructions[0]["text"],
                "enhance_prompts": enhance_prompts
            }
            
            return image_all
        
        except Exception as e:
            print(f"Error processing entry: {e}")
            return None

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

    # Check if temporary file exists (for resuming)
    temp_output_file = output_file_path + '.tmp'
    if os.path.exists(temp_output_file):
        with open(temp_output_file, 'r') as f:
            existing_results = json.load(f)
        # Create a set of processed original_image_paths
        processed_paths = {result['original_image_path'] for result in existing_results}
        # Filter out already processed entries
        entry_list = [entry for entry in entry_list if entry.get('original_image_path') not in processed_paths]
    else:
        existing_results = []
        # Initialize the output file
        with open(temp_output_file, 'w') as f:
            json.dump([], f)

    # Track processed entries
    processed_count = 0
    total_entries = len(entry_list)

    # Use ThreadPoolExecutor for parallel processing of entries
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_single_entry, entry): entry for entry in entry_list}
        
        for future in as_completed(futures):
            result = future.result()
            
            if result is not None:
                # Append to the temporary file
                with open(temp_output_file, 'r+') as f:
                    # Read existing content
                    f.seek(0)
                    existing_data = json.load(f)
                    
                    # Append new result
                    existing_data.append(result)
                    
                    # Move file pointer to the beginning and write updated content
                    f.seek(0)
                    json.dump(existing_data, f, indent=4)
                    f.truncate()
                
                processed_count += 1
                print(f"Processed {processed_count}/{total_entries} entries")

    # Rename temporary file to final output file
    os.replace(temp_output_file, output_file_path)
    
    print(f"Processing complete. Total processed entries: {processed_count}")
    print(f"Results saved to {output_file_path}")

# Main execution
def main():
    # Read input file
    with open(SELECT_INPUT_FILE_PATH, 'r') as f:
        entry_list = json.load(f)[:]

    # Generate prompts in parallel with incremental saving
    parallel_save_incremental_prompts(
        entry_list=entry_list, 
        num_responses=1,
        output_file_path=ENHANCED_PROMPTS_OUTPUT_FILE_PATH,
        max_workers=20,  # Use system default worker count
    )

# Add this at the end of the file to print statistics
def print_unique_combinations():
    print(f"Total unique combinations: {len(unique_combinations)}")
    print("Unique combinations:")
    for combination in unique_combinations:
        print(combination)

if __name__ == "__main__":
    main()