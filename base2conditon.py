import os
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any
from utils_tools import *
from condition_def import *
from conditions.GenCondition import GenCondition

class BaseImageProcessor:
    def __init__(self, base_dataset_path: str, base_save_path: str, 
                 image_dataset_path: str, add_base_name: bool, 
                 condition_list='all',replace=False):
        self.base_dataset_path = base_dataset_path
        self.base_save_path = base_save_path
        self.image_dataset_path = image_dataset_path  # 子类传入
        self.add_base_name = add_base_name           # 子类传入
        self.condition_list = condition_list         # 子类传入
        self.replace = replace
        # 初始化 GenCondition（所有子类通用）
        self.gen_condition = GenCondition(
            self.add_base_name,
            self.base_save_path
        )

    def filter_parallel_tasks(self, parallel_tasks: Dict[str, Any], existing_conditions: Dict[str, Any]) -> Dict[str, Any]:
        if self.condition_list == 'all':
            condition_list = list(parallel_tasks.keys())
        else:
            condition_list = self.condition_list

        # Filter tasks based on replace flag and check if the existing_conditions for the key is empty or None
        filtered_tasks = {
            k: v for k, v in parallel_tasks.items()
            if k in condition_list and (
                self.replace or k not in existing_conditions or not existing_conditions[k]
            )
        }
        return filtered_tasks

    def collect_image_files_from_dataset(self) -> List[Dict[str, Any]]:
        """
        Collects information about images in the dataset. 
        Child classes should override this if they have a specific directory structure or logic.

        Expected return format: a list of dicts, each with at least:
            {
                "image_path": Path(...),
                "save_name": "some_unique_name"
                ...  # anything else you want
            }
        """
        raise NotImplementedError(
            "collect_image_files_from_dataset must be implemented by child classes."
        )

    def generate_conditions(self, image_path: str, existing_conditions: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Generate any necessary conditions or metadata for a given image.
        Child classes may override this with their own logic.
        
        By default, we raise NotImplementedError to ensure child classes provide an implementation 
        unless there's nothing to do.
        """
        raise NotImplementedError(
            "generate_conditions is not implemented. Override in child classes if needed.")
        
    def process_single_image(self, image_path: str, save_name: str) -> Dict[str, Any]:
        """
        Processes a single image:
        1. If JSON exists, loads its conditions and passes them to generate_conditions.
        2. If JSON doesn't exist, passes an empty conditions dictionary to generate_conditions.
        3. Saves the updated conditions to JSON and returns the result.
        """
        # Define the JSON path
        json_path = os.path.join(self.base_save_path, 'json', f"{save_name}.json")
        os.makedirs(os.path.dirname(json_path), exist_ok=True)

        # Step 1: Load existing conditions if JSON exists
        existing_conditions = {}
        if os.path.exists(json_path):
            try:
                result = load_json(json_path)
                existing_conditions = result.get("conditions", {})
                print(f"[INFO] JSON already exists. Loaded conditions from: {json_path}")
            except Exception as e:
                print(f"[ERROR] Failed to read JSON {json_path}. Proceeding with empty conditions: {e}")

        # Step 2: Generate conditions, passing the existing ones
        try:
            updated_conditions = self.generate_conditions(image_path, existing_conditions)
        except Exception as e:
            #print(f"[ERROR] Failed to generate conditions for {image_path}: {e}")
            updated_conditions = {}

        # Step 3: Build the result dictionary
        result = {
            "original_image_path": image_path,
            "conditions": updated_conditions
        }

        # Step 4: Save the updated JSON
        try:
            save_json(json_path, result)
            #print(f"[INFO] Successfully saved JSON: {json_path}")
        except Exception as e:
            print(f"[ERROR] Failed to save JSON {json_path}: {e}")

        return result


    @measure_time
    def process_dataset(self, max_workers: int = 2, limit: int = None) -> None:
        """
        Collects images via collect_image_files_from_dataset,
        then processes them in parallel using process_single_image.
        """
        all_images_info = self.collect_image_files_from_dataset()
        if limit is not None:
            all_images_info = all_images_info[:limit]

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for info in all_images_info:
                image_path = info["image_path"]
                save_name = info["save_name"]
                
                futures.append(
                    executor.submit(
                        self.process_single_image, 
                        str(image_path), 
                        save_name
                    )
                )

            # Wrap the as_completed in tqdm for progress tracking
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing images"):
                try:
                    _ = future.result()
                except Exception as e:
                    print(f"[ERROR] Exception while processing an image: {e}")

        print("[INFO] All images processed.")