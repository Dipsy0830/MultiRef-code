import os,sys
sys.path.append('.')
import numpy as np
import json
from typing import List, Dict, Any
from pathlib import Path
from utils_tools import *
from condition_def import *
from base2conditon import BaseImageProcessor
import traceback
from tqdm import tqdm


class Stylebooth(BaseImageProcessor):
    def __init__(self, base_dataset_path: str, base_save_path: str, condition_list='all'):
        image_dataset_path = [os.path.join(base_dataset_path, 'BatchA')]
        print(image_dataset_path)
        add_base_name = True  
        super().__init__(base_dataset_path, base_save_path, 
                         image_dataset_path, add_base_name, 
                         condition_list)
        
        # Load the stylebooth-1.jsonl file
        json_path = os.path.join(os.path.dirname(base_dataset_path), 'json path')
        self.stylebooth_data = []
        with open(json_path, 'r') as f:
            for line in f:
                if line.strip():  # Skip empty lines
                    self.stylebooth_data.append(json.loads(line))
        
    def collect_image_files_from_dataset(self) -> List[Dict[str, Any]]:
        dataset_paths = [Path(p) for p in self.image_dataset_path]  

        image_files = []
        for dataset_path in dataset_paths:
            image_files.extend(dataset_path.rglob("*.jpg"))
            image_files.extend(dataset_path.rglob("*.png"))
            image_files.extend(dataset_path.rglob("*.jpeg"))

        image_info_list = []
        for image_file in image_files:
            if image_file.is_file():
                # Find corresponding entry in stylebooth-1-fixed.jsonl
                for entry in self.stylebooth_data:
                    if 'path to /MetaData/X2I-mm-instruction/'+ str(entry['output_image']) == str(image_file):
                        # Extract style and image number from the path
                        path_parts = str(image_file).split('/')
                        style = path_parts[-2]  # Get style from path (e.g., AbstractExpressionism)
                        image_num = path_parts[-1].split('.')[0]  # Get image number (e.g., 100)
                        save_name = f"{style}_{image_num}"
                        # save_name = f"{image_file.stem}"
                        image_info_list.append({
                            "image_path": str(image_file),
                            "save_name": save_name
                        })
                        
        print(len(image_info_list))
        return image_info_list
        
    @measure_time
    def generate_conditions(self, image_path: str, existing_conditions: Dict[str, Any] = None) -> Dict[str, Any]:
        gc = self.gen_condition
        conditions = existing_conditions or {}
        
        parallel_tasks = {
            "caption": lambda: gc.generate_caption(image_path),
            'style_path': lambda: gc.generate_style_stylebooth(image_path),
            "semantic_map_path": lambda: gc.generate_semantic_map_low_iou(image_path),
        }
        parallel_tasks.update(common_tasks(gc, image_path))
        # parallel_tasks.update(img_edit_tasks(gc, image_path))
        parallel_tasks = self.filter_parallel_tasks(parallel_tasks, conditions,)
        
        new_conditions = execute_parallel_tasks(parallel_tasks)
        conditions.update(new_conditions)

        if self.replace or not all(k in conditions for k in ["bbox_path", "bbox_value", "single_mask_path"]):
            bbox_path, mask_list, selected_bbox = gc.generate_bbox_and_single_mask(
                image_path,
                conditions['caption'],
                top_n=1
            )
            conditions['bbox_path'] = bbox_path
            conditions['bbox_value'] = selected_bbox.tolist()
            conditions['single_mask_path'] = mask_list
    
        return conditions
    
    def process_single_image(self, image_path: str, save_name: str) -> Dict[str, Any]:
        """
        调用父类的方法处理单张图片
        """
        return super().process_single_image(image_path, save_name)
    
    def process_dataset(self, max_workers: int = 1) -> None:
        """
        处理数据集中的所有图片
        """
        image_info_list = self.collect_image_files_from_dataset()
        
        for image_info in tqdm(image_info_list, total=len(image_info_list), desc="Processing images"):
            image_path = image_info["image_path"]
            save_name = image_info["save_name"]
            
            # 处理单张图片
            self.process_single_image(image_path, save_name)
    
if __name__ == '__main__':
    base_dataset_path = '../../MetaData/X2I-mm-instruction/stylebooth'
    save_path = '../../Condition_stylebooth'
    dataset_class = Stylebooth(base_dataset_path, save_path, condition_list='all')
    dataset_class.process_dataset(max_workers=1) 