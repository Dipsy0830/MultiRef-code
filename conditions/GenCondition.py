import matplotlib.pyplot as plt
import os
import cv2
import warnings
import glob
import time
warnings.filterwarnings("ignore")
from typing import List, Dict, Any
import os
from pathlib import Path
from transformers import AutoProcessor, AutoModelForCausalLM
# from transformers import pipeline, AutoProcessor
import sys
sys.path.append('/media/sata4/Contextaware/Depth-Anything-V2/')
from depth_anything_v2.dpt import DepthAnythingV2
from .to_depth import get_depth

from .to_sketch_new import get_sketch
# from .to_caption import get_caption
from .to_caption_qwen import get_caption
from .to_extrapolation import *
from .to_ground_sam import *

# from to_sam import get_mask, save_masks_as_image
# environment settings
# use bfloat16

# torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

# if torch.cuda.get_device_properties(0).major >= 8:
#     # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
#     torch.backends.cuda.matmul.allow_tf32 = True
#     torch.backends.cudnn.allow_tf32 = True




import os
import shutil
from PIL import Image



class GenCondition:
    '''
    base path is the target condition directory for the whole dataset
    '''
    def __init__(self, 
                 add_base_name:bool=False,
                 base_save_path:str=None,
                 florence_model_id=None, 
                 sam2_config=None, 
                 sam2_checkpoint=None, 
                 device="cuda"):
        sam2_config = "../../Grounded-SAM-2/sam2/configs/sam2.1/sam2.1_hiera_l.yaml"
        sam2_checkpoint = "../../Grounded-SAM-2/checkpoints/sam2.1_hiera_large.pt"
        florence_model_id="microsoft/Florence-2-large"
        self.add_base_name=add_base_name
        self.base_save_path=base_save_path
        self.device = device
        
        # torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

        # if torch.cuda.get_device_properties(0).major >= 8:
        #     # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        #     torch.backends.cuda.matmul.allow_tf32 = True
        #     torch.backends.cudnn.allow_tf32 = True

        # Load models during initialization
        if florence_model_id:
            self.florence_model = AutoModelForCausalLM.from_pretrained(
                florence_model_id, trust_remote_code=True, torch_dtype='auto'
            ).eval().to(self.device)
            self.florence_processor = AutoProcessor.from_pretrained(
                florence_model_id, trust_remote_code=True,device=self.device
            )
        else:
            self.florence_model = None
            self.florence_processor = None

        if sam2_config and sam2_checkpoint:
            print("Building SAM2 model...")
            self.sam2_model = build_sam2_no_hydra(sam2_config, sam2_checkpoint, device=self.device)
        else:
            print("No!!!!! SAM2 model...")
            self.sam2_model = None
        ### 关闭！！！！！ 混合进度可以吗？？
        
        
        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }

        encoder = 'vitb' # or 'vits', 'vitb', 'vitg'
        depth_model = DepthAnythingV2(**model_configs[encoder])
        depth_model.load_state_dict(torch.load(f'/media/sata4/Contextaware/Depth-Anything-V2/checkpoints/depth_anything_v2_{encoder}.pth', map_location='cpu'))
        self.depth_model = depth_model.to(self.device).eval()


    def _get_condition_path(self, ori_image_path, condition_type):
        if self.add_base_name:
            base_name = os.path.join(os.path.basename(os.path.dirname(ori_image_path)),
                                     os.path.basename(ori_image_path)).replace('/', '_')
        else:
            base_name = os.path.basename(ori_image_path)

        condition_dir = os.path.join(self.base_save_path, condition_type)
        os.makedirs(condition_dir, exist_ok=True)
        
        condition_dir = os.path.join(self.base_save_path, condition_type)
        return os.path.join(condition_type, base_name)
    
    def _get_condition_paths(self, ori_image_path, condition_name):
        condition_path = self._get_condition_path(ori_image_path, condition_name)
        output_condition_path = os.path.join(self.base_save_path, condition_path)
        return condition_path, output_condition_path
    
    def get_condition_paths_list(self, ori_image_paths, condition_name):
        """
        接收一个包含多个 ori_image_path 的列表，将每一个进行 _get_condition_paths 处理，
        最后返回一个列表，列表内的每一项都是 (condition_path, output_condition_path) 二元组。
        """
        results = []
        for ori_image_path in ori_image_paths:
            condition_path, output_condition_path = self._get_condition_paths(ori_image_path, condition_name)
            results.append((condition_path, output_condition_path))
        return results
    def get_condition_paths_list(self, ori_image_paths, condition_name):
        """
        支持两种输入：
        1. 如果 ori_image_paths 是单个字符串，则直接调用 self._get_condition_paths 返回 (condition_path, output_condition_path)。
        2. 如果 ori_image_paths 是一个列表，则对每个元素调用 self._get_condition_paths，返回一个列表，每个元素是 (condition_path, output_condition_path)。
        """
        # 如果是单个字符串，则直接调用
        if isinstance(ori_image_paths, str):
            return self._get_condition_paths(ori_image_paths, condition_name)

        # 如果是列表，逐个处理
        elif isinstance(ori_image_paths, (list, tuple)):
            results = []
            for ori_image_path in ori_image_paths:
                condition_path, output_condition_path = self._get_condition_paths(ori_image_path, condition_name)
                results.append(output_condition_path)
            return results

        else:
            raise TypeError(
                "ori_image_paths 必须是字符串或字符串列表，"
                f"当前类型: {type(ori_image_paths)}"
            )


    def generate_canny(self, ori_image_path):
        relative_path, output_path = self._get_condition_paths(ori_image_path, 'canny')
        canny_image = get_canny(ori_image_path)
        cv2.imwrite(output_path, canny_image)
        return relative_path

    def generate_depth(self, ori_image_path):
        relative_path, output_path = self._get_condition_paths(ori_image_path, 'depth')
        depth_image = get_depth(ori_image_path, self.depth_model)
        cv2.imwrite(output_path, depth_image)
        return relative_path

    def generate_sketch(self, ori_image_path):
        relative_path, output_path = self._get_condition_paths(ori_image_path, 'sketch')
        sketch_image = get_sketch(ori_image_path)
        cv2.imwrite(output_path, sketch_image)
        return relative_path

    def generate_extrapolation(self, ori_image_path):
        relative_path, output_path = self._get_condition_paths(ori_image_path, 'extrapolation')
        crop_image = random_crop(ori_image_path)
        cv2.imwrite(output_path, crop_image)
        return relative_path

    def generate_super_res(self, ori_image_path):
        relative_path, output_path = self._get_condition_paths(ori_image_path, 'super_res')
        low_resolution_image = generate_low_resolution_image(ori_image_path)
        cv2.imwrite(output_path, low_resolution_image)
        return relative_path

    def generate_light_enhance(self, ori_image_path):
        relative_path, output_path = self._get_condition_paths(ori_image_path, 'light_enhance')
        low_light_image = generate_low_light_image(ori_image_path)
        cv2.imwrite(output_path, low_light_image)
        return relative_path

    def generate_super_res_light_enhance(self, ori_image_path):
        relative_path, output_path = self._get_condition_paths(ori_image_path, 'super_res_light_enhance')
        low_res_low_light_image = generate_low_res_low_light(ori_image_path)
        cv2.imwrite(output_path, low_res_low_light_image)
        return relative_path

    def generate_super_res_extrapolation(self, ori_image_path):
        relative_path, output_path = self._get_condition_paths(ori_image_path, 'super_res_extrapolation')
        low_res_crop_image = generate_low_res_crop_image(ori_image_path)
        cv2.imwrite(output_path, low_res_crop_image)
        return relative_path

    def generate_light_enhance_extrapolation(self, ori_image_path):
        relative_path, output_path = self._get_condition_paths(ori_image_path, 'light_enhance_extrapolation')
        low_light_crop_image = generate_low_light_crop_image(ori_image_path)
        cv2.imwrite(output_path, low_light_crop_image)
        return relative_path

    def generate_semantic_map(self, ori_image_path):
        relative_path, output_path = self._get_condition_paths(ori_image_path, 'semantic_map')
        get_semantic_map(ori_image_path, output_path, self.sam2_model)
        return relative_path
    
    def generate_semantic_map_low_iou(self, ori_image_path):
        relative_path, output_path = self._get_condition_paths(ori_image_path, 'semantic_map')
        get_semantic_map_low_iou(ori_image_path, output_path, self.sam2_model)
        return relative_path

    def generate_semantic_map_human(self, ori_image_path):
        relative_path, output_path = self._get_condition_paths(ori_image_path, 'semantic_map')
        get_semantic_map_human(ori_image_path, output_path, self.sam2_model)
        return relative_path
    
    ##sam
    def generate_bbox_and_single_mask(self, ori_image_path, input_text, top_n):
        bbox_relative_path, bbox_output_path = self._get_condition_paths(ori_image_path, 'bbox')
        mask_relative_path, mask_output_path = self._get_condition_paths(ori_image_path, 'single_mask')
        mask_list, selected_bbox = get_grounding_bbox_and_single_mask(
            ori_image_path, input_text, self.sam2_model,
            self.florence_model, self.florence_processor, top_n,
            bbox_output_path, mask_output_path
        )
        mask_list = trim_base_path(mask_list, self.base_save_path)
        return bbox_relative_path, mask_list, selected_bbox
    
    # def generate_bbox_and_single_mask_tryon(self, ori_image_path, input_text='upper cloth', top_n=1):
    #     bbox_relative_path, bbox_output_path = self._get_condition_paths(ori_image_path, 'bbox')
    #     mask_relative_path, mask_output_path = self._get_condition_paths(ori_image_path, 'single_mask')
    #     mask_list, selected_bbox = get_grounding_bbox_and_single_mask(
    #         ori_image_path, input_text, self.sam2_model,
    #         self.florence_model, self.florence_processor, top_n,
    #         bbox_output_path, mask_output_path
    #     )
    #     mask_list = trim_base_path(mask_list, self.base_save_path)
    #     return bbox_relative_path, mask_list, selected_bbox
    
    def generate_multi_bbox_and_merge_mask(self, ori_image_path, input_text):   
        bbox_relative_path, bbox_output_path = self.        _get_condition_paths(ori_image_path, 'bbox')
        mask_relative_path, mask_output_path = self._get_condition_paths(ori_image_path, 'single_mask')
        mask_path,bbox = get_multi_bbox_and_mask(ori_image_path, input_text, self.sam2_model, self.florence_model, self.florence_processor, bbox_output_path, mask_output_path)
        mask_path = trim_base_path(mask_path, self.base_save_path)
        return bbox_relative_path, mask_path, bbox
        
    
    ##sam
    def generate_single_mask_human(self, ori_image_path, bbox_array_list):
        relative_path, output_path = self._get_condition_paths(ori_image_path, 'single_mask')
        mask_path_list = get_mask_human(ori_image_path, bbox_array_list, self.sam2_model, save_path=output_path)
        return mask_path_list

    def generate_pose_and_bbox_human(self, ori_image_path, keypoints_2d_list, bbox_list):
        pose_relative_path, pose_output_path = self._get_condition_paths(ori_image_path, 'pose')
        bbox_relative_path, bbox_output_path = self._get_condition_paths(ori_image_path, 'bbox')
        pose_path_list, bbox_path_list = get_keypoints_and_bbox_human(
            keypoints_2d_list, bbox_list, ori_image_path,
            pose_output_path, bbox_output_path
        )
        return pose_path_list, bbox_path_list

    def generate_subject_db(self, ori_image_path):
        relative_paths=[]
        subject_path = get_subject_db(ori_image_path)
        for path in subject_path:
            output_path = self._get_condition_path(path, 'subject')
            copy_file_as_image(path, os.path.join(self.base_save_path,output_path))
            relative_paths.append(output_path)
        return relative_paths

    
    def generate_subject_200k(self, ori_image_path):
        relative_path, output_path = self._get_condition_paths(ori_image_path, 'subject')
        subject_path = get_subject_200k(ori_image_path)
        target_file_path = os.path.join(self.base_save_path, 'subject')
        copied_file_path = copy_file(subject_path, target_file_path)
        relative_path = trim_base_path(copied_file_path, self.base_save_path)
        return relative_path
    
    def generate_subject_tryon(self, ori_image_path):
        relative_path, output_path = self._get_condition_paths(ori_image_path, 'subject')
        subject_path = get_subject_tryon(ori_image_path)
        target_file_path = os.path.join(self.base_save_path, 'subject')
        copied_file_path = copy_file(subject_path, target_file_path)
        relative_path = trim_base_path(copied_file_path, self.base_save_path)
        return relative_path
    
    def generate_pose_tryon(self, ori_image_path):
        relative_path, output_path = self._get_condition_paths(ori_image_path, 'pose')
        pose_path = get_pose_tryon(ori_image_path)
        target_file_path = os.path.join(self.base_save_path, 'pose')
        copied_file_path = copy_file(pose_path, target_file_path)
        relative_path = trim_base_path(copied_file_path, self.base_save_path)
        return relative_path
    
    def generate_style_2d_human(self, ori_image_path):
        relative_path, output_path = self._get_condition_paths(ori_image_path, 'style')
        style_path = get_style_2d_human(ori_image_path, N=3)
        target_file_path = os.path.join(self.base_save_path, 'style')
        copied_file_path = copy_file(style_path, target_file_path)
        relative_path = trim_base_path(copied_file_path, self.base_save_path)
        return relative_path
    
    def generate_style_wikiart(self, ori_image_path):
        relative_path, output_path = self._get_condition_paths(ori_image_path, 'style')
        style_path = get_style_wikiart(ori_image_path, N=3)
        target_file_path = os.path.join(self.base_save_path, 'style')
        copied_file_path = copy_file(style_path, target_file_path)
        relative_path = trim_base_path(copied_file_path, self.base_save_path)
        return relative_path

    def generate_style_stylebooth(self, ori_image_path):
        relative_path, output_path = self._get_condition_paths(ori_image_path, 'style')
        style_path = get_style_stylebooth(ori_image_path, N=3)
        target_file_path = os.path.join(self.base_save_path, 'style')
        all_path = []
        for i in style_path:
            path_parts = str(i).split('/')
            style = path_parts[-2]  # Get style from path (e.g., AbstractExpressionism)
            image_num = path_parts[-1].split('.')[0]  # Get image number (e.g., 100)
            save_name = f"{style}_{image_num}"
            copied_file_path = copy_file_as_image(i, target_file_path+'/'+ save_name+'.jpg')
            # append target_file_path
            all_path.append(target_file_path+'/'+ save_name+'.jpg')
        relative_path = trim_base_path(all_path, self.base_save_path)
        return relative_path
        
    
    
    def generate_caption(self, ori_image_path, max_retries=5, retry_delay=1):
        """
        Generate image caption with retry mechanism

        Args:
            ori_image_path: Path to the original image
            base_path: Base path for saving
            max_retries: Maximum number of retry attempts, default 5
            retry_delay: Delay between retries in seconds, default 1

        Returns:
            caption: Generated image caption
        """
        for attempt in range(max_retries):
            try:
                caption = get_caption(ori_image_path,)
                return caption
            except Exception as e:
                if attempt < max_retries - 1:  # If not the last attempt
                    print(f"Caption generation failed (attempt {attempt + 1}/{max_retries}): {str(e)}")
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    continue
                else:   # If last attempt failed
                    print(f"Caption generation failed after {max_retries} attempts: {str(e)}")
                    raise  # Re-raise the last exception
    
    
    
def copy_file(src_paths, dst_path):
    if isinstance(src_paths, str):
        src_paths = [src_paths]  # 如果是单个路径，则转为列表处理
    
    copied_files = []
    for src_path in src_paths:
        try:
            os.makedirs(dst_path, exist_ok=True)
            file_name = os.path.basename(src_path)
            target_file_path = os.path.join(dst_path, file_name)
            shutil.copy(src_path, target_file_path)
            #print(f"[INFO] Successfully copied {file_name} to {target_file_path}")
            copied_files.append(target_file_path)
        except Exception as e:
            #print(f"[ERROR] Failed to copy {src_path} to {dst_path}: {e}")
            copied_files.append(None)

    return copied_files


def copy_file_as_image(src_path, dst_path):
    """
    读取 src_path（完整路径）的图像，并将其保存到 dst_path（完整路径）。

    :param src_path: 源图像文件的完整路径
    :param dst_path: 目标文件的完整路径
    :return: 如果成功，则返回保存成功的目标路径；否则返回 None
    """
    try:
        # 创建目标文件夹（如果不存在）
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)

        # 读取图像
        with Image.open(src_path) as img:
            # 保存到目标路径
            img.save(dst_path)

        #print(f"[INFO] Successfully processed {src_path} -> {dst_path}")
        return dst_path
    except Exception as e:
        #print(f"[ERROR] Failed to process {src_path}: {e}")
        return None


def trim_base_path(paths, base_path):
    """
    这个函数将返回去掉 base_path 部分后的相对路径。
    
    参数:
    - paths: 可以是单个路径字符串，或者路径字符串的列表。
    - base_path: 要去掉的 base 路径。

    返回:
    - 如果输入是一个列表，返回相对路径列表。
    - 如果输入是一个单个路径，返回单个相对路径字符串。
    """
    # 确保 base_path 是绝对路径
    base_path = os.path.abspath(base_path)

    # 处理单个路径或路径列表
    if isinstance(paths, str):
        # 如果是单个路径字符串，直接返回相对路径
        return os.path.relpath(paths, base_path)
    
    elif isinstance(paths, list):
        # 如果是路径列表，返回每个路径相对于 base_path 的相对路径
        return [os.path.relpath(path, base_path) for path in paths]

    else:
        raise ValueError("The 'paths' parameter should be a string or a list of strings.")

