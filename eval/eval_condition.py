import matplotlib.pyplot as plt
import os
import torch 
import cv2
import numpy as np
from typing import List, Dict, Any, Union
from pathlib import Path
import sys
import json
import traceback
print("Importing basic libraries...")

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.insert(0, project_root)
print(f"Added project root to path: {project_root}")

try:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../Depth-Anything-V2/')))
    from depth_anything_v2.dpt import DepthAnythingV2
    print("Imported DepthAnythingV2")
except Exception as e:
    print(f"Error importing DepthAnythingV2: {str(e)}")
    traceback.print_exc()

try:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../Grounded-SAM-2/')))
    from sam2.build_sam import build_sam2_no_hydra
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    print("Imported SAM2 modules")
except Exception as e:
    print(f"Error importing SAM2 modules: {str(e)}")
    traceback.print_exc()

try:
    from conditions.to_depth import get_depth
    from conditions.to_sketch_new import get_sketch
    from conditions.to_caption import get_caption
    # from conditions.to_caption_qwen import get_caption
    from conditions.to_extrapolation import get_canny
    from conditions.to_ground_sam import get_semantic_map_wo_save, get_bbox_and_single_mask_for_eval 
    from conditions.to_pose_no_args import get_pose_and_save
    print("Imported condition processing functions")
except Exception as e:
    print(f"Error importing condition processing functions: {str(e)}")
    traceback.print_exc()

try:
    from traditional_metrics import calculate_MSE, calculate_IoU, convert_to_bbx, calculate_bbox_iou, convert_to_mask, convert_to_label_map,calculate_image_text_clip_score, calculate_image_clip_score, calculate_FID, calculate_aesthetics_score, compute_pixel_mAP
    print("Imported metric calculation functions")
except Exception as e:
    print(f"Error importing metric calculation functions: {str(e)}")
    traceback.print_exc()

try:
    from transformers import AutoProcessor, AutoModelForCausalLM
    from transformers import pipeline, AutoProcessor
    from scipy.optimize import linear_sum_assignment
    print("Imported transformers and optimization modules")
except Exception as e:
    print(f"Error importing transformers and optimization modules: {str(e)}")
    traceback.print_exc()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# torch.backends.cuda.matmul.allow_tf32 = True  # 允许Tensor Core加速
# torch.set_float32_matmul_precision('high')  # 统一矩阵运算精度

class ImageConditionEvaluator:
    def __init__(self):
        print("Initializing ImageConditionEvaluator...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            self._setup_device_optimizations()
        except Exception as e:
            print(f"Error setting up device optimizations: {str(e)}")
            traceback.print_exc()
        self.depth_model = None
        self.sam2_model = None
        self.florence_model = None
        self.florence_processor = None
        #self._load_depth_model()
        self._load_florence_models()
        self._load_sam2_model()
        print("ImageConditionEvaluator initialized")

    def _setup_device_optimizations(self):
        print("Setting up device optimizations...")
        if self.device.type == "cuda":
            pass
            # torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
            # if torch.cuda.get_device_properties(0).major >= 8:
            #     torch.backends.cuda.matmul.allow_tf32 = True
            #     torch.backends.cudnn.allow_tf32 = True
        print("Device optimizations complete")

    def _load_depth_model(self):
        if self.depth_model is None:
            print("Loading depth model...")
            try:
                model_configs = {
                    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
                    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
                    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
                    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
                }
                encoder = 'vitb'
                self.depth_model = DepthAnythingV2(**model_configs[encoder])
                self.depth_model.load_state_dict(torch.load(f'../../Depth-Anything-V2/checkpoints/depth_anything_v2_{encoder}.pth', map_location='cpu'))
                self.depth_model = self.depth_model.to(self.device).eval()
                print("Depth model loaded and ready")
            except Exception as e:
                print(f"Error loading depth model: {str(e)}")
                traceback.print_exc()

    def _load_sam2_model(self):
        if self.sam2_model is None:
            print("Loading SAM2 model...")
            try:
                sam2_config = "../../Grounded-SAM-2/sam2/configs/sam2.1/sam2.1_hiera_l.yaml"
                sam2_checkpoint = "../../Grounded-SAM-2/checkpoints/sam2.1_hiera_large.pt"
                # Load SAM2 on GPU 0
                sam2_device = "cuda"
                self.sam2_model = build_sam2_no_hydra(sam2_config, sam2_checkpoint, device=sam2_device)
                print(f"SAM2 model loaded and ready on {sam2_device}")
            except Exception as e:
                print(f"Error loading SAM2 model: {str(e)}")
                traceback.print_exc()

    def _load_florence_models(self):
        if self.florence_model is None or self.florence_processor is None:
            print("Loading Florence models...")
            try:
                # Load Florence on GPU 1
                florence_device = "cuda"
                self.florence_model = AutoModelForCausalLM.from_pretrained(
                    "microsoft/Florence-2-large", trust_remote_code=True, torch_dtype='auto'
                ).eval().to(florence_device)
                self.florence_processor = AutoProcessor.from_pretrained(
                    "microsoft/Florence-2-large", trust_remote_code=True, device=florence_device
                )
                print(f"Florence models loaded and ready on {florence_device}")
            except Exception as e:
                print(f"Error loading Florence models: {str(e)}")
                traceback.print_exc()

    def caption_for_image(self, image_path):
        print(f"Generating caption for image: {image_path}")
        try:
            return get_caption(image_path, subject=True)
        except Exception as e:
            print(f"Error generating caption: {str(e)}")
            traceback.print_exc()
            return ""

    def eval_fid(self, gen_image_path):
        print(f"Calculating FID for image: {gen_image_path}")
        try:
            return calculate_FID(gen_image_path)
        except Exception as e:
            print(f"Error calculating FID: {str(e)}")
            traceback.print_exc()
            return float('nan')

    def eval_aesthetics(self, gen_image_path):
        print(f"Calculating aesthetics score for image: {gen_image_path}")
        try:
            return calculate_aesthetics_score(gen_image_path)
        except Exception as e:
            print(f"Error calculating aesthetics score: {str(e)}")
            traceback.print_exc()
            return float('nan')

    def eval_canny(self, gen_image_path, canny_path):
        print(f"Evaluating canny edge detection for image: {gen_image_path}")
        try:
            gen_canny = get_canny(gen_image_path)
            ref_canny = cv2.imread(canny_path, 0)
            return calculate_MSE(gen_canny, ref_canny)
        except Exception as e:
            print(f"Error evaluating canny: {str(e)}")
            traceback.print_exc()
            return float('nan')

    def eval_depth(self, gen_image_path, depth_path):
        print(f"Evaluating depth for image: {gen_image_path}")
        try:
            self._load_depth_model()
            gen_depth = get_depth(gen_image_path, self.depth_model)
            ref_depth = cv2.imread(depth_path, 0)
            print(gen_image_path,depth_path)
            mse=calculate_MSE(gen_depth, ref_depth)
            print(mse)
            return mse
        
        except Exception as e:
            print(f"Error evaluating depth: {str(e)}")
            traceback.print_exc()
            return float('nan')

    def eval_sketch(self, gen_image_path, sketch_path):
        print(f"Evaluating sketch for image: {gen_image_path}")
        gen_sketch = get_sketch(gen_image_path)
        ref_sketch = cv2.imread(sketch_path, 0)
        return calculate_MSE(gen_sketch, ref_sketch)

    def eval_caption(self, gen_image_path, text):
        print(f"Evaluating caption for image: {gen_image_path}")
        return calculate_image_text_clip_score(gen_image_path, text)

    def eval_style(self, gen_image_path, style_path):
        print(f"Evaluating style for image: {gen_image_path}")
        return calculate_image_clip_score(gen_image_path, style_path)

    def eval_subject(self, gen_image_path, subject_path):
        print(f"Evaluating subject for image: {gen_image_path}")
        return calculate_image_clip_score(gen_image_path, subject_path)

    def eval_semantic_map(self, gen_image_path, semantic_map_path):
        print(f"Evaluating semantic map for image: {gen_image_path}")
        self._load_sam2_model()
        
        gen_semantic_map = get_semantic_map_wo_save(gen_image_path, self.sam2_model)
        ref_semantic_map = cv2.imread(semantic_map_path, 0)
        
        if gen_semantic_map.shape[:2] != ref_semantic_map.shape[:2]:
            print("Resizing reference semantic map to match generated map")
            ref_semantic_map = cv2.resize(ref_semantic_map, 
                                        (gen_semantic_map.shape[1], gen_semantic_map.shape[0]),
                                        interpolation=cv2.INTER_NEAREST)
        
        if len(ref_semantic_map.shape) == 2:
            print("Converting grayscale semantic map to RGB")
            ref_semantic_map = cv2.cvtColor(ref_semantic_map, cv2.COLOR_GRAY2RGB)
        
        print("Converting maps to label maps")
        gen_labels = convert_to_label_map(gen_semantic_map)
        ref_labels = convert_to_label_map(ref_semantic_map)
        
        min_region_size = 300
        print("Finding valid regions in generated map")
        valid_gen_masks = []
        for label in range(1, np.max(gen_labels) + 1):
            mask = (gen_labels == label)
            if np.sum(mask) >= min_region_size:
                valid_gen_masks.append(mask)
        
        valid_ref_masks = []
        for label in range(1, np.max(ref_labels) + 1):
            mask = (ref_labels == label)
            if np.sum(mask) >= min_region_size:
                valid_ref_masks.append(mask)
        
        if not valid_gen_masks or not valid_ref_masks:
            return 0.0
        
        iou_matrix = np.zeros((len(valid_gen_masks), len(valid_ref_masks)))
        for i, gen_mask in enumerate(valid_gen_masks):
            for j, ref_mask in enumerate(valid_ref_masks):
                iou = calculate_IoU(gen_mask, ref_mask)
                iou_matrix[i, j] = iou
        
        row_ind, col_ind = linear_sum_assignment(-iou_matrix)
        matched_ious = iou_matrix[row_ind, col_ind]
        return np.mean(matched_ious)

    def eval_bbox(self, gen_image_path, bbx_path):
        self._load_sam2_model()
        self._load_florence_models()
        ref_img = cv2.imread(bbx_path)
        ref_bbx_list = convert_to_bbx(ref_img)
        image_caption = self.caption_for_image(gen_image_path)
        _, selected_bbx = get_bbox_and_single_mask_for_eval(
            gen_image_path, image_caption, self.sam2_model,
            self.florence_model, self.florence_processor, top_n=1
        )
        return calculate_bbox_iou(np.array(ref_bbx_list[0]), np.array(selected_bbx[0]))

    def resize_mask(self, mask, target_shape):
        resized = cv2.resize(mask, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)
        return (resized > 0).astype(np.uint8)

    def eval_mask(self, gen_image_path, single_mask_path):
        self._load_sam2_model()
        self._load_florence_models()
        ref_img = cv2.imread(single_mask_path)
        ref_mask_list = convert_to_mask(ref_img)
        image_caption = self.caption_for_image(gen_image_path)
        selected_mask, _ = get_bbox_and_single_mask_for_eval(
            gen_image_path, image_caption, self.sam2_model,
            self.florence_model, self.florence_processor, top_n=1
        )
        ref_mask = np.array(ref_mask_list)
        sel_mask = np.array(selected_mask[0])
        if ref_mask.shape != sel_mask.shape:
            ref_mask = self.resize_mask(ref_mask, sel_mask.shape)
        return calculate_IoU(ref_mask, sel_mask)

    def eval_pose(self, gen_image_path, pose_path):
        parent = os.path.basename(os.path.dirname(os.path.dirname(gen_image_path)))
        child = os.path.basename(os.path.dirname(gen_image_path))
        filename = os.path.basename(gen_image_path)
        new_name = f"{parent}_{child}_{filename}"
        save_path = os.path.join('../pose_cache', new_name)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        get_pose_and_save(gen_image_path, save_path)
        return compute_pixel_mAP(save_path, pose_path, threshold=0)
    
    
    def evaluate_all(self, gen_image_path: str, reference_paths: Dict[str, str], conditions_to_evaluate: List[str] = []) -> Dict[str, float]:
        """
        Run all available evaluations based on provided reference paths and conditions to evaluate.
        
        Args:
            gen_image_path: Path to the generated image.
            reference_paths: Dictionary mapping evaluation type to reference path.
                Possible keys: 'depth', 'sketch', 'canny', 'semantic_map', 'bbox', 
                'mask', 'pose', 'caption', 'style', 'subject'
            conditions_to_evaluate: List of conditions to evaluate. If empty, all conditions are evaluated.
        
        Returns:
            Dictionary of evaluation results.
        """
        results = {}
        
        def try_evaluation(eval_type, eval_func, *args):
            try:
                results[eval_type] = eval_func(*args)
            except Exception as e:
                traceback.print_exc()
                print(f"Error evaluating {eval_type}: {str(e)}")
                results[f"{eval_type}_error"] = str(e)
        
        # Standalone evaluations (no reference needed)
        if not conditions_to_evaluate or 'fid' in conditions_to_evaluate:
            try_evaluation('fid', self.eval_fid, gen_image_path)
        if not conditions_to_evaluate or 'aesthetics' in conditions_to_evaluate:
            try_evaluation('aesthetics', self.eval_aesthetics, gen_image_path)
        
        # Evaluations requiring reference paths
        if not conditions_to_evaluate or 'depth' in conditions_to_evaluate and 'depth' in reference_paths:
            try_evaluation('depth', self.eval_depth, gen_image_path, reference_paths['depth'])
        
        if not conditions_to_evaluate or 'sketch' in conditions_to_evaluate and 'sketch' in reference_paths:
            try_evaluation('sketch', self.eval_sketch, gen_image_path, reference_paths['sketch'])
        
        if not conditions_to_evaluate or 'canny' in conditions_to_evaluate and 'canny' in reference_paths:
            try_evaluation('canny', self.eval_canny, gen_image_path, reference_paths['canny'])

        if not conditions_to_evaluate or 'semantic_map' in conditions_to_evaluate and 'semantic_map' in reference_paths:
            try_evaluation('semantic_map', self.eval_semantic_map, gen_image_path, reference_paths['semantic_map'])
        
        if not conditions_to_evaluate or 'bbox' in conditions_to_evaluate and 'bbox' in reference_paths:
            print(f"Evaluating bbox for image: {gen_image_path}")
            print(f"Reference bbox path: {reference_paths['bbox']}")
            try_evaluation('bbox', self.eval_bbox, gen_image_path, reference_paths['bbox'])
        
        if not conditions_to_evaluate or 'mask' in conditions_to_evaluate and 'mask' in reference_paths:
            try_evaluation('mask', self.eval_mask, gen_image_path, reference_paths['mask'])
        
        if not conditions_to_evaluate or 'pose' in conditions_to_evaluate and 'pose' in reference_paths:
            try_evaluation('pose', self.eval_pose, gen_image_path, reference_paths['pose'])
        
        if not conditions_to_evaluate or 'caption' in conditions_to_evaluate and 'caption' in reference_paths:
            try:
                # Caption can be a path or text
                caption_text = reference_paths['caption']
                if os.path.exists(reference_paths['caption']):
                    with open(reference_paths['caption'], 'r') as f:
                        caption_text = f.read().strip()
                try_evaluation('caption', self.eval_caption, gen_image_path, caption_text)
            except Exception as e:
                print(f"Error evaluating caption: {str(e)}")
                results['caption_error'] = str(e)
        
        if not conditions_to_evaluate or 'style' in conditions_to_evaluate and 'style' in reference_paths:
            try_evaluation('style', self.eval_style, gen_image_path, reference_paths['style'])
        
        if not conditions_to_evaluate or 'subject' in conditions_to_evaluate and 'subject' in reference_paths:
            try_evaluation('subject', self.eval_subject, gen_image_path, reference_paths['subject'])
        
        return results

    # def evaluate_all(self, result_image_path, reference_paths):
    #     results = {}
    #     for condition, reference_path in reference_paths.items():
    #         if condition == "canny":
    #             results[condition] = self.eval_canny(result_image_path, reference_path)
    #         elif condition == "subject":
    #             results[condition] = self.eval_subject(result_image_path, reference_path)
    #         elif condition == "caption":
    #             results[condition] = self.eval_caption(result_image_path, reference_path)
    #         elif condition == "semantic_map":
    #             results[condition] = self.eval_semantic_map(result_image_path, reference_path)
    #         elif condition == "sketch":
    #             results[condition] = self.eval_sketch(result_image_path, reference_path)
    #         elif condition == "bbox":
    #             results[condition] = self.eval_bbox(result_image_path, reference_path)
    #         elif condition == "mask":
    #             results[condition] = self.eval_mask(result_image_path, reference_path)
    #         elif condition == "pose":
    #             results[condition] = self.eval_pose(result_image_path, reference_path)
    #     return results

    # def batch_evaluate(self, json_data: Union[str, List[Dict]]) -> List[Dict]:
    #     """
    #     Evaluate multiple entries in batch mode.
        
    #     Args:
    #         json_data: JSON string or list of entry dictionaries
            
    #     Returns:
    #         List of updated entries with evaluation results
    #     """
    #     if isinstance(json_data, str):
    #         try:
    #             entries = json.loads(json_data)
    #             if not isinstance(entries, list):
    #                 entries = [entries]
    #         except json.JSONDecodeError:
    #             return [{"error": "Invalid JSON format"}]
    #     else:
    #         if isinstance(json_data, dict):
    #             entries = [json_data]
    #         else:
    #             entries = json_data
        
    #     for entry in entries:
    #         result_image_path = entry.get('results', {}).get('image_path')
    #         if not result_image_path:
    #             entry["evaluations"] = {"error": "No result image path found"}
    #             continue
            
    #         # Get conditions and reference paths
    #         conditions = entry.get('instructions', {}).get('conditions', [])
    #         image_path_mapping = entry.get('instructions', {}).get('image_path_mapping', {})
            
    #         reference_paths = {}
    #         for condition in conditions:
    #             if condition == "canny":
    #                 reference_paths['canny'] = image_path_mapping.get('canny_image')
    #             elif condition == "subject":
    #                 reference_paths['subject'] = image_path_mapping.get('subject_1')
    #             elif condition == "caption":
    #                 reference_paths['caption'] = entry.get('conditions', {}).get('caption')
    #             elif condition == "semantic_map":
    #                 reference_paths['semantic_map'] = entry.get('conditions', {}).get('semantic_map_path')
    #             elif condition == "sketch":
    #                 reference_paths['sketch'] = entry.get('conditions', {}).get('sketch_path')
    #             elif condition == "bbox":
    #                 reference_paths['bbox'] = entry.get('conditions', {}).get('bbox_path')
    #             elif condition == "depth":
    #                 reference_paths['depth'] = entry.get('conditions', {}).get('depth_path')
    #             elif condition == "mask":
    #                 reference_paths['mask'] = entry.get('conditions', {}).get('mask_path')
    #             elif condition == "pose":
    #                 reference_paths['pose'] = entry.get('conditions', {}).get('pose_path')
            
    #         # Run evaluations
    #         entry["evaluations"] = self.evaluate_all(result_image_path, reference_paths)
        
    #     return entries

if __name__ == "__main__":
    bbx_path = ''
    gen_img_path = ''
    evaluator = ImageConditionEvaluator()
    mse = evaluator.eval_depth(gen_img_path, bbx_path)
    print(mse)
    
