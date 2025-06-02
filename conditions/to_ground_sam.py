import os
import sys
sys.path.append('/media/sata4/Contextaware/Grounded-SAM-2/')
import cv2
import torch
import argparse
import numpy as np
import supervision as sv
from PIL import Image
from sam2.build_sam import build_sam2_no_hydra
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator



# os.environ["CUDA_VISIBLE_DEVICES"] = "1" if torch.cuda.is_available() else ""
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# if device.type == "cuda":
#     # use bfloat16 for the entire notebook
#     torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
#     # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
#     if torch.cuda.get_device_properties(0).major >= 8:
#         torch.backends.cuda.matmul.allow_tf32 = True
#         torch.backends.cudnn.allow_tf32 = True



def run_florence2(task_prompt, text_input, model, processor, image):
    assert model is not None, "You should pass the init florence-2 model here"
    assert processor is not None, "You should set florence-2 processor here"

    device = model.device

    if text_input is None:
        prompt = task_prompt
    else:
        prompt = task_prompt + text_input

    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device, torch.float16)
    generated_ids = model.generate(
        input_ids=inputs["input_ids"].to(device),
        pixel_values=inputs["pixel_values"].to(device),
        max_new_tokens=1024,
        early_stopping=False,
        do_sample=False,
        num_beams=3,
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(
        generated_text,
        task=task_prompt,
        image_size=(image.width, image.height)
    )
    return parsed_answer

def phrase_grounding_and_segmentation(
        florence2_model,
        florence2_processor,
        sam2_predictor,
        image_path,
        task_prompt="<CAPTION_TO_PHRASE_GROUNDING>",
        text_input=None
):
    # run florence-2 object detection in demo
    image = Image.open(image_path).convert("RGB")
    results = run_florence2(task_prompt, text_input, florence2_model, florence2_processor, image)

    """ Florence-2 Object Detection Output Format
    {'<CAPTION_TO_PHRASE_GROUNDING>': 
        {
            'bboxes':  #xyxy
                [
                    [34.23999786376953, 159.1199951171875, 582.0800170898438, 374.6399841308594], 
                    [1.5999999046325684, 4.079999923706055, 639.0399780273438, 305.03997802734375]
                ], 
            'labels': ['A green car', 'a yellow building']
        }
    }
    """
    assert text_input is not None, "Text input should not be None when calling phrase grounding pipeline."
    results = results[task_prompt]
    # parse florence-2 detection results
    input_boxes = np.array(results["bboxes"])

    class_names = results["labels"]
    class_ids = np.array(list(range(len(class_names))))

    # predict mask with SAM 2
    sam2_predictor.set_image(np.array(image))
    masks, scores, logits = sam2_predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_boxes,
        multimask_output=False,
    )

    if masks.ndim == 4:
        masks = masks.squeeze(1)

    return input_boxes, masks, class_ids    #input_box in xyxy

def get_grounding_bbox_and_single_mask(image_path, input_text,sam2_model,florence2_model,florence2_processor,top_n,full_bbox_path,full_single_mask_path):
    sam2_predictor = SAM2ImagePredictor(sam2_model)
    bbox, masks, class_ids = phrase_grounding_and_segmentation(
        florence2_model=florence2_model,
        florence2_processor=florence2_processor,
        sam2_predictor=sam2_predictor,
        image_path=image_path,
        text_input=input_text
    )    # bbox in xyxy
    selected_bbox, selected_mask, selected_class = select_bbox_and_mask(bbox, masks, class_ids, top_n)
    visualize_bbox(image_path, selected_bbox, selected_mask, selected_class, save_path=full_bbox_path)
    mask_list = visualize_single_mask(image_path, selected_mask, save_path=full_single_mask_path)

    return  mask_list,selected_bbox    #selected box in xyxy

def get_bbox_and_single_mask_for_eval(image_path, input_text,sam2_model,florence2_model,florence2_processor,top_n):
    sam2_predictor = SAM2ImagePredictor(sam2_model)
    bbox, masks, class_ids = phrase_grounding_and_segmentation(
        florence2_model=florence2_model,
        florence2_processor=florence2_processor,
        sam2_predictor=sam2_predictor,
        image_path=image_path,
        text_input=input_text
    )    # bbox in xyxy
    selected_bbox, selected_mask, _ = select_bbox_and_mask(bbox, masks, class_ids, top_n)

    return  selected_mask, selected_bbox 

def open_vocabulary_detection_and_segmentation(
        florence2_model,
        florence2_processor,
        sam2_predictor,
        image_path,
        task_prompt="<OPEN_VOCABULARY_DETECTION>",
        text_input=None
):
    # run florence-2 object detection in demo
    image = Image.open(image_path).convert("RGB")
    results = run_florence2(task_prompt, text_input, florence2_model, florence2_processor, image)
    
    """ Florence-2 Open-Vocabulary Detection Output Format
    {'<OPEN_VOCABULARY_DETECTION>': 
        {
            'bboxes': 
                [
                    [34.23999786376953, 159.1199951171875, 582.0800170898438, 374.6399841308594]
                ], 
            'bboxes_labels': ['A green car'],
            'polygons': [], 
            'polygons_labels': []
        }
    }
    """
    assert text_input is not None, "Text input should not be None when calling open-vocabulary detection pipeline."
    results = results[task_prompt]
    # parse florence-2 detection results
    input_boxes = np.array(results["bboxes"])
    # print(results)
    class_names = results["bboxes_labels"]
    class_ids = np.array(list(range(len(class_names))))
    
    # predict mask with SAM 2
    sam2_predictor.set_image(np.array(image))
    masks, scores, logits = sam2_predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_boxes,
        multimask_output=False,
    )
    
    if masks.ndim == 4:
        masks = masks.squeeze(1)

    return input_boxes, masks, class_ids


def get_multi_bbox_and_mask(image_path, input_text,sam2_model,florence2_model,florence2_processor,full_bbox_path,full_single_mask_path):
    sam2_predictor = SAM2ImagePredictor(sam2_model)
    bbox, masks, class_ids = open_vocabulary_detection_and_segmentation(
        florence2_model=florence2_model,
        florence2_processor=florence2_processor,
        sam2_predictor=sam2_predictor,
        image_path=image_path,
        text_input=input_text
    )    
    visualize_multi_bbox(image_path, bbox, masks, class_ids, save_path=full_bbox_path)
    mask_path = visualize_all_masks(image_path, masks, save_path=full_single_mask_path)
    return mask_path, bbox

def get_shape(image_path):
    image = Image.open(image_path)
    image = np.array(image.convert("RGB"))
    return image.shape


def select_bbox_and_mask(bbox_list, mask_list, class_ids, top_n=2):
    """
    从mask矩阵列表中筛选前n个元素

    Args:
        mask_list: 包含mask矩阵的列表或numpy数组
        top_n: 需要返回的元素数量，默认为2

    Returns:
        filtered_masks: 筛选后的mask列表
    """
    # 确保输入是numpy数组
    if not isinstance(mask_list, np.ndarray):
        mask_list = np.array(mask_list)
    if not isinstance(bbox_list, np.ndarray):
        bbox_list = np.array(bbox_list)

    # 确保top_n不超过列表长度
    top_n = min(top_n, len(mask_list))

    # 截取前top_n个元素
    filtered_bbox = bbox_list[:top_n]
    filtered_mask = mask_list[:top_n]
    filtered_ids = class_ids[:top_n]

    return filtered_bbox, filtered_mask, filtered_ids

def visualize_bbox(image_path, bbox, masks, class_ids, save_path):
   """
   在黑色背景上可视化bbox

   Args:
       image_shape: 原图的形状 (H, W, C)
       top_bboxes: 经过筛选的top N个bbox列表，格式为[x, y, w, h]
       save_path: 保存路径
   """
   image_shape = get_shape(image_path)
   # 创建黑色背景图像
   black_image = np.zeros(image_shape, dtype=np.uint8)
#    img = cv2.imread(image_path)
   detections = sv.Detections(
       xyxy=bbox,
       mask=masks.astype(bool),
       class_id=class_ids
   )
   box_annotator = sv.BoxAnnotator()
   annotated_frame = box_annotator.annotate(scene=black_image.copy(), detections=detections)
   cv2.imwrite(save_path, annotated_frame)
   
def visualize_multi_bbox(image_path, bbox_list, masks, class_ids, save_path):
    """
    在黑色背景上可视化多个bbox

    Args:
        image_path: 原始图像路径
        bbox_list: 包含多个bbox的列表，格式为[[x1, y1, w1, h1], [x2, y2, w2, h2], ...]
        masks: 每个bbox对应的掩膜
        class_ids: 每个bbox对应的类别ID
        save_path: 保存路径
    """
    image_shape = get_shape(image_path)
    
    # 创建黑色背景图像
    black_image = np.zeros(image_shape, dtype=np.uint8)
    
    # 将bbox列表合并为一个二维数组 [x1, y1, w1, h1, x2, y2, w2, h2, ...]
    flattened_bbox = np.array(bbox_list).reshape(-1, 4)
    
    # 创建检测对象
    detections = sv.Detections(
        xyxy=flattened_bbox,
        mask=masks.astype(bool),
        class_id=class_ids
    )
    
    # 设置框的标注器
    box_annotator = sv.BoxAnnotator()
    
    # 在黑色背景图像上绘制所有bbox
    annotated_frame = box_annotator.annotate(scene=black_image.copy(), detections=detections)
    
    # 保存标注后的图像
    cv2.imwrite(save_path, annotated_frame)


def visualize_single_mask(image_path, top_semantic_masks, save_path):
    """
    对每个top mask进行可视化，生成黑白图像

    Args:
        image_path: 原始图像路径
        top_semantic_masks: 经过筛选的top N个mask列表，每个mask是二值化的布尔数组
        save_path: 保存路径（完整路径，到目录级别）

    Returns:
        one_mask_list: 保存的mask图像路径列表
    """
    # 从图像路径获取basename
    basename = os.path.splitext(os.path.basename(image_path))[0]

    # 获取图像shape
    image_shape = get_shape(image_path)

    # 确保保存目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    save_dir = os.path.dirname(save_path)
    save_basename = os.path.basename(save_path)
    save_basename = os.path.splitext(save_basename)[0]

    # 处理每个mask
    one_mask_list = []
    for i, mask in enumerate(top_semantic_masks):
        # 创建黑色背景
        mask_image = np.zeros(image_shape[:2], dtype=np.uint8)

        if isinstance(mask, np.ndarray):
            bool_mask = mask.astype(bool)
        else:
            # 如果mask不是numpy数组，先转换为numpy数组
            bool_mask = np.array(mask, dtype=bool)

        # 将mask区域设为白色
        mask_image[bool_mask] = 255

        # 构建完整保存路径
        mask_path = os.path.join(save_dir, f"{save_basename}mask_{i + 1}.jpg")

        # 保存mask图像
        cv2.imwrite(mask_path, mask_image)

        # 将路径添加到列表
        one_mask_list.append(mask_path)
    # print(f"Saved mask {i + 1} to {mask_path}")

    return one_mask_list

def visualize_all_masks(image_path, all_semantic_masks, save_path):
    """
    将所有的mask叠加在一张图像上，并保存合成后的mask图像

    Args:
        image_path: 原始图像路径
        all_semantic_masks: 所有mask的列表，每个mask是二值化的布尔数组
        save_path: 保存路径（完整路径，到目录级别）

    Returns:
        mask_path: 合成的mask图像保存路径
    """
    # 从图像路径获取basename
    basename = os.path.splitext(os.path.basename(image_path))[0]
    # print(basename)

    # 获取图像shape
    image_shape = get_shape(image_path)

    # 确保保存目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    save_dir = os.path.dirname(save_path)
    save_basename = os.path.basename(save_path)

    save_basename = os.path.splitext(save_basename)[0]

    # 创建一个黑色背景图像，用于合成所有mask
    combined_mask = np.zeros(image_shape[:2], dtype=np.uint8)

    # 处理每个mask并叠加到合成图像中
    for i, mask in enumerate(all_semantic_masks):
        if isinstance(mask, np.ndarray):
            bool_mask = mask.astype(bool)
        else:
            # 如果mask不是numpy数组，先转换为numpy数组
            bool_mask = np.array(mask, dtype=bool)

        # 将当前mask区域叠加到合成图像中（将其区域设置为255）
        combined_mask[bool_mask] = 255

    # 构建合成的mask图像保存路径
    mask_path = os.path.join(save_dir, f"{basename}_merge_mask.jpg")

    # 保存合成后的mask图像
    cv2.imwrite(mask_path, combined_mask)

    return mask_path

def get_semantic_map(image_path, output_path,sam2_model, borders=False):   #generate sam mask for semantic map
    image = Image.open(image_path)
    image = np.array(image.convert("RGB"))
    mask_generator = SAM2AutomaticMaskGenerator(
        model=sam2_model,
        points_per_side=24,
        points_per_batch=32,
        pred_iou_thresh=0.93,
        stability_score_thresh=0.96,
        stability_score_offset=0.85,
        crop_n_layers=0,
        box_nms_thresh=0.85,
        min_mask_region_area=300,
        use_m2m=True,
    )
    # mask_generator = SAM2AutomaticMaskGenerator(
    #     model=sam2_model
    # )
    anns = mask_generator.generate(image)
    height, width = anns[0]['segmentation'].shape[:2]
    combined_img = np.zeros((height, width, 3), dtype=np.float32)

    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)

    for ann in sorted_anns:
        m = ann['segmentation']
        # 为每个mask生成随机颜色
        color_mask = np.random.random(3) * 0.6 + 0.4  # 随机生成 RGB 颜色
        combined_img[m] = color_mask

        # 如果需要边框
        if borders:
            contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            # 尝试平滑轮廓并弱化边缘
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            cv2.drawContours(combined_img, contours, -1, (0, 0, 1), thickness=0.5)

            # 保存合成的图像
    cv2.imwrite(output_path, (combined_img * 255).astype(np.uint8))
    
def get_semantic_map_low_iou(image_path, output_path,sam2_model, borders=False):   #generate sam mask for semantic map
    image = Image.open(image_path)
    image = np.array(image.convert("RGB"))
    mask_generator = SAM2AutomaticMaskGenerator(
        model=sam2_model,
        points_per_side=24,
        points_per_batch=32,
        pred_iou_thresh=0.80,
        stability_score_thresh=0.8,
        stability_score_offset=0.8,
        crop_n_layers=0,
        box_nms_thresh=0.85,
        min_mask_region_area=300,
        use_m2m=True,
    )
    # mask_generator = SAM2AutomaticMaskGenerator(
    #     model=sam2_model
    # )
    anns = mask_generator.generate(image)
    height, width = anns[0]['segmentation'].shape[:2]
    combined_img = np.zeros((height, width, 3), dtype=np.float32)

    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)

    for ann in sorted_anns:
        m = ann['segmentation']
        # 为每个mask生成随机颜色
        color_mask = np.random.random(3) * 0.6 + 0.4  # 随机生成 RGB 颜色
        combined_img[m] = color_mask

        # 如果需要边框
        if borders:
            contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            # 尝试平滑轮廓并弱化边缘
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            cv2.drawContours(combined_img, contours, -1, (0, 0, 1), thickness=0.5)

            # 保存合成的图像
    cv2.imwrite(output_path, (combined_img * 255).astype(np.uint8))

def get_semantic_map_human(image_path, output_path,sam2_model, borders=False,):   #generate sam mask for semantic map
    image = Image.open(image_path)
    image = np.array(image.convert("RGB"))
    mask_generator = SAM2AutomaticMaskGenerator(
        model=sam2_model
    )
    anns = mask_generator.generate(image)
    height, width = anns[0]['segmentation'].shape[:2]
    combined_img = np.zeros((height, width, 3), dtype=np.float32)

    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)

    for ann in sorted_anns:
        m = ann['segmentation']
        # 为每个mask生成随机颜色
        color_mask = np.random.random(3) * 0.6 + 0.4  # 随机生成 RGB 颜色
        combined_img[m] = color_mask

        # 如果需要边框
        if borders:
            contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            # 尝试平滑轮廓并弱化边缘
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            cv2.drawContours(combined_img, contours, -1, (0, 0, 1), thickness=0.5)

            # 保存合成的图像
    cv2.imwrite(output_path, (combined_img * 255).astype(np.uint8))

def coco_to_xy(bboxes):
    """
    将 COCO 格式的边界框转换为 (x_min, y_min, x_max, y_max) 格式。

    Args:
        bboxes (numpy.ndarray): COCO 格式的边界框，形状为 (N, 4)，每行为 [x_min, y_min, width, height]。

    Returns:
        numpy.ndarray: 转换后的边界框，形状为 (N, 4)，每行为 [x_min, y_min, x_max, y_max]。
    """
    bboxes = np.array(bboxes)  # 确保输入是 NumPy 数组
    x_min, y_min, width, height = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
    x_max = x_min + width
    y_max = y_min + height
    return np.stack([x_min, y_min, x_max, y_max], axis=1)

def process_all_masks(all_masks):
    """
    将 all_masks 转换为 N 个布尔数组的列表，每个掩码为二维布尔数组。

    Args:
        all_masks (list): 包含 N 个 3D NumPy 数组的列表，每个数组形状为 (1, height, width)。

    Returns:
        list: 包含 N 个布尔数组的列表，每个数组形状为 (height, width)。
    """
    binary_masks = []
    for mask in all_masks:
        # 检查掩码是否为 3D 数组，取第一个通道
        if mask.ndim == 3 and mask.shape[0] == 1:
            mask_2d = mask[0]  # 去掉第一个维度
        else:
            mask_2d = mask  # 如果已经是 2D，直接使用

        # 转换为布尔数组
        binary_mask = mask_2d > 0  # 非零值转换为 True，零值转换为 False
        binary_masks.append(binary_mask)

    return binary_masks




# import json
# image_id = 15000000000006
# image_path = "/media/sata4/Contextaware/MetaData/HumanArt/images/real_human/acrobatics/000000000006.jpg"
# image = cv2.imread(image_path)
# # 将图像数据转为 float16
#
# json_file_path = '/media/sata4/Contextaware/MetaData/HumanArt/annotations/training_humanart_acrobatics.json'
# with open(json_file_path, "r") as file:
#     data = json.load(file)

def get_mask_human(image_path, bbox_list, model, save_path):
    """""
    bbox_list must be float16 NumPy 数组列表
    """
    # annotations = [item for item in data['annotations'] if item["image_id"] == image_id]
    # annotations = sorted(annotations, key=lambda x: x["id"])
    image = cv2.imread(image_path)
    predictor = SAM2ImagePredictor(model)
    predictor.set_image(image)
    # bbox_array = np.array(bbox_list, dtype=np.float16)
    # bbox_list = [np.array(item["bbox"], dtype=np.float16) for item in data['annotations'] if item["image_id"] == image_id]
    all_masks = []
    for idx in range(len(bbox_list)):
        singe_bbox_coco = np.array(bbox_list[idx],)[None, :]
        single_bbox = coco_to_xy(singe_bbox_coco)
        masks, scores, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=single_bbox,  # 转为 float16
            multimask_output=False,
        )
        all_masks.append(masks)

    processed_masks = process_all_masks(all_masks)
    mask_list = visualize_single_mask(image_path, processed_masks, save_path)
    return mask_list

def get_semantic_map_wo_save(image_path, sam2_model, borders=False):   #generate sam mask for semantic map
    image = Image.open(image_path)
    image = np.array(image.convert("RGB"))
    
    # 判断图像尺寸
    image_shape = image.shape
    base_size = max(image_shape[0], image_shape[1])
    
    if base_size > 2000:  # 大图像
        mask_generator = SAM2AutomaticMaskGenerator(
            model=sam2_model,
            points_per_side=24,
            points_per_batch=32,
            pred_iou_thresh=0.93,
            stability_score_thresh=0.96,
            stability_score_offset=0.85,
            crop_n_layers=0,
            box_nms_thresh=0.85,
            min_mask_region_area=300,
            use_m2m=True,
        )  # for large images
    else:  # 小图像
        mask_generator = SAM2AutomaticMaskGenerator(
            model=sam2_model
        )
    
    anns = mask_generator.generate(image)
    height, width = anns[0]['segmentation'].shape[:2]
    combined_img = np.zeros((height, width, 3), dtype=np.float32)

    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)

    for ann in sorted_anns:
        m = ann['segmentation']
        # 为每个mask生成随机颜色
        color_mask = np.random.random(3) * 0.6 + 0.4  # 随机生成 RGB 颜色
        combined_img[m] = color_mask

        # 如果需要边框
        if borders:
            contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            # 尝试平滑轮廓并弱化边缘
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            cv2.drawContours(combined_img, contours, -1, (0, 0, 1), thickness=0.5)

    return (combined_img * 255).astype(np.uint8)

# mask_list = get_mask_human(data,sam2_model,image_id,save_path='/media/sata4/Contextaware/image2condition_model/mask')

# annotations = [item for item in data['annotations'] if item["image_id"] == image_id]
# annotations = sorted(annotations, key=lambda x: x["id"])
#
# predictor = SAM2ImagePredictor(sam2_model)
# predictor.set_image(image)
# bbox_list = [np.array(item["bbox"], dtype=np.float16) for item in data['annotations'] if item["image_id"] == image_id]
# mask_list=get_mask_human(bbox_list, sam2_model, save_path='/media/sata4/Contextaware/image2condition_model/mask/')
# print(mask_list)
# all_masks = []
# for idx in range(len(bbox_list)):
#     singe_bbox_coco = np.array(bbox_list[idx],)[None, :]
#     single_bbox = coco_to_xy(singe_bbox_coco)
#     masks, scores, _ = predictor.predict(
#         point_coords=None,
#         point_labels=None,
#         box=single_bbox,  # 转为 float16
#         multimask_output=False,
#     )
#     all_masks.append(masks)
# processed_masks = process_all_masks(all_masks)
# mask_list = visualize_single_mask(image_path, processed_masks, save_path='/media/sata4/Contextaware/image2condition_model/mask')
# get_semantic_map(image_path, output_path="/media/sata4/Contextaware/image2condition_model/human_semantic_map.png", sam2_model=sam2_model, borders=False)
# import os
# os.environ["TORCH_CUDNN_SDPA_ENABLED"] = "1"
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

# sam2_config = "/media/sata4/Contextaware/Grounded-SAM-2/sam2/configs/sam2.1/sam2.1_hiera_l.yaml"
# sam2_checkpoint = "/media/sata4/Contextaware/Grounded-SAM-2/checkpoints/sam2.1_hiera_large.pt"
# sam2_model = build_sam2_no_hydra(sam2_config, sam2_checkpoint, device=device)

# florence_model_id = "microsoft/Florence-2-large"
# florence_model = AutoModelForCausalLM.from_pretrained(
#     florence_model_id, trust_remote_code=True, torch_dtype='auto'
# ).eval().to(device)
# florence_processor = AutoProcessor.from_pretrained(
#     florence_model_id, trust_remote_code=True,device=device
# )

# image_path = '/media/sata4/Contextaware/MetaData/Subjects200K/data/dataset1_images/image_0_left.png'
# input_text = "A Lounge Chair is placed in a modern city living room"
# bbox, masks, class_ids = get_grounding_bbox_and_single_mask(image_path, input_text,sam2_model,florence_model,florence_processor)
# selected_bbox, selected_mask, selected_class = select_bbox_and_mask(bbox, masks, class_ids, top_n=1)
# visualize_bbox(image_path, selected_bbox, selected_mask, selected_class, save_path="/media/sata4/Contextaware/image2condition_model/chair_black_bounding.jpg")
# # exit(0)
# # selected_bbox, selected_mask = select_bbox_and_mask(bbox, masks, top_n=1)
# # visualize_bbox(image_path, selected_bbox, save_path="/media/sata4/Contextaware/image2condition_model/02bbox.jpg")
# mask_list = visualize_single_mask(image_path, selected_mask, save_path="/media/sata4/Contextaware/image2condition_model/")
# print(mask_list)
# get_semantic_map_human(image_path, output_path="/media/sata4/Contextaware/image2condition_model/chair_semantic_map.png", sam2_model=sam2_model, borders=False)


