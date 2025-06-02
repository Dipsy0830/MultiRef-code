import cv2
import numpy as np
import shutil
from pathlib import Path
from typing import List
import json
import os
import random
import glob

def get_canny(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return edges

def random_crop(image_path, min_crop_ratio=0.3, max_crop_ratio=0.6):
    """
    对输入图像进行随机裁剪，裁剪比例为 [min_crop_ratio, max_crop_ratio] 内的随机值。

    Args:
        image (numpy.ndarray): 输入图像 (H, W, C) 或 (H, W) 格式。
        min_crop_ratio (float): 最小裁剪比例。
        max_crop_ratio (float): 最大裁剪比例。

    Returns:
        cropped_image (numpy.ndarray): 裁剪后的图像。
    """

    assert 0 < min_crop_ratio <= max_crop_ratio <= 1, "裁剪比例应在 (0, 1] 范围内，并且 min_crop_ratio <= max_crop_ratio"
    image = cv2.imread(image_path)
    h, w = image.shape[:2]

    # 随机生成裁剪比例
    crop_ratio = np.random.uniform(min_crop_ratio, max_crop_ratio)

    # 计算随机裁剪的尺寸
    crop_h = int(h * crop_ratio)
    crop_w = int(w * crop_ratio)

    # 随机选择裁剪起始位置
    start_y = np.random.randint(0, h - crop_h + 1)
    start_x = np.random.randint(0, w - crop_w + 1)

    # 裁剪图像
    cropped_image = image[start_y:start_y + crop_h, start_x:start_x + crop_w]

    return cropped_image

def generate_low_resolution_image(image_path):
    """
    Generate a low-resolution version of the input image.

    Parameters:
    - image_path (str): Path to the input image.
    - scale_factor (float): Factor by which to downscale the image.

    Returns:
    - low_res_image (numpy.ndarray): The downscaled, low-resolution version of the image.
    """
    # 读取输入图像
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image at path '{image_path}' not found.")

    # 获取原图像的尺寸
    original_height, original_width = image.shape[:2]
    # Randomize scale_factor
    scale_factor = random.uniform(0.1, 0.5)  # Random scale factor between 0.1 and 0.5

    # 计算新的尺寸
    new_width = int(original_width * scale_factor)
    new_height = int(original_height * scale_factor)

    # 调整图像大小到低分辨率
    low_res_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    return low_res_image

def generate_low_light_image(image_path):
    """
    Generate a low-light version of the input image.

    Parameters:
    - image_path (str): Path to the input image.
    - alpha (float): Factor by which to reduce the brightness.

    Returns:
    - low_light_image (numpy.ndarray): The darkened version of the image.
    """
    # 读取输入图像
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image at path '{image_path}' not found.")

    alpha = random.uniform(0.1, 0.5)  # Random brightness reduction factor between 0.1 and 0.5

    # 将图像亮度调低
    low_light_image = cv2.convertScaleAbs(image, alpha=alpha, beta=0)

    return low_light_image

def generate_low_res_low_light(image_path):   # super_res + light_enhance
    """
        Generate a low-resolution, low-light version of the input image with random scale factor and alpha.

        Parameters:
        - image_path (str): Path to the input image.

        Returns:
        - processed_image (numpy.ndarray): The processed image with low resolution and reduced brightness.
        """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image at path '{image_path}' not found.")

    # Randomize scale_factor and alpha
    scale_factor = random.uniform(0.1, 0.5)  # Random scale factor between 0.1 and 0.5
    alpha = random.uniform(0.1, 0.5)  # Random brightness reduction factor between 0.1 and 0.5

    # Get original image dimensions
    original_height, original_width = image.shape[:2]

    # Compute new dimensions for low resolution
    new_width = int(original_width * scale_factor)
    new_height = int(original_height * scale_factor)

    # Resize the image to low resolution
    low_res_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    # Reduce the brightness of the low-resolution image
    low_res_low_light_image = cv2.convertScaleAbs(low_res_image, alpha=alpha, beta=0)

    return low_res_low_light_image

def generate_low_res_crop_image(image_path):   #super_res + extrapolation
    """
        Generate a low-resolution, cropped version of the input image with random scale factor and crop ratio.

        Parameters:
        - image_path (str): Path to the input image.

        Returns:
        - cropped_image (numpy.ndarray): The processed image with low resolution and random cropping.
        """
    # Read the input image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image at path '{image_path}' not found.")

    # Randomize scale_factor
    scale_factor = random.uniform(0.1, 0.5)

    # Get original image dimensions
    original_height, original_width = image.shape[:2]

    # Compute new dimensions for low resolution
    new_width = int(original_width * scale_factor)
    new_height = int(original_height * scale_factor)

    # Resize the image to low resolution
    low_res_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    # Randomize crop ratio
    min_crop_ratio = 0.3
    max_crop_ratio = 0.6
    crop_ratio = random.uniform(min_crop_ratio, max_crop_ratio)

    # Compute crop dimensions
    crop_h = int(new_height * crop_ratio)
    crop_w = int(new_width * crop_ratio)

    # Randomize crop start position
    start_y = random.randint(0, new_height - crop_h)
    start_x = random.randint(0, new_width - crop_w)

    # Crop the image
    low_res_crop_image = low_res_image[start_y:start_y + crop_h, start_x:start_x + crop_w]

    return low_res_crop_image

def generate_low_light_crop_image(image_path):   #light_enhance + extrapolation
    """
        Generate a low-resolution, cropped version of the input image with random scale factor and crop ratio.

        Parameters:
        - image_path (str): Path to the input image.

        Returns:
        - cropped_image (numpy.ndarray): The processed image with low resolution and random cropping.
        """
    # Read the input image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image at path '{image_path}' not found.")

    # Randomize scale_factor
    scale_factor = random.uniform(0.1, 0.5)

    # Get original image dimensions
    original_height, original_width = image.shape[:2]

    # Compute new dimensions for low resolution
    new_width = int(original_width * scale_factor)
    new_height = int(original_height * scale_factor)

    # Resize the image to low resolution
    low_res_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    # Randomize crop ratio
    min_crop_ratio = 0.3
    max_crop_ratio = 0.6
    crop_ratio = random.uniform(min_crop_ratio, max_crop_ratio)

    # Compute crop dimensions
    crop_h = int(new_height * crop_ratio)
    crop_w = int(new_width * crop_ratio)

    # Randomize crop start position
    start_y = random.randint(0, new_height - crop_h)
    start_x = random.randint(0, new_width - crop_w)

    # Crop the image
    low_light_crop_image = low_res_image[start_y:start_y + crop_h, start_x:start_x + crop_w]

    return low_light_crop_image

def get_subject_db(image_path):
    """
        Get list of all jpg files in the same directory.

        Args:
            image_path (str): Path to the original image (e.g., "backpack_dog/00.jpg")

        Returns:
            list: List of jpg file paths in the same directory
        """
    # Get the directory containing the image
    source_dir = Path(image_path).parent
    input_filename = Path(image_path).name  # 原图文件名
    subject_paths = []

    for f in sorted(os.listdir(source_dir)):
        # 拼出完整路径
        file_path = source_dir / f  
        
        # 1) 确保是文件而非文件夹
        if not file_path.is_file():
            continue

        # Define a list of image extensions
        image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff']

        # 2) 统一转小写处理，避免后缀或文件名大小写差异导致的比较失效
        if file_path.suffix.lower() in image_extensions and f.lower() != input_filename.lower():
            subject_paths.append(str(file_path))
    
    return subject_paths

import os
from pathlib import Path

def get_subject_200k(image_path):
    """
    Get related image files with matching base names from the same directory.
    E.g.: For 'image_5_left.png' finds and returns 'image_5_right.png'
    
    Args:
        image_path (str): Path to the original image
        
    Returns:
        list: List of related image paths with matching base names
    """
    source_dir = str(Path(image_path).parent)
    input_filename = Path(image_path).name
    
    # Get base name without _left/_right suffix (split by "_")
    base_name_parts = input_filename.split('_')[:2]  # Example: ['image', '2429']
    base_name = '_'.join(base_name_parts)
    
    # Find all matching files in the directory
    subject_paths = [
        os.path.join(source_dir, f)
        for f in sorted(os.listdir(source_dir))
        if f != input_filename  # Avoid the same file as the input
        and f.endswith(('.png', '.jpg', '.jpeg'))  # Ensure it's an image file
        and '_'.join(f.split('_')[:2]) == base_name  # Ensure matching base name
    ]
    
    return subject_paths

def get_subject_tryon(image_path):
    subject_path = glob.glob(os.path.join('/media/sata4/Contextaware/MetaData/fashion_tryon/VITON-HD/train/cloth', '*.jpg'))
    subject_path = [p for p in subject_path if os.path.basename(p) == os.path.basename(image_path)]
    subject_path = subject_path[0] if subject_path else None
    return subject_path

def get_pose_tryon(image_path):
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    # Look for rendered pose file
    pose_path = glob.glob(os.path.join('/media/sata4/Contextaware/MetaData/fashion_tryon/VITON-HD/train/openpose_img', f'{base_name}_rendered.png'))
    pose_path = pose_path[0] if pose_path else None
    return pose_path

def get_style_wikiart(image_path: str, N: int = 3) -> List[str]:
    """
    Find N images with the same style as the input image.
    
    Args:
        image_path: Absolute path to the target image
        N: Number of similar style images to return (default: 3)
        
    Returns:
        List of absolute paths to images with the same style
    """
    # Get the root directory (WikiArt_post) from image path
    root_dir = str(Path(image_path).parent.parent)  # Go up two levels: from image to images to WikiArt_post
    image_name = os.path.basename(image_path)
    base_dir = str(Path(image_path).parent)   #images directory
    json_path = os.path.join(root_dir, 'data.json')
    with open(json_path, 'r') as f:
        data = json.load(f)
        target_style = None
    for item in data:
        if item['path'] == image_name:
            target_style = item['style']
            break
            
    if target_style is None:
        return []
    similar_images = []
    for item in data:
        if item['style'] == target_style and item['path'] != image_name:
            # Ensure absolute path
            full_path = os.path.abspath(os.path.join(root_dir, 'images', item['path']))
            similar_images.append(full_path)
    
    # Randomly select N images
    if len(similar_images) <= N:
        return similar_images
    else:
        return random.sample(similar_images, N)

def get_style_stylebooth(image_path: str, N: int = 3) -> List[str]:
    """
        Get list of all jpg files in the same directory.

        Args:
            image_path (str): Path to the original image (e.g., "backpack_dog/00.jpg")

        Returns:
            list: List of jpg file paths in the same directory
        """
    # Get the directory containing the image
    source_dir = Path(image_path).parent
    input_filename = Path(image_path).name  # 原图文件名
    style_paths = []

    for f in sorted(os.listdir(source_dir)):
        # 拼出完整路径
        file_path = source_dir / f  
        
        # 1) 确保是文件而非文件夹
        if not file_path.is_file():
            continue

        # Define a list of image extensions
        image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff']

        # 2) 统一转小写处理，避免后缀或文件名大小写差异导致的比较失效
        if file_path.suffix.lower() in image_extensions and f.lower() != input_filename.lower():
            style_paths.append(str(file_path))
    
    return random.sample(style_paths, N)

def get_style_2d_human(image_path, N=3):
    """
        Get N random jpg files in the same directory. default N=3.

        Args:
            image_path (str): Path to the original image (e.g., "backpack_dog/00.jpg")
            N (int): Number of random jpg files to retrieve

        Returns:
            list: List of N random jpg file paths in the same directory
        """
    # Get the directory containing the image
    source_dir = str(Path(image_path).parent)
    input_filename = Path(image_path).name
    style_paths = [
        os.path.join(source_dir, f)
        for f in sorted(os.listdir(source_dir))
        if f.lower().endswith('.jpg') and f != input_filename
    ]

    # Get N random jpg files
    random_style_paths = random.sample(style_paths, min(N, len(style_paths)))

    return random_style_paths

def add_joints(image, joints, color, dataset='COCO'):
    import math
    import cv2
    coco_part_labels = [
        'nose', 'eye_l', 'eye_r', 'ear_l', 'ear_r',
        'sho_l', 'sho_r', 'elb_l', 'elb_r', 'wri_l', 'wri_r',
        'hip_l', 'hip_r', 'kne_l', 'kne_r', 'ank_l', 'ank_r'
    ]
    coco_part_idx = {
        b: a for a, b in enumerate(coco_part_labels)
    }
    coco_part_orders = [
        ('nose', 'eye_l'), ('eye_l', 'eye_r'), ('eye_r', 'nose'),
        ('eye_l', 'ear_l'), ('eye_r', 'ear_r'), ('ear_l', 'sho_l'),
        ('ear_r', 'sho_r'), ('sho_l', 'sho_r'), ('sho_l', 'hip_l'),
        ('sho_r', 'hip_r'), ('hip_l', 'hip_r'), ('sho_l', 'elb_l'),
        ('elb_l', 'wri_l'), ('sho_r', 'elb_r'), ('elb_r', 'wri_r'),
        ('hip_l', 'kne_l'), ('kne_l', 'ank_l'), ('hip_r', 'kne_r'),
        ('kne_r', 'ank_r')
    ]

    crowd_pose_part_labels = [
        'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
        'left_knee', 'right_knee', 'left_ankle', 'right_ankle',
        'head', 'neck'
    ]
    crowd_pose_part_idx = {
        b: a for a, b in enumerate(crowd_pose_part_labels)
    }
    crowd_pose_part_orders = [
        ('head', 'neck'), ('neck', 'left_shoulder'), ('neck', 'right_shoulder'),
        ('left_shoulder', 'right_shoulder'), ('left_shoulder', 'left_hip'),
        ('right_shoulder', 'right_hip'), ('left_hip', 'right_hip'), ('left_shoulder', 'left_elbow'),
        ('left_elbow', 'left_wrist'), ('right_shoulder', 'right_elbow'), ('right_elbow', 'right_wrist'),
        ('left_hip', 'left_knee'), ('left_knee', 'left_ankle'), ('right_hip', 'right_knee'),
        ('right_knee', 'right_ankle')
    ]

    VIS_CONFIG = {
        'COCO': {
            'part_labels': coco_part_labels,
            'part_idx': coco_part_idx,
            'part_orders': coco_part_orders
        },
        'CROWDPOSE': {
            'part_labels': crowd_pose_part_labels,
            'part_idx': crowd_pose_part_idx,
            'part_orders': crowd_pose_part_orders
        }
    }
    part_idx = VIS_CONFIG[dataset]['part_idx']
    part_orders = VIS_CONFIG[dataset]['part_orders']

    base_size = max(image.shape[0], image.shape[1])
    if base_size > 2000:  # 大图像
        link_thickness = int(base_size * 0.004)  # 适度粗一点
        joint_radius = int(base_size * 0.009)  # 更大的关键点半径
    elif base_size > 1000:  # 中等图像
        link_thickness = max(2, int(base_size * 0.0008))
        joint_radius = max(3, int(base_size * 0.0015))
    else:  # 小图像
        link_thickness = max(1, int(base_size * 0.0005))
        joint_radius = max(2, int(base_size * 0.001))

    def link(a, b, color):
        if part_idx[a] < joints.shape[0] and part_idx[b] < joints.shape[0]:
            jointa = joints[part_idx[a]]
            jointb = joints[part_idx[b]]
            if jointa[2] > 0 and jointb[2] > 0:
                cv2.line(
                    image,
                    (int(jointa[0]), int(jointa[1])),
                    (int(jointb[0]), int(jointb[1])),
                    color,
                    link_thickness
                )

    # add joints
    for joint in joints:
        if joint[2] > 0:
            cv2.circle(image, (int(joint[0]), int(joint[1])), 1, color, joint_radius)

    # add link
    for pair in part_orders:
        link(pair[0], pair[1], color)

    return image

def get_shape(image_path):
    from PIL import Image
    image = Image.open(image_path)
    image = np.array(image.convert("RGB"))
    # print(image.shape)
    return image.shape

def save_pose_image_resize(image, image_path, joints, file_name, dataset='COCO'):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    black_image = np.zeros_like(image)

    for person in joints:
        color = np.random.randint(128, 256, size=3)
        color = [int(i) for i in color]
        add_joints(black_image, person, color, dataset=dataset)

    # 调整图像大小并保存
    result_image = black_image
    tgt_shape = get_shape(image_path)
    resized_image=cv2.resize(result_image, (tgt_shape[1], tgt_shape[0]))
    cv2.imwrite(file_name, resized_image)

def visualize_bbox(image_path, bbox, save_path):
    """
    在黑色背景上可视化 bbox in COCO xywh

    Args:
        image: 输入图像 (H, W, C) 的 NumPy 数组
        bbox: COCO 格式的边界框 [x_min, y_min, width, height]
        color: 矩形框的颜色 (默认绿色，格式为 BGR)
        thickness: 矩形框的线条粗细 (默认 2)
    """
    # 获取原图形状
    image_shape = get_shape(image_path)  # 替换为实际的获取形状逻辑
    # 创建黑色背景
    black_image = np.zeros(image_shape, dtype=np.uint8)
    # 从 COCO 格式转换为 [x_min, y_min, x_max, y_max]
    x_min, y_min, width, height = bbox
    x_max, y_max = x_min + width, y_min + height
    color = np.random.randint(128, 256, size=3)
    color = [int(i) for i in color]

    base_size = max(image_shape[0], image_shape[1])
    if base_size > 2000:  # 大图像
        link_thickness = int(base_size * 0.006)  # 适度粗一点
    elif base_size > 1000:  # 中等图像
        link_thickness = max(2, int(base_size * 0.0008))
    else:  # 小图像
        link_thickness = max(1, int(base_size * 0.0005))

    cv2.rectangle(black_image,  (int(x_min), int(y_min)), (int(x_max), int(y_max)), color, link_thickness)
    cv2.imwrite(save_path, black_image)

def get_keypoints_and_bbox_human(keypoints_2d_list, bbox_list, image_path, pose_path, bbox_path):
    image = cv2.imread(image_path)
    save_basename = os.path.basename(image_path)
    save_basename = os.path.splitext(save_basename)[0]
    pose_image_path_list = []
    bbox_image_path_list = []

    for idx, (keypoints, bbox) in enumerate(zip(keypoints_2d_list, bbox_list)):
        keypoints = np.array(keypoints).reshape(-1, 3)
        pose_name = pose_path.split('.')[0]
        bbox_name = bbox_path.split('.')[0]

        current_pose_path = f"{pose_name}_{idx + 1}.jpg"
        current_bbox_path = f"{bbox_name}_{idx + 1}.jpg"
        save_pose_image_resize(image, image_path, [keypoints], current_pose_path, dataset='COCO')
        visualize_bbox(image_path, bbox, current_bbox_path)
        pose_image_path_list.append(current_pose_path)
        bbox_image_path_list.append(current_bbox_path)
    return pose_image_path_list, bbox_image_path_list

# def generate_pose_and_bbox_human(ori_image_path, base_path, keypoints_2d_list, bbox_list):
#     pose_path = _get_condition_path(ori_image_path, base_path, 'pose')
#     bbox_path = _get_condition_path(ori_image_path, base_path, 'bbox')
#     pose_path_list, bbox_path_list = get_keypoints_and_bbox_human(keypoints_2d_list, bbox_list, ori_image_path, pose_path, bbox_path)
#     return pose_path_list, bbox_path_list

# import json
# image_id = 15000000000006
# image_path = "/media/sata4/Contextaware/MetaData/HumanArt/images/real_human/acrobatics/000000000006.jpg"
# image = cv2.imread(image_path)
# json_file_path = '/media/sata4/Contextaware/MetaData/HumanArt/annotations/training_humanart_acrobatics.json'
# with open(json_file_path, "r") as file:
#     data = json.load(file)
# annotations = [item for item in data['annotations'] if item["image_id"] == image_id]
# annotations = sorted(annotations, key=lambda x: x["id"])
# keypoints_list = [item["keypoints"] for item in data['annotations'] if item["image_id"] == image_id]
# bbox_list = [item["bbox"] for item in data['annotations'] if item["image_id"] == image_id]
# keypoints_2d_list = [
#     np.array(keypoints).reshape(-1, 3).tolist() for keypoints in keypoints_list
# ]
# assert len(keypoints_2d_list) == len(bbox_list), "Mismatch between keypoints and bbox lengths!"
# # print(keypoints_2d_list)
#
# pose_path_list, bbox_path_list = generate_pose_and_bbox_human(image_path, "/media/sata4/Contextaware/image2condition_model/test", keypoints_2d_list, bbox_list)
# print(pose_path_list)
# print(bbox_path_list)
# pose_path_list, bbox_path_list = get_keypoints_and_bbox_human(keypoints_2d_list, bbox_list, image_path, "/media/sata4/Contextaware/image2condition_model/pose", "/media/sata4/Contextaware/image2condition_model/bbox")
# print(pose_path_list)
# print(bbox_path_list)

# 调用pose并保存
# image_path = "/media/sata4/Contextaware/MetaData/HumanArt/images/real_human/acrobatics/000000000011.jpg"
# pose_dict = np.load("/media/sata4/Contextaware/MetaData/HumanArt/pose/real_human/acrobatics/000000000011.npz", allow_pickle=True)
# pose = pose_dict['arr_0'][0]['keypoints']   #[17,3]
# image = cv2.imread(image_path)
# save_pose_image_resize(image, image_path, [pose], "/media/sata4/Contextaware/image2condition_model/pose0011.jpg", dataset='COCO')




# image_path = '/media/sata4/Contextaware/MetaData/dreambooth/dataset/backpack/01.jpg' # 加载为 (H, W, C) 格式
# # condition_dir = '/media/sata4/Contextaware/Condition_set/subject'
# path_list = get_subject_db(image_path)
# print(path_list)
# # 随机裁剪
# cropped_image = random_crop(image_path)
# cv2.imwrite("/media/sata4/Contextaware/image2condition_model/random_crop.jpg", cropped_image)

# import json
# image_path = '/media/sata4/Contextaware/MetaData/WikiArt_post/images/image_0.jpg'
# style_list = get_style_wikiart(image_path, N=3)
# print(style_list)

