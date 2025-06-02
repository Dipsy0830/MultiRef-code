import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import inception_v3
from torchmetrics.multimodal import CLIPImageQualityAssessment
import numpy as np
import cv2
from PIL import Image
from functools import partial
from transformers import CLIPProcessor, CLIPModel, AutoImageProcessor, AutoModel
from aesthetics_predictor import AestheticsPredictorV1
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr
from lpips import LPIPS
from brisque import BRISQUE
from functools import partial
from torchmetrics.multimodal.clip_score import CLIPScore  
# Add this import at the top of your file


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def convert_to_label_map(image):
    """
    Convert RGB image to label map where each unique color gets a unique label.
    """
    # Flatten the image to get unique colors
    pixels = image.reshape(-1, image.shape[-1])
    unique_colors = np.unique(pixels, axis=0)
    
    # Create label map
    label_map = np.zeros(image.shape[:2], dtype=np.int32)
    for i, color in enumerate(unique_colors):
        mask = np.all(image == color, axis=-1)
        label_map[mask] = i + 1
    
    return label_map

def convert_to_bbx(image):
    """
    Convert CV2 image to single bounding box array.
    输入的图像是全黑的底色，上面有一个非黑色边框
    
    Args:
        image: numpy array, OpenCV格式的图像 (H, W) 或 (H, W, C)
        
    Returns:
        List[List]: 格式为 [[x1, y1, x2, y2]] 的bbox坐标
    """
     # 如果是彩色图，先转换为灰度图
    if len(image.shape) == 3:
        # 对于彩色线条，使用最大值而不是平均值来保留线条信息
        channels = cv2.split(image)
        gray = np.maximum.reduce(channels)
    else:
        gray = image
    
    # 使用更低的阈值进行二值化，对于细线条可能需要更低的值
    # _, binary = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    
    # 或者尝试自适应阈值
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY, 11, 2)
    
    # 可选：形态学操作来增强线条
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.dilate(binary, kernel, iterations=1)
    
    # 找到所有非零点的坐标
    y_coords, x_coords = np.nonzero(binary)
    
    if len(x_coords) == 0 or len(y_coords) == 0:
        return None
    
    # 获取边界框坐标
    x1 = float(np.min(x_coords))
    y1 = float(np.min(y_coords))
    x2 = float(np.max(x_coords))
    y2 = float(np.max(y_coords))
    
    # 返回格式化的bbox
    bbox = [[x1, y1, x2, y2]]
    return bbox

def convert_to_mask(image):
    """
    Convert CV2 image to binary mask array (0-1).
    输入的图像是全黑的底色，上面有白色的mask
    
    Args:
        image: numpy array, OpenCV 格式的图像 (H, W) 或 (H, W, C)
        
    Returns:
        numpy.ndarray: 二值mask数组，0表示背景，1表示前景
    """
    # 确保图像是灰度格式
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # 二值化处理
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    
    # 转换为0-1 mask
    mask = (binary > 0).astype(np.uint8)
    
    return mask

def compute_pixel_mAP(image_path_1, image_path_2, threshold=0):
    """
    读取两张图，提取其中"不是彩色的像素点"（或"非黑色像素"），
    以第一张图为"真值"，第二张图为"预测"，
    在像素级别计算一个简单的单阈值 AP（precision * recall）。
    
    参数：
        image_path_1 (str): 第1张图片路径（视作真值）。
        image_path_2 (str): 第2张图片路径（视作预测）。
        threshold (int): 判断是否为"非黑色像素"的阈值。
                         当像素在BGR三通道之和 > threshold 时，视为前景像素。
    
    返回：
        ap (float): 单阈值下的"平均精度"近似值，计算方式为 precision * recall。
    """
    # 读取图像（彩色，BGR格式）
    img1 = cv2.imread(image_path_1)
    img2 = cv2.imread(image_path_2)
    
    # 如果图像读取失败，做一下简单检查
    if img1 is None:
        raise ValueError(f"无法读取图像: {image_path_1}")
    if img2 is None:
        raise ValueError(f"无法读取图像: {image_path_2}")
    
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    if (h1 != h2) or (w1 != w2):
        # 将 img2 调整为和 img1 一样大小，这里使用最近邻插值
        img2 = cv2.resize(img2, (w1, h1), interpolation=cv2.INTER_NEAREST)
    
    # 生成二值掩码：若像素(B+G+R)之和 > threshold，则视为"前景"像素
    mask1 = (np.sum(img1, axis=2) > threshold)
    mask2 = (np.sum(img2, axis=2) > threshold)
    
    # 计算混淆矩阵
    # TP：真值是前景，预测也是前景
    TP = np.sum(mask1 & mask2)
    # FP：真值是背景，预测是前景
    FP = np.sum(~mask1 & mask2)
    # FN：真值是前景，预测是背景
    FN = np.sum(mask1 & ~mask2)
    
    # 计算precision和recall
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    
    # 这里简单地把单阈值时的AP定义为 precision * recall
    # 仅作示例
    ap = precision * recall
    
    return ap

def calculate_IoU(matrix1, matrix2):
    """
    Calculate the Intersection-over-Union (IoU) between two binary matrices.
    
    Args:
        matrix1 (numpy.ndarray): The first binary matrix.
        matrix2 (numpy.ndarray): The second binary matrix.
    
    Returns:
        float: IoU value between 0 and 1.
    """
    # Ensure the matrices have the same shape
    if matrix1.shape != matrix2.shape:
        
        raise ValueError("Both matrices must have the same shape.")
    
    # Calculate intersection and union
    intersection = np.logical_and(matrix1, matrix2).sum()
    union = np.logical_or(matrix1, matrix2).sum()
    
    # Avoid division by zero
    if union == 0:
        return 0.0
    
    # Compute IoU
    iou = intersection / union
    return iou

def calculate_bbox_iou(bbox1, bbox2):
    """
    Calculate the Intersection-over-Union (IoU) between two bounding boxes.

    Args:
        bbox1 (list or numpy.ndarray): The first bounding box in [x1, y1, x2, y2] format.
        bbox2 (list or numpy.ndarray): The second bounding box in [x1, y1, x2, y2] format.

    Returns:
        float: IoU value between 0 and 1.
    """
    # Ensure bbox1 and bbox2 are in numpy format
    bbox1 = np.array(bbox1)
    bbox2 = np.array(bbox2)

    # Calculate intersection coordinates
    x_left = max(bbox1[0], bbox2[0])
    y_top = max(bbox1[1], bbox2[1])
    x_right = min(bbox1[2], bbox2[2])
    y_bottom = min(bbox1[3], bbox2[3])

    # Calculate the area of intersection rectangle
    if x_right < x_left or y_bottom < y_top:
        intersection_area = 0.0
    else:
        intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # Calculate the area of both bounding boxes
    bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

    # Calculate the union area
    union_area = bbox1_area + bbox2_area - intersection_area

    # Avoid division by zero
    if union_area == 0:
        return 0.0

    # Compute IoU
    iou = intersection_area / union_area
    return iou

def calculate_MSE(image1, image2):   # canny, map, sketch, depth
    """
    Calculate the Mean Squared Error (MSE) between two images.
     Args:
        image1 (numpy.ndarray): single condition of generated image
        image2 (numpy.ndarray): condition image given in the query (ref image)
        Ensure the pixel values are in the range [0, 1].
    
    Returns:
        float: The MSE value between the two images.
    """
     # Normalize pixel values to [0, 1] if they are in the range [0, 255]
    if image1.max() > 1:
        image1 = image1 / 255.0
    if image2.max() > 1:
        image2 = image2 / 255.0
        
    # Resize the second image if dimensions differ
    if image1.shape != image2.shape:
        image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]), interpolation=cv2.INTER_AREA)
    
    # Calculate MSE
    mse = np.mean((image1 - image2) ** 2)
    return mse   # Lower MSE values indicate higher similarity.

def calculate_FID(image_path):
    """
    Calculate Fréchet Inception Distance for a single image
    
    Args:
        image_path (str): Path to the image file
    
    Returns:
        float: FID score
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load pretrained InceptionV3
    model = inception_v3(pretrained=True, transform_input=False)
    model.fc = nn.Identity()  # Remove final FC layer
    model.eval()
    model.to(device)
    
    # Image preprocessing
    preprocess = transforms.Compose([
        transforms.Resize(299),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Load and preprocess image
    img = Image.open(image_path).convert('RGB')
    img = preprocess(img).unsqueeze(0).to(device)
    
    # Get features
    with torch.no_grad():
        features = model(img).squeeze().float().cpu().numpy()
    
    # Calculate statistics (mean and covariance)
    mu = np.mean(features)
    sigma = np.cov(features.reshape(1, -1))
    
    # FID score (compared to zero mean and identity covariance as reference)
    ref_mu = np.zeros_like(mu)
    ref_sigma = np.eye(len(features))
    
    # Calculate FID
    diff = mu - ref_mu
    covmean = (sigma + ref_sigma) / 2
    # Use multiplication instead of dot for scalars
    fid = diff * diff + np.trace(sigma + ref_sigma - 2*covmean)
    fid_score = float(fid)
    return fid_score


def calculate_aesthetics_score(image_path: str) -> float:
    """
    Calculate aesthetics score for an image from path
    
    Args:
        image_path (str): Path to the input image
        
    Returns:
        float: Aesthetics score
    """
    model_id = "shunk031/aesthetics-predictor-v1-vit-large-patch14"
    predictor = AestheticsPredictorV1.from_pretrained(model_id)
    processor = CLIPProcessor.from_pretrained(model_id)
    # Open and convert image to RGB
    image = Image.open(image_path).convert('RGB')
    inputs = processor(images=image, return_tensors="pt")
    device = "cuda"
    predictor = predictor.to(device)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad(): # or `torch.inference_model` in torch 1.9+
        outputs = predictor(**inputs)
    prediction = outputs.logits.cpu().float().item()
    
    return prediction

def calculate_ssim_psnr_lpips(ref_image_path, gen_image_path):
    """
    Calculate SSIM between original and generated images compared to reference image
    SSIM, PSNR, and LPIPS are metrics that require a reference image to calculate the similarity between two images
    
    Args:
        ref_image_path (str): Path to the input image (image in the prompt)
        gen_image_path (str): Path to the generated image
        
    Returns:
        float: SSIM, PSNR, and LPIPS
    """
     # Load LPIPS model
    lpips_model = LPIPS(net='alex').eval()
    # Load images
    ref_image = Image.open(ref_image_path).convert("RGB")
    gen_image = Image.open(gen_image_path).convert("RGB")
    
    # Convert images to NumPy arrays for shape comparison
    ref_image_np = np.array(ref_image)
    gen_image_np = np.array(gen_image)
    
    # Resize if shapes are different
    if ref_image_np.shape != gen_image_np.shape:
        gen_image = gen_image.resize(ref_image.size, Image.Resampling.LANCZOS)
        gen_image_np = np.array(gen_image)
    
    # Dynamically determine the `win_size` for SSIM
    min_dim = min(ref_image_np.shape[:2])  # Smaller of height and width
    win_size = min(7, min_dim if min_dim % 2 != 0 else min_dim - 1)  # Ensure `win_size` is odd and <= min_dim

    # SSIM calculations; higher, better
    # Measures perceptual similarity between two images based on luminance, contrast, and structure.
    ssim_gen, _ = ssim(gen_image_np, ref_image_np, full=True, channel_axis=-1, win_size=win_size)
    
    # PSNR calculations; Higher PSNR indicates higher similarity
    # Measures the similarity based on the difference in pixel intensities
    psnr_gen = psnr(ref_image_np, gen_image_np, data_range=255)
    
    # LPIPS calculations (convert images to tensors) Lower LPIPS values indicate higher similarity.
    # Measures the perceptual difference between images
    ref_image_tensor = torch.tensor(np.array(ref_image).transpose(2, 0, 1)).unsqueeze(0).float() / 255.0
    gen_image_tensor = torch.tensor(np.array(gen_image).transpose(2, 0, 1)).unsqueeze(0).float() / 255.0
    
    lpips_gen = lpips_model(ref_image_tensor, gen_image_tensor).item()
    
    return ssim_gen, psnr_gen, lpips_gen

def calculate_CLIPIQA(image_path):   # for super resolution
    # Open and convert image to RGB
    img = Image.open(image_path).convert('RGB')
    
    # Convert PIL image to numpy array
    img_tensor = torch.tensor(np.array(img)).permute(2, 0, 1).unsqueeze(0).float() / 255.0

    metric = CLIPImageQualityAssessment(model_name_or_path='clip_iqa')
    return metric(img_tensor).item()

def calculate_brisque(image_path):
    """
    Calculate BRISQUE (Blind/Referenceless Image Spatial Quality Evaluator) score
    
    Args:
        image_path (str): Path to the input image
        from https://github.com/rehanguha/brisque
    Returns:
        float: BRISQUE score (lower is better, typically 0-100)
    """
    img = Image.open(image_path)
    ndarray = np.asarray(img)

    obj = BRISQUE(url=False)
    brisque = obj.score(img=ndarray)
    return brisque
        

def calculate_image_text_clip_score(image_path: str, prompts) -> float:   # image-text similarity   
    """
    Calculate CLIP score for an image from path
    
    Args:
        image_path (str): Path to the input image
        prompts: Text prompts to calculate CLIP score against
        
    Returns:
        float: Rounded CLIP score
    """
    image = Image.open(image_path).convert('RGB')
    
    # Use standard image preprocessing for CLIP
    preprocess = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])
    
    # Convert image to tensor with proper shape
    img_tensor = preprocess(image).unsqueeze(0)  # Add batch dimension
    
    # Create CLIP scorer and calculate score
    clip_scorer = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16")
    return clip_scorer(img_tensor, prompts).item()


def calculate_image_clip_score(image_path1, image_path2):  #subject, style
    """
    Calculate CLIP score between two images using cosine similarity
    
    Args:
        image_path1 (str): Path to first image
        image_path2 (str): Path to second image
        
    Returns:
        float: CLIP similarity score between the images
    """
  
    # Load CLIP model
    model_name = "openai/clip-vit-base-patch16"  # Pretrained CLIP model
    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)
    model.eval()
    
    def preprocess_image(image_path):
        image = Image.open(image_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")
        return inputs["pixel_values"]

    image1_tensor = preprocess_image(image_path1)
    image2_tensor = preprocess_image(image_path2)
    
    with torch.no_grad():
        image1_features = model.get_image_features(image1_tensor)
        image2_features = model.get_image_features(image2_tensor)

    # Normalize embeddings
    image1_features = image1_features / image1_features.norm(dim=-1, keepdim=True)
    image2_features = image2_features / image2_features.norm(dim=-1, keepdim=True)
    
    clip_score = torch.nn.functional.cosine_similarity(image1_features, image2_features).item()
    
    return clip_score

def calculate_dino_score(image_path1, image_path2):
    """
    Calculate the similarity score between two images using the DINOv2 model.
    
    :param image_path1: Path to the first image.
    :param image_path2: Path to the second image.
    :param model_name: Name of the pre-trained model to use. Default is 'facebook/dinov2-base'.
    :param device: Device to use for computation ('cuda' or 'cpu'). Default is 'cuda'.
    :return: Similarity score (float) between the two images.
    """
    # Load model and processor
    model_name='facebook/dinov2-base'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    # Define a helper function for processing and extracting features
    def extract_features(image_path):
        image = Image.open(image_path).convert("RGB")
        with torch.no_grad():
            inputs = processor(images=image, return_tensors="pt").to(device)
            outputs = model(**inputs)
            features = outputs.last_hidden_state.mean(dim=1)  # Global average pooling
        return features

    # Extract features for both images
    image_features1 = extract_features(image_path1)
    image_features2 = extract_features(image_path2)

    # Compute cosine similarity
    cos = nn.CosineSimilarity(dim=1)
    similarity = cos(image_features1, image_features2).item()
    
    # Normalize the similarity score to [0, 1]
    normalized_similarity = (similarity + 1) / 2

    return normalized_similarity


