import sys

import cv2
import torch
import numpy as np



# raw_img = cv2.imread('/media/sata4/Contextaware/image2condition_model/000000000002.jpg')
# depth = model.infer_image(raw_img) # HxW raw depth map in numpy
# depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
# depth = depth.astype(np.uint8)
# cv2.imwrite('/media/sata4/Contextaware/image2condition_model/depth.jpg', depth)
def get_depth(img_path,model):
    raw_img = cv2.imread(img_path)
    depth = model.infer_image(raw_img)  # HxW raw depth map in numpy
    depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
    depth = depth.astype(np.uint8)
    return depth
