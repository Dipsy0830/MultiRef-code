# example command:


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
sys.path.append('../../HigherHRNet-Human-Pose-Estimation/tools')
# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (leoxiaobin@gmail.com)
# Modified by Bowen Cheng (bcheng9@illinois.edu)
# ------------------------------------------------------------------------------

import os
import pprint
import torch
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms
import torch.multiprocessing
from tqdm import tqdm
import _init_paths
import models
from config import cfg
from config import check_config
from core.inference import get_multi_stage_outputs
from core.inference import aggregate_results
from core.group import HeatmapParser
from dataset import make_test_dataloader
from dataset import VIS_CONFIG
from fp16_utils.fp16util import network_to_half
from utils.hr_utils import create_logger
from utils.hr_utils import get_model_summary
from utils.vis import save_debug_images
from PIL import Image
import numpy as np
import cv2
from utils.transforms import resize_align_multi_scale
from utils.transforms import get_final_preds
from utils.transforms import get_multi_scale_size

torch.multiprocessing.set_sharing_strategy('file_system')

def update_config(cfg, cfg_file, opts=None):
    cfg.defrost()
    cfg.merge_from_file(cfg_file)
    if opts:
        cfg.merge_from_list(opts)

    if not os.path.exists(cfg.DATASET.ROOT):
        cfg.DATASET.ROOT = os.path.join(
            cfg.DATA_DIR, cfg.DATASET.ROOT
        )

    cfg.MODEL.PRETRAINED = os.path.join(
        cfg.DATA_DIR, cfg.MODEL.PRETRAINED
    )

    # 直接设置 MODEL_FILE 的值
    cfg.TEST.MODEL_FILE = '../../HigherHRNet-Human-Pose-Estimation/models/pytorch/pose_coco/pose_higher_hrnet_w32_512.pth'

    if cfg.DATASET.WITH_CENTER:
        cfg.DATASET.NUM_JOINTS += 1
        cfg.MODEL.NUM_JOINTS = cfg.DATASET.NUM_JOINTS

    if not isinstance(cfg.DATASET.OUTPUT_SIZE, (list, tuple)):
        cfg.DATASET.OUTPUT_SIZE = [cfg.DATASET.OUTPUT_SIZE]
    if not isinstance(cfg.LOSS.WITH_HEATMAPS_LOSS, (list, tuple)):
        cfg.LOSS.WITH_HEATMAPS_LOSS = (cfg.LOSS.WITH_HEATMAPS_LOSS)

    if not isinstance(cfg.LOSS.HEATMAPS_LOSS_FACTOR, (list, tuple)):
        cfg.LOSS.HEATMAPS_LOSS_FACTOR = (cfg.LOSS.HEATMAPS_LOSS_FACTOR)

    if not isinstance(cfg.LOSS.WITH_AE_LOSS, (list, tuple)):
        cfg.LOSS.WITH_AE_LOSS = (cfg.LOSS.WITH_AE_LOSS)

    if not isinstance(cfg.LOSS.PUSH_LOSS_FACTOR, (list, tuple)):
        cfg.LOSS.PUSH_LOSS_FACTOR = (cfg.LOSS.PUSH_LOSS_FACTOR)

    if not isinstance(cfg.LOSS.PULL_LOSS_FACTOR, (list, tuple)):
        cfg.LOSS.PULL_LOSS_FACTOR = (cfg.LOSS.PULL_LOSS_FACTOR)

    cfg.freeze()


config_file = '../../HigherHRNet-Human-Pose-Estimation/experiments/coco/higher_hrnet/w32_512_adam_lr1e-3.yaml'  # 替代 args.cfg 的路径
update_config(cfg, config_file)
check_config(cfg)
cudnn.benchmark = cfg.CUDNN.BENCHMARK
torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

model = eval('models.' + cfg.MODEL.NAME + '.get_pose_net')(
    cfg, is_train=False
)

dump_input = torch.rand(
    (1, 3, cfg.DATASET.INPUT_SIZE, cfg.DATASET.INPUT_SIZE)
)
# logger.info(get_model_summary(model, dump_input, verbose=cfg.VERBOSE))

if cfg.FP16.ENABLED:
    model = network_to_half(model)

if cfg.TEST.MODEL_FILE:
    # logger.info('=> loading model from {}'.format(cfg.TEST.MODEL_FILE))
    model.load_state_dict(torch.load(cfg.TEST.MODEL_FILE), strict=True)
else:
    model_state_file = os.path.join(
        final_output_dir, 'model_best.pth.tar'
    )
    # logger.info('=> loading model from {}'.format(model_state_file))
    model.load_state_dict(torch.load(model_state_file))

model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).cuda()
model.eval()

# print('Successfully loaded model')

# data_loader, test_dataset = make_test_dataloader(cfg)
transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ]
        )

# 读取单张图片
def get_pose(image_path):
    image = Image.open(image_path).convert('RGB')
    original_width, original_height = image.size
    image = Image.open(image_path).convert('RGB')
    image_resized = image.resize((cfg.DATASET.INPUT_SIZE, cfg.DATASET.INPUT_SIZE))
    # Convert image to NumPy array for further processing
    image_np = np.array(image_resized)
    image_tensor = transforms(image_resized).unsqueeze(0).cuda()

    with torch.no_grad():
        final_heatmaps, tags_list = None, []
        base_size, center, scale = get_multi_scale_size(
            image_np, cfg.DATASET.INPUT_SIZE, 1.0, min(cfg.TEST.SCALE_FACTOR)
        )

        # Multi-scale inference
        for s in sorted(cfg.TEST.SCALE_FACTOR, reverse=True):
            resized, center, scale = resize_align_multi_scale(
                image_np, cfg.DATASET.INPUT_SIZE, s, min(cfg.TEST.SCALE_FACTOR)
            )
            resized_tensor = transforms(Image.fromarray(resized)).unsqueeze(0).cuda()

            outputs, heatmaps, tags = get_multi_stage_outputs(
                cfg, model, resized_tensor, cfg.TEST.FLIP_TEST,
                cfg.TEST.PROJECT2IMAGE, base_size
            )

            final_heatmaps, tags_list = aggregate_results(
                cfg, s, final_heatmaps, tags_list, heatmaps, tags
            )

        final_heatmaps /= float(len(cfg.TEST.SCALE_FACTOR))
        tags = torch.cat(tags_list, dim=4)
        parser = HeatmapParser(cfg)
        grouped, scores = parser.parse(
            final_heatmaps, tags, cfg.TEST.ADJUST, cfg.TEST.REFINE
        )

        final_results = get_final_preds(
            grouped, center, scale,
            [final_heatmaps.size(3), final_heatmaps.size(2)]
        )
        return final_results, image_np

def add_joints(image, joints, color, dataset='COCO'):
    part_idx = VIS_CONFIG[dataset]['part_idx']
    part_orders = VIS_CONFIG[dataset]['part_orders']

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
                    2
                )

    # add joints
    for joint in joints:
        if joint[2] > 0:
            cv2.circle(image, (int(joint[0]), int(joint[1])), 1, color, 2)

    # add link
    for pair in part_orders:
        link(pair[0], pair[1], color)

    return image
def get_shape(image_path):
    image = Image.open(image_path)
    image = np.array(image.convert("RGB"))
    return image.shape

def save_pose_image_resize(image, image_path, joints, file_name, dataset='COCO'):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    black_image = np.zeros_like(image)

    for person in joints:
        color = np.random.randint(0, 255, size=3)
        color = [int(i) for i in color]
        add_joints(black_image, person, color, dataset=dataset)
    result_image = black_image
    tgt_shape = get_shape(image_path)
    cv2.imwrite(file_name, cv2.resize(result_image, (tgt_shape[1], tgt_shape[0])))

def get_pose_and_save(image_path, save_path):
    pose_pred, image_np = get_pose(image_path)
    save_pose_image_resize(image_np, image_path, pose_pred, save_path)  # Use NumPy array here
    # np.save(save_numpy_path, pose_pred)
    
def save_resize_img(image_path, save_path):
    image = Image.open(image_path).convert('RGB')
    image_resized = image.resize((cfg.DATASET.INPUT_SIZE, cfg.DATASET.INPUT_SIZE))
    image_resized.save(save_path)

# def main():
#     image_path = '../../MetaData/HumanArt/images/real_human/drama/000000000001.jpg'
#     save_pose_path = '../../image2condition_model/temp/000000000001_pose.jpg'
#     # save_numpy_path = '/media/sata4/Contextaware/image2condition_model/000000000000_pose.npy'
#     get_pose_and_save(image_path, save_pose_path)

def main():
    input_dir = '../../MetaData/HumanArt/images/2D_virtual_human/oil_painting'
    output_dir = '../../Test_case/HRNet_pose_2d'
    images = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.png'))]
    images.sort()  # 排序后保证顺序一致
    images = images[:50]
    for image_file in images:
        # 构造输入图片的完整路径
        image_path = os.path.join(input_dir, image_file)
        
        # 生成保存姿态图的文件名：在原文件名中添加 '_pose'
        base, ext = os.path.splitext(image_file)
        save_pose_file = f"{base}_pose{ext}"
        save_pose_path = os.path.join(output_dir, save_pose_file)
        
        # 调用 get_pose_and_save 函数处理图片
        get_pose_and_save(image_path, save_pose_path)
        print(image_path)
    
if __name__ == '__main__':
    main()
# image_path = '/media/sata4/Contextaware/image2condition_model/000000000000.jpg'
# pose_pred, image_np = get_pose(image_path)
# save_pose_image_resize(image_np, pose_pred, 'savepath')  # Use NumPy array here
# save_pose_np(pose_pred, '{}.npy'.format(prefix))

# logger.info('Result saved to {}'.format(prefix))
# npy_path = '{}.npy'.format(prefix)
# np.save(npy_path, final_results)
# logger.info('Pose results saved as .npy to {}'.format(npy_path))
# 生成的pose还要缩放到原图的尺寸才可以给模型作为condition
# 评估结果的时候 图片又要缩成512


