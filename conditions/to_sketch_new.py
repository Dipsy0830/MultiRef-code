import os
import argparse
import sys
sys.path.append('/media/sata4/Contextaware/informative-drawings')
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch
import cv2
from model import Generator, GlobalGenerator2, InceptionV3
# from dataset import UnpairedDepthDataset
from PIL import Image
import numpy as np
from sk_utils import channel2width
import torch.utils.data as data
from torchvision import transforms
import random

# input_path = '/media/sata4/Contextaware/image2condition_model/000000000002.jpg'
# results_dir = '/media/sata4/Contextaware/image2condition_model/sketch2.jpg'

input_nc = 3
output_nc = 1
n_blocks = 3
checkpoints_dir = '/media/sata4/Contextaware/informative-drawings/checkpoints/model'
name = 'contour_style'
which_epoch = 'latest'
size = 256
mode = 'test'
midas = 0
geom_name = 'feats2Geom'
batchSize = 1
dataroot = ''
load_size = 256
crop_size = 256
how_many = 100
depthroot = ''
preprocess = 'resize_and_crop'
no_flip = False

def get_params(size):
    w, h = size
    new_h = h
    new_w = w
    if preprocess == 'resize_and_crop':
        new_h = new_w = load_size

    x = random.randint(0, np.maximum(0, new_w - crop_size))
    y = random.randint(0, np.maximum(0, new_h - crop_size))

    flip = random.random() > 0.5

    return {'crop_pos': (x, y), 'flip': flip}

def make_dataset(dir, stop=10000):
    images = []
    count = 0
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
                count += 1
            if count >= stop:
                return images
    return images

def get_transform(preprocess, params=None, grayscale=False, method=Image.BICUBIC, convert=True, norm=True):
    transform_list = []
    if grayscale:
        transform_list.append(transforms.Grayscale(1))
    if 'resize' in preprocess:
        osize = [load_size, load_size]
        transform_list.append(transforms.Resize(osize, method))
    elif 'scale_width' in preprocess:
        transform_list.append(transforms.Lambda(lambda img: __scale_width(img, load_size, method)))

    if 'crop' in preprocess:
        if params is None:
            transform_list.append(transforms.RandomCrop(crop_size))
        else:
            transform_list.append(transforms.Lambda(lambda img: __crop(img, params['crop_pos'], crop_size)))

    if not no_flip:
        if params is None:
            transform_list.append(transforms.RandomHorizontalFlip())
        elif params['flip']:
            transform_list.append(transforms.Lambda(lambda img: __flip(img, params['flip'])))

    if convert:
        transform_list += [transforms.ToTensor()]
        if not grayscale:
            if norm:
                transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)

class UnpairedDepthDataset(data.Dataset):
    def __init__(self, root, root2, transforms_r=None, mode='test', midas=False, depthroot=''):

        self.root = root
        self.mode = mode
        self.midas = midas

        # all_img = make_dataset(self.root)

        self.depth_maps = 0

        if self.midas:

            depth = []
            print(depthroot)
            if os.path.exists(depthroot):
                depth = make_dataset(depthroot)
            else:
                print('could not find %s'%depthroot)
                import sys
                sys.exit(0)

            newimages = []
            self.depth_maps = []

            for dmap in depth:
                lastname = os.path.basename(dmap)
                trainName1 = os.path.join(self.root, lastname)
                trainName2 = os.path.join(self.root, lastname.split('.')[0] + '.jpg')
                if os.path.exists(trainName1):
                    newimages.append(trainName1)
                elif os.path.exists(trainName2):
                    newimages.append(trainName2)

            print(f"Found {len(newimages)} correspondences")
            self.depth_maps = depth
            all_img = newimages
        else:
            all_img = [self.root]  # Single image mode

        self.data = all_img
        self.transform_r = transforms.Compose(transforms_r) if transforms_r else None
        self.min_length = len(self.data)

    def __getitem__(self, index=0):
        img_path = self.data[index]
        base = os.path.splitext(os.path.basename(img_path))[0]

        img_r = Image.open(img_path).convert('RGB')
        transform_params = get_params(img_r.size)
        A_transform = get_transform(transform_params, grayscale=(input_nc == 1), norm=False)
        if self.mode != 'train':
            A_transform = self.transform_r or A_transform

        img_r = A_transform(img_r)

        img_depth = None
        if self.midas:
            depth_map_path = self.depth_maps[index]
            if os.path.exists(depth_map_path):
                depth_img = cv2.imread(depth_map_path, cv2.IMREAD_UNCHANGED)
                img_depth = A_transform(Image.fromarray(depth_img.astype(np.uint8)).convert('RGB'))
            else:
                print(f"Depth map not found for {depth_map_path}")

        input_dict = {'r': img_r, 'depth': img_depth, 'path': img_path, 'index': index, 'name': base, 'label': 0}
        return input_dict

    def __len__(self):
        return self.min_length



def get_sketch(input_path):
    name = 'contour_style'
    with torch.no_grad():
        # Networks
        net_G = 0
        net_G = Generator(input_nc, output_nc, n_blocks)
        net_G.cuda()

        net_GB = 0

        netGeom = 0

        # Load state dicts
        net_G.load_state_dict(torch.load(os.path.join(checkpoints_dir, name, 'netG_A_%s.pth' % which_epoch)))
        # print('loaded', os.path.join(checkpoints_dir, name, 'netG_A_%s.pth' % which_epoch))

        # Set model's test mode
        net_G.eval()

        transforms_r = [transforms.Resize(int(size), Image.BICUBIC),
                        transforms.ToTensor()]

        test_data = UnpairedDepthDataset(input_path, '', transforms_r=transforms_r,
                                         mode=mode, midas=midas > 0, depthroot=depthroot)
        batch = test_data[0]

        img_r = Variable(batch['r']).unsqueeze(0).cuda()  # 添加 batch 维度
        img_depth = Variable(batch['depth']).unsqueeze(0).cuda() if batch['depth'] is not None else None
        real_A = img_r
        name = batch['name']

        input_image = real_A
        image = net_G(input_image)
        # image_numpy = image.data.squeeze().cpu().numpy()
        image_numpy = image.data.squeeze().float().cpu().numpy()
        image_numpy = (image_numpy * 255).astype(np.uint8)
    return image_numpy
        # cv2.imwrite(full_output_dir, image_numpy)
        
def main():
    sketch_image = get_sketch('/media/sata4/Contextaware/plot/image1.png')
    cv2.imwrite('/media/sata4/Contextaware/plot/sketch1.png', sketch_image)
    
if __name__ == '__main__':
    main()
    
