from PIL import Image
import os
from torch.utils.data.dataset import Dataset
import random
import torch
import numpy as np
from skimage import measure

def read_disp(filename, subset=False):
    # Scene Flow dataset
    if filename.endswith('pfm'):
        # For finalpass and cleanpass, gt disparity is positive, subset is negative
        disp = np.ascontiguousarray(_read_pfm(filename)[0])
        if subset:
            disp = -disp
    # KITTI
    elif filename.endswith('png'):
        disp = _read_kitti_disp(filename)
    elif filename.endswith('npy'):
        disp = np.load(filename)
    else:
        raise Exception('Invalid disparity file format!')
    return disp  # [H, W]


def _read_pfm(file):
    file = open(file, 'rb')

    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().rstrip()
    if header.decode("ascii") == 'PF':
        color = True
    elif header.decode("ascii") == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode("ascii"))
    if dim_match:
        width, height = list(map(int, dim_match.groups()))
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().decode("ascii").rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data, scale

def _read_kitti_disp(filename):
    depth = np.array(Image.open(filename))
    depth = depth.astype(np.float32) / 256.
    return depth

class TrainSetLoader(Dataset):
    def __init__(self, cfg):
        super(TrainSetLoader, self).__init__()
        self.dataset_dir = cfg.trainset_dir + '/patches_x' + str(cfg.scale_factor)
        self.file_list = os.listdir(self.dataset_dir)
    def __getitem__(self, index):
        img_hr_left  = Image.open(self.dataset_dir + '/' + self.file_list[index] + '/hr0.png')
        img_hr_right = Image.open(self.dataset_dir + '/' + self.file_list[index] + '/hr1.png')
        img_lr_left  = Image.open(self.dataset_dir + '/' + self.file_list[index] + '/lr0.png')
        img_lr_right = Image.open(self.dataset_dir + '/' + self.file_list[index] + '/lr1.png')
        img_hr_left  = np.array(img_hr_left,  dtype=np.float32)
        img_hr_right = np.array(img_hr_right, dtype=np.float32)
        img_lr_left  = np.array(img_lr_left,  dtype=np.float32)
        img_lr_right = np.array(img_lr_right, dtype=np.float32)
        img_hr_left, img_hr_right, img_lr_left, img_lr_right = augmentation(img_hr_left, img_hr_right, img_lr_left, img_lr_right)
        return toTensor(img_hr_left), toTensor(img_hr_right), toTensor(img_lr_left), toTensor(img_lr_right)

    def __len__(self):
        return len(self.file_list)

def cal_psnr(img1, img2):
    img1_np = np.array(img1)
    img2_np = np.array(img2)
    return measure.compare_psnr(img1_np, img2_np)

def augmentation(hr_image_left, hr_image_right, lr_image_left, lr_image_right):
        flag=0
        if random.random()<0.5:     # flip horizonly
            lr_image_left_ = lr_image_right[:, ::-1, :]
            lr_image_right_ = lr_image_left[:, ::-1, :]
            hr_image_left_ = hr_image_right[:, ::-1, :]
            hr_image_right_ = hr_image_left[:, ::-1, :]
            lr_image_left, lr_image_right = lr_image_left_, lr_image_right_
            hr_image_left, hr_image_right = hr_image_left_, hr_image_right_
            flag=1

        if random.random()<0.5:     #flip vertically
            lr_image_left = lr_image_left[::-1, :, :]
            lr_image_right = lr_image_right[::-1, :, :]
            hr_image_left = hr_image_left[::-1, :, :]
            hr_image_right = hr_image_right[::-1, :, :]

        return np.ascontiguousarray(hr_image_left), np.ascontiguousarray(hr_image_right), \
                np.ascontiguousarray(lr_image_left), np.ascontiguousarray(lr_image_right)

def toTensor(img):
    img = torch.from_numpy(img.transpose((2, 0, 1)))
    return img.float().div(255)

def D1_metric(D_est, D_gt, mask, threshold=3):
    mask = mask.byte()
    error = []
    for i in range(D_gt.size(0)):
        D_est_, D_gt_ = D_est[i,...][mask[i,...]], D_gt[i,...][mask[i,...]]
        if len(D_gt_) > 0:
            E = torch.abs(D_gt_ - D_est_)
            err_mask = (E > threshold) & (E / D_gt_.abs() > 0.05)
            error.append(torch.mean(err_mask.float()).data.cpu())
    return error


def EPE_metric(D_est, D_gt, mask):
    mask = mask.byte()
    error = []
    for i in range(D_gt.size(0)):
        D_est_, D_gt_ = D_est[i,...][mask[i,...]], D_gt[i,...][mask[i,...]]
        if len(D_gt_) > 0:
            error.append(F.l1_loss(D_est_, D_gt_, size_average=True).data.cpu())
    return error



'''
from PIL import Image
import os
from torch.utils.data.dataset import Dataset
import random
import torch
import numpy as np
from skimage import measure
import torch.nn.functional as F

def read_disp(filename, subset=False):
    # Scene Flow dataset
    if filename.endswith('pfm'):
        # For finalpass and cleanpass, gt disparity is positive, subset is negative
        disp = np.ascontiguousarray(_read_pfm(filename)[0])
        if subset:
            disp = -disp
    # KITTI
    elif filename.endswith('png'):
        disp = _read_kitti_disp(filename)
    elif filename.endswith('npy'):
        disp = np.load(filename)
    else:
        raise Exception('Invalid disparity file format!')
    return disp  # [H, W]


def _read_pfm(file):
    file = open(file, 'rb')

    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().rstrip()
    if header.decode("ascii") == 'PF':
        color = True
    elif header.decode("ascii") == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode("ascii"))
    if dim_match:
        width, height = list(map(int, dim_match.groups()))
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().decode("ascii").rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data, scale

def _read_kitti_disp(filename):
    depth = np.array(Image.open(filename))
    depth = depth.astype(np.float32) / 256.
    return depth

class TrainSetLoader(Dataset):
    def __init__(self, cfg):
        super(TrainSetLoader, self).__init__()
        self.dataset_dir = cfg.trainset_dir + '/patches_x' + str(cfg.scale_factor)
        self.file_list = os.listdir(self.dataset_dir)
    def __getitem__(self, index):
        img_hr_left  = Image.open(self.dataset_dir + '/' + self.file_list[index] + '/hr0.png')
        img_hr_right = Image.open(self.dataset_dir + '/' + self.file_list[index] + '/hr1.png')
        img_lr_left  = Image.open(self.dataset_dir + '/' + self.file_list[index] + '/lr0.png')
        img_lr_right = Image.open(self.dataset_dir + '/' + self.file_list[index] + '/lr1.png')
        img_hr_left  = np.array(img_hr_left,  dtype=np.float32)
        img_hr_right = np.array(img_hr_right, dtype=np.float32)
        img_lr_left  = np.array(img_lr_left,  dtype=np.float32)
        img_lr_right = np.array(img_lr_right, dtype=np.float32)
        img_hr_left, img_hr_right, img_lr_left, img_lr_right = augmentation(img_hr_left, img_hr_right, img_lr_left, img_lr_right)
        return toTensor(img_hr_left), toTensor(img_hr_right), toTensor(img_lr_left), toTensor(img_lr_right)

    def __len__(self):
        return len(self.file_list)

def cal_psnr(img1, img2):
    img1_np = np.array(img1)
    img2_np = np.array(img2)
    return measure.compare_psnr(img1_np, img2_np)

def augmentation(hr_image_left, hr_image_right, lr_image_left, lr_image_right):

        if random.random()<0.5:     # flip horizonly
            lr_image_left_ = lr_image_right[:, ::-1, :]
            lr_image_right_ = lr_image_left[:, ::-1, :]
            hr_image_left_ = hr_image_right[:, ::-1, :]
            hr_image_right_ = hr_image_left[:, ::-1, :]
            lr_image_left, lr_image_right = lr_image_left_, lr_image_right_
            hr_image_left, hr_image_right = hr_image_left_, hr_image_right_


        if random.random()<0.5:     #flip vertically
            lr_image_left = lr_image_left[::-1, :, :]
            lr_image_right = lr_image_right[::-1, :, :]
            hr_image_left = hr_image_left[::-1, :, :]
            hr_image_right = hr_image_right[::-1, :, :]

        return np.ascontiguousarray(hr_image_left), np.ascontiguousarray(hr_image_right), \
                np.ascontiguousarray(lr_image_left), np.ascontiguousarray(lr_image_right)

def toTensor(img):
    img = torch.from_numpy(img.transpose((2, 0, 1)))
    return img.float().div(255)

def D1_metric(D_est, D_gt, mask, threshold=3):
    mask = mask.byte()
    error = []
    for i in range(D_gt.size(0)):
        D_est_, D_gt_ = D_est[i,...][mask[i,...]], D_gt[i,...][mask[i,...]]
        if len(D_gt_) > 0:
            E = torch.abs(D_gt_ - D_est_)
            err_mask = (E > threshold) & (E / D_gt_.abs() > 0.05)
            error.append(torch.mean(err_mask.float()).data.cpu())
    return error


def EPE_metric(D_est, D_gt, mask):
    mask = mask.byte()
    error = []
    for i in range(D_gt.size(0)):
        D_est_, D_gt_ = D_est[i,...][mask[i,...]], D_gt[i,...][mask[i,...]]
        if len(D_gt_) > 0:
            error.append(F.l1_loss(D_est_, D_gt_, size_average=True).data.cpu())
    return error
'''