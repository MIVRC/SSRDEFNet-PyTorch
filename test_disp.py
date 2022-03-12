from torch.autograd import Variable
from PIL import Image
from torchvision.transforms import ToTensor
import argparse
import os
from model import *
from utils import *
import numpy as np
import torch.nn.functional as F
import re
from metric import *
import imageio

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--testset_dir', type=str, default='./data/test/')
    parser.add_argument('--scale_factor', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--model_name', type=str, default='SSRDEF_4xSR')
    return parser.parse_args()

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
    depth = depth.astype(np.float32)/256.
    return depth

def test(cfg, loadname, net):
    print(loadname)
    psnr_list = []
    psnr_list_r = []
    psnr_list_m = []
    psnr_list_r_m = []
    disphigh_occ0 = []
    disphigh_noc0 = []
    disp_occ0 = []
    disp_noc0 = []
    disphigh_occ1 = []
    disphigh_noc1 = []
    disp_occ1 = []
    disp_noc1 = []

    file_list = os.listdir(cfg.testset_dir + cfg.dataset + '/hr')
    for idx in range(len(file_list)):
        LR_left = Image.open(cfg.testset_dir + cfg.dataset + '/lr_x' + str(cfg.scale_factor) + '/' + file_list[idx] + '/lr0.png')
        LR_right = Image.open(cfg.testset_dir + cfg.dataset + '/lr_x' + str(cfg.scale_factor) + '/' + file_list[idx] + '/lr1.png')
        HR_left = Image.open(cfg.testset_dir + cfg.dataset + '/hr/' + file_list[idx] + '/hr0.png')
        HR_right = Image.open(cfg.testset_dir + cfg.dataset + '/hr/' + file_list[idx] + '/hr1.png')
        disp_left = read_disp(cfg.testset_dir + cfg.dataset + '/hr/' + file_list[idx] + '/dispocc0.png')
        disp_leftall = read_disp(cfg.testset_dir + cfg.dataset + '/hr/' + file_list[idx] + '/dispnoc0.png')

        LR_left, LR_right, HR_left, HR_right, disp_left, disp_leftall = ToTensor()(LR_left), ToTensor()(LR_right), ToTensor()(HR_left), ToTensor()(HR_right), ToTensor()(disp_left), ToTensor()(disp_leftall)
        LR_left, LR_right, HR_left, HR_right, disp_left, disp_leftall = LR_left.unsqueeze(0), LR_right.unsqueeze(0), HR_left.unsqueeze(0), HR_right.unsqueeze(0), disp_left.unsqueeze(0), disp_leftall.unsqueeze(0)
        LR_left, LR_right, HR_left, HR_right, disp_left, disp_leftall = Variable(LR_left).cuda(), Variable(LR_right).cuda(), Variable(HR_left).cuda(), Variable(HR_right).cuda(), Variable(disp_left).cuda(), Variable(disp_leftall).cuda()
        scene_name = file_list[idx]

        _,_,h,w=disp_left.shape


        disp_left=disp_left.view(1,h,w)
        disp_leftall=disp_leftall.view(1,h,w)

        mask0 = (disp_left > 0) & (disp_left < 192)
        mask1 = (disp_leftall > 0) & (disp_leftall < 192)
        _,h,w=mask0.shape


        #print('Running Scene ' + scene_name + ' of ' + cfg.dataset + ' Dataset......')
        with torch.no_grad():
            _,_,_,_,_,_,SR_left, SR_right, (disp1, disp2), (disp1_3, disp2_3), (disp1_high, disp2_high), (disp1_high2, disp2_high2) = net(LR_left, LR_right, is_training=0)
            
            SR_left, SR_right = torch.clamp(SR_left, 0, 1), torch.clamp(SR_right, 0, 1)

            psnr_list.append(cal_psnr(HR_left[:,:,:,64:].data.cpu(), SR_left[:,:,:,64:].data.cpu()))
            psnr_list_r.append(cal_psnr(HR_right.data.cpu(), SR_right.data.cpu()))
            psnr_list_r.append(cal_psnr(HR_left.data.cpu(), SR_left.data.cpu()))
            '''
            psnr_list_m.append(cal_psnr(HR_left[:,:,:,64:].data.cpu(), SR_left[0][:,:,:,64:].data.cpu()))
            psnr_list_r_m.append(cal_psnr(HR_right.data.cpu(), SR_right[0].data.cpu()))
            psnr_list_r_m.append(cal_psnr(HR_left.data.cpu(), SR_left[0].data.cpu()))
            '''         

            disphigh_occ0.append((disp1_high[mask0].cpu()-disp_left[mask0].cpu()).abs().mean())
            disphigh_noc0.append((disp1_high[mask1].cpu()-disp_leftall[mask1].cpu()).abs().mean())
            disphigh_occ1.append((disp1_high2[mask0].cpu()-disp_left[mask0].cpu()).abs().mean())
            disphigh_noc1.append((disp1_high2[mask1].cpu()-disp_leftall[mask1].cpu()).abs().mean())

            disp_occ0.append((disp1[mask0].cpu()-disp_left[mask0].cpu()).abs().mean())
            disp_noc0.append((disp1[mask1].cpu()-disp_leftall[mask1].cpu()).abs().mean())
            disp_occ1.append((disp1_3[mask0].cpu()-disp_left[mask0].cpu()).abs().mean())
            disp_noc1.append((disp1_3[mask1].cpu()-disp_leftall[mask1].cpu()).abs().mean())
            

            #psnr_list_m.append(cal_psnr(HR_left[:,:,:,64:].data.cpu(), out1_left[:,:,:,64:].data.cpu()))
            #psnr_list_r_m.append(cal_psnr(HR_right.data.cpu(), out1_right.data.cpu()))
            #psnr_list_r_m.append(cal_psnr(HR_left.data.cpu(), out1_left.data.cpu()))
            #print(torch.mean(V_left2))

        save_path = './results/' + cfg.model_name + '/' + cfg.dataset
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        SR_left_img = transforms.ToPILImage()(torch.squeeze(SR_left.data.cpu(), 0))
        SR_left_img.save(save_path + '/' + scene_name + '_L.png')
        SR_right_img = transforms.ToPILImage()(torch.squeeze(SR_right.data.cpu(), 0))
        SR_right_img.save(save_path + '/' + scene_name + '_R.png')

        imageio.imsave(save_path + '/' + scene_name + '_Lhdisp.png', torch.squeeze(disp1_high2.data.cpu(), 0))
        imageio.imsave(save_path + '/' + scene_name + '_Rhdisp.png', torch.squeeze(disp2_high2.data.cpu(), 0))

    print(cfg.dataset + ' mean psnr left: ', float(np.array(psnr_list).mean()))
    print(cfg.dataset + ' mean psnr average: ', float(np.array(psnr_list_r).mean()))
    #print(cfg.dataset + ' mean psnr left intermedite: ', float(np.array(psnr_list_m).mean()))
    #print(cfg.dataset + ' mean psnr average intermedite: ', float(np.array(psnr_list_r_m).mean()))

    print(cfg.dataset + ' disp high left all1: ', float(np.array(disphigh_occ1).mean()))
    print(cfg.dataset + ' disp high left noc1: ', float(np.array(disphigh_noc1).mean()))
    print(cfg.dataset + ' disp high left all0: ', float(np.array(disphigh_occ0).mean()))
    print(cfg.dataset + ' disp high left noc0: ', float(np.array(disphigh_noc0).mean()))

    print(cfg.dataset + ' disp left all1: ', float(np.array(disp_occ1).mean()))
    print(cfg.dataset + ' disp left noc1: ', float(np.array(disp_noc1).mean()))
    print(cfg.dataset + ' disp left all0: ', float(np.array(disp_occ0).mean()))
    print(cfg.dataset + ' disp left noc0: ', float(np.array(disp_noc0).mean()))



if __name__ == '__main__':
    cfg = parse_args()
    dataset_list = ['K2012','K2015']
    net = SSRDEFNet(cfg.scale_factor).cuda()
    net = torch.nn.DataParallel(net)
    net.eval()
    
    for j in range(80,81):
        loadname = './checkpoints/SSRDEF_4xSR_epoch' + str(j) + '.pth.tar'
        model = torch.load(loadname)
        net.load_state_dict(model['state_dict'])
        for i in range(len(dataset_list)):
            cfg.dataset = dataset_list[i]
            test(cfg, loadname, net)
    print('Finished!')
