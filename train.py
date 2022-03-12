from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import argparse
from utils import *
from model import *
from torchvision.transforms import ToTensor
import os
import torch.nn.functional as F
from loss import *


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'


def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale_factor", type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--batch_size', type=int, default=12)
    parser.add_argument('--lr', type=float, default=2e-4, help='initial learning rate')
    parser.add_argument('--gamma', type=float, default=0.5, help='')
    parser.add_argument('--start_epoch', type=int, default=0, help='start epoch')
    parser.add_argument('--n_epochs', type=int, default=80, help='number of epochs to train')
    parser.add_argument('--n_steps', type=int, default=30, help='number of epochs to update learning rate')
    parser.add_argument('--trainset_dir', type=str, default='./data/train/Flickr1024_patches')
    parser.add_argument('--model_name', type=str, default='SSRDEF')
    parser.add_argument('--load_pretrain', type=bool, default=False)
    parser.add_argument('--model_path', type=str, default='')
    parser.add_argument('--testset_dir', type=str, default='./data/test/')
    return parser.parse_args()

def warpfeature(feat, disp_range_samples, cost_prob, ndisp):
    bs, channels, height, width = feat.size()
    mh,_ = torch.meshgrid([torch.arange(0, height, dtype=feat.dtype, device=feat.device), torch.arange(0, width, dtype=feat.dtype, device=feat.device)])  # (H *W)
    mh = mh.reshape(1, 1, height, width).repeat(bs, ndisp, 1, 1)

    cur_disp_coords_y = mh
    cur_disp_coords_x = disp_range_samples

    coords_x = cur_disp_coords_x / ((width - 1.0) / 2.0) - 1.0  # trans to -1 - 1
    coords_y = cur_disp_coords_y / ((height - 1.0) / 2.0) - 1.0
    grid = torch.stack([coords_x, coords_y], dim=4).view(bs, ndisp * height, width, 2)   #(B, D, H, W, 2)->(B, D*H, W, 2)

    #warped = F.grid_sample(feat, grid.view(bs, ndisp * height, width, 2), mode='bilinear', padding_mode='zeros').view(bs, channels, ndisp, height, width) 
    warped_feat = cost_prob[:, 0, :, :].unsqueeze(1) * F.grid_sample(feat, grid[:, :height, :, :], mode='bilinear', padding_mode='zeros').view(bs, channels,height, width) 
    for i in range(1, ndisp):
        warped_feat +=  cost_prob[:, i, :, :].unsqueeze(1) * F.grid_sample(feat, grid[:, i*height:(i+1)*height, :, :], mode='bilinear', padding_mode='zeros').view(bs, channels,height, width) 

    return warped_feat

def dispwarpfeature(feat, disp):
    bs, channels, height, width = feat.size()
    mh,_ = torch.meshgrid([torch.arange(0, height, dtype=feat.dtype, device=feat.device), torch.arange(0, width, dtype=feat.dtype, device=feat.device)])  # (H *W)
    mh = mh.reshape(1, 1, height, width).repeat(bs, 1, 1, 1)

    cur_disp_coords_y = mh
    cur_disp_coords_x = disp

    coords_x = cur_disp_coords_x / ((width - 1.0) / 2.0) - 1.0  # trans to -1 - 1
    coords_y = cur_disp_coords_y / ((height - 1.0) / 2.0) - 1.0
    grid = torch.stack([coords_x, coords_y], dim=4).view(bs, height, width, 2)   #(B, D, H, W, 2)->(B, D*H, W, 2)

    #warped = F.grid_sample(feat, grid.view(bs, ndisp * height, width, 2), mode='bilinear', padding_mode='zeros').view(bs, channels, ndisp, height, width) 
    warped_feat = F.grid_sample(feat, grid, mode='bilinear', padding_mode='zeros').view(bs, channels,height, width) 

    return warped_feat
    
def cal_grad(image):
    """
    Calculate the image-edge-aware second-order smoothness loss for flo 
    """

    def gradient(pred):
        D_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        D_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        D_dy = F.pad(D_dy, pad=(0,0,0,1), mode="constant", value=0)
        D_dx = F.pad(D_dx, pad=(0,1,0,0), mode="constant", value=0)
        return D_dx, D_dy
    
    
    img_grad_x, img_grad_y = gradient(image)
    weights_x = torch.exp(-10.0 * torch.mean(torch.abs(img_grad_x), 1, keepdim=True))
    weights_y = torch.exp(-10.0 * torch.mean(torch.abs(img_grad_y), 1, keepdim=True))

    return weights_x, weights_y

def load_pretrain(model, pretrained_dict):
    torch_params =  model.state_dict()
    for k,v in pretrained_dict.items():
        print(k)
    pretrained_dict_1 = {k: v for k, v in pretrained_dict.items() if k in torch_params}
    torch_params.update(pretrained_dict_1)
    model.load_state_dict(torch_params)

def train(train_loader, cfg):
    net = SSRDEFNet(cfg.scale_factor).cuda()
    print(get_parameter_number(net))
    cudnn.benchmark = True
    scale = cfg.scale_factor

    net = torch.nn.DataParallel(net)

    if cfg.load_pretrain:
        if os.path.isfile(cfg.model_path):
            model = torch.load(cfg.model_path)
            net.load_state_dict(model['state_dict'])
            cfg.start_epoch = model["epoch"]
        else:
            print("=> no model found at '{}'".format(cfg.load_model))


    # net = torch.nn.DataParallel(net, device_ids=[0, 1])
    criterion_L1 = torch.nn.L1Loss().cuda()
    optimizer = torch.optim.Adam([paras for paras in net.parameters() if paras.requires_grad == True], lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.n_steps, gamma=cfg.gamma)

    loss_epoch = []
    loss_list = []
    psnr_epoch = []
    psnr_epoch_r = []
    psnr_epoch_m = []
    psnr_epoch_r_m = []

    for idx_epoch in range(cfg.start_epoch, cfg.n_epochs):

        for idx_iter, (HR_left, HR_right, LR_left, LR_right) in enumerate(train_loader):
            b, c, h, w = LR_left.shape
            _, _, h2, w2 = HR_left.shape
            HR_left, HR_right, LR_left, LR_right  = Variable(HR_left).cuda(), Variable(HR_right).cuda(),\
                                                    Variable(LR_left).cuda(), Variable(LR_right).cuda()

            SR_left, SR_right, SR_left2, SR_right2, SR_left3, SR_right3, SR_left4, SR_right4,\
            (M_right_to_left, M_left_to_right), (disp1, disp2), (V_left, V_right), (V_left2, V_right2), (disp1_high, disp2_high),\
            (M_right_to_left3, M_left_to_right3), (disp1_3, disp2_3), (V_left3, V_right3), (V_left4, V_right4), (disp1_high_2, disp2_high_2)\
            =net(LR_left, LR_right, is_training=1)

            ''' SR Loss '''
            loss_SR = criterion_L1(SR_left, HR_left) + criterion_L1(SR_right, HR_right) + criterion_L1(SR_left2, HR_left) + criterion_L1(SR_right2, HR_right) +\
            criterion_L1(SR_left3, HR_left) + criterion_L1(SR_right3, HR_right) + criterion_L1(SR_left4, HR_left) + criterion_L1(SR_right4, HR_right)

            loss_S = loss_disp_smoothness(disp1_high, HR_left) + loss_disp_smoothness(disp2_high, HR_right) + \
            loss_disp_smoothness(disp1_high_2, HR_left) + loss_disp_smoothness(disp2_high_2, HR_right)

            loss_P = loss_disp_unsupervised(HR_left, HR_right, disp1, F.interpolate(V_left, scale_factor=4, mode='nearest')) + loss_disp_unsupervised(HR_right, HR_left, disp2, F.interpolate(V_right, scale_factor=4, mode='nearest')) +\
            loss_disp_unsupervised(HR_left, HR_right, disp1_high, V_left2) + loss_disp_unsupervised(HR_right, HR_left, disp2_high, V_right2) + \
            loss_disp_unsupervised(HR_left, HR_right, disp1_3, F.interpolate(V_left3, scale_factor=4, mode='nearest')) + loss_disp_unsupervised(HR_right, HR_left, disp2_3, F.interpolate(V_right3, scale_factor=4, mode='nearest')) +\
            loss_disp_unsupervised(HR_left, HR_right, disp1_high_2, V_left4) + loss_disp_unsupervised(HR_right, HR_left, disp2_high_2, V_right4)

            ''' Photometric Loss '''
            Res_left = torch.abs(HR_left - F.interpolate(LR_left, scale_factor=scale, mode='bicubic', align_corners=False))
            Res_left_low = F.interpolate(Res_left, scale_factor=1 / scale, mode='bicubic', align_corners=False)
            Res_right = torch.abs(HR_right - F.interpolate(LR_right, scale_factor=scale, mode='bicubic', align_corners=False))
            Res_right_low = F.interpolate(Res_right, scale_factor=1 / scale, mode='bicubic', align_corners=False)
            Res_leftT_low = torch.bmm(M_right_to_left.contiguous().view(b * h, w, w), Res_right_low.permute(0, 2, 3, 1).contiguous().view(b * h, w, c)
                                  ).view(b, h, w, c).contiguous().permute(0, 3, 1, 2)
            Res_rightT_low = torch.bmm(M_left_to_right.contiguous().view(b * h, w, w), Res_left_low.permute(0, 2, 3, 1).contiguous().view(b * h, w, c)
                                   ).view(b, h, w, c).contiguous().permute(0, 3, 1, 2)
            Res_leftT_low2 = torch.bmm(M_right_to_left3.contiguous().view(b * h, w, w), Res_right_low.permute(0, 2, 3, 1).contiguous().view(b * h, w, c)
                                  ).view(b, h, w, c).contiguous().permute(0, 3, 1, 2)
            Res_rightT_low2 = torch.bmm(M_left_to_right3.contiguous().view(b * h, w, w), Res_left_low.permute(0, 2, 3, 1).contiguous().view(b * h, w, c)
                                   ).view(b, h, w, c).contiguous().permute(0, 3, 1, 2)
            Res_leftT = dispwarpfeature(Res_right, disp1_high)
            Res_rightT = dispwarpfeature(Res_left, disp2_high)
            Res_leftT2 = dispwarpfeature(Res_right, disp1_high_2)
            Res_rightT2 = dispwarpfeature(Res_left, disp2_high_2)

            loss_photo = criterion_L1(Res_left_low * V_left.repeat(1, 3, 1, 1), Res_leftT_low * V_left.repeat(1, 3, 1, 1)) + \
                         criterion_L1(Res_right_low * V_right.repeat(1, 3, 1, 1), Res_rightT_low * V_right.repeat(1, 3, 1, 1)) + \
                         criterion_L1(Res_left * V_left2.repeat(1, 3, 1, 1), Res_leftT * V_left2.repeat(1, 3, 1, 1)) + \
                         criterion_L1(Res_right * V_right2.repeat(1, 3, 1, 1), Res_rightT * V_right2.repeat(1, 3, 1, 1)) +\
                         criterion_L1(Res_left_low * V_left3.repeat(1, 3, 1, 1), Res_leftT_low2 * V_left3.repeat(1, 3, 1, 1)) + \
                         criterion_L1(Res_right_low * V_right3.repeat(1, 3, 1, 1), Res_rightT_low2 * V_right3.repeat(1, 3, 1, 1)) + \
                         criterion_L1(Res_left * V_left4.repeat(1, 3, 1, 1), Res_leftT2 * V_left4.repeat(1, 3, 1, 1)) + \
                         criterion_L1(Res_right * V_right4.repeat(1, 3, 1, 1), Res_rightT2 * V_right4.repeat(1, 3, 1, 1))

            loss_h = criterion_L1(M_right_to_left[:, :-1, :, :], M_right_to_left[:, 1:, :, :]) + \
                     criterion_L1(M_left_to_right[:, :-1, :, :], M_left_to_right[:, 1:, :, :]) + \
                     criterion_L1(M_right_to_left3[:, :-1, :, :], M_right_to_left3[:, 1:, :, :]) + \
                     criterion_L1(M_left_to_right3[:, :-1, :, :], M_left_to_right3[:, 1:, :, :])

            loss_w = criterion_L1(M_right_to_left[:, :, :-1, :-1], M_right_to_left[:, :, 1:, 1:]) + \
                     criterion_L1(M_left_to_right[:, :, :-1, :-1], M_left_to_right[:, :, 1:, 1:]) + \
                     criterion_L1(M_right_to_left3[:, :, :-1, :-1], M_right_to_left3[:, :, 1:, 1:]) + \
                     criterion_L1(M_left_to_right3[:, :, :-1, :-1], M_left_to_right3[:, :, 1:, 1:])

            loss_smooth = loss_w + loss_h

            ''' Cycle Loss '''
            Res_left_cycle_low = torch.bmm(M_right_to_left.contiguous().view(b * h, w, w), Res_rightT_low.permute(0, 2, 3, 1).contiguous().view(b * h, w, c)
                                       ).view(b, h, w, c).contiguous().permute(0, 3, 1, 2)
            Res_right_cycle_low = torch.bmm(M_left_to_right.contiguous().view(b * h, w, w), Res_leftT_low.permute(0, 2, 3, 1).contiguous().view(b * h, w, c)
                                        ).view(b, h, w, c).contiguous().permute(0, 3, 1, 2)
            Res_left_cycle = dispwarpfeature(Res_rightT, disp1_high)
            Res_right_cycle = dispwarpfeature(Res_leftT, disp2_high)

            Res_left_cycle_low2 = torch.bmm(M_right_to_left3.contiguous().view(b * h, w, w), Res_rightT_low2.permute(0, 2, 3, 1).contiguous().view(b * h, w, c)
                                       ).view(b, h, w, c).contiguous().permute(0, 3, 1, 2)
            Res_right_cycle_low2 = torch.bmm(M_left_to_right3.contiguous().view(b * h, w, w), Res_leftT_low2.permute(0, 2, 3, 1).contiguous().view(b * h, w, c)
                                        ).view(b, h, w, c).contiguous().permute(0, 3, 1, 2)
            Res_left_cycle2 = dispwarpfeature(Res_rightT2, disp1_high_2)
            Res_right_cycle2 = dispwarpfeature(Res_leftT2, disp2_high_2)

            loss_cycle = criterion_L1(Res_left_low * V_left.repeat(1, 3, 1, 1), Res_left_cycle_low * V_left.repeat(1, 3, 1, 1)) + \
                         criterion_L1(Res_right_low * V_right.repeat(1, 3, 1, 1), Res_right_cycle_low * V_right.repeat(1, 3, 1, 1)) + \
                         criterion_L1(Res_left * V_left2.repeat(1, 3, 1, 1), Res_left_cycle * V_left2.repeat(1, 3, 1, 1)) + \
                         criterion_L1(Res_right * V_right2.repeat(1, 3, 1, 1), Res_right_cycle * V_right2.repeat(1, 3, 1, 1)) +\
                         criterion_L1(Res_left_low * V_left3.repeat(1, 3, 1, 1), Res_left_cycle_low2 * V_left3.repeat(1, 3, 1, 1)) + \
                         criterion_L1(Res_right_low * V_right3.repeat(1, 3, 1, 1), Res_right_cycle_low2 * V_right3.repeat(1, 3, 1, 1)) + \
                         criterion_L1(Res_left * V_left4.repeat(1, 3, 1, 1), Res_left_cycle2 * V_left4.repeat(1, 3, 1, 1)) + \
                         criterion_L1(Res_right * V_right4.repeat(1, 3, 1, 1), Res_right_cycle2 * V_right4.repeat(1, 3, 1, 1))

            ''' Consistency Loss '''
            SR_left_res = F.interpolate(torch.abs(HR_left - SR_left), scale_factor=1 / scale, mode='bicubic', align_corners=False)
            SR_right_res = F.interpolate(torch.abs(HR_right - SR_right), scale_factor=1 / scale, mode='bicubic', align_corners=False)
            SR_left_res3 = F.interpolate(torch.abs(HR_left - SR_left3), scale_factor=1 / scale, mode='bicubic', align_corners=False)
            SR_right_res3 = F.interpolate(torch.abs(HR_right - SR_right3), scale_factor=1 / scale, mode='bicubic', align_corners=False)
            
            SR_left_res2 = torch.abs(HR_left - SR_left2)
            SR_right_res2 = torch.abs(HR_right - SR_right2)
            SR_left_res4 = torch.abs(HR_left - SR_left4)
            SR_right_res4 = torch.abs(HR_right - SR_right4)

            SR_left_resT = torch.bmm(M_right_to_left.detach().contiguous().view(b * h, w, w), SR_right_res.permute(0, 2, 3, 1).contiguous().view(b * h, w, c)
                                     ).view(b, h, w, c).contiguous().permute(0, 3, 1, 2)
            SR_right_resT = torch.bmm(M_left_to_right.detach().contiguous().view(b * h, w, w), SR_left_res.permute(0, 2, 3, 1).contiguous().view(b * h, w, c)
                                      ).view(b, h, w, c).contiguous().permute(0, 3, 1, 2)
            SR_left_resT2 = dispwarpfeature(SR_right_res2, disp1_high)
            SR_right_resT2 = dispwarpfeature(SR_left_res2, disp2_high)

            SR_left_resT3 = torch.bmm(M_right_to_left3.detach().contiguous().view(b * h, w, w), SR_right_res3.permute(0, 2, 3, 1).contiguous().view(b * h, w, c)
                                     ).view(b, h, w, c).contiguous().permute(0, 3, 1, 2)
            SR_right_resT3 = torch.bmm(M_left_to_right3.detach().contiguous().view(b * h, w, w), SR_left_res3.permute(0, 2, 3, 1).contiguous().view(b * h, w, c)
                                      ).view(b, h, w, c).contiguous().permute(0, 3, 1, 2)
            SR_left_resT4 = dispwarpfeature(SR_right_res4, disp1_high_2)
            SR_right_resT4 = dispwarpfeature(SR_left_res4, disp2_high_2)

            loss_cons = criterion_L1(SR_left_res * V_left.repeat(1, 3, 1, 1), SR_left_resT * V_left.repeat(1, 3, 1, 1)) + \
                       criterion_L1(SR_right_res * V_right.repeat(1, 3, 1, 1), SR_right_resT * V_right.repeat(1, 3, 1, 1)) + \
                       criterion_L1(SR_left_res2 * V_left2.repeat(1, 3, 1, 1), SR_left_resT2 * V_left2.repeat(1, 3, 1, 1)) + \
                       criterion_L1(SR_right_res2 * V_right2.repeat(1, 3, 1, 1), SR_right_resT2 * V_right2.repeat(1, 3, 1, 1)) + \
                       criterion_L1(SR_left_res3 * V_left3.repeat(1, 3, 1, 1), SR_left_resT3 * V_left3.repeat(1, 3, 1, 1)) + \
                       criterion_L1(SR_right_res3 * V_right3.repeat(1, 3, 1, 1), SR_right_resT3 * V_right3.repeat(1, 3, 1, 1)) + \
                       criterion_L1(SR_left_res4 * V_left4.repeat(1, 3, 1, 1), SR_left_resT4 * V_left4.repeat(1, 3, 1, 1)) + \
                       criterion_L1(SR_right_res4 * V_right4.repeat(1, 3, 1, 1), SR_right_resT4 * V_right4.repeat(1, 3, 1, 1))
            ''' Total Loss '''
            loss = loss_SR + 0.1 * loss_cons + 0.1 * (loss_photo + loss_smooth + loss_cycle) + 0.001*loss_S + 0.01*loss_P
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            psnr_epoch.append(cal_psnr(HR_left[:,:,:,64:].data.cpu(), SR_left4[:,:,:,64:].data.cpu()))
            psnr_epoch_r.append(cal_psnr(HR_right[:,:,:,:HR_right.shape[3]-64].data.cpu(), SR_right4[:,:,:,:HR_right.shape[3]-64].data.cpu()))

            psnr_epoch_m.append(cal_psnr(HR_left[:,:,:,64:].data.cpu(), SR_left2[:,:,:,64:].data.cpu()))
            psnr_epoch_r_m.append(cal_psnr(HR_right[:,:,:,:HR_right.shape[3]-64].data.cpu(), SR_right2[:,:,:,:HR_right.shape[3]-64].data.cpu()))
            loss_epoch.append(loss.data.cpu())
            if idx_iter%300==0:
                print("SRloss: {:.4f} Photoloss: {:.5f} Smoothloss: {:.5f} Cycleloss: {:.5f} Consloss: {:.5f} Ploss: {:.5f} Sloss: {:.5f}".format(loss_SR.item(), 0.1*loss_photo.item(), 0.1*loss_smooth.item(), 0.1*loss_cycle.item(), 0.1*loss_cons.item(), 0.02*loss_P.item(), 0.001*loss_S.item()))
                print(torch.mean(V_left2))
                print(torch.mean(V_left4))

        scheduler.step()
            

        if idx_epoch % 1 == 0:
            loss_list.append(float(np.array(loss_epoch).mean()))

            print('Epoch--%4d, loss--%f, loss_SR--%f, loss_photo--%f, loss_smooth--%f, loss_cycle--%f, loss_cons--%f' %
                  (idx_epoch + 1, float(np.array(loss_epoch).mean()), float(np.array(loss_SR.data.cpu()).mean()),
                   float(np.array(loss_photo.data.cpu()).mean()), float(np.array(loss_smooth.data.cpu()).mean()),
                   float(np.array(loss_cycle.data.cpu()).mean()), float(np.array(loss_cons.data.cpu()).mean())))
            print('PSNR left---%f, PSNR right---%f' % (float(np.array(psnr_epoch).mean()), float(np.array(psnr_epoch_r).mean())))
            print('intermediate PSNR left---%f, PSNR right---%f' % (float(np.array(psnr_epoch_m).mean()), float(np.array(psnr_epoch_r_m).mean())))
            loss_epoch = []
            psnr_epoch = []
            psnr_epoch_r = []
            psnr_epoch_m = []
            psnr_epoch_r_m = []

        torch.save({'epoch': idx_epoch + 1, 'state_dict': net.state_dict()},
                'checkpoints/' + cfg.model_name + '_' + str(cfg.scale_factor) + 'xSR_epoch' + str(idx_epoch + 1) + '.pth.tar')


def main(cfg):
    train_set = TrainSetLoader(cfg)
    train_loader = DataLoader(dataset=train_set, num_workers=6, batch_size=cfg.batch_size, shuffle=True)
    train(train_loader, cfg)

if __name__ == '__main__':
    cfg = parse_args()
    main(cfg)
