from torch.autograd import Variable
from PIL import Image
from torchvision.transforms import ToTensor
import argparse
import os
from model import *
from utils import *
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--testset_dir', type=str, default='./data/test/')
    parser.add_argument('--scale_factor', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--model_name', type=str, default='SSRDEF_4xSR')
    return parser.parse_args()


def test(cfg, loadname, net):
    print(loadname)
    psnr_list = []
    psnr_list_r = []
    psnr_list_m1 = []
    psnr_list_r_m1 = []
    psnr_list_m2 = []
    psnr_list_r_m2 = []
    psnr_list_m3 = []
    psnr_list_r_m3 = []

    file_list = os.listdir(cfg.testset_dir + cfg.dataset + '/hr')
    for idx in range(len(file_list)):
        LR_left = Image.open(cfg.testset_dir + cfg.dataset + '/lr_x' + str(cfg.scale_factor) + '/' + file_list[idx] + '/lr0.png')
        LR_right = Image.open(cfg.testset_dir + cfg.dataset + '/lr_x' + str(cfg.scale_factor) + '/' + file_list[idx] + '/lr1.png')
        HR_left = Image.open(cfg.testset_dir + cfg.dataset + '/hr/' + file_list[idx] + '/hr0.png')
        HR_right = Image.open(cfg.testset_dir + cfg.dataset + '/hr/' + file_list[idx] + '/hr1.png')

        LR_left, LR_right, HR_left, HR_right = ToTensor()(LR_left), ToTensor()(LR_right), ToTensor()(HR_left), ToTensor()(HR_right)
        LR_left, LR_right, HR_left, HR_right = LR_left.unsqueeze(0), LR_right.unsqueeze(0), HR_left.unsqueeze(0), HR_right.unsqueeze(0)
        LR_left, LR_right, HR_left, HR_right = Variable(LR_left).cuda(), Variable(LR_right).cuda(), Variable(HR_left).cuda(), Variable(HR_right).cuda()
        scene_name = file_list[idx]
        #print('Running Scene ' + scene_name + ' of ' + cfg.dataset + ' Dataset......')
        with torch.no_grad():
            SR_left1, SR_right1, SR_left2, SR_right2, SR_left3, SR_right3, SR_left4, SR_right4,_,_,_,_ = net(LR_left, LR_right, is_training=0)
            SR_left1, SR_right1 = torch.clamp(SR_left1, 0, 1), torch.clamp(SR_right1, 0, 1)
            SR_left2, SR_right2 = torch.clamp(SR_left2, 0, 1), torch.clamp(SR_right2, 0, 1)
            SR_left3, SR_right3 = torch.clamp(SR_left3, 0, 1), torch.clamp(SR_right3, 0, 1)
            SR_left4, SR_right4 = torch.clamp(SR_left4, 0, 1), torch.clamp(SR_right4, 0, 1)

            psnr_list.append(cal_psnr(HR_left[:,:,:,64:].data.cpu(), SR_left4[:,:,:,64:].data.cpu()))
            psnr_list_r.append(cal_psnr(HR_right.data.cpu(), SR_right4.data.cpu()))
            psnr_list_r.append(cal_psnr(HR_left.data.cpu(), SR_left4.data.cpu()))

            psnr_l = cal_psnr(HR_left.data.cpu(), SR_left4.data.cpu())
            psnr_r = cal_psnr(HR_right.data.cpu(), SR_right4.data.cpu())

            psnr_l1 = cal_psnr(HR_left.data.cpu(), SR_left1.data.cpu())
            psnr_r1 = cal_psnr(HR_right.data.cpu(), SR_right1.data.cpu())

            psnr_l2 = cal_psnr(HR_left.data.cpu(), SR_left2.data.cpu())
            psnr_r2 = cal_psnr(HR_right.data.cpu(), SR_right2.data.cpu())

            psnr_l3 = cal_psnr(HR_left.data.cpu(), SR_left3.data.cpu())
            psnr_r3 = cal_psnr(HR_right.data.cpu(), SR_right3.data.cpu())
            
            psnr_list_m1.append(cal_psnr(HR_left[:,:,:,64:].data.cpu(), SR_left1[:,:,:,64:].data.cpu()))
            psnr_list_r_m1.append(cal_psnr(HR_right.data.cpu(), SR_right1.data.cpu()))
            psnr_list_r_m1.append(cal_psnr(HR_left.data.cpu(), SR_left1.data.cpu()))
            
            psnr_list_m2.append(cal_psnr(HR_left[:,:,:,64:].data.cpu(), SR_left2[:,:,:,64:].data.cpu()))
            psnr_list_r_m2.append(cal_psnr(HR_right.data.cpu(), SR_right2.data.cpu()))
            psnr_list_r_m2.append(cal_psnr(HR_left.data.cpu(), SR_left2.data.cpu()))

            psnr_list_m3.append(cal_psnr(HR_left[:,:,:,64:].data.cpu(), SR_left3[:,:,:,64:].data.cpu()))
            psnr_list_r_m3.append(cal_psnr(HR_right.data.cpu(), SR_right3.data.cpu()))
            psnr_list_r_m3.append(cal_psnr(HR_left.data.cpu(), SR_left3.data.cpu()))

        save_path = './results/' + cfg.model_name + '/' + cfg.dataset
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        SR_left_img = transforms.ToPILImage()(torch.squeeze(SR_left4.data.cpu(), 0))
        SR_left_img.save(save_path + '/' + scene_name + '_L%.2f.png'%psnr_l)
        SR_right_img = transforms.ToPILImage()(torch.squeeze(SR_right4.data.cpu(), 0))
        SR_right_img.save(save_path + '/' + scene_name + '_R%.2f.png'%psnr_r)
        '''
        SR_left_img = transforms.ToPILImage()(torch.squeeze(SR_left3.data.cpu(), 0))
        SR_left_img.save(save_path + '/' + scene_name + '_L%.2f.png'%psnr_l3)
        SR_right_img = transforms.ToPILImage()(torch.squeeze(SR_right3.data.cpu(), 0))
        SR_right_img.save(save_path + '/' + scene_name + '_R%.2f.png'%psnr_r3)

        SR_left_img = transforms.ToPILImage()(torch.squeeze(SR_left2.data.cpu(), 0))
        SR_left_img.save(save_path + '/' + scene_name + '_L%.2f.png'%psnr_l2)
        SR_right_img = transforms.ToPILImage()(torch.squeeze(SR_right2.data.cpu(), 0))
        SR_right_img.save(save_path + '/' + scene_name + '_R%.2f.png'%psnr_r2)

        SR_left_img = transforms.ToPILImage()(torch.squeeze(SR_left1.data.cpu(), 0))
        SR_left_img.save(save_path + '/' + scene_name + '_L%.2f.png'%psnr_l1)
        SR_right_img = transforms.ToPILImage()(torch.squeeze(SR_right1.data.cpu(), 0))
        SR_right_img.save(save_path + '/' + scene_name + '_R%.2f.png'%psnr_r1)
        '''
        

    print(cfg.dataset + ' mean psnr left: ', float(np.array(psnr_list).mean()))
    print(cfg.dataset + ' mean psnr average: ', float(np.array(psnr_list_r).mean()))
    print(cfg.dataset + ' mean psnr left intermediate3: ', float(np.array(psnr_list_m3).mean()))
    print(cfg.dataset + ' mean psnr average intermediate3: ', float(np.array(psnr_list_r_m3).mean()))
    print(cfg.dataset + ' mean psnr left intermediate2: ', float(np.array(psnr_list_m2).mean()))
    print(cfg.dataset + ' mean psnr average intermediate2: ', float(np.array(psnr_list_r_m2).mean()))
    print(cfg.dataset + ' mean psnr left intermediate1: ', float(np.array(psnr_list_m1).mean()))
    print(cfg.dataset + ' mean psnr average intermediate1: ', float(np.array(psnr_list_r_m1).mean()))

if __name__ == '__main__':
    cfg = parse_args()
    dataset_list = ['Middlebury', 'KITTI2012','KITTI2015','Flickr1024']
    net = SSRDEFNet(cfg.scale_factor).cuda()
    net = torch.nn.DataParallel(net)
    net.eval()
    
    for j in range(80,81):
        loadname = './checkpoints/SSRDEF_4xSR_epoch' + str(j) + '.pth.tar'
        print(loadname)
        model = torch.load(loadname)
        net.load_state_dict(model['state_dict'])
        for i in range(len(dataset_list)):
            cfg.dataset = dataset_list[i]
            test(cfg, loadname, net)
    print('Finished!')
