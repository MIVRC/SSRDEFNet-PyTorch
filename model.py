import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from skimage import morphology
from torchvision import transforms
from common import *

class SSRDEFNet(nn.Module):
    def __init__(self, upscale_factor):
        super(SSRDEFNet, self).__init__()
        self.upscale_factor = upscale_factor
        if upscale_factor == 2:
            kernel = 6
            stride = 2
            padding = 2
        elif upscale_factor == 4:
            kernel = 8
            stride = 4
            padding = 2
        self.init_feature = nn.Conv2d(3, 64, 3, 1, 1, bias=True)
        self.deep_feature = RDG(G0=64, C=4, G=24, n_RDB=4)
        self.pam = PAM(64)
        self.transition = nn.Sequential(
                nn.BatchNorm2d(64),
                ResB(64)
        )
        self.transition2 = nn.Sequential(
                nn.BatchNorm2d(64),
                ResB(64)
        )
        self.StereoFea = Stereo_feature()
        self.StereoFeaHigh = hStereo_feature()     

        self.encode = RDB(G0=64, C=6, G=24)
        self.encoder2 = RDB(G0=64, C=6, G=24)
        self.CALayer2 = CALayer(64, 8)
        self.CALayer = CALayer(64, 8)

        self.reconstruct = RDG(G0=64, C=4, G=24, n_RDB=4)
        self.upscale = nn.Sequential(
            nn.Conv2d(64, 64 * upscale_factor ** 2, 1, 1, 0, bias=True),
            nn.PixelShuffle(upscale_factor))
        self.final = nn.Conv2d(64, 3, 3, 1, 1, bias=True)
        self.final2 = nn.Conv2d(64, 3, 3, 1, 1, bias=True)

        self.get_cv = GetCostVolume(24)
        self.dres0 = nn.Sequential(convbn(24, 24, 3, 1, 1, 1),
                                     nn.PReLU(),
                                     convbn(24, 24, 3, 1, 1, 1),
                                     nn.PReLU())

        self.dres1 = nn.Sequential(convbn(24, 24, 3, 1, 1, 1),
                                   nn.PReLU(),
                                   convbn(24, 24, 3, 1, 1, 1))
        self.dres2 = hourglass(24)
        self.softmax = nn.Softmax(1)

        self.att1 = nn.Conv2d(64, 32, 1, 1, 0)
        self.att2 = nn.Conv2d(64, 32, 1, 1, 0)

        self.backatt1 = nn.Conv2d(64, 32, 1, 1, 0)
        self.backatt2 = nn.Conv2d(64, 32, 1, 1, 0)

        self.feedbackatt = FeedbackBlock(64, 64, kernel, stride, padding)
        self.down = ConvBlock(64, 64, kernel, stride, padding, activation='prelu', norm=None)
        self.compress_in = nn.Conv2d(64*2, 64, 1, bias=True)
        self.resblock = RDB(G0=64, C=2, G=24)

    def forward(self, x_left, x_right, is_training):
        x_left_upscale = F.interpolate(x_left, scale_factor=self.upscale_factor, mode='bicubic', align_corners=False)
        x_right_upscale = F.interpolate(x_right, scale_factor=self.upscale_factor, mode='bicubic', align_corners=False)
        buffer_left = self.init_feature(x_left)
        buffer_right = self.init_feature(x_right)
        buffer_left = self.deep_feature(buffer_left)
        buffer_right = self.deep_feature(buffer_right)

        stereo_left = self.StereoFea(self.transition(buffer_left))
        stereo_right = self.StereoFea(self.transition(buffer_right))

        b,c,h,w = buffer_left.shape

        cost = [
            torch.zeros(b*h, w, w).to(buffer_left.device),
            torch.zeros(b*h, w, w).to(buffer_left.device)
        ]

        buffer_leftT, buffer_rightT, disp1, disp2, (M_right_to_left, M_left_to_right), (V_left, V_right)\
            = self.pam(buffer_left, buffer_right, stereo_left, stereo_right, cost, is_training)
        
        
        buffer_leftF = self.CALayer(buffer_left + self.encode(buffer_leftT - buffer_left))
        buffer_rightF = self.CALayer(buffer_right + self.encode(buffer_rightT - buffer_right))
        

        buffer_leftF = self.reconstruct(buffer_leftF)
        buffer_rightF = self.reconstruct(buffer_rightF)
        feat_left = self.upscale(buffer_leftF)
        feat_right = self.upscale(buffer_rightF)
        out1_left = self.final(feat_left)+x_left_upscale
        out1_right = self.final(feat_right)+x_right_upscale

        hstereo_left = self.StereoFeaHigh(self.transition2(feat_left))
        hstereo_right = self.StereoFeaHigh(self.transition2(feat_right))

            
        disp1 = F.interpolate(disp1 * self.upscale_factor, scale_factor=self.upscale_factor, mode='bilinear', align_corners=False)
        disp2 = F.interpolate(disp2 * self.upscale_factor, scale_factor=self.upscale_factor, mode='bilinear', align_corners=False)
        maxdisp = x_left.shape[3]*self.upscale_factor


        disp_range_samples1 = get_disp_range_samples(cur_disp=disp1.detach().squeeze(1), ndisp=24, 
                                                        shape=[x_left.shape[0], x_left.shape[2]*self.upscale_factor, x_left.shape[3]*self.upscale_factor],
                                                        max_disp=maxdisp)
        disp_range_samples2 = get_disp_range_samples(cur_disp=disp2.detach().squeeze(1), ndisp=24, 
                                                        shape=[x_left.shape[0], x_left.shape[2]*self.upscale_factor, x_left.shape[3]*self.upscale_factor],
                                                        max_disp=maxdisp)

        cost1, cost2 = self.get_cv(hstereo_left, hstereo_right, disp_range_samples1, disp_range_samples2, 24)       
        
        cost1 = cost1.contiguous()
        cost2 = cost2.contiguous()

        cost1_0 = self.dres0(cost1)
        cost1_0 = self.dres1(cost1_0) + cost1_0
        cost2_0 = self.dres0(cost2)
        cost2_0 = self.dres1(cost2_0) + cost2_0

        out1 = self.dres2(cost1_0, None, None) 
        cost1_1 = out1+cost1_0  
        out2 = self.dres2(cost2_0, None, None) 
        cost2_1 = out2+cost2_0         
        
        cost_prob1 = self.softmax(cost1_1)
        cost_prob2 = self.softmax(cost2_1)

        disp1_high = torch.sum(disp_range_samples1 * cost_prob1, dim=1).unsqueeze(1)
        disp2_high = torch.sum(disp_range_samples2 * cost_prob2, dim=1).unsqueeze(1)

        feat_leftW = dispwarpfeature(feat_right, disp1_high)
        feat_rightW = dispwarpfeature(feat_left, disp2_high)

        geoerror_left = torch.abs(disp1_high - dispwarpfeature(disp2_high, disp1_high)).detach()
        geoerror_right = torch.abs(disp2_high - dispwarpfeature(disp1_high, disp2_high)).detach()

        V_left2 = 1 - torch.tanh(0.1*geoerror_left)
        V_right2 = 1 - torch.tanh(0.1*geoerror_right)

        V_left2 = torch.max(F.interpolate(V_left, scale_factor=self.upscale_factor, mode='nearest'), V_left2)
        V_right2 = torch.max(F.interpolate(V_right, scale_factor=self.upscale_factor, mode='nearest'), V_right2)

        left_att = self.att1(feat_left)
        leftW_att = self.att2(feat_leftW)
        corrleft = (torch.tanh(5*torch.sum(left_att*leftW_att, 1).unsqueeze(1))+1)/2

        right_att = self.att1(feat_right)
        rightW_att = self.att2(feat_rightW)
        corrright = (torch.tanh(5*torch.sum(right_att*rightW_att, 1).unsqueeze(1))+1)/2

        err1 = self.encoder2((feat_leftW - feat_left)*corrleft)
        buffer_leftF2 = self.CALayer2(err1 + feat_left) #high resolution feature that contains high resolution information of the other image through high res stereo matching

        err2 = self.encoder2((feat_rightW - feat_right)*corrright)
        buffer_rightF2 = self.CALayer2(err2 + feat_right)

        out2_left = self.final2(buffer_leftF2)+x_left_upscale
        out2_right = self.final2(buffer_rightF2)+x_right_upscale

        #feedback start

        left_back = self.down(buffer_leftF2)
        right_back = self.down(buffer_rightF2)

        att1 = self.feedbackatt(left_back)
        att2 = self.feedbackatt(right_back)

        left_back = left_back + 0.1*(left_back*att1)
        right_back = right_back + 0.1*(right_back*att2)


        bufferleft_att = self.backatt1(buffer_left)
        bufferright_att = self.backatt1(buffer_right)

        
        for ii in range(self.upscale_factor):
            for jj in range(self.upscale_factor):
                draft_l = dispwarpfeature(buffer_right, disp1_high[:, :, ii::self.upscale_factor, jj::self.upscale_factor]/self.upscale_factor)
                draft_r = dispwarpfeature(buffer_left, disp2_high[:, :, ii::self.upscale_factor, jj::self.upscale_factor]/self.upscale_factor)
                draftl_att = self.backatt2(draft_l)
                draftr_att = self.backatt2(draft_r)            
                corrleft = (torch.tanh(5*torch.sum(bufferleft_att*draftl_att, 1).unsqueeze(1))+1)/2
                corrright = (torch.tanh(5*torch.sum(bufferright_att*draftr_att, 1).unsqueeze(1))+1)/2
                draft_l = (1-corrleft)*buffer_left+corrleft*draft_l
                draft_r = (1-corrright)*buffer_right+corrright*draft_r
                if ii==0 and jj==0:
                    draft_left = buffer_left + self.resblock(draft_l - buffer_left)
                    draft_right = buffer_right + self.resblock(draft_r - buffer_right)
                else:
                    draft_left += buffer_left + self.resblock(draft_l - buffer_left)
                    draft_right += buffer_right + self.resblock(draft_r - buffer_right)

        draft_left = draft_left/(self.upscale_factor**2)
        draft_right = draft_right/(self.upscale_factor**2)
        
        buffer_left = self.compress_in(torch.cat([draft_left, left_back], 1))
        buffer_right = self.compress_in(torch.cat([draft_right, right_back], 1))

        stereo_left = self.StereoFea(self.transition(buffer_left))
        stereo_right = self.StereoFea(self.transition(buffer_right))

        cost = [
            torch.zeros(b*h, w, w).to(buffer_left.device),
            torch.zeros(b*h, w, w).to(buffer_left.device)
        ]

        buffer_leftT, buffer_rightT, disp1_3, disp2_3, (M_right_to_left3, M_left_to_right3), (V_left3, V_right3)\
            = self.pam(buffer_left, buffer_right, stereo_left, stereo_right, cost, is_training)
        
        buffer_leftF = self.CALayer(buffer_left + self.encode(buffer_leftT - buffer_left))
        buffer_rightF = self.CALayer(buffer_right + self.encode(buffer_rightT - buffer_right))
        

        buffer_leftF = self.reconstruct(buffer_leftF)
        buffer_rightF = self.reconstruct(buffer_rightF)
        feat_left = self.upscale(buffer_leftF)
        feat_right = self.upscale(buffer_rightF)
        out3_left = self.final(feat_left)+x_left_upscale
        out3_right = self.final(feat_right)+x_right_upscale

        hstereo_left = self.StereoFeaHigh(self.transition2(feat_left))
        hstereo_right = self.StereoFeaHigh(self.transition2(feat_right))

            
        disp1_3 = F.interpolate(disp1_3 * self.upscale_factor, scale_factor=self.upscale_factor, mode='bilinear', align_corners=False)
        disp2_3 = F.interpolate(disp2_3 * self.upscale_factor, scale_factor=self.upscale_factor, mode='bilinear', align_corners=False)
        maxdisp = x_left.shape[3]*self.upscale_factor


        disp_range_samples1 = get_disp_range_samples(cur_disp=disp1_3.detach().squeeze(1), ndisp=24, 
                                                        shape=[x_left.shape[0], x_left.shape[2]*self.upscale_factor, x_left.shape[3]*self.upscale_factor],
                                                        max_disp=maxdisp)
        disp_range_samples2 = get_disp_range_samples(cur_disp=disp2_3.detach().squeeze(1), ndisp=24, 
                                                        shape=[x_left.shape[0], x_left.shape[2]*self.upscale_factor, x_left.shape[3]*self.upscale_factor],
                                                        max_disp=maxdisp)

        cost1, cost2 = self.get_cv(hstereo_left, hstereo_right, disp_range_samples1, disp_range_samples2, 24)       
        cost1 = cost1.contiguous()
        cost2 = cost2.contiguous()

        cost1_0 = self.dres0(cost1)
        cost1_0 = self.dres1(cost1_0) + cost1_0
        cost2_0 = self.dres0(cost2)
        cost2_0 = self.dres1(cost2_0) + cost2_0

        out1 = self.dres2(cost1_0, None, None) 
        cost1_1 = out1+cost1_0  
        out2 = self.dres2(cost2_0, None, None) 
        cost2_1 = out2+cost2_0   
        
        
        cost_prob1_2 = self.softmax(cost1_1)
        cost_prob2_2 = self.softmax(cost2_1)

        disp1_high2 = torch.sum(disp_range_samples1 * cost_prob1_2, dim=1).unsqueeze(1)
        disp2_high2 = torch.sum(disp_range_samples2 * cost_prob2_2, dim=1).unsqueeze(1)

        feat_leftW = dispwarpfeature(feat_right, disp1_high2)
        feat_rightW = dispwarpfeature(feat_left, disp2_high2)

        geoerror_left = torch.abs(disp1_high2 - dispwarpfeature(disp2_high2, disp1_high2)).detach()
        geoerror_right = torch.abs(disp2_high2 - dispwarpfeature(disp1_high2, disp2_high2)).detach()

        V_left4 = 1 - torch.tanh(0.1*geoerror_left)
        V_right4 = 1 - torch.tanh(0.1*geoerror_right)

        V_left4 = torch.max(F.interpolate(V_left3, scale_factor=self.upscale_factor, mode='nearest'), V_left4)
        V_right4 = torch.max(F.interpolate(V_right3, scale_factor=self.upscale_factor, mode='nearest'), V_right4)

        left_att = self.att1(feat_left)
        leftW_att = self.att2(feat_leftW)
        corrleft = (torch.tanh(5*torch.sum(left_att*leftW_att, 1).unsqueeze(1))+1)/2

        right_att = self.att1(feat_right)
        rightW_att = self.att2(feat_rightW)
        corrright = (torch.tanh(5*torch.sum(right_att*rightW_att, 1).unsqueeze(1))+1)/2

        err1 = self.encoder2((feat_leftW - feat_left)*corrleft)
        buffer_leftF2 = self.CALayer2(err1 + feat_left)

        err2 = self.encoder2((feat_rightW - feat_right)*corrright)
        buffer_rightF2 = self.CALayer2(err2 + feat_right)

        out4_left = self.final2(buffer_leftF2)+x_left_upscale
        out4_right = self.final2(buffer_rightF2)+x_right_upscale

        if is_training == 0:
            index=(torch.arange(w*self.upscale_factor).view(1, 1, w*self.upscale_factor).repeat(1,h*self.upscale_factor,1)).to(buffer_left.device)

            disp1 = index-disp1.view(1,h*self.upscale_factor,w*self.upscale_factor)
            disp2 = disp2.view(1,h*self.upscale_factor,w*self.upscale_factor)-index
            disp1[disp1<0]=0
            disp2[disp2<0]=0
            disp1[disp1>192]=192
            disp2[disp2>192]=192

            disp1_3 = index-disp1_3.view(1,h*self.upscale_factor,w*self.upscale_factor)
            disp2_3 = disp2_3.view(1,h*self.upscale_factor,w*self.upscale_factor)-index
            disp1_3[disp1_3<0]=0
            disp2_3[disp2_3<0]=0
            disp1_3[disp1_3>192]=192
            disp2_3[disp2_3>192]=192

            disp1_high = index-disp1_high.view(1,h*self.upscale_factor,w*self.upscale_factor)
            disp2_high = disp2_high.view(1,h*self.upscale_factor,w*self.upscale_factor)-index
            disp1_high[disp1_high<0]=0
            disp2_high[disp2_high<0]=0
            disp1_high[disp1_high>192]=192
            disp2_high[disp2_high>192]=192

            disp1_high2 = index-disp1_high2.view(1,h*self.upscale_factor,w*self.upscale_factor)
            disp2_high2 = disp2_high2.view(1,h*self.upscale_factor,w*self.upscale_factor)-index
            disp1_high2[disp1_high2<0]=0
            disp2_high2[disp2_high2<0]=0
            disp1_high2[disp1_high2>192]=192
            disp2_high2[disp2_high2>192]=192

            return out1_left, out1_right, out2_left, out2_right, out3_left, out3_right, out4_left, out4_right, (disp1, disp2), (disp1_3, disp2_3), (disp1_high, disp2_high), (disp1_high2, disp2_high2)
        
        if is_training == 1:
            return out1_left, out1_right, out2_left, out2_right, out3_left, out3_right, out4_left, out4_right,\
            (M_right_to_left, M_left_to_right), (disp1, disp2), (V_left, V_right), (V_left2, V_right2), (disp1_high, disp2_high),\
            (M_right_to_left3, M_left_to_right3), (disp1_3, disp2_3), (V_left3, V_right3), (V_left4, V_right4), (disp1_high2, disp2_high2)



def get_disp_range_samples(cur_disp, ndisp, shape, max_disp):
    #shape, (B, H, W)
    #cur_disp: (B, H, W)
    #return disp_range_samples: (B, D, H, W)
    cur_disp_min = (cur_disp - ndisp / 2).clamp(min=0.0)  # (B, H, W)
    cur_disp_max = (cur_disp + ndisp / 2).clamp(max=max_disp)

    assert cur_disp.shape == torch.Size(shape), "cur_disp:{}, input shape:{}".format(cur_disp.shape, shape)
    new_interval = (cur_disp_max - cur_disp_min) / (ndisp - 1)  # (B, H, W)

    disp_range_samples = cur_disp_min.unsqueeze(1) + (torch.arange(0, ndisp, device=cur_disp.device,
                                                                      dtype=cur_disp.dtype,
                                                                      requires_grad=False).reshape(1, -1, 1,
                                                                                                   1) * new_interval.unsqueeze(1))
    return disp_range_samples


class Stereo_feature(nn.Module):
    def __init__(self):
        super(Stereo_feature, self).__init__()

        self.branch1 = nn.Sequential(nn.AvgPool2d((30, 30), stride=(30,30)),
                                     convbn(64, 32, 1, 1, 0, 1),
                                     nn.PReLU())

        self.branch2 = nn.Sequential(nn.AvgPool2d((10, 10), stride=(10,10)),
                                     convbn(64, 32, 1, 1, 0, 1),
                                     nn.PReLU())

        self.branch3 = nn.Sequential(nn.AvgPool2d((5, 5), stride=(5,5)),
                                     convbn(64, 32, 1, 1, 0, 1),
                                     nn.PReLU())

        self.lastconv = nn.Sequential(convbn(160, 64, 3, 1, 1, 1),
                                      nn.PReLU(),
                                      nn.Conv2d(64, 64, kernel_size=1, padding=0, stride = 1, bias=False))

    def forward(self, output_skip):

        output_branch1 = self.branch1(output_skip)
        output_branch1 = F.upsample(output_branch1, (output_skip.size()[2],output_skip.size()[3]),mode='bilinear')

        output_branch2 = self.branch2(output_skip)
        output_branch2 = F.upsample(output_branch2, (output_skip.size()[2],output_skip.size()[3]),mode='bilinear')

        output_branch3 = self.branch3(output_skip)
        output_branch3 = F.upsample(output_branch3, (output_skip.size()[2],output_skip.size()[3]),mode='bilinear')

        output_feature = torch.cat((output_skip, output_branch3, output_branch2, output_branch1), 1)
        output_feature = self.lastconv(output_feature)

        return output_feature


class hourglass(nn.Module):
    def __init__(self, inplanes):
        super(hourglass, self).__init__()

        self.conv1 = nn.Sequential(convbn(inplanes, inplanes, kernel_size=3, stride=2, pad=1, dilation=1),
                                   nn.PReLU())

        self.conv2 = convbn(inplanes, inplanes, kernel_size=3, stride=1, pad=1, dilation=1)

        self.conv3 = nn.Sequential(convbn(inplanes, inplanes, kernel_size=3, stride=2, pad=1, dilation=1),
                                   nn.PReLU())

        self.conv4 = nn.Sequential(convbn(inplanes, inplanes, kernel_size=3, stride=1, pad=1, dilation=1),
                                   nn.PReLU())

        self.conv5 = nn.Sequential(nn.ConvTranspose2d(inplanes, inplanes, kernel_size=3, padding=1, output_padding=1, stride=2,bias=False),
                                   nn.BatchNorm2d(inplanes)) #+conv2

        self.conv6 = nn.Sequential(nn.ConvTranspose2d(inplanes, inplanes, kernel_size=3, padding=1, output_padding=1, stride=2,bias=False),
                                   nn.BatchNorm2d(inplanes)) #+x

        self.prelu = nn.PReLU()

    def forward(self, x ,presqu, postsqu):
        
        out  = self.conv1(x) #in:1/4 out:1/8
        pre  = self.conv2(out) #in:1/8 out:1/8
        if postsqu is not None:
           pre = self.prelu(pre + postsqu)
        else:
           pre = self.prelu(pre)

        out  = self.conv3(pre) #in:1/8 out:1/16
        out  = self.conv4(out) #in:1/16 out:1/16

        if presqu is not None:
           post = self.prelu(self.conv5(out)+presqu) #in:1/16 out:1/8
        else:
           post = self.prelu(self.conv5(out)+pre) 

        out  = self.conv6(post)  #in:1/8 out:1/4

        return out


class hStereo_feature(nn.Module):
    def __init__(self):
        super(hStereo_feature, self).__init__()

        self.branch2 = nn.Sequential(nn.AvgPool2d((16, 16), stride=(16,16)),
                                     convbn(64, 24, 1, 1, 0, 1),
                                     nn.PReLU())

        self.branch3 = nn.Sequential(nn.AvgPool2d((8, 8), stride=(8,8)),
                                     convbn(64, 24, 1, 1, 0, 1),
                                     nn.PReLU())

        self.lastconv = nn.Sequential(convbn(48+64, 64, 3, 1, 1, 1),
                                      nn.PReLU(),
                                      nn.Conv2d(64, 24, kernel_size=1, padding=0, stride = 1, bias=False))

    def forward(self, output_skip):

        output_branch2 = self.branch2(output_skip)
        output_branch2 = F.upsample(output_branch2, (output_skip.size()[2],output_skip.size()[3]),mode='bilinear')

        output_branch3 = self.branch3(output_skip)
        output_branch3 = F.upsample(output_branch3, (output_skip.size()[2],output_skip.size()[3]),mode='bilinear')

        output_feature = torch.cat((output_skip, output_branch3, output_branch2), 1)
        output_feature = self.lastconv(output_feature)

        return output_feature

class FeedbackBlock(torch.nn.Module):
    def __init__(self, in_filter, num_filter, kernel_size=8, stride=4, padding=2, bias=True, activation='prelu', norm=None):
        super(FeedbackBlock, self).__init__()
        #self.conv1 = ConvBlock(in_filter, num_filter, 1, 1, 0, activation='prelu', norm=None)
        self.avgpool_1 = torch.nn.AvgPool2d(4, 4, 0)
        self.up_1 = DeconvBlock(num_filter, num_filter , 8, 4, 2, activation='prelu', norm=None)
        self.act_1 = torch.nn.ReLU(True)

    def forward(self, x):
        
        #x = self.conv1(x)
        p1 = self.avgpool_1(x)
        l00 = self.up_1(p1)
        l00 = F.upsample(l00, x.size()[2:], mode='bilinear')
        act1 = self.act_1(x - l00)
        return act1

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

class GetCostVolume(nn.Module):
    def __init__(self, channels):
        super(GetCostVolume, self).__init__()
        self.query = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.key = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)

    def forward(self, x, y, disp_range_samples1, disp_range_samples2, ndisp):
        assert (x.is_contiguous() == True)

        Q = self.query(x)
        K = self.key(y)

        bs, channels, height, width = x.size()
        cost1 = x.new().resize_(bs, ndisp, height, width).zero_()
        cost2 = x.new().resize_(bs, ndisp, height, width).zero_()
        # cost = y.unsqueeze(2).repeat(1, 2, ndisp, 1, 1) #(B, D, H, W)

        mh, mw = torch.meshgrid([torch.arange(0, height, dtype=x.dtype, device=x.device),
                                 torch.arange(0, width, dtype=x.dtype, device=x.device)])  # (H *W)
        mh = mh.reshape(1, 1, height, width).repeat(bs, ndisp, 1, 1)
        mw = mw.reshape(1, 1, height, width).repeat(bs, ndisp, 1, 1)  # (B, D, H, W)

        cur_disp_coords_y = mh
        cur_disp_coords_x = disp_range_samples1

        coords_x = cur_disp_coords_x / ((width - 1.0) / 2.0) - 1.0  # trans to -1 - 1
        coords_y = cur_disp_coords_y / ((height - 1.0) / 2.0) - 1.0
        grid = torch.stack([coords_x, coords_y], dim=4).view(bs, ndisp * height, width, 2)   #(B, D, H, W, 2)

        for i in range(ndisp):
            cost1[:, i, :, :] = (Q * F.grid_sample(K, grid[:, i*height:(i+1)*height, :, :], mode='bilinear', padding_mode='zeros').view(bs, channels,height, width)).mean(dim=1)

        Q = self.query(y)
        K = self.key(x)
        cur_disp_coords_x = disp_range_samples2
        coords_y = cur_disp_coords_y / ((height - 1.0) / 2.0) - 1.0
        grid = torch.stack([coords_x, coords_y], dim=4).view(bs, ndisp * height, width, 2)

        for i in range(ndisp):
            cost2[:, i, :, :] = (Q * F.grid_sample(K, grid[:, i*height:(i+1)*height, :, :], mode='bilinear', padding_mode='zeros').view(bs, channels,height, width)).mean(dim=1)

        return cost1, cost2

class one_conv(nn.Module):
    def __init__(self, G0, G):
        super(one_conv, self).__init__()
        self.conv = nn.Conv2d(G0, G, kernel_size=3, stride=1, padding=1, bias=True)
        #self.relu = nn.LeakyReLU(0.1, inplace=True)
        self.relu = nn.PReLU()
    def forward(self, x):
        output = self.relu(self.conv(x))
        return torch.cat((x, output), dim=1)


class RDB(nn.Module):
    def __init__(self, G0, C, G):
        super(RDB, self).__init__()
        convs = []
        for i in range(C):
            convs.append(one_conv(G0+i*G, G))
        self.conv = nn.Sequential(*convs)
        self.LFF = nn.Conv2d(G0+C*G, G0, kernel_size=1, stride=1, padding=0, bias=True)
    def forward(self, x):
        out = self.conv(x)
        lff = self.LFF(out)
        return lff + x


class RDG(nn.Module):
    def __init__(self, G0, C, G, n_RDB):
        super(RDG, self).__init__()
        self.n_RDB = n_RDB
        RDBs = []
        for i in range(n_RDB):
            RDBs.append(RDB(G0, C, G))
        self.RDB = nn.Sequential(*RDBs)
        self.conv = nn.Conv2d(G0*n_RDB, G0, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        buffer = x
        temp = []
        for i in range(self.n_RDB):
            buffer = self.RDB[i](buffer)
            temp.append(buffer)
        buffer_cat = torch.cat(temp, dim=1)
        out = self.conv(buffer_cat)
        return out


class CALayer(nn.Module):
    def __init__(self, channel, reduction):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel//reduction, 1, padding=0, bias=True),
                #nn.LeakyReLU(0.1, inplace=True),
                nn.PReLU(),
                nn.Conv2d(channel//reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid())

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class ResB(nn.Module):
    def __init__(self, channels):
        super(ResB, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, groups=4, bias=True),
            #nn.LeakyReLU(0.1, inplace=True),
            nn.PReLU(),
            nn.Conv2d(channels, channels, 3, 1, 1, groups=4, bias=True),
        )
    def __call__(self,x):
        out = self.body(x)
        return out + x


class PAM(nn.Module):
    def __init__(self, channels):
        super(PAM, self).__init__()
        self.pab1 = PAB(channels)
        self.pab2 = PAB(channels)
        self.pab3 = PAB(channels)
        self.pab4 = PAB(channels)
        self.softmax = nn.Softmax(-1)

    def forward(self, x_left, x_right, fea_left, fea_right, cost, is_training):
        b, c, h, w = fea_left.shape
        fea_left, fea_right, cost = self.pab1(fea_left, fea_right, cost)
        fea_left, fea_right, cost = self.pab2(fea_left, fea_right, cost)
        fea_left, fea_right, cost = self.pab3(fea_left, fea_right, cost)
        fea_left, fea_right, cost = self.pab4(fea_left, fea_right, cost)

        M_right_to_left = self.softmax(cost[0])                                                   # (B*H) * Wl * Wr
        M_left_to_right = self.softmax(cost[1])                                  # (B*H) * Wr * Wl
        

        M_right_to_leftp = M_right_to_left.view(b,h,w,w).permute(0,3,1,2).contiguous()
        M_left_to_rightp = M_left_to_right.view(b,h,w,w).permute(0,3,1,2).contiguous()

        M_right_to_left_relaxed = M_Relax(M_right_to_left, num_pixels=2)
        V_left = torch.bmm(M_right_to_left_relaxed.contiguous().view(-1, w).unsqueeze(1),
                           M_left_to_right.permute(0, 2, 1).contiguous().view(-1, w).unsqueeze(2)
                           ).detach().contiguous().view(b, 1, h, w)  # (B*H*Wr) * Wl * 1
        M_left_to_right_relaxed = M_Relax(M_left_to_right, num_pixels=2)
        V_right = torch.bmm(M_left_to_right_relaxed.contiguous().view(-1, w).unsqueeze(1),  # (B*H*Wl) * 1 * Wr
                            M_right_to_left.permute(0, 2, 1).contiguous().view(-1, w).unsqueeze(2)
                                  ).detach().contiguous().view(b, 1, h, w)   # (B*H*Wr) * Wl * 1

        V_left_tanh = torch.tanh(5 * V_left)
        V_right_tanh = torch.tanh(5 * V_right)

        x_leftT = torch.bmm(M_right_to_left, x_right.permute(0, 2, 3, 1).contiguous().view(-1, w, c)
                            ).contiguous().view(b, h, w, c).permute(0, 3, 1, 2)                           #  B, C0, H0, W0
        x_rightT = torch.bmm(M_left_to_right, x_left.permute(0, 2, 3, 1).contiguous().view(-1, w, c)
                            ).contiguous().view(b, h, w, c).permute(0, 3, 1, 2)                              #  B, C0, H0, W0
        out_left = x_left * (1 - V_left_tanh.repeat(1, c, 1, 1)) + x_leftT * V_left_tanh.repeat(1, c, 1, 1)
        out_right = x_right * (1 - V_right_tanh.repeat(1, c, 1, 1)) +  x_rightT * V_right_tanh.repeat(1, c, 1, 1)

        index = torch.arange(w).view(1, 1, 1, w).to(M_right_to_left.device).float()    # index: 1*1*1*w
        disp1 = torch.sum(M_right_to_left * index, dim=-1).view(b, 1, h, w) # x axis of the corresponding point
        disp2 = torch.sum(M_left_to_right * index, dim=-1).view(b, 1, h, w)


        return out_left, out_right, disp1, disp2,\
                (M_right_to_left.contiguous().view(b, h, w, w), M_left_to_right.contiguous().view(b, h, w, w)),\
                (V_left_tanh, V_right_tanh)

class PAB(nn.Module):
    def __init__(self, channels):
        super(PAB, self).__init__()
        self.head = nn.Sequential(
            nn.Conv2d(channels, channels//2, 3, 1, 1, bias=True),
            nn.BatchNorm2d(channels//2),
            nn.PReLU(),
            nn.Conv2d(channels//2, channels, 3, 1, 1, bias=True),
            nn.BatchNorm2d(channels),
            nn.PReLU(),
        )
        self.bq = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.bs = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)

    def __call__(self, fea_left, fea_right, cost):
        b, c0, h0, w0 = fea_left.shape
        fea_left1 = self.head(fea_left)
        fea_right1 = self.head(fea_right)
        Q = self.bq(fea_left1)
        b, c, h, w = Q.shape
        Q = Q - torch.mean(Q, 3).unsqueeze(3).repeat(1, 1, 1, w)
        K = self.bs(fea_right1)
        K = K - torch.mean(K, 3).unsqueeze(3).repeat(1, 1, 1, w)

        score = torch.bmm(Q.permute(0, 2, 3, 1).contiguous().view(-1, w, c),                    # (B*H) * Wl * C
                          K.permute(0, 2, 1, 3).contiguous().view(-1, c, w))  

        cost[0] += score
        cost[1] += score.permute(0, 2, 1) 
        return fea_left+fea_left1, fea_right+fea_right1, cost


def M_Relax(M, num_pixels):
    _, u, v = M.shape
    M_list = []
    M_list.append(M.unsqueeze(1))
    for i in range(num_pixels):
        pad = nn.ZeroPad2d(padding=(0, 0, i+1, 0))
        pad_M = pad(M[:, :-1-i, :])
        M_list.append(pad_M.unsqueeze(1))
    for i in range(num_pixels):
        pad = nn.ZeroPad2d(padding=(0, 0, 0, i+1))
        pad_M = pad(M[:, i+1:, :])
        M_list.append(pad_M.unsqueeze(1))
    M_relaxed = torch.sum(torch.cat(M_list, 1), dim=1)
    return M_relaxed

if __name__ == "__main__":
    net = Net(upscale_factor=4)
    total = sum([param.nelement() for param in net.parameters()])
    print('   Number of params: %.2fM' % (total / 1e6))