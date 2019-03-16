import torch
from torch.autograd import Variable as Vb
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torchvision.utils as tov
import cv2
import datetime
import numpy as np
import utils.ops
from utils.utils import *
from models.vgg_utils import my_vgg
from utils import ops
from torchvision import transforms as trn


preprocess = trn.Compose([
    # trn.ToTensor(),
    trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

mean = Vb(torch.FloatTensor([0.485, 0.456, 0.406])).view([1,3,1,1])
std = Vb(torch.FloatTensor([0.229, 0.224, 0.225])).view([1,3,1,1])


def normalize(x):
    gpu_id = x.get_device()
    return (x- mean.cuda(gpu_id))/std.cuda(gpu_id)


class TrainingLoss(object):

    def __init__(self, opt, flowwarpper):
        self.opt = opt
        self.flowwarp = flowwarpper

    def gdloss(self, a, b):
        xloss = torch.sum(
            torch.abs(torch.abs(a[:, :, 1:, :] - a[:, :, :-1, :]) - torch.abs(b[:, :, 1:, :] - b[:, :, :-1, :])))
        yloss = torch.sum(
            torch.abs(torch.abs(a[:, :, :, 1:] - a[:, :, :, :-1]) - torch.abs(b[:, :, :, 1:] - b[:, :, :, :-1])))
        return (xloss + yloss) / (a.size()[0] * a.size()[1] * a.size()[2] * a.size()[3])

    def vgg_loss(self, y_pred_feat, y_true_feat):
        loss = 0
        for i in range(len(y_pred_feat)):
            loss += (y_true_feat[i] - y_pred_feat[i]).abs().mean()
        return loss

    def _quickflowloss(self, flow, img, neighber=5, alpha=1):
        flow = flow * 128
        img = img * 256
        bs, c, h, w = img.size()
        center = int((neighber - 1) / 2)
        loss = []
        neighberrange = list(range(neighber))
        neighberrange.remove(center)
        for i in neighberrange:
            for j in neighberrange:
                flowsub = (flow[:, :, center:-center, center:-center] -
                           flow[:, :, i:h - (neighber - i - 1), j:w - (neighber - j - 1)]) ** 2
                imgsub = (img[:, :, center:-center, center:-center] -
                          img[:, :, i:h - (neighber - i - 1), j:w - (neighber - j - 1)]) ** 2
                flowsub = flowsub.sum(1)
                imgsub = imgsub.sum(1)
                indexsub = (i - center) ** 2 + (j - center) ** 2
                loss.append(flowsub * torch.exp(-alpha * imgsub - indexsub))
        return torch.stack(loss).sum() / (bs * w * h)

    def quickflowloss(self, flow, img, t=1):
        flowloss = 0.
        for ii in range(t):
                flowloss += self._quickflowloss(flow[:, :, ii, :, :], img[:, ii, :, :, :])
        return flowloss

    def _flowgradloss(self, flow, image):
        flow = flow * 128
        image = image * 256
        flowgradx = ops.gradientx(flow)
        flowgrady = ops.gradienty(flow)
        imggradx = ops.gradientx(image)
        imggrady = ops.gradienty(image)
        weightx = torch.exp(-torch.mean(torch.abs(imggradx), 1, keepdim=True))
        weighty = torch.exp(-torch.mean(torch.abs(imggrady), 1, keepdim=True))
        lossx = flowgradx * weightx
        lossy = flowgrady * weighty
        # return torch.mean(torch.abs(lossx + lossy))
        return torch.mean(torch.abs(lossx)) + torch.mean(torch.abs(lossy))

    def flowgradloss(self, flow, image, t=1):

        flow_gradient_loss = 0.
        for ii in range(t):
            flow_gradient_loss += self._flowgradloss(flow[:, :, ii, :, :], image[:, ii, :, :, :])
        return flow_gradient_loss

    def imagegradloss(self, input, target):
        input_gradx = ops.gradientx(input)
        input_grady = ops.gradienty(input)

        target_gradx = ops.gradientx(target)
        target_grady = ops.gradienty(target)

        return F.l1_loss(torch.abs(target_gradx), torch.abs(input_gradx)) \
               + F.l1_loss(torch.abs(target_grady), torch.abs(input_grady))

    def SSIM(self, x, y):
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        mu_x = F.avg_pool2d(x, 3, 1)
        mu_y = F.avg_pool2d(y, 3, 1)

        sigma_x = F.avg_pool2d(x ** 2, 3, 1) - mu_x ** 2
        sigma_y = F.avg_pool2d(y ** 2, 3, 1) - mu_y ** 2
        sigma_xy = F.avg_pool2d(x * y, 3, 1) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)

        SSIM = SSIM_n / SSIM_d

        return torch.clamp((1 - SSIM) / 2, 0, 1).mean()

    def image_similarity(self, x, y, opt=None):
        sim = 0
        # for ii in range(opt.num_predicted_frames):
        for ii in range(x.size()[1]):
            sim += opt.alpha_recon_image * self.SSIM(x[:, ii, ...], y[:, ii, ...]) \
                  + (1 - opt.alpha_recon_image) * F.l1_loss(x[:, ii, ...], y[:, ii, ...])
        return sim
    
    def loss_function(self, mu, logvar, bs):
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        KLD /= 1000
        return KLD

    def kl_criterion(self, mu, logvar, bs):
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        KLD /= self.opt.batch_size
        return KLD

    def _flowconsist(self, flow, flowback, mask_fw=None, mask_bw=None):
        if mask_fw is not None:
            # mask_fw, mask_bw = occlusion(flow, flowback, self.flowwarp)
            prevloss = (mask_bw * torch.abs(self.flowwarp(flow, -flowback) - flowback)).mean()
            nextloss = (mask_fw * torch.abs(self.flowwarp(flowback, flow) - flow)).mean()
        else:
            prevloss = torch.abs(self.flowwarp(flow, -flowback) - flowback).mean()
            nextloss = torch.abs(self.flowwarp(flowback, flow) - flow).mean()
        return prevloss + nextloss

    def flowconsist(self, flow, flowback, mask_fw=None, mask_bw=None, t=4):
        flowcon = 0.
        if mask_bw is not None:
            for ii in range(t):
                flowcon += self._flowconsist(flow[:, :, ii, :, :], flowback[:, :, ii, :, :],
                                                 mask_fw=mask_fw[:, ii:ii + 1, ...],
                                                 mask_bw=mask_bw[:, ii:ii + 1, ...])
        else:
            for ii in range(t):
                flowcon += self._flowconsist(flow[:, :, ii, :, :], flowback[:, :, ii, :, :])
        return flowcon

    def reconlossT(self, x, y, t=4, mask=None):
        if mask is not None:
            x = x * mask.unsqueeze(2)
            y = y * mask.unsqueeze(2)

        loss = (x.contiguous() - y.contiguous()).abs().mean()
        return loss


class losses_multigpu_only_mask(nn.Module):
    def __init__(self, opt, flowwarpper):
        super(losses_multigpu_only_mask, self).__init__()
        self.tl = TrainingLoss(opt, flowwarpper)
        self.flowwarpper = flowwarpper
        self.opt = opt

    def forward(self, frame1, frame2, y_pred, mu, logvar, flow, flowback, mask_fw, mask_bw, prediction_vgg_feature, gt_vgg_feature,  y_pred_before_refine=None):
        opt = self.opt
        flowwarpper = self.flowwarpper
        tl = self.tl
        output = y_pred

        '''flowloss'''
        flowloss = tl.quickflowloss(flow, frame2)
        flowloss += tl.quickflowloss(flowback, frame1.unsqueeze(1))
        flowloss *= 0.01

        '''flow consist'''
        flowcon = tl.flowconsist(flow, flowback, mask_fw, mask_bw, t=opt.num_predicted_frames)

        '''kldloss'''
        kldloss = tl.loss_function(mu, logvar, opt.batch_size)

        '''flow gradient loss'''
        # flow_gradient_loss = tl.flowgradloss(flow, frame2)
        # flow_gradient_loss += tl.flowgradloss(flowback, frame1)
        # flow_gradient_loss *= 0.01

        '''Image Similarity loss'''
        sim_loss = tl.image_similarity(output, frame2, opt)

        '''reconstruct loss'''
        prevframe = [torch.unsqueeze(flowwarpper(frame2[:, ii, :, :, :], -flowback[:, :, ii, :, :]* mask_bw[:, ii:ii + 1, ...]), 1)
                     for ii in range(opt.num_predicted_frames)]
        prevframe = torch.cat(prevframe, 1)

        reconloss_back = tl.reconlossT(prevframe,
                                  torch.unsqueeze(frame1, 1).repeat(1, opt.num_predicted_frames, 1, 1, 1),
                                  mask=mask_bw, t=opt.num_predicted_frames)
        reconloss = tl.reconlossT(output, frame2, t=opt.num_predicted_frames)

        if y_pred_before_refine is not None:
            reconloss_before = tl.reconlossT(y_pred_before_refine, frame2, mask=mask_fw, t=opt.num_predicted_frames)
        else:
            reconloss_before = 0.

        '''vgg loss'''
        vgg_loss = tl.vgg_loss(prediction_vgg_feature, gt_vgg_feature)

        '''mask loss'''
        mask_loss = (1 - mask_bw).mean() + (1 - mask_fw).mean()

        return flowloss, reconloss, reconloss_back, reconloss_before, kldloss, flowcon, sim_loss, vgg_loss, mask_loss
