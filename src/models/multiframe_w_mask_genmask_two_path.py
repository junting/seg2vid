import torch
from torch.autograd import Variable as Vb
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torchvision.models
import torch.optim as optim
import os
import logging
import torchvision.utils as tov

import sys
sys.path.insert(0, '../utils')
from utils import utils
from utils import ops
from models.vgg_utils import my_vgg


class motion_net(nn.Module):
    def __init__(self, opt, input_channel, output_channel=int(1024/2)):
        super(motion_net, self).__init__()
         # input 3*128*128
        self.main = nn.Sequential(
            nn.Conv2d(input_channel, 32, 4, 2, 1, bias=False),  # 64
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1, bias=False),  # 32
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1, bias=False),  # 32
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),  # 16
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 128, 3, 1, 1, bias=False),  # 16
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 64, 4, 2, 1, bias=False)  # 8
        )
        self.fc1 = nn.Linear(1024, output_channel)
        self.fc2 = nn.Linear(1024, output_channel)

    def forward(self, x):
        temp = self.main(x).view(-1, 1024)
        mu = self.fc1(temp)
        # print 'mu: '+str(mu.size())
        logvar = self.fc2(temp)
        return mu, logvar


class gateconv3d_bak(nn.Module):
    def __init__(self, innum, outnum, kernel, stride, pad):
        super(gateconv3d, self).__init__()
        self.conv = nn.Conv3d(innum, outnum * 2, kernel, stride, pad, bias=True)
        self.bn = nn.BatchNorm3d(outnum * 2)

    def forward(self, x):
        return F.glu(self.bn(self.conv(x)), 1) + x


class gateconv3d(nn.Module):
    def __init__(self, innum, outnum, kernel, stride, pad):
        super(gateconv3d, self).__init__()
        self.conv = nn.Conv3d(innum, outnum, kernel, stride, pad, bias=True)
        self.bn = nn.BatchNorm3d(outnum)

    def forward(self, x):
        return F.leaky_relu(self.bn(self.conv(x)), 0.2)


class convblock(nn.Module):
    def __init__(self, innum, outnum, kernel, stride, pad):
        super(convblock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(innum, outnum, kernel, stride, pad, bias=False),
            nn.BatchNorm2d(outnum),
            nn.LeakyReLU(0.2, inplace=True))

    def forward(self, x):
        return self.main(x)


class convbase(nn.Module):
    def __init__(self, innum, outnum, kernel, stride, pad):
        super(convbase, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(innum, outnum, kernel, stride, pad),
            nn.LeakyReLU(0.2, inplace=True))

    def forward(self, x):
        return self.main(x)


class upconv(nn.Module):
    def __init__(self, innum, outnum, kernel, stride, pad):
        super(upconv, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(innum, outnum * 2, kernel, stride, pad),
            nn.BatchNorm2d(outnum * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(outnum * 2, outnum, kernel, stride, pad),
            nn.BatchNorm2d(outnum),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear')
        )

    def forward(self, x):
        return self.main(x)


class getflow(nn.Module):
    def __init__(self, output_channel=2):
        super(getflow, self).__init__()
        self.main = nn.Sequential(
            upconv(64, 16, 5, 1, 2),
            nn.Conv2d(16, output_channel, 5, 1, 2),
        )

    def forward(self, x):
        return self.main(x)


class get_occlusion_mask(nn.Module):
    def __init__(self):
        super(get_occlusion_mask, self).__init__()
        self.main = nn.Sequential(
            upconv(64, 16, 5, 1, 2),
            nn.Conv2d(16, 2, 5, 1, 2),
        )

    def forward(self, x):
        return torch.sigmoid(self.main(x))


class get_frames(nn.Module):
    def __init__(self, opt):
        super(get_frames, self).__init__()
        opt = opt
        self.main = nn.Sequential(
            upconv(64, 16, 5, 1, 2),
            nn.Conv2d(16, opt.input_channel, 5, 1, 2)
        )

    def forward(self, x):
        return torch.sigmoid(self.main(x))


class encoder(nn.Module):
    def __init__(self, opt):
        super(encoder, self).__init__()
        self.econv1 = convbase(opt.input_channel + opt.mask_channel, 32, 4, 2, 1)  # 32,64,64
        self.econv2 = convblock(32, 64, 4, 2, 1)  # 64,32,32
        self.econv3 = convblock(64, 128, 4, 2, 1)  # 128,16,16
        self.econv4 = convblock(128, 256, 4, 2, 1)  # 256,8,8

    def forward(self, x):
        enco1 = self.econv1(x)  # 32
        enco2 = self.econv2(enco1)  # 64
        enco3 = self.econv3(enco2)  # 128
        codex = self.econv4(enco3)  # 256
        return enco1, enco2, enco3, codex


class decoder(nn.Module):
    def __init__(self, opt):
        super(decoder, self).__init__()
        self.opt = opt
        self.dconv1 = convblock(256 + 16, 256, 3, 1, 1)  # 256,8,8
        self.dconv2 = upconv(256, 128, 3, 1, 1)  # 128,16,16
        self.dconv3 = upconv(256, 64, 3, 1, 1)  # 64,32,32
        self.dconv4 = upconv(128, 32, 3, 1, 1)  # 32,64,64
        self.gateconv1 = gateconv3d(64, 64, 3, 1, 1)
        self.gateconv2 = gateconv3d(32, 32, 3, 1, 1)

    def forward(self, enco1, enco2, enco3, z):
        opt = self.opt
        deco1 = self.dconv1(z)  # .view(-1,256,4,4,4)# bs*4,256,8,8
        deco2 = torch.cat(torch.chunk(self.dconv2(deco1).unsqueeze(2), opt.num_predicted_frames, 0), 2)  # bs*4,128,16,16
        deco2 = torch.cat(torch.unbind(torch.cat([deco2, torch.unsqueeze(enco3, 2).repeat(1, 1, opt.num_predicted_frames, 1, 1)], 1), 2), 0)
        deco3 = torch.cat(self.dconv3(deco2).unsqueeze(2).chunk(opt.num_predicted_frames, 0), 2)  # 128,32,32
        deco3 = self.gateconv1(deco3)
        deco3 = torch.cat(torch.unbind(torch.cat([deco3, torch.unsqueeze(enco2, 2).repeat(1, 1, opt.num_predicted_frames, 1, 1)], 1), 2), 0)
        deco4 = torch.cat(self.dconv4(deco3).unsqueeze(2).chunk(opt.num_predicted_frames, 0), 2)  # 32,4,64,64
        deco4 = self.gateconv2(deco4)
        deco4 = torch.cat(torch.unbind(torch.cat([deco4, torch.unsqueeze(enco1, 2).repeat(1, 1, opt.num_predicted_frames, 1, 1)], 1), 2), 0)
        return deco4


mean = Vb(torch.FloatTensor([0.485, 0.456, 0.406])).view([1,3,1,1])
std = Vb(torch.FloatTensor([0.229, 0.224, 0.225])).view([1,3,1,1])


class VAE(nn.Module):
    def __init__(self, hallucination=False, opt=None, refine=True, bg=512, fg=512):
        super(VAE, self).__init__()

        self.opt = opt
        self.hallucination = hallucination

        # BG
        self.motion_net_bg = motion_net(opt, int(opt.num_frames*opt.input_channel)+11, bg)
        # FG
        self.motion_net_fg = motion_net(opt, int(opt.num_frames*opt.input_channel)+9, fg)

        self.encoder = encoder(opt)
        self.flow_decoder = decoder(opt)
        if self.hallucination:
            self.raw_decoder = decoder(opt)
            self.predict = get_frames(opt)

        self.zconv = convbase(256 + 64, 16*self.opt.num_predicted_frames, 3, 1, 1)
        self.floww = ops.flowwrapper()
        self.fc = nn.Linear(1024, 1024)
        self.flownext = getflow()
        self.flowprev = getflow()
        self.get_mask = get_occlusion_mask()
        self.refine = refine
        if self.refine:
            from models.vgg_128 import RefineNet
            self.refine_net = RefineNet(num_channels=opt.input_channel)

        vgg19 = torchvision.models.vgg19(pretrained=True)
        self.vgg_net = my_vgg(vgg19)
        for param in self.vgg_net.parameters():
            param.requires_grad = False

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Vb(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return Vb(mu.data.new(mu.size()).normal_())

    def _normalize(self, x):
        gpu_id = x.get_device()
        return (x - mean.cuda(gpu_id)) / std.cuda(gpu_id)

    def forward(self, x, data, bg_mask, fg_mask, noise_bg, z_m=None):

        frame1 = data[:, 0, :, :, :]
        frame2 = data[:, 1:, :, :, :]
        mask = torch.cat([bg_mask, fg_mask], 1)
        input = torch.cat([x, mask], 1)
        opt = self.opt

        y = torch.cat(
            [frame1, frame2.contiguous().view(-1, opt.num_predicted_frames * opt.input_channel, opt.input_size[0],
                                              opt.input_size[1]) -
             frame1.repeat(1, opt.num_predicted_frames, 1, 1)], 1)

        # Encoder Network --> encode input frames
        enco1, enco2, enco3, codex = self.encoder(input)

        # Motion Network --> compute latent vector

        # BG
        mu_bg, logvar_bg = self.motion_net_bg(torch.cat([y, bg_mask], 1).contiguous())
        # FG
        mu_fg, logvar_fg = self.motion_net_fg(torch.cat([y , fg_mask], 1).contiguous())

        mu = torch.cat([mu_bg, mu_fg], 1)
        logvar = torch.cat([logvar_bg, logvar_fg], 1)

        # mu = mu_bg + mu_fg
        # logvar = logvar_bg + logvar_fg
        # print (mu.size())
        # z_m = self.reparameterize(mu, logvar)
        # print (z_m.size())
        if z_m is None:
            z_m = self.reparameterize(mu, logvar)

        codey = self.zconv(torch.cat([self.fc(z_m).view(-1, 64, int(opt.input_size[0]/16), int(opt.input_size[1]/16)), codex], 1))
        codex = torch.unsqueeze(codex, 2).repeat(1, 1, opt.num_predicted_frames, 1, 1)  # bs,256,4,8,8
        codey = torch.cat(torch.chunk(codey.unsqueeze(2), opt.num_predicted_frames, 1), 2)  # bs,16,4,8,8
        z = torch.cat(torch.unbind(torch.cat([codex, codey], 1), 2), 0)  # (256L, 272L, 8L, 8L)   272-256=16

        # Flow Decoder Network --> decode latent vectors into flow fields.
        flow_deco4 = self.flow_decoder(enco1, enco2, enco3, z)  # (256, 64, 64, 64)
        flow = torch.cat(self.flownext(flow_deco4).unsqueeze(2).chunk(opt.num_predicted_frames, 0), 2)  # (64, 2, 4, 128, 128)
        flowback = torch.cat(self.flowprev(flow_deco4).unsqueeze(2).chunk(opt.num_predicted_frames, 0), 2) # (64, 2, 4, 128, 128)

        # Warp frames using computed flows
        # out = [torch.unsqueeze(self.floww(x, flow[:, :, i, :, :]), 1) for i in range(opt.num_predicted_frames)]
        # out = torch.cat(out, 1)  # (64, 4, 3, 128, 128)

        '''Compute Occlusion Mask'''
        # mask_fw, mask_bw = ops.get_occlusion_mask(flow, flowback, self.floww, opt, t=opt.num_predicted_frames)
        masks = torch.cat(self.get_mask(flow_deco4).unsqueeze(2).chunk(opt.num_predicted_frames, 0),
                          2)  # (64, 2, 4, 128, 128)
        mask_fw = masks[:, 0, ...]
        mask_bw = masks[:, 1, ...]

        '''Use mask before warpping'''
        output = ops.warp(x, flow, opt, self.floww, mask_fw)

        y_pred = output

        '''Go through the refine network.'''
        if self.refine:
            y_pred = ops.refine(output, flow, mask_fw, self.refine_net, opt, noise_bg)

        if self.training:
            # y_pred_vgg_feature = self.vgg_net(
            #     self._normalize(y_pred.contiguous().view(-1, opt.input_channel, opt.input_size, opt.input_size)))
            prediction_vgg_feature = self.vgg_net(
                self._normalize(output.contiguous().view(-1, opt.input_channel, opt.input_size[0], opt.input_size[1])))
            gt_vgg_feature = self.vgg_net(
                self._normalize(frame2.contiguous().view(-1, opt.input_channel, opt.input_size[0], opt.input_size[1])))

            return output, y_pred, mu, logvar, flow, flowback, mask_fw, mask_bw, prediction_vgg_feature, gt_vgg_feature#, y_pred_vgg_feature
        else:
            return output, y_pred, mu, logvar, flow, flowback, mask_fw, mask_bw



