from __future__ import print_function
import torch
from torch.autograd import Variable as Vb
from torch.utils.data import DataLoader

import os, time, sys
from tqdm import tqdm

from models.multiframe_w_mask_genmask import *
from dataset import get_test_set
from utils import utils
from opts import parse_opts

args = parse_opts()
print (args)


def make_save_dir(output_image_dir):
    val_cities = ['frankfurt', 'lindau', 'munster']
    for city in val_cities:
        pathOutputImages = os.path.join(output_image_dir, city)
        if not os.path.isdir(pathOutputImages):
            os.makedirs(pathOutputImages)


class flowgen(object):

    def __init__(self, opt):

        self.opt = opt

        print("Random Seed: ", self.opt.seed)
        torch.manual_seed(self.opt.seed)
        torch.cuda.manual_seed_all(self.opt.seed)

        dataset = opt.dataset
        self.suffix = '_' + opt.suffix

        self.refine = True
        self.useHallucination = False
        self.jobname = dataset + self.suffix
        self.modeldir = self.jobname + 'model'

        # whether to start training from an existing snapshot
        self.load = True
        self.iter_to_load = opt.iter_to_load

        ''' Cityscapes'''
        test_Dataset = get_test_set(opt)

        self.sampledir = os.path.join('../city_scapes_test_results', self.jobname,
                                      self.suffix + '_' + str(self.iter_to_load)+'_'+str(opt.seed))

        if not os.path.exists(self.sampledir):
            os.makedirs(self.sampledir)

        self.testloader = DataLoader(test_Dataset, batch_size=opt.batch_size, shuffle=False, pin_memory=True, num_workers=8)

        # Create Folder for test images.

        self.output_image_dir = self.sampledir + '_images'
        self.output_image_before_dir = self.sampledir + '_images_before'
        self.output_bw_flow_dir = self.sampledir + '_bw_flow'
        self.output_fw_flow_dir = self.sampledir + '_fw_flow'

        self.output_bw_mask_dir = self.sampledir + '_bw_mask'
        self.output_fw_mask_dir = self.sampledir + '_fw_mask'

        make_save_dir(self.output_image_dir)
        make_save_dir(self.output_image_before_dir)

        make_save_dir(self.output_bw_flow_dir)
        make_save_dir(self.output_fw_flow_dir)

        make_save_dir(self.output_fw_mask_dir)
        make_save_dir(self.output_bw_mask_dir)

    def test(self):

        opt = self.opt
        gpu_ids = range(torch.cuda.device_count())
        print ('Number of GPUs in use {}'.format(gpu_ids))

        iteration = 0

        if torch.cuda.device_count() > 1:
            vae = nn.DataParallel(VAE(hallucination=self.useHallucination, opt=opt, refine=self.refine), device_ids=gpu_ids).cuda()
        else:
            vae = VAE(hallucination=self.useHallucination, opt=opt).cuda()

        print(self.jobname)

        if self.load:
            # model_name = '../' + self.jobname + '/{:06d}_model.pth.tar'.format(self.iter_to_load)
            model_name = '../pretrained_models/refine_genmask_w_mask_098000.pth.tar'

            print ("loading model from {}".format(model_name))

            state_dict = torch.load(model_name)
            if torch.cuda.device_count() > 1:
                vae.module.load_state_dict(state_dict['vae'])
            else:
                vae.load_state_dict(state_dict['vae'])

        z_noise = torch.ones(1, 1024).normal_()

        for sample, mask, paths in tqdm(iter(self.testloader)):
            # Set to evaluation mode (randomly sample z from the whole distribution)
            vae.eval()

            # Read data
            data = Vb(sample)
            mask = Vb(mask)

            # If test on generated images
            # data = data.unsqueeze(1)
            # data = data.repeat(1, opt.num_frames, 1, 1, 1)

            frame1 = data[:, 0, :, :, :]
            noise_bg = Vb(torch.randn(frame1.size())).cuda()
            z_m = Vb(z_noise.repeat(frame1.size()[0] * 8, 1))

            y_pred_before_refine, y_pred, mu, logvar, flow, flowback, mask_fw, mask_bw = vae(frame1, data, mask, noise_bg, z_m)

            # y_pred = y_pred * mask_fw.unsqueeze(2)

            utils.save_samples(data, y_pred_before_refine, y_pred, flow, mask_fw, mask_bw, iteration, self.sampledir, opt,
                         eval=True, useMask=True, grid=[4, 4])

            utils.save_images(self.output_image_before_dir, data, y_pred_before_refine, paths, opt)
            utils.save_images(self.output_image_dir, data, y_pred, paths, opt)

            utils.save_flows(self.output_fw_flow_dir, flow, paths)
            utils.save_flows(self.output_bw_flow_dir, flowback, paths)

            utils.save_occ_map(self.output_fw_mask_dir, mask_fw, paths)
            utils.save_occ_map(self.output_bw_mask_dir, mask_bw, paths)

            data = data.cpu().data.transpose(2, 3).transpose(3, 4).numpy()
            utils.save_gif(data * 255, opt.num_frames, [4, 4], self.sampledir + '/{:06d}_real.gif'.format(iteration))

            iteration += 1


if __name__ == '__main__':
    a = flowgen(opt=args)
    a.test()