import torch
from torch.autograd import Variable as Vb
import torch.optim as optim
import os, time, sys

from models.multiframe_w_mask_genmask_two_path import *
from utils import utils
from uitls import ops
import losses
from dataset import get_training_set, get_test_set
from opts import parse_opts

opt = parse_opts()
print (opt)


class flowgen(object):

    def __init__(self, opt):

        self.opt = opt
        dataset = 'cityscapes_seq_full'
        self.workspace = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')

        self.jobname = dataset + '_gpu8_refine_genmask_linklink_256_1node'
        self.modeldir = self.jobname + 'model'
        self.sampledir = os.path.join(self.workspace, self.jobname)
        self.parameterdir = self.sampledir + '/params'
        self.useHallucination = False

        if not os.path.exists(self.parameterdir):
            os.makedirs(self.parameterdir)

        # whether to start training from an existing snapshot
        self.load = False
        self.iter_to_load = 62000

        # Write parameters setting file
        if os.path.exists(self.parameterdir):
            utils.save_parameters(self)

        ''' Cityscapes'''
        train_Dataset = get_training_set(opt)
        test_Dataset = get_test_set(opt)

        self.trainloader = DataLoader(train_Dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.workers,
                                      pin_memory=True, drop_last=True)
        self.testloader = DataLoader(test_Dataset, batch_size=2, shuffle=False, num_workers=opt.workers,
                                     pin_memory=True, drop_last=True)

    def train(self):

        opt = self.opt
        gpu_ids = range(torch.cuda.device_count())
        print ('Number of GPUs in use {}'.format(gpu_ids))

        iteration = 0

        vae = VAE(hallucination=self.useHallucination, opt=opt).cuda()
        if torch.cuda.device_count() > 1:
            vae = nn.DataParallel(vae, opt.sync).cuda()

        objective_func = losses.losses_multigpu_only_mask(opt, vae.module.floww)

        print(self.jobname)
        cudnn.benchmark = True

        optimizer = optim.Adam(vae.parameters(), lr=opt.lr_rate)

        if self.load:

            model_name = self.sampledir + '/{:06d}_model.pth.tar'.format(self.iter_to_load)
            print ("loading model from {}".format(model_name))

            state_dict = torch.load(model_name)
            if torch.cuda.device_count() > 1:
                vae.module.load_state_dict(state_dict['vae'])
                optimizer.load_state_dict(state_dict['optimizer'])
            else:
                vae.load_state_dict(state_dict['vae'])
                optimizer.load_state_dict(state_dict['optimizer'])
            iteration = self.iter_to_load + 1

        for epoch in range(opt.num_epochs):

            print('Epoch {}/{}'.format(epoch, opt.num_epochs - 1))
            print('-' * 10)

            for sample, bg_mask, fg_mask in iter(self.trainloader):

                # get the inputs
                data = sample.cuda()
                # mask = mask.cuda()
                bg_mask = bg_mask.cuda()
                fg_mask = fg_mask.cuda()
                # print('loaded data')

                frame1 = data[:, 0, :, :, :]
                frame2 = data[:, 1:, :, :, :]
                noise_bg = torch.randn(frame1.size()).cuda()

                start = time.time()

                # Set train mode
                vae.train()

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                y_pred_before_refine, y_pred, mu, logvar, flow, flowback, mask_fw, mask_bw, prediction_vgg_feature, gt_vgg_feature = vae(
                    frame1, data, bg_mask, fg_mask, noise_bg)

                # Compute losses
                flowloss, reconloss, reconloss_back, reconloss_before, kldloss, flowcon, sim_loss, vgg_loss, mask_loss = objective_func(
                    frame1, frame2,
                    y_pred, mu, logvar, flow, flowback,
                    mask_fw, mask_bw, prediction_vgg_feature, gt_vgg_feature,
                    y_pred_before_refine=y_pred_before_refine)

                loss = (flowloss + 2. * reconloss + reconloss_back + reconloss_before + kldloss * self.opt.lamda + flowcon + sim_loss + vgg_loss + 0.1 * mask_loss) / world_size

                # backward
                loss.backward()

                # Update
                optimizer.step()
                end = time.time()

                # print statistics
                if iteration % 20 == 0 and rank == 0:
                    print(
                        "iter {} (epoch {}), recon_loss = {:.6f}, recon_loss_back = {:.3f}, "
                        "recon_loss_before = {:.3f}, flow_loss = {:.6f}, flow_consist = {:.3f}, kl_loss = {:.6f}, "
                        "img_sim_loss= {:.3f}, vgg_loss= {:.3f}, mask_loss={:.3f}, time/batch = {:.3f}"
                        .format(iteration, epoch, reconloss.item(), reconloss_back.item(), reconloss_before.item(),
                                flowloss.item(), flowcon.item(),
                                kldloss.item(), sim_loss.item(), vgg_loss.item(), mask_loss.item(), end - start))

                if iteration % 500 == 0:
                    utils.save_samples(data, y_pred_before_refine, y_pred, flow, mask_fw, mask_bw, iteration,
                                       self.sampledir, opt)

                if iteration % 2000 == 0:
                    # Set to evaluation mode (randomly sample z from the whole distribution)
                    with torch.no_grad():
                        vae.eval()
                        val_sample, val_bg_mask, val_fg_mask = iter(self.testloader).next()

                        # Read data
                        data = val_sample.cuda()
                        bg_mask = val_bg_mask.cuda()
                        fg_mask = val_fg_mask.cuda()
                        frame1 = data[:, 0, :, :, :]

                        noise_bg = torch.randn(frame1.size()).cuda()
                        y_pred_before_refine, y_pred, mu, logvar, flow, flowback, mask_fw, mask_bw = vae(frame1, data,
                                                                                                         bg_mask,
                                                                                                         fg_mask,
                                                                                                         noise_bg)

                    utils.save_samples(data, y_pred_before_refine, y_pred, flow, mask_fw, mask_bw, iteration,
                                       self.sampledir, opt,
                                       eval=True, useMask=True)

                    # Save model's parameter
                    checkpoint_path = self.sampledir + '/{:06d}_model.pth.tar'.format(iteration)
                    print("model saved to {}".format(checkpoint_path))

                    if torch.cuda.device_count() > 1:
                        torch.save({'vae': vae.state_dict(), 'optimizer': optimizer.state_dict()},
                                   checkpoint_path)
                    else:
                        torch.save({'vae': vae.module.state_dict(), 'optimizer': optimizer.state_dict()},
                                   checkpoint_path)

                iteration += 1


if __name__ == '__main__':

    a = flowgen(opt)
    a.train()

