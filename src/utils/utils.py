from  __future__ import division
import math
import os
import numpy as np
from PIL import Image
import imageio
import cv2
import torch
import sys
from utils import ops


def save_images(images, size, image_path):
    # images = (images+1.)/2.
    # import pdb
    # pdb.set_trace()
    num_images = size[0] * size[1]
    puzzle = merge(images[0:num_images], size)

    im = Image.fromarray(np.uint8(puzzle))
    return im.save(image_path)


def save_gif(images, length, size, gifpath):
    num_images = size[0] * size[1]
    images = np.array(images[0:num_images])
    savegif = [np.uint8(merge(images[:, times, :, :, :], size)) for times in range(0, length)]
    imageio.mimsave(gifpath, savegif, fps=int(length))


def merge(images, size):
    cdim = images.shape[-1]
    h, w = images.shape[1], images.shape[2]
    if cdim == 1:
        img = np.zeros((h * size[0], w * size[1]))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w] = np.squeeze(image)
        return img
    else:
        img = np.zeros((h * size[0], w * size[1], cdim))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w, :] = image
        # print img.shape
        return img


def psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def sharpness(img1, img2):
    dxI = np.abs(img1[:, 1:, 1:, :] - img1[:, :-1, 1:, :])
    dyI = np.abs(img1[:, 1:, 1:, :] - img1[:, 1:, :-1, :])
    dxJ = np.abs(img2[:, 1:, 1:, :] - img2[:, :-1, 1:, :])
    dyJ = np.abs(img2[:, 1:, 1:, :] - img2[:, 1:, :-1, :])
    PIXEL_MAX = 255.0
    grad = np.mean(np.abs(dxI + dyI - dxJ - dyJ))
    if grad == 0:
        return 100
    else:
        return 20 * math.log10(PIXEL_MAX / math.sqrt(grad))


def save_samples(data, y_pred_before_refine, y_pred, flow, mask_fw, mask_bw, iteration, sampledir, opt, eval=False, useMask=True,
                grid=[8, 4], single=False, bidirectional=False):

    frame1 = data[:, 0, :, :, :]

    num_predicted_frames = y_pred.size()[1] -1
    num_frames = y_pred.size()[1]

    if useMask:
        save_gif(mask_fw.unsqueeze(4).data.cpu().numpy() * 255., num_predicted_frames, grid, sampledir +
                 '/{:06d}_foward_occ_map.gif'.format(iteration))
        save_gif(mask_bw.unsqueeze(4).data.cpu().numpy() * 255., num_predicted_frames, grid, sampledir +
                 '/{:06d}_backward_occ_map.gif'.format(iteration))


    # Save results before refinement
    frame1_ = torch.unsqueeze(frame1, 1)
    if bidirectional:
        fakegif_before_refinement = torch.cat([y_pred_before_refine[:, 0:3, ...], frame1_.cuda(), y_pred_before_refine[:, 3::, ...]], 1)
    else:
        fakegif_before_refinement = torch.cat([frame1_.cuda(), y_pred_before_refine], 1)
    fakegif_before_refinement = fakegif_before_refinement.transpose(2, 3).transpose(3, 4).data.cpu().numpy()


    # Save reconstructed or sampled video
    if bidirectional:
        fakegif = torch.cat([y_pred[:,0:3,...], frame1_.cuda(), y_pred[:,3::,...]], 1)

    else:
        fakegif = torch.cat([frame1_.cuda(), y_pred], 1)
    fakegif = fakegif.transpose(2, 3).transpose(3, 4).data.cpu().numpy()

    # Save flow field
    # _flow = flow[:, :, -1, :, :]
    # _flow = _flow.cpu().data.transpose(1, 2).transpose(2, 3).numpy()
    _flow = flow.permute(0, 2, 3, 4, 1)
    _flow = _flow.cpu().data.numpy()

    if eval:
        save_file_name = 'sample'
        # Save ground truth sample
        if bidirectional:
            data = data[:, [1, 2, 3, 0, 4, 5, 6, 7], ...].cpu().data.transpose(2, 3).transpose(3, 4).numpy()
        else:
            data = data.cpu().data.transpose(2, 3).transpose(3, 4).numpy()
        save_gif(data * 255, opt.num_frames, [8, 4], sampledir + '/{:06d}_gt.gif'.format(iteration))
    else:
        save_file_name = 'recon'

    save_gif(fakegif * 255, num_frames, grid, sampledir + '/{:06d}_%s.gif'.format(iteration)%save_file_name)
    save_gif(fakegif_before_refinement * 255, num_frames, grid,
             sampledir + '/{:06d}_%s_bf_refine.gif'.format(iteration)%save_file_name)
    # ops.saveflow(_flow, opt.input_size, grid, sampledir + '/{:06d}_%s_flow.jpg'.format(iteration)%save_file_name)
    ops.save_flow_sequence(_flow, num_predicted_frames, opt.input_size, grid, sampledir + '/{:06d}_%s_flow.gif'.format(iteration) % save_file_name)

    if single:
        import scipy.misc
        for i in range(5):
            scipy.misc.imsave(sampledir +'/{:06d}_'.format(iteration) + str(i)+'.png', fakegif[0,i,...])


def save_parameters(flowgen):
    '''Write parameters setting file'''
    with open(os.path.join(flowgen.parameterdir, 'params.txt'), 'w') as file:
        file.write(flowgen.jobname)
        file.write('Training Parameters: \n')
        file.write(str(flowgen.opt) + '\n')
        if flowgen.load:
            file.write('Load pretrained model: ' + str(flowgen.load) + '\n')
            file.write('Iteration to load:' + str(flowgen.iter_to_load) + '\n')
import cv2

def save_images(root_dir, data, y_pred, paths, opt):

    frame1 = data[:, 0, :, :, :]
    frame1_ = torch.unsqueeze(frame1, 1)
    frame_sequence = torch.cat([frame1_.cuda(), y_pred], 1)
    frame_sequence = frame_sequence.permute((0, 1, 3, 4, 2)).cpu().data.numpy()* 255 # batch, num_frame, H, W, C

    for i in range(y_pred.size()[0]):

        #  save images as gif
        frames_fo_save = [np.uint8(frame_sequence[i][frame_id]) for frame_id in range(y_pred.size()[1]+1)]
        # 3fps
        aux_dir = os.path.join(root_dir, paths[0][i][0:-22])
        if not os.path.isdir(aux_dir):
            os.makedirs(aux_dir)

        imageio.mimsave(os.path.join(root_dir, paths[0][i][0:-4] + '.gif'), frames_fo_save, fps=int(len(paths)*2))


        # new added

        for j, frame in enumerate(frames_fo_save):
            # import pdb
            # pdb.set_trace()
            cv2.imwrite(os.path.join(root_dir, paths[0][i][0:-4] + '{:02d}.png'.format(j)), cv2.cvtColor(frame,cv2.COLOR_RGB2BGR))

        # for j in range(len(frame_sequence[0])):
        #     # aux_dir = os.path.join(root_dir, paths[j][i][0:-22])
        #     # if not os.path.isdir(aux_dir):
        #     #     os.makedirs(aux_dir)
        #     frame = frame_sequence[i][j]
        #     # frameResized = cv2.resize(frame, (256, 128), interpolation=cv2.INTER_LINEAR)
        #     # cv2.imwrite(os.path.join(root_dir, paths[j][i]), frame)
        #     # cv2.imwrite(os.path.join(root_dir, paths[j][i]), cv2.cvtColor(frame,cv2.COLOR_RGB2BGR))
        #     cv2.imwrite(os.path.join(root_dir, paths[j][i]), frame)



def save_images_ucf(root_dir, data, y_pred, paths, opt):

    frame1 = data[:, 0, :, :, :]
    frame1_ = torch.unsqueeze(frame1, 1)
    frame_sequence = torch.cat([frame1_.cuda(), y_pred], 1)
    frame_sequence = frame_sequence.permute((0, 1, 3, 4, 2)).cpu().data.numpy()* 255 # batch, num_frame, H, W, C

    for i in range(y_pred.size()[0]):

        #  save images as gif
        frames_fo_save = [np.uint8(frame_sequence[i][frame_id]) for frame_id in range(y_pred.size()[1]+1)]
        # 3fps
        # import pdb
        # pdb.set_trace()
        aux_dir = os.path.join(root_dir, paths[0][i])
        if not os.path.isdir(aux_dir):
            os.makedirs(aux_dir)

        imageio.mimsave(os.path.join(root_dir, paths[0][i] + '.gif'), frames_fo_save, fps=int(len(paths)*2))

        # new added

        for j, frame in enumerate(frames_fo_save):
            # import pdb
            # pdb.set_trace()
            cv2.imwrite(os.path.join(root_dir, paths[0][i], '{:02d}.png'.format(j)), cv2.cvtColor(frame,cv2.COLOR_RGB2BGR))



def save_images_kitti(root_dir, data, y_pred, paths, opt):

    frame1 = data[:, 0, :, :, :]
    frame1_ = torch.unsqueeze(frame1, 1)
    frame_sequence = torch.cat([frame1_.cuda(), y_pred], 1)
    frame_sequence = frame_sequence.permute((0, 1, 3, 4, 2)).cpu().data.numpy()* 255 # batch, num_frame, H, W, C

    for i in range(y_pred.size()[0]):

        #  save images as gif
        frames_fo_save = [np.uint8(frame_sequence[i][frame_id]) for frame_id in range(y_pred.size()[1]+1)]
        # 3fps
        # import pdb
        # pdb.set_trace()
        aux_dir = os.path.join(root_dir, paths[i])
        if not os.path.isdir(aux_dir):
            os.makedirs(aux_dir)

        imageio.mimsave(os.path.join(root_dir, paths[i] + '.gif'), frames_fo_save, fps=int(len(paths)*2))

        # new added

        for j, frame in enumerate(frames_fo_save):
            # import pdb
            # pdb.set_trace()
            frame = cv2.resize(frame, (256, 78))
            cv2.imwrite(os.path.join(root_dir, paths[i], '{:02d}.png'.format(j)), cv2.cvtColor(frame,cv2.COLOR_RGB2BGR))



def save_flows(root_dir, flow, paths):
    # print(flow.size())
    _flow = flow.permute(0, 2, 3, 4, 1)
    _flow = _flow.cpu().data.numpy()
    # mask  = mask.unsqueeze(4)
    # # print (mask.size())
    # mask = mask.data.cpu().numpy() * 255.

    for i in range(flow.size()[0]):

        # save flow*mask as gif
        # *mask[i][frame_id])
        flow_fo_save = [np.uint8(ops.compute_flow_color_map(_flow[i][frame_id])) for frame_id in range(len(paths)-1)]
        # 3fps
        imageio.mimsave(os.path.join(root_dir, paths[0][i][0:-4] + '.gif'), flow_fo_save, fps=int(len(paths)-1-2))

        for j in range(flow.size()[2]):
            ops.saveflow(_flow[i][j], (256, 128), os.path.join(root_dir, paths[j+1][i]))


def save_occ_map(root_dir, mask, paths):
    mask = mask.data.cpu().numpy() * 255.
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            cv2.imwrite(os.path.join(root_dir, paths[j+1][i]), mask[i][j])


def save_samples_no_flow(data, y_pred, iteration, sampledir, opt, eval=False,
                grid=[8, 4], single=False, bidirectional=False):

    frame1 = data[:, 0, :, :, :]
    num_frames = y_pred.size()[1]

    # Save results before refinement
    frame1_ = torch.unsqueeze(frame1, 1)

    # Save reconstructed or sampled video
    if bidirectional:
        fakegif = torch.cat([y_pred[:,0:3,...], frame1_.cuda(), y_pred[:,3::,...]], 1)

    else:
        fakegif = torch.cat([frame1_.cuda(), y_pred], 1)
    fakegif = fakegif.transpose(2, 3).transpose(3, 4).data.cpu().numpy()

    if eval:
        save_file_name = 'sample'
        # Save ground truth sample
        if bidirectional:
            data = data[:, [1, 2, 3, 0, 4, 5, 6, 7], ...].cpu().data.transpose(2, 3).transpose(3, 4).numpy()
        else:
            data = data.cpu().data.transpose(2, 3).transpose(3, 4).numpy()
        save_gif(data * 255, opt.num_frames, [8, 4], sampledir + '/{:06d}_gt.gif'.format(iteration))
    else:
        save_file_name = 'recon'

    save_gif(fakegif * 255, num_frames, grid, sampledir + '/{:06d}_%s.gif'.format(iteration)%save_file_name)

    if single:
        import scipy.misc
        for i in range(5):
            scipy.misc.imsave(sampledir +'/{:06d}_'.format(iteration) + str(i)+'.png', fakegif[0,i,...])
