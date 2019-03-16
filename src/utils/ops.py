from __future__ import division
import torch
# import vgg
from torch.autograd import Variable as Vb
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torchvision.utils  as tov
import cv2
import datetime
import numpy as np


def make_color_wheel():
    """
    Generate color wheel according Middlebury color code
    :return: Color wheel
    """
    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR

    colorwheel = np.zeros([ncols, 3])

    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.transpose(np.floor(255 * np.arange(0, RY) / RY))
    col += RY

    # YG
    colorwheel[col:col + YG, 0] = 255 - np.transpose(np.floor(255 * np.arange(0, YG) / YG))
    colorwheel[col:col + YG, 1] = 255
    col += YG

    # GC
    colorwheel[col:col + GC, 1] = 255
    colorwheel[col:col + GC, 2] = np.transpose(np.floor(255 * np.arange(0, GC) / GC))
    col += GC

    # CB
    colorwheel[col:col + CB, 1] = 255 - np.transpose(np.floor(255 * np.arange(0, CB) / CB))
    colorwheel[col:col + CB, 2] = 255
    col += CB

    # BM
    colorwheel[col:col + BM, 2] = 255
    colorwheel[col:col + BM, 0] = np.transpose(np.floor(255 * np.arange(0, BM) / BM))
    col += + BM

    # MR
    colorwheel[col:col + MR, 2] = 255 - np.transpose(np.floor(255 * np.arange(0, MR) / MR))
    colorwheel[col:col + MR, 0] = 255

    return colorwheel


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


def compute_color(u, v):
    """
    compute optical flow color map
    :param u: optical flow horizontal map
    :param v: optical flow vertical map
    :return: optical flow in color code
    """
    [h, w] = u.shape
    img = np.zeros([h, w, 3])
    nanIdx = np.isnan(u) | np.isnan(v)
    u[nanIdx] = 0
    v[nanIdx] = 0

    colorwheel = make_color_wheel()
    ncols = np.size(colorwheel, 0)

    rad = np.sqrt(u ** 2 + v ** 2)

    a = np.arctan2(-v, -u) / np.pi

    fk = (a + 1) / 2 * (ncols - 1) + 1

    k0 = np.floor(fk).astype(int)

    k1 = k0 + 1
    k1[k1 == ncols + 1] = 1
    f = fk - k0

    for i in range(0, np.size(colorwheel, 1)):
        tmp = colorwheel[:, i]
        col0 = tmp[k0 - 1] / 255
        col1 = tmp[k1 - 1] / 255
        col = (1 - f) * col0 + f * col1

        idx = rad <= 1
        col[idx] = 1 - rad[idx] * (1 - col[idx])
        notidx = np.logical_not(idx)

        col[notidx] *= 0.75
        img[:, :, i] = np.uint8(np.floor(255 * col * (1 - nanIdx)))

    return img


# def saveflow(flows, imgsize, size, savepath):
#     num_images = size[0] * size[1]
#     flows = merge(flows[0:num_images], size) * 32
#     u = flows[:, :, 0]
#     v = flows[:, :, 1]
#     image = compute_color(u, v)
#     flow = cv.resize(image, imgsize)
#     cv2.imwrite(savepath, flow)

def saveflow(flows, imgsize, savepath):
    u = flows[:, :, 0]*3
    v = flows[:, :, 1]*3
    image = compute_color(u, v)
    flow = cv2.resize(image, imgsize)
    cv2.imwrite(savepath, flow)


def compute_flow_color_map(flows):
    u = flows[:, :, 0] * 3
    v = flows[:, :, 1] * 3
    flow = compute_color(u, v)
    # flow = cv2.resize(image, imgsize)
    return flow

def compute_flow_img(flows, imgsize, size):
    # import pdb
    # pdb.set_trace()

    num_images = size[0] * size[1]
    flows = merge(flows[0:num_images], size) * 3
    u = flows[:, :, 0]
    v = flows[:, :, 1]
    image = compute_color(u, v)
    return image
    # cv2.imwrite(savepath, image)

import imageio

def save_flow_sequence(flows, length, imgsize, size, savepath):
    flow_seq = [np.uint8(compute_flow_img(flows[:,i,...], imgsize, size)) for i in range(length)]
    imageio.mimsave(savepath, flow_seq, fps=int(length))


def saveflowopencv(flows, imgsize, size, savepath):
    # print flows.shape
    hsv = np.uint8(np.zeros([128 * 4, 128 * 4, 3]))
    hsv[..., 1] = 255
    flows = np.clip(merge(flows, size), -1, 1)

    mag, ang = cv2.cartToPolar(flows[..., 0], flows[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    bgr = bgr * 10
    cv2.imwrite(savepath, bgr)


class flowwrapper(nn.Module):
    def __init__(self):
        super(flowwrapper, self).__init__()

    def forward(self, x, flow):
        # flow: (batch size, 2, height, width)
        # x = x.cuda()
        N = x.size()[0]
        H = x.size()[2]
        W = x.size()[3]
        base_grid = torch.zeros([N, H, W, 2])
        linear_points = torch.linspace(-1, 1, W) if W > 1 else torch.Tensor([-1])
        base_grid[:, :, :, 0] = torch.ger(torch.ones(H), linear_points).expand_as(base_grid[:, :, :, 0])
        linear_points = torch.linspace(-1, 1, H) if H > 1 else torch.Tensor([-1])
        base_grid[:, :, :, 1] = torch.ger(linear_points, torch.ones(W)).expand_as(base_grid[:, :, :, 1])
        if x.is_cuda:
            base_grid = Vb(base_grid).cuda()
        else:
            base_grid = Vb(base_grid)
        # print flow.shape
        flow = flow.transpose(1, 2).transpose(2, 3)
        # print flow.size()
        grid = base_grid - flow
        # print grid.size()
        out = F.grid_sample(x, grid)
        return out


def testcode():
    a = flowwrapper()
    img = cv2.imread('image.jpg')
    b, g, r = cv2.split(img)
    img = cv2.merge([r, g, b])
    image = torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes()))
    image = image.view(img.shape[0], img.shape[1], 3)
    image = image.transpose(0, 2).transpose(1, 2).contiguous()
    image = image.float().div(255)
    img = Vb(torch.stack([image])).cuda()
    flow = Vb(torch.randn([1, 2, img.size()[2], img.size()[3]]).div(40)).cuda()
    newimg = a(img, flow)
    tov.save_image(newimg.data, 'img1.jpg')


def viewflow(filename):
    '''
    cap=cv2.VideoCapture(filename)
    _,frame1=cap.read()
    _,frame2=cap.read()
    prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prvs, next, 0.5, 3, 15, 3, 5, 1.2, 0)
    flow=np.array(flow)
    print flow.shape
    u=flow[:,:,0]
    v=flow[:,:,1]
    '''
    u = -np.ones([128, 128]) * 40
    v = np.zeros([128, 128]) * 40
    image = compute_color(u, v)
    cv2.imwrite('flow.jpg', image)
    # cv2.imwrite('frame1.jpg',frame1)
    # cv2.imwrite('frame2.jpg',frame2)
    # cap.release()


def gradientx(img):
    return img[:, :, :, :-1] - img[:, :, :, 1:]


def gradienty(img):
    return img[:, :, :-1, :] - img[:, :, 1:, :]


def length_sq(x):
    return torch.sum(x**2, dim=1)


def occlusion(flow, flowback, flowwarp, opt):
    flow_bw_warped = flowwarp(flow, -flowback)
    flow_fw_warped = flowwarp(flowback, flow)

    # flow_diff_fw = torch.abs(flow - flow_bw_warped)
    # flow_diff_bw = torch.abs(flowback - flow_fw_warped)

    flow_diff_fw = torch.abs(flow - flow_fw_warped)
    flow_diff_bw = torch.abs(flowback - flow_bw_warped)

    occ_thresh = opt.alpha1 * (length_sq(flow) + length_sq(flowback)) + opt.alpha2

    occ_fw = (length_sq(flow_diff_fw) > occ_thresh).float().unsqueeze(1)
    occ_bw = (length_sq(flow_diff_bw) > occ_thresh).float().unsqueeze(1)

    return 1-occ_fw, 1-occ_bw


def get_occlusion_mask(flow, flowback, flowwarpper, opt, t=4):
    mask_fw = []
    mask_bw = []
    for i in xrange(t):
        tmp_mask_fw, tmp_mask_bw = occlusion(flow[:, :, i, :, :], flowback[:, :, i, :, :], flowwarpper, opt)

        mask_fw.append(tmp_mask_fw)
        mask_bw.append(tmp_mask_bw)

    mask_bw = torch.cat(mask_bw, 1)
    mask_fw = torch.cat(mask_fw, 1)
    return mask_fw, mask_bw

def warp(frame, flow, opt, flowwarpper, mask):
    '''Use mask before warpping'''
    out = [torch.unsqueeze(flowwarpper(frame, flow[:, :, i, :, :] * mask[:, i:i + 1, ...]), 1)
           for i in range(opt.num_predicted_frames)]
    output = torch.cat(out, 1)  # (64, 4, 3, 128, 128)
    return output


def warp_back(frame2, flowback, opt, flowwarpper, mask):
    prevframe = [
        torch.unsqueeze(flowwarpper(frame2[:, ii, :, :, :], -flowback[:, :, ii, :, :] * mask[:, ii:ii + 1, ...]), 1)
        for ii in range(opt.num_predicted_frames)]
    output = torch.cat(prevframe, 1)
    return output


def refine(input, flow, mask, refine_net, opt, noise_bg):
    '''Go through the refine network.'''
    # apply mask to the warpped image
    out = [torch.unsqueeze(refine_net(input[:, i, ...] * mask[:, i:i + 1, ...] + noise_bg * (1. - mask[:, i:i + 1, ...])
                                           , flow[:, :, i, :, :]
                                      ), 1) for i in range(opt.num_predicted_frames)]

    out = torch.cat(out, 1)
    return out

def refine_id(input, flow, mask, refine_net, opt, noise_bg):
    '''Go through the refine network.'''
    # apply mask to the warpped image
    out = [torch.unsqueeze(refine_net(input[:, i+1, ...] * mask[:, i:i + 1, ...] + noise_bg * (1. - mask[:, i:i + 1, ...])
                                           , flow[:, :, i, :, :]
                                      ), 1) for i in range(opt.num_predicted_frames)]
    out1 = [refine_net(input[:, 0, ...], flow[:, :, 0, :, :]).unsqueeze(1)]

    out = torch.cat(out1+out, 1)
    return out

def refine_w_mask(input, ssmask, flow, mask, refine_net, opt, noise_bg):
    '''Go through the refine network.'''
    # apply mask to the warpped image
    out = [torch.unsqueeze(refine_net(input[:, i, ...] * mask[:, i:i + 1, ...] + noise_bg * (1. - mask[:, i:i + 1, ...])
                                           , flow[:, :, i, :, :], ssmask[:, i, ...]
                                      ), 1) for i in range(opt.num_predicted_frames)]

    out = torch.cat(out, 1)
    return out


if __name__ == '__main__':

    viewflow('a')
    # viewflow('/ssd/10.10.20.21/share/guojiaming/UCF-101/Surfing/v_Surfing_g15_c02.avi')

    img = Vb(torch.randn([16, 3, 128, 128]).div(40)).cuda()
    flow = Vb(torch.randn([16, 2, img.size()[2], img.size()[3]]).div(40)).cuda()
    begin = datetime.datetime.now()
    print (quickflowloss(flow, img))
    end = datetime.datetime.now()
    time2 = end-begin
    print (time2.total_seconds())


    neighber=5
    bound = ((neighber-1)/2)
    x = torch.zeros([neighber, neighber])
    linear_points = torch.linspace(-bound, bound, neighber)
    x = torch.ger(torch.ones(neighber), linear_points).expand_as(x)

    y = torch.ger(linear_points, torch.ones(neighber)).expand_as(x)
    dst = x**2+y**2
    print (dst)

    testcode()
