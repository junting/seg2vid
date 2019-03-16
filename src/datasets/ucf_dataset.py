import torch
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import utils
import random

import time
random.seed(1234)


class UCF101(Dataset):
    def __init__(self, datapath, datalist, num_frame=5, size=128, returnpath=False):
        self.datapath = datapath
        self.datalist = open(datalist).readlines()
        self.numframe = num_frame
        self.size = size
        self.returnpath = returnpath

    def __len__(self):
        return len(self.datalist)

    def get_path_list(self, image_dir, num_frame, start):
        # new_dirs = [image_dir[1:-15] + str(int(image_dir[-15:-12]) + indx).zfill(3) + image_dir[-12::] for indx in
        #             range(num_frame)]
        #
        video_folder = os.path.join(image_dir[0:-4], str(start))
        # new_dirs = [os.path.join(video_folder, '%d.png'%indx) for indx in range(num_frame)]
        new_dirs = [video_folder for indx in range(num_frame)]
        return new_dirs

    def __getitem__(self, idx):

        item = np.load(os.path.join(self.datapath, self.datalist[idx].split(' ')[0]).strip())
        start = int(self.datalist[idx].split(' ')[1])
        item = item[start:start + self.numframe, :, :, :] / 255.0

        data = torch.from_numpy(np.array(item))
        data = data.contiguous()
        # print data.shape
        data = data.transpose(2, 3).transpose(1, 2)
        data = data.float()

        if self.size == 64:
            bs, T, c, h, w = data.size()
            data = Vb(data, requires_grad=False)
            data = F.avg_pool2d(data.view(-1, c, h, w), 2, 2)
            data = data.view(bs, T, c, h / 2, w / 2).data

        if self.returnpath:
            paths = self.get_path_list(self.datalist[idx].split(' ')[0].strip(), self.numframe, start)
            return data, paths
        return data


if __name__ == '__main__':
    start_time = time.time()

    # v_IceDancing_g06_c01.npy

    Path = '/mnt/lustre/panjunting/f2video2.0/UCF-101'

    # train_Dataset = UCF101(datapath=os.path.join(Path, 'IceDancing'),
    #                        datalist=os.path.join(Path, 'list/trainicedancing.txt'))
    test_Dataset = UCF101(datapath=os.path.join(Path, 'IceDancing'),
                          datalist=os.path.join(Path, 'list/testicedancing.txt'), returnpath=True)

    dataloader = DataLoader(test_Dataset, batch_size=32, shuffle=True, num_workers=8)

    sample, path = iter(dataloader).next()
    # print sample.size()
    import pdb
    pdb.set_trace()
    spent_time = time.time() - start_time
    # print spent_time
    # from tqdm import tqdm
    # i = 0
    # for sample in tqdm(iter(dataloader)):
    #     if i ==0:
    #         x = sample.shape
    #         i=1
    #     # print sample.shape
    #     if sample.shape != x:
    #         print sample.shape