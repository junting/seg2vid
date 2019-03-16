import torch
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import utils
import random
import cv2
import re
import time
random.seed(1234)


def cv2_tensor(pic):
    img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
    img = img.view(pic.shape[0], pic.shape[1], 3)
    img = img.transpose(0, 2).transpose(1, 2).contiguous()
    return img.float().div(255)


class KITTI(Dataset):
    def __init__(self, datapath, datalist, size=(128, 128), returnpath=False):
        self.datapath = datapath
        # self.datalist = open(datalist).readlines()
        self.datalist = datalist
        self.size = size
        self.returnpath = returnpath

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        image_name = os.path.join(self.datapath, self.datalist[idx].strip())
        img = cv2.resize(cv2.cvtColor(cv2.imread(image_name, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB), (self.size[1], self.size[0]))
        sample = cv2_tensor(img)
        if self.returnpath:
            return sample, self.datalist[idx][0:-4]
        return sample


if __name__ == '__main__':

    start_time = time.time()

    kitti_dataset_path = '/mnt/lustre/panjunting/kitti'
    kitti_dataset_list = os.listdir(kitti_dataset_path)

    kitti_Dataset = KITTI(datapath=kitti_dataset_path,
                                    datalist=kitti_dataset_list,
                                    size=(128, 256), returnpath=True)

    dataloader = DataLoader(kitti_Dataset, batch_size=32, shuffle=False, num_workers=8)

    sample, path = iter(dataloader).next()
    import pdb
    pdb.set_trace()
    print (sample.shape)    # #
    # spent_time = time.time() - start_time
    # print spent_time
    # from tqdm import tqdm
    # a= [ 1 for sample in tqdm(iter(dataloader))]
