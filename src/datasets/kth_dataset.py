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
    img = img.view(pic.shape[0], pic.shape[1], 1)
    img = img.transpose(0, 2).transpose(1, 2).contiguous()
    return img.float().div(255)

def replace_index_and_read(image_dir, indx, size):
    new_dir = image_dir[0:-15] + str(int(image_dir[-15:-12]) + indx).zfill(3) + image_dir[-12::]
    try:
        img = cv2.resize(cv2.imread(new_dir, 0), size)
    except:
        print ('orgin_dir: ' + image_dir)
        print ('new_dir: ' + new_dir)
    # center crop
    # img = cv2.resize(cv2.imread(new_dir, 0), size)
    frame = cv2_tensor(img)
    return frame

def imagetoframe(image_dir, size, num_frame):

    samples = [replace_index_and_read(image_dir, indx, size) for indx in range(num_frame)]
    return torch.stack(samples)

def get_path_list(image_dir, num_frame):
    new_dirs = [image_dir[1:-15] + str(int(image_dir[-15:-12]) + indx).zfill(3) + image_dir[-12::] for indx in range(num_frame)]
    return new_dirs

class KTH(Dataset):
    def __init__(self, dataset_root,  datalist, num_frames=5, size=(128, 128), returnpath=False):
        self.datalist = open(datalist).readlines()
        self.size = size
        self.num_frame = num_frames
        self.dataset_root = dataset_root
        self.returnpath = returnpath
    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        sample = imagetoframe(self.dataset_root+self.datalist[idx].strip(), self.size, self.num_frame)

        if self.returnpath:
            paths = get_path_list(self.datalist[idx].strip(), self.num_frame)
            return sample, paths
        else:
            return sample


if __name__ == '__main__':

    start_time = time.time()
    cityscapes_Dataset = KTH(dataset_root='/mnt/lustrenew/panjunting/kth/KTH/processed', datalist='kth_train_16.txt',
                                size=(128, 128), num_frames=16)

    dataloader = DataLoader(cityscapes_Dataset, batch_size=32, shuffle=False, num_workers=1)

    sample = iter(dataloader).next()
    print (sample.size())
    # from tqdm import tqdm
    # a= [ 1 for sample in tqdm(iter(dataloader))]
