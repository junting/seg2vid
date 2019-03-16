import torch
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import glob
from tqdm import tqdm
from multiprocessing.dummy import Pool as ThreadPool

# image_root_dir = '/mnt/lustre/panjunting/video_generation/cityscapes/leftImg8bit/train_extra/*'
# image_root_dir = '/mnt/lustre/panjunting/video_generation/cityscapes/leftImg8bit/demoVideo/'
image_root_dir = '/mnt/lustre/panjunting/video_generation/cityscapes/leftImg8bit_sequence/train/'
listfile = open("cityscapes_train_sequence_full_8.txt", 'r')
num_frame_to_predict = 8

file_names = [file_name.strip() for file_name in listfile.readlines()]

for file_name in tqdm(file_names):
    image_dir = image_root_dir + file_name

    for i in range(num_frame_to_predict):
        new_dir = image_dir[0:-22] + str(int(image_dir[-22:-16]) + i).zfill(6) + image_dir[-16::]
        if not os.path.isfile(new_dir):
            print new_dir
            print file_dir