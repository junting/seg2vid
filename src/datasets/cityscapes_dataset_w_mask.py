import torch
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import utils
import random
import cv2
import re
import time
from scipy.misc import imread
random.seed(1234)


# num_class = 20
def cv2_tensor(pic):
    img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
    img = img.view(pic.shape[0], pic.shape[1], 3)
    img = img.transpose(0, 2).transpose(1, 2).contiguous()
    return img.float().div(255)


def replace_index_and_read(image_dir, indx, size):
    new_dir = image_dir[0:-22] + str(int(image_dir[-22:-16]) + indx).zfill(6) + image_dir[-16::]

    img = cv2.resize(cv2.cvtColor(cv2.imread(new_dir, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB), (size[1], size[0]))
    # except:
    #     print ('orgin_dir: ' + image_dir)
    #     print ('new_dir: ' + new_dir)
    # # center crop
    # if img.shape[0] != img.shape[1]:
    #     frame = cv2_tensor(img[:, 64:64+128])
    # else:
    frame = cv2_tensor(img)
    return frame


def load_mask(mask_dir, size):
    mask = imread(mask_dir)
    mask = cv2.resize(mask, (size[1], size[0]), interpolation=cv2.INTER_NEAREST)
    mask_volume = np.concatenate([np.expand_dims(mask, 0) == i for i in range(0, 20)], 0).astype(int)
    mask_volume = torch.from_numpy(mask_volume).contiguous().type(torch.FloatTensor)
    return mask_volume


def imagetoframe(image_dir, size, num_frame):

    samples = [replace_index_and_read(image_dir, indx, size) for indx in range(num_frame)]
    return torch.stack(samples)


def complete_full_list(image_dir, num_frames, output_name):
    dir_list = [image_dir[0:-22] + str(int(image_dir[-22:-16]) + i).zfill(6) + '_' + output_name for i in range(num_frames)]
    return dir_list


class Cityscapes(Dataset):
    def __init__(self, datapath, mask_data_path, datalist, num_frames=5, size=(128, 128), split='train', split_num=1600, mask_suffix='gtFine_labelIds.png', returnpath=False):
        self.datapath = datapath
        # if split is 'train':
        #     self.datalist = open(datalist).readlines()[0:-split_num]
        # else:
        #     self.datalist = open(datalist).readlines()[-split_num::]
        self.datalist = open(datalist).readlines()
        self.size = size
        self.num_frame = num_frames
        self.mask_root = mask_data_path
        self.mask_suffix = mask_suffix
        self.returnPath = returnpath
        
    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        image_dir = os.path.join(self.datapath, self.datalist[idx].strip())
        sample = imagetoframe(image_dir, self.size, self.num_frame)
        # mask_dir = os.path.join(self.mask_root, self.datalist[idx].strip()[0:-15]+'gtFine_labelIds.png')
        mask_dir = os.path.join(self.mask_root, self.datalist[idx].strip()[0:-15]+self.mask_suffix)
        mask = load_mask(mask_dir, self.size)
        if self.returnPath:
            return  sample, mask, complete_full_list(self.datalist[idx].strip(), self.num_frame, 'pred.png')
        else:
            return sample, mask
if __name__ == '__main__':

    start_time = time.time()
    from dataset_path import *
    cityscapes_Dataset = Cityscapes(datapath=CITYSCAPES_VAL_DATA_PATH, mask_data_path=CITYSCAPES_VAL_DATA_SEGMASK_PATH,
                                    datalist=CITYSCAPES_VAL_DATA_MASK_LIST,
                                    size=(128, 256), split='train', split_num=1,
                                    num_frames=5, mask_suffix='ssmask.png')

    dataloader = DataLoader(cityscapes_Dataset, batch_size=32, shuffle=False, num_workers=4)

    sample, mask = iter(dataloader).next()
    print (sample.shape)
    print (mask.shape)

