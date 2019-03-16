import torch
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import glob
from tqdm import tqdm
from multiprocessing.dummy import Pool as ThreadPool

# image_root_dir = '/mnt/lustre/panjunting/video_generation/cityscapes/leftImg8bit/train_extra/*'
# image_root_dir = '/mnt/lustre/panjunting/video_generation/cityscapes/leftImg8bit/demoVideo/'

image_root_dir = '/mnt/lustrenew/DATAshare/leftImg8bit_sequence/val/'
listfile = open("cityscapes_val_sequence_full_18.txt", 'a')

# image_root_dir = '/mnt/lustrenew/DATAshare/gtFine/val/'
# listfile = open("cityscapes_val_sequence_w_mask_8.txt", 'a')
print (image_root_dir)
# max = [6299, 599, 4599]
# i = 0
num_frame_to_predict = 18

def gen_list_per_city(sub_dir):
    # image_list = glob.glob(sub_dir + "/*_gtFine_labelIds.png")
    image_list = glob.glob(sub_dir + "/*.png")
    for image_dir in tqdm(image_list):
        flag = True
        for j in range(1, num_frame_to_predict):
            new_dir = image_dir[0:-22] + str(int(image_dir[-22:-16]) + j).zfill(6) + image_dir[-16::]
            if not os.path.isfile(new_dir):
                flag = False
        if flag:
            # Replace mask suffix for image suffix
            # img_dir = image_dir.split(image_root_dir)[-1].split('_gtFine_labelIds.png')[0] + '_leftImg8bit.png'
            listfile.write(image_dir.split(image_root_dir)[-1] + "\n")
            # i += 1

        # new_dir = image_dir[0:-22] + str(int(image_dir[-22:-16]) + num_frame_to_predict).zfill(6) + image_dir[-16::]
        # if os.path.isfile(new_dir):
        #     listfile.write(image_dir.split(image_root_dir)[-1]+"\n")
        #     i += 1
    # print i

#25 26 27 28 29

cities = [sub_dir for sub_dir in glob.glob(image_root_dir + '*')]
print (cities)
# for city in cities:
#     gen_list_per_city(city)
# # make the Pool of workers
pool = ThreadPool(len(cities))

# open the urls in their own threads
# and return the results
results = pool.map(gen_list_per_city, cities)
listfile.close()
# close the pool and wait for the work to finish
pool.close()
pool.join()