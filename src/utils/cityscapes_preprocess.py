import os
import cv2
import glob
from tqdm import tqdm

from multiprocessing.dummy import Pool as ThreadPool

# image_root_dir = '/mnt/lustre/panjunting/video_generation/cityscapes/leftImg8bit/train_extra/*'
# resized_image_root_dir = '/mnt/lustre/panjunting/video_generation/cityscapes/leftImg8bit/train_extra_256x512/'

image_root_dir = '/mnt/lustrenew/DATAshare/leftImg8bit_sequence/val/*'
resized_image_root_dir = '/mnt/lustrenew/DATAshare/leftImg8bit_sequence/val_256x128'

img_size = (256, 128)


# for sub_dir in glob.glob(image_root_dir):
#     for image_dir in  tqdm(glob.glob(sub_dir + "/*.png")):
#         imageResized = cv2.resize(cv2.imread(image_dir, cv2.IMREAD_COLOR), img_size, interpolation=cv2.INTER_AREA)
#         filename = image_dir.split('/')[-1]
#         city_name = sub_dir.split('/')[-1]
#
#         pathOutputImages = os.path.join(resized_image_root_dir, city_name)
#
#         if not os.path.isdir(pathOutputImages):
#             os.makedirs(pathOutputImages)
#         cv2.imwrite(os.path.join(pathOutputImages, filename), imageResized)


def resize_and_save(sub_dir):
    print sub_dir
    for image_dir in tqdm(glob.glob(sub_dir + "/*.png")):
        imageResized = cv2.resize(cv2.imread(image_dir, cv2.IMREAD_COLOR), img_size, interpolation=cv2.INTER_AREA)
        filename = image_dir.split('/')[-1]
        city_name = sub_dir.split('/')[-1]

        pathOutputImages = os.path.join(resized_image_root_dir, city_name)

        if not os.path.isdir(pathOutputImages):
            os.makedirs(pathOutputImages)
        cv2.imwrite(os.path.join(pathOutputImages, filename), imageResized)


cities = [sub_dir for sub_dir in glob.glob(image_root_dir)]

# make the Pool of workers
pool = ThreadPool(len(cities))

# open the urls in their own threads
# and return the results
results = pool.map(resize_and_save, cities)

# close the pool and wait for the work to finish
pool.close()
pool.join()