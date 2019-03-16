import os
import glob
from tqdm import tqdm

image_root_dir = '/mnt/lustre/share/charlie/synth/images/'
image_list_file = open('cityscapes_test_pix2pixImage_list.txt', 'w')


for file_name in tqdm(glob.glob(image_root_dir+'*_synthesized_image.jpg')):
    image_list_file.write(file_name.split(image_root_dir)[-1]+'\n')
image_list_file.close()
