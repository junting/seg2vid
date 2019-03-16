import glob
import os
from tqdm import tqdm

import multiprocessing as mp


cities = glob.glob('/mnt/lustrenew/DATAshare/leftImg8bit_sequence/val/*')

cities = [city[50:] for city in cities]

# print (cities)

# root_folder = '/mnt/lustrenew/DATAshare/leftImg8bit_sequence/train_semantic_segmask'
root_folder = '/mnt/lustrenew/DATAshare/unzip/leftImg8bit_sequence/val_pix2pixHD'

if not os.path.exists(root_folder):
    os.makedirs(root_folder)

for city in cities:
    if not os.path.exists(os.path.join(root_folder, city)):
        os.makedirs(os.path.join(root_folder, city))

mask_folder = '/mnt/lustrenew/DATAshare/synth/images/*synthesized_image.jpg'

segmask_list = glob.glob(mask_folder)

# print (len(segmask_list))
# print (segmask_list[0])
# print(segmask_list[0][0:97]+ '\\' + segmask_list[0][97:-5] + '\\' + segmask_list[0][-5::])
# print(segmask_list[0][98:-5]+'_ssmask.png')

# for segmask in tqdm(segmask_list):
#
#     city = segmask[98:-5].split('_')[0]
#     new_segmask_name = segmask[98:-5]+'_ssmask.png'
#
#     target_folder = os.path.join(root_folder, city)
#     command = 'mv ' + segmask[0:97]+ '\\' + segmask[97:-5] + '\\' + segmask[-5::] + ' ' + target_folder + '/' +new_segmask_name
#     os.system(command)
#     print (command)
#
#     break

def processing(segmask):
    city = segmask[38::].split('_')[0]
    new_segmask_name = segmask[38:-37] + 'pix2pixHD.png'

    target_folder = os.path.join(root_folder, city)
    # command = 'mv ' + segmask[0:97] + '\\' + segmask[97:-5] + '\\' + segmask[
    #                                                                  -5::] + ' ' + target_folder + '/' + new_segmask_name

    command = 'mv ' + segmask + ' ' + target_folder + '/' + new_segmask_name

    os.system(command)
    print (command)

mp.Pool(mp.cpu_count()).map(processing, segmask_list)
