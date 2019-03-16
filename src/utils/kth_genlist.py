import os
import numpy as np
import glob
from tqdm import tqdm
from multiprocessing.dummy import Pool as ThreadPool


def gen_list_per_city(image_root_dir, category, split, datalist, num_frame_to_predict):
    listfile = open("kth_" + split + "_" + category + "_%d_ok.txt" % num_frame_to_predict, 'w')
    for image_dir in tqdm(datalist):
        flag = True
        for j in range(1, num_frame_to_predict):
            new_dir = image_dir[0:-15] + str(int(image_dir[-15:-12]) + j).zfill(3) + image_dir[-12::]
            new_dir = image_root_dir + new_dir
            if not os.path.isfile(new_dir):
                flag = False
        if flag:
            # Replace mask suffix for image suffix
            # img_dir = image_dir.split(image_root_dir)[-1].split('_gtFine_labelIds.png')[0] + '_leftImg8bit.png'
            listfile.write(image_dir + "\n")
    listfile.close()



def get_list(category, num_frame_to_predict, split):
    # image_root_dir = '/mnt/lustre/panjunting/video_generation/cityscapes/leftImg8bit/train_extra/*'
    # image_root_dir = '/mnt/lustre/panjunting/video_generation/cityscapes/leftImg8bit/demoVideo/'

    # image_root_dir = '/mnt/lustrenew/DATAshare/leftImg8bit_sequence/val/'
    # listfile = open("cityscapes_val_sequence_full_18.txt", 'a')

    image_root_dir = '/mnt/lustrenew/panjunting/kth/KTH/processed/'
    # listfile = open("kth_"+split+"_"+category+"_%d_ok.txt" %num_frame_to_predict, 'w')


    # image_root_dir = '/mnt/lustrenew/DATAshare/gtFine/val/'
    # listfile = open("cityscapes_val_sequence_w_mask_8.txt", 'a')
    print (image_root_dir)
    # max = [6299, 599, 4599]
    # i = 0
    # cities = [sub_dir for sub_dir in glob.glob(image_root_dir + '*')]
    # print (cities)
    # for city in cities:
    #     gen_list_per_city(city)
    # # make the Pool of workers
    # pool = ThreadPool(len(cities))


    datalist= open('kth_'+split+'_'+category+'_16_ok.txt', 'r')
    datalist = [l.strip() for l in datalist.readlines()]
    gen_list_per_city(image_root_dir, category, split, datalist, num_frame_to_predict)
    # open the urls in their own threads
    # and return the results
    # results = pool.map(gen_list_per_city, cities)
    # close the pool and wait for the work to finish
    # pool.close()
    # pool.join()


def process_per_class():
    listfile = open("kth_train_walking_16.txt", 'w')
    datalist = open('kth_train_16.txt','r')
    datalist = [l.strip() for l in datalist.readlines()]

    for image_dir in tqdm(datalist):
        if image_dir.split('/')[1] == 'walking':
            listfile.write(image_dir+'\n')
    listfile.close()


def main():
    # process_per_class()
    # main()

    #  self.classes = ['boxing', 'handclapping', 'handwaving', 'jogging', 'running', 'walking']
    datalist = open('kth_train_walking_16.txt','r')
    listfile = open('kth_train_walking_16_ok.txt', 'w')
    datalist = [l.strip() for l in datalist.readlines()]
    print(len(datalist))
    datalist = set(datalist)
    for l in tqdm(datalist):
        listfile.write(l+'\n')
    print (len(datalist))
    # process_per_class()

def new_main(class_name='hadwaving', split='train'):

    data_list = open('kth_' + split + '_16.txt', 'r')
    datalist = [l.strip() for l in data_list.readlines()]

    listfile = open("kth_" + split + "_%s_16_ok.txt"%class_name, 'w')

    class_specific_list = [image_dir for image_dir in datalist if image_dir.split('/')[1] == class_name]
    print(len(class_specific_list))

    class_specific_list_unique = set(class_specific_list)

    for l in tqdm(class_specific_list_unique):
        listfile.write(l+'\n')
    print (len(class_specific_list_unique))
    listfile.close()
    data_list.close()

# new_main('handwaving', split='train')
# new_main('handwaving', split='test')

# get_list('handwaving', 18, split='train')
get_list('handwaving', 18, split='test')