import numpy as np
import random
import os




def listinit(path, testfilename='testbaskeball.txt', trainfilename='trainbaskeball.txt'):

    output_dir = '/mnt/lustre/panjunting/f2video2.0/UCF-101/list'

    fop1 = open(os.path.join(output_dir, testfilename), 'w')
    fop2 = open(os.path.join(output_dir, trainfilename), 'w')

    for item in os.listdir(path):
        if item.endswith('.npy'):
            k = random.randint(0, 9)
            print (item)
            try:
                a = np.load(os.path.join(path, item))
                num = a.shape[0]

                if k == 9:
                    for i in range(num - 4):
                        fop1.write(item + ' ' + str(i) + '\n')
                else:
                    for i in range(num - 4):
                        fop2.write(item + ' ' + str(i) + '\n')
            except:
                print ("invalid npy")
                print (item)

# class toframe(object):
#     def __init__(self, path):
#         self.path = path
#         self.datalist = os.listdir(path)
#
#     def run1(self):
#         for item in self.datalist:
#             if mode == 'dir':
#                 dirpath = os.path.join(self.path, item.strip('.avi'))
#                 if tf.gfile.Exists(dirpath):
#                     break
#                 else:
#                     tf.gfile.MkDir(dirpath)
#             elif mode == 'npy':
#                 savef = []
#                 if item.endswith('.avi'):  # and item.replace('.avi','.npy') not in self.datalist:
#                     avipath = os.path.join(self.path, item)
#                     cap = cv2.VideoCapture(avipath)
#                     print avipath
#                     ret = 1
#                     while (ret):
#                         ret, frame = cap.read()
#                         if (ret):
#                             try:
#                                 frame = cv2.resize(np.array(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), (64, 64),
#                                                    interpolation=cv2.INTER_AREA)
#                                 savef.append(frame)
#                             except:
#                                 print 'noframe'
#                     savef = np.array(savef)
#                     np.save(avipath.replace('.avi', '_64.npy'), savef)


from tqdm import tqdm

def get_ucf_list(num_frames, in_filename, category):

    datapath = '/mnt/lustre/panjunting/f2video2.0/UCF-101'
    datalist = open(os.path.join(datapath, 'list', in_filename)).readlines()

    out_filename = in_filename[0:-4] + '_%dframes.txt'%num_frames

    f = open(os.path.join(datapath, out_filename), 'w')
    count = 0

    for idx in tqdm(range(len(datalist))):
        item = np.load(os.path.join(datapath, category, datalist[idx].split(' ')[0]).strip())
        num = item.shape[0]
        start = int(datalist[idx].split(' ')[1])
        if start + num_frames < num:
            f.write(datalist[idx])
            count += 1
    print len(datalist)
    print count
    f.close()





if __name__ == '__main__':
    # listinit(path='/mnt/lustre/panjunting/f2video2.0/UCF-101/Skiing',
    #          testfilename='testskiing.txt',
    #          trainfilename='trainskiing.txt')
    # get_ucf_list(18, 'trainskiing.txt', 'trainskiing_18frames.txt', 'Skiing')
    get_ucf_list(18, 'trainplayingviolin.txt', 'PlayingViolin')
    get_ucf_list(18, 'testplayingviolin.txt', 'PlayingViolin')
