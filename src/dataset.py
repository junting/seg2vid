from datasets.dataset_path import *


def get_training_set(opt):

    assert opt.datset in ['cityscapes', 'cityscapes_two_path', 'kth']

    if opt.dataset == 'cityscapes':
        from datasets.cityscapes_dataset_w_mask import Cityscapes

        train_Dataset = Cityscapes(datapath=CITYSCAPES_TRAIN_DATA_PATH, datalist=CITYSCAPES_TRAIN_DATA_LIST,
                                   size=opt.input_size, split='train', split_num=1, num_frames=opt.num_frames)

    elif opt.dataset == 'cityscapes_two_path':
        from datasets.cityscapes_dataset_w_mask_two_path import Cityscpes
        train_Dataset = Cityscapes(datapath=CITYSCAPES_TRAIN_DATA_PATH,
                                   mask_data_path=CITYSCAPES_TRAIN_DATA_SEGMASK_PATH,
                                   datalist=CITYSCAPES_TRAIN_DATA_LIST,
                                   size=opt.input_size, split='train', split_num=1, num_frames=opt.num_frames,
                                   mask_suffix='ssmask.png')

    elif opt.dataset == 'kth':

        from datasets.kth_dataset import KTH
        train_Dataset = KTH(dataset_root=KTH_DATA_PATH,
                            datalist=KTH_DATA_PATH_LIST,
                            size=opt.input_size, num_frames=opt.num_frames)

    return train_Dataset


def get_test_set(opt):

    assert opt.dataset in ['cityscapes', 'cityscapes_two_path', 'kth', 'ucf101', 'KITTI']

    if opt.dataset == 'cityscapes':
        from datasets.cityscapes_dataset_w_mask import Cityscapes
        test_Dataset = Cityscapes(datapath=CITYSCAPES_VAL_DATA_PATH, mask_data_path=CITYSCAPES_VAL_DATA_SEGMASK_PATH,
                                  datalist=CITYSCAPES_VAL_DATA_LIST,
                                  size=opt.input_size, split='train', split_num=1, num_frames=opt.num_frames,
                                  mask_suffix='ssmask.png', returnpath=True)

    elif opt.dataset == 'cityscapes_two_path':
        from datasets.cityscapes_dataset_w_mask_two_path import Cityscapes
        test_Dataset = Cityscapes(datapath=CITYSCAPES_VAL_DATA_PATH, mask_data_path=CITYSCAPES_VAL_DATA_SEGMASK_PATH,
                                  datalist=CITYSCAPES_VAL_DATA_LIST,
                                  size=opt.input_size, split='train', split_num=1, num_frames=opt.num_frames,
                                  mask_suffix='ssmask.png', returnpath=True)

    elif opt.dataset == 'cityscapes_pix2pixHD':
        from cityscapes_dataloader_w_mask_pix2pixHD import Cityscapes
        test_Dataset = Cityscapes(datapath=CITYSCAPES_TEST_DATA_PATH,
                                  mask_data_path=CITYSCAPES_VAL_DATA_SEGMASK_PATH,
                                  datalist=CITYSCAPES_VAL_DATA_MASK_LIST,
                                  size= opt.input_size, split='test', split_num=1,
                                  num_frames=opt.num_frames, mask_suffix='ssmask.png', returnpath=True)

    elif opt.dataset == 'kth':
        from datasets.kth_dataset import KTH
        test_Dataset = KTH(dataset_root=KTH_DATA_PATH,
                           datalist='./file_list/kth_test_%s_16_ok.txt' % opt.category,
                           size=opt.input_size, num_frames=opt.num_frames)

    elif opt.dataset == 'KITTI':
        from datasets.kitti_dataset import KITTI
        kitti_dataset_list = os.listdir(KITTI_DATA_PATH)
        test_Dataset = KITTI(datapath=KITTI_DATA_PATH, datalist=kitti_dataset_list, size=opt.input_size,
                             returnpath=True)

    elif opt.dataset == 'ucf101':
        from datasets.ucf101_dataset import UCF101
        test_Dataset = UCF101(datapath=os.path.join(UCF_101_DATA_PATH, category),
                              datalist=os.path.join(UCF_101_DATA_PATH, 'list/test%s.txt' % (opt.category.lower())), returnpath=True)

    return test_Dataset