from .dataset_path import *
from .cityscapes_dataset_w_mask_pix2pixHD import *
from .cityscapes_dataset_w_mask_pix2pixHD_two_path import *
from .cityscapes_dataset_w_mask import *
from .cityscapes_dataset_w_mask_two_path import *
from .kitti_dataset import *
from .kth_dataset import *
from .ucf_dataset import *


# def model_entry(config):
#     return globals()[config['arch']](**config['kwargs'])
