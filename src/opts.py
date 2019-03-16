import argparse


def parse_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--batch_size',
        default=16,
        type=int,
        help='batch size')
    parser.add_argument(
        '--input_channel',
        default=3,
        type=int,
        help='input image channel (3 for RGB, 1 for Grayscale)')
    parser.add_argument(
        '--input_size',
        default=(128, 256),
        type=tuple,
        help='input image size')
    parser.add_argument(
        '--num_frames',
        default=5,
        type=int,
        help='number of frames for each video clip')
    parser.add_argument(
        '--num_predicted_frames',
        default=4,
        type=int,
        help='number of frames to predict')
    parser.add_argument(
        '--num_epochs',
        default=1000,
        type=int,
        help=
        'Max. number of epochs to train.'
    )
    parser.add_argument(
        '--lr_rate',
        default=0.001,
        type=float,
        help='learning rate used for training.'
    )
    parser.add_argument(
        '--lamda',
        default=0.1,
        type=float,
        help='weight use to penalize the generated occlusion mask.'
    )
    parser.add_argument(
        '--workers',
        default=3,
        type=int,
        help='number of workers used for data loading.'
    )
    parser.add_argument(
        '--dataset',
        default='cityscapes',
        type=str,
        help=
        'Used dataset (cityscpes | cityscapes_two_path | kth | ucf101).'
    )
    parser.add_argument(
        '--iter_to_load',
        default=1,
        type=int,
        help='iteration to load'
    )
    parser.add_argument(
        '--mask_channel',
        default=20,
        type=int,
        help='channel of the input semantic lable map'
    )
    parser.add_argument(
        '--category',
        default='walking',
        type=str,
        help='class category of the video to train (only apply to KTH and UCF101)'
    )
    parser.add_argument(
        '--seed',
        default=31415,
        type=int,
        help='Manually set random seed'
    )
    parser.add_argument(
        '--suffix',
        default='',
        type=str,
        help='model suffix'
    )

    args = parser.parse_args()

    return args