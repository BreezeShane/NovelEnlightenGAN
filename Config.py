from utils.Metrics import *
import os
import torch
import argparse

ROOT_PATH = os.getcwd()
GPU_IDs = [id for id in range(torch.cuda.device_count())]
IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

# todo: make obj opt for more args
parser = argparse.ArgumentParser(prefix_chars='-_')
parser.add_argument("--train", action='store_true')
parser.add_argument("--predict", action='store_true')

parser.add_argument("--use_gpu", type=bool, default=True if torch.cuda.is_available() else False)
parser.add_argument("--gpu_id", type=int, default=0)
parser.add_argument("--batchSize", type=int, default=8)
parser.add_argument('--which_direction', type=str, default='AtoB', help='AtoB or BtoA')
parser.add_argument("--fineSize", type=int, default=256)
parser.add_argument("--input_nc", type=int, default=3)
parser.add_argument("--output_nc", type=int, default=3)
parser.add_argument("--no_vgg_instance", type=bool, default=False)
parser.add_argument("--skip", type=float, default=0.8)
parser.add_argument('--norm', type=str, default='instance')
parser.add_argument('--no_dropout', type=bool, default=True)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument("--patchD", type=bool, default=True)
parser.add_argument("--syn_norm", type=bool, default=False)
parser.add_argument("--use_norm", type=float, default=1.0)
parser.add_argument("--tanh", type=bool, default=False)
parser.add_argument("--which_model_netD", type=str, default='no_norm')
parser.add_argument("--which_model_netG", type=str, default='Unet_resize_conv')
parser.add_argument('--vary', type=int, default=1, help='use light data augmentation')
parser.add_argument('--low_times', type=int, default=200, help='choose the number of crop for patch discriminator')
parser.add_argument('--lighten', action='store_true', help='normalize attention map')
parser.add_argument('--high_times', type=int, default=400,
                    help='choose the number of crop for patch discriminator')
parser.add_argument('--noise', type=float, default=0, help='variance of noise')
parser.add_argument('--input_linear', action='store_true', help='lieanr scaling input')
parser.add_argument('--times_residual', action='store_true', help='output = input + residual*attention')
parser.add_argument('--linear_add', action='store_true', help='lieanr scaling input')
parser.add_argument('--latent_threshold', action='store_true', help='lieanr scaling input')
parser.add_argument('--latent_norm', action='store_true', help='lieanr scaling input')
parser.add_argument('--linear', action='store_true', help='tanh')
parser.add_argument('--patchSize', type=int, default=32, help='then crop to this size')
parser.add_argument('--patchD_3', type=int, default=0, help='choose the number of crop for patch discriminator')
parser.add_argument('--use_wgan', type=float, default=0, help='use wgan-gp')
parser.add_argument('--use_ragan', action='store_true', help='use ragan')
parser.add_argument('--hybrid_loss', action='store_true', help='use lsgan and ragan separately')
parser.add_argument('--D_P_times2', action='store_true', help='loss_D_P *= 2')
parser.add_argument('--IN_vgg', action='store_true', help='patch vgg individual')
parser.add_argument('--vgg_mean', action='store_true', help='substract mean in vgg loss')
parser.add_argument('--vgg_choose', type=str, default='relu5_3', help='choose layer for vgg')
parser.add_argument('--vgg_maxpooling', action='store_true', help='normalize attention map')
parser.add_argument('--patch_vgg', action='store_true', help='use vgg loss between each patch')
parser.add_argument('--new_lr', action='store_true', help='tanh')

# Train args

parser.add_argument('--continue_train', action='store_true')
parser.add_argument('--which_epoch', type=str, default='latest')
parser.add_argument("--lr", type=float, default=0.0001)
parser.add_argument('--resize_or_crop', type=str, default='crop')
parser.add_argument("--serial_batches", type=bool, default=False)
parser.add_argument("--no_flip", type=bool, default=False)
parser.add_argument('--max_dataset_size', type=int, default=float("inf"))
parser.add_argument('--niter', type=int, default=100)
parser.add_argument('--niter_decay', type=int, default=100)
parser.add_argument('--vgg', type=float, default=0)
parser.add_argument('--fcn', type=float, default=0)
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--use_avgpool', type=float, default=0)
parser.add_argument('--n_layers_D', type=int, default=3)
parser.add_argument('--n_layers_patchD', type=int, default=3)
parser.add_argument('--pool_size', type=int, default=50,
                    help='the size of image buffer that stores previously generated images')
parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
parser.add_argument('--save_latest_freq', type=int, default=5000, help='frequency of saving the latest results')
parser.add_argument('--save_epoch_freq', type=int, default=5,
                    help='frequency of saving checkpoints at the end of epochs')

# Test args
parser.add_argument("--use_models",action='store_true')

opt = parser.parse_args()
opt.isTrain = opt.train or opt.continue_train
torch.cuda.set_device(device=GPU_IDs[opt.gpu_id])
