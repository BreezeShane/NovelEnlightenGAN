import os
import torch
import argparse


ROOT_PATH = os.getcwd()
GPU_IDs = [id for id in range(torch.cuda.device_count())]
IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

parser = argparse.ArgumentParser(prefix_chars='-_')
parser.add_argument("--train", action='store_true', help='Run train mode.')
parser.add_argument("--predict", action='store_true', help='Run predict mode. ')
parser.add_argument("--isWeb", action='store_true', help='Set the status if the project should run on website.')
parser.add_argument("--is_on_colab", action='store_true', help='Set the status if the project is run on colab.')

# parser.add_argument("--use_gpu", type=bool, default=True if torch.cuda.is_available() else False)
parser.add_argument("--use_gpu", type=bool, default=True, help='Set whether you would like to use gpu or not.')
parser.add_argument("--gpu_id", type=int, default=0, help='Set up on which gpu you want to run.')
parser.add_argument("--batchSize", type=int, default=8, help='input batch size')
parser.add_argument('--which_direction', type=str, default='AtoB', help='AtoB or BtoA')
parser.add_argument("--fineSize", type=int, default=256, help='then crop to this size')
parser.add_argument("--input_nc", type=int, default=3, help='of input image channels')
parser.add_argument("--output_nc", type=int, default=3, help='# of output image channels')
parser.add_argument("--no_vgg_instance", type=bool, default=False, help='vgg instance normalization')
parser.add_argument("--skip", type=float, default=1, help='B = net.forward(A) + skip*A')
parser.add_argument('--norm', type=str, default='instance', help='instance normalization or batch normalization')
parser.add_argument('--no_dropout', type=bool, default=True, help='no dropout for the generator')
parser.add_argument('--ndf', type=int, default=64, help='of discrim filters in first conv layer')
parser.add_argument("--patchD", type=bool, default=True, help='use patch discriminator')
parser.add_argument("--syn_norm", type=bool, default=False, help='use synchronize batch normalization')
parser.add_argument("--use_norm", type=float, default=1.0, help='L1 loss weight is 10.0')
parser.add_argument("--tanh", type=bool, default=False, help='Add tanh activation layer in the end of Generator')
parser.add_argument("--which_model_netD", type=str, default='no_norm', help='selects model to use for netD')
# parser.add_argument("--which_model_netG", type=str, default='Unet_resize_conv')
parser.add_argument('--vary', type=int, default=1, help='use light data augmentation')
parser.add_argument('--low_times', type=int, default=200, help='choose the number of crop for patch discriminator')
parser.add_argument('--lighten', action='store_true', help='normalize attention map')
parser.add_argument('--high_times', type=int, default=400,
                    help='choose the number of crop for patch discriminator')
parser.add_argument('--noise', type=float, default=0, help='variance of noise')
# parser.add_argument('--times_residual', action='store_true', help='output = input + residual*attention')
parser.add_argument('--times_residual', type=bool, default=True, help='output = input + residual*attention')
parser.add_argument('--input_linear', action='store_true', help='lieanr scaling input')
parser.add_argument('--linear_add', action='store_true', help='lieanr scaling input')
parser.add_argument('--latent_threshold', action='store_true', help='lieanr scaling input')
parser.add_argument('--latent_norm', action='store_true', help='lieanr scaling input')
parser.add_argument('--linear', action='store_true', help='Normalize the output. # tanh')
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
parser.add_argument('--self_attention', action='store_true', help='adding attention on the input of generator')
parser.add_argument('--resize_or_crop', type=str, default='no', help='scaling and cropping of images at load time [resize_and_crop|crop|scale_width|scale_width_and_crop]') # would be crop when training.
parser.add_argument("--serial_batches", type=bool, default=False, help='if true, takes images in order to make batches, otherwise takes them randomly')
parser.add_argument("--no_flip", type=bool, default=False, help='if specified, do not flip the images for data augmentation')
parser.add_argument('--max_dataset_size', type=int, default=float("inf"), help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
parser.add_argument('--vgg', type=float, default=0, help='use perceptrual loss')
parser.add_argument('--fcn', type=float, default=0, help='use semantic loss')
parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer')
parser.add_argument('--use_avgpool', type=float, default=0, help='use to perceptrual loss')

# Train args

parser.add_argument('--continue_train', action='store_true', help='To continue train from last time.')
parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
parser.add_argument("--lr", type=float, default=0.0001, help='initial learning rate for adam')
parser.add_argument('--niter', type=int, default=100, help='# of iter at starting learning rate')
parser.add_argument('--niter_decay', type=int, default=100, help='# of iter to linearly decay learning rate to zero')
parser.add_argument('--n_layers_D', type=int, default=3, help='only used if which_model_netD==n_layers')
parser.add_argument('--n_layers_patchD', type=int, default=3, help='only used if which_model_netD==n_layers')
parser.add_argument('--pool_size', type=int, default=50,
                    help='the size of images buffer that stores previously generated images')
parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
parser.add_argument('--save_latest_freq', type=int, default=5000, help='frequency of saving the latest results')
parser.add_argument('--save_epoch_freq', type=int, default=5,
                    help='frequency of saving checkpoints at the end of epochs')

# Test args
parser.add_argument("--use_models", action='store_true', help='To use multiple models to predict Test_data.')
parser.add_argument("--testMetrics", action='store_true', help='To use Metrics to evaluate the quality of model.')

opt = parser.parse_args()
opt.isTrain = opt.train or opt.continue_train

if opt.gpu_id not in GPU_IDs:
    print(f"GPU {opt.gpu_id} is invalid! ")
    exit()
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu_id)

if opt.is_on_colab:
    SAVE_ROOT_PATH = '/content/drive/MyDrive/EnlightenGAN-Customed/'
else:
    SAVE_ROOT_PATH = ROOT_PATH

folder_paths = [
    os.path.join(SAVE_ROOT_PATH, 'log'),
    os.path.join(SAVE_ROOT_PATH, 'Processing'),
]

for path in folder_paths:
    if not os.path.exists(path):
        os.mkdir(path)
