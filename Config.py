import os
import argparse
import torch

ROOT_PATH = os.getcwd()
GPU_IDs = [id for id in range(torch.cuda.device_count())]


# todo: make obj opt for more args
parser = argparse.ArgumentParser(prefix_chars='-_')
parser.add_argument("--train", action='store_true')
parser.add_argument('--continue_train', action='store_true')
parser.add_argument('--which_epoch', type=str, default='latest')
parser.add_argument("--predict", action='store_true')

parser.add_argument("--use_gpu", type=bool, default=True if torch.cuda.is_available() else False)
parser.add_argument("--gpu_id", type=int, default=0)
parser.add_argument("--batchSize", type=int, default=8)
parser.add_argument("--fineSize", type=int, default=256)
parser.add_argument("--input_nc", type=int, default=3)
parser.add_argument("--output_nc", type=int, default=3)
parser.add_argument("--no_vgg_instance", type=bool, default=False)
parser.add_argument("--skip", type=float, default=0.8)
parser.add_argument('--norm', type=str, default='instance')
parser.add_argument('--no_dropout', type=bool, default=True)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument("--patchD", type=bool, default=True)
parser.add_argument("--lr", type=float, default=0.0001)
parser.add_argument("--syn_norm", type=bool, default=False)
parser.add_argument("--use_norm", type=float, default=1.0)
parser.add_argument("--tanh", type=bool, default=False)

opt = parser.parse_args(args=[])
torch.cuda.set_device(device=GPU_IDs[opt.gpu_id])