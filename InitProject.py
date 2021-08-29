from Config import *
from os import mkdir
from os.path import join, exists

if opt.is_on_colab:
    root_path = '/content/drive/MyDrive/EnlightenGAN-Customed/'
else:
    root_path = ROOT_PATH

folder_paths = [
    join(root_path, 'log'),
    join(root_path, 'log', 'D_A'),
    join(root_path, 'log', 'G_A'),
    join(root_path, 'log', 'VGG'),
    join(root_path, 'log', 'D_P'),
    join(root_path, 'log', 'Discriminator_Global_Struct'),
    join(root_path, 'log', 'Discriminator_Local_Struct'),
    join(root_path, 'log', 'Generator_Struct'),
    join(root_path, 'Processing'),
]


def initialize():
    for path in folder_paths:
        if not exists(path):
            mkdir(path)
