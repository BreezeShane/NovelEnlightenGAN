# Standard importing

# Customed importing
from Config import *
import data_loader
import EnlightenGAN_Network

# todo: Create utils.py to store some unaligned methods.

# todo: Build up these two whole classes
DataLoader = data_loader.loading_data()
GAN_Network = EnlightenGAN_Network.Network()


def train():
    pass


def predict():
    pass


if __name__ == '__main__':
    if opt.train:
        # todo: save train images & port them to the website
        train()
    elif opt.predict:
        # todo: save test images & port them to the website
        predict()
