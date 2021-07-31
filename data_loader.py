import os
from utils import *


class EnlightenGAN_DataLoader:
    def __init__(self):
        super(EnlightenGAN_DataLoader, self).__init__()

    def initialize(self, opt):
        self.opt = opt
        self.dataroot = os.path.join(ROOT_PATH, 'Data/')
        self.dir_A = os.path.join(self.dataroot, 'Train_data/')

        self.A_paths = make_dataset(self.dir_A)

        self.A_paths = sorted(self.A_paths)

        self.transform = get_transform(opt)

    def __getitem__(self, index):
        A_path = self.A_paths[index]

        A_img = Image.open(A_path).convert('RGB')
        A_size = A_img.size
        A_size = (A_size[0] // 16 * 16, A_size[1] // 16 * 16)
        A_img = A_img.resize(A_size, Image.BICUBIC)

        A_img = self.transform(A_img)

        return {'A': A_img, 'A_paths': A_path}

    def __len__(self):
        return len(self.A_paths)

    def name(self):
        return 'EnlightenGAN_ImageDataset'


class Data_Loader:
    def __init__(self):
        pass

    def initialize(self, opt):
        self.dataset = EnlightenGAN_DataLoader()
        print("dataset [%s] was created" % (self.dataset.name()))
        self.dataset.initialize(opt)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batchSize,
            shuffle=not opt.serial_batches)

    def load_data(self):
        return self.dataloader

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)
