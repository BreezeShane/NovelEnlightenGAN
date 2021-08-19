from utils.utils import *


class UnalignedDataset:
    def __init__(self, isWeb=False):
        super(UnalignedDataset, self).__init__()
        self.opt = opt
        self.root = os.path.join(ROOT_PATH, 'Data/')
        if isWeb:
            self.dir_A = os.path.join(os.path.join(ROOT_PATH, 'front-end', 'Static_Files', 'uploads'))
            self.dir_B = os.path.join(os.path.join(ROOT_PATH, 'front-end', 'Static_Files', 'downloads'))
        else:
            if self.opt.isTrain:
                self.dir_A = os.path.join(self.root, 'Train_data_A/')
                self.dir_B = os.path.join(self.root, 'Train_data_B/')
            else:
                self.dir_A = os.path.join(self.root, 'Test_data/')
                self.dir_B = os.path.join(self.root, 'Targets/')
        # self.A_paths = make_dataset(self.dir_A)
        # self.B_paths = make_dataset(self.dir_B)
        self.A_imgs, self.A_paths = store_dataset(self.dir_A)
        self.B_imgs, self.B_paths = store_dataset(self.dir_B)

        # self.A_paths = sorted(self.A_paths)
        # self.B_paths = sorted(self.B_paths)
        self.A_size = len(self.A_paths)
        if self.opt.isTrain:
            self.B_size = len(self.B_paths)

        self.transform = get_transform(opt)

    def __getitem__(self, index):
        # A_path = self.A_paths[index % self.A_size]
        # B_path = self.B_paths[index % self.B_size]

        # A_img = Image.open(A_path).convert('RGB')
        # B_img = Image.open(B_path).convert('RGB')
        A_img = self.A_imgs[index % self.A_size]
        B_img = []
        if self.opt.isTrain:
            B_img = self.B_imgs[index % self.B_size]
        A_path = self.A_paths[index % self.A_size]
        B_path = []
        if self.opt.isTrain:
            B_path = self.B_paths[index % self.B_size]
        # A_size = A_img.size
        # B_size = B_img.size
        # A_size = A_size = (A_size[0]//16*16, A_size[1]//16*16)
        # B_size = B_size = (B_size[0]//16*16, B_size[1]//16*16)
        # A_img = A_img.resize(A_size, Image.BICUBIC)
        # B_img = B_img.resize(B_size, Image.BICUBIC)
        # A_gray = A_img.convert('LA')
        # A_gray = 255.0-A_gray

        A_img = self.transform(A_img)
        if self.opt.isTrain:
            B_img = self.transform(B_img)

        if self.opt.resize_or_crop == 'no':
            r, g, b = A_img[0] + 1, A_img[1] + 1, A_img[2] + 1
            A_gray = 1. - (0.299 * r + 0.587 * g + 0.114 * b) / 2.
            A_gray = torch.unsqueeze(A_gray, 0)
            input_img = A_img
            # A_gray = (1./A_gray)/255.
        else:
            w = A_img.size(2)
            h = A_img.size(1)

            # A_gray = (1./A_gray)/255.
            if (not self.opt.no_flip) and random.random() < 0.5:
                idx = [i for i in range(A_img.size(2) - 1, -1, -1)]
                idx = torch.LongTensor(idx)
                A_img = A_img.index_select(2, idx)
                if self.opt.isTrain:
                    B_img = B_img.index_select(2, idx)
            if (not self.opt.no_flip) and random.random() < 0.5:
                idx = [i for i in range(A_img.size(1) - 1, -1, -1)]
                idx = torch.LongTensor(idx)
                A_img = A_img.index_select(1, idx)
                if self.opt.isTrain:
                    B_img = B_img.index_select(1, idx)
            if self.opt.vary == 1 and (not self.opt.no_flip) and random.random() < 0.5:
                times = random.randint(self.opt.low_times, self.opt.high_times) / 100.
                input_img = (A_img + 1) / 2. / times
                input_img = input_img * 2 - 1
            else:
                input_img = A_img
            if self.opt.lighten and self.opt.isTrain:
                B_img = (B_img + 1) / 2.
                B_img = (B_img - torch.min(B_img)) / (torch.max(B_img) - torch.min(B_img))
                B_img = B_img * 2. - 1
            r, g, b = input_img[0] + 1, input_img[1] + 1, input_img[2] + 1
            A_gray = 1. - (0.299 * r + 0.587 * g + 0.114 * b) / 2.
            A_gray = torch.unsqueeze(A_gray, 0)
        return {'A': A_img, 'B': B_img, 'A_gray': A_gray, 'input_img': input_img,
                'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        if self.opt.isTrain:
            return max(self.A_size, self.B_size)
        else:
            return self.A_size

    def name(self):
        return 'UnalignedDataset'


# class SingleDataset:
#     def __init__(self, isWeb=False):
#         self.opt = opt
#         self.root = os.path.join(ROOT_PATH, 'Data')
#         if isWeb:
#             self.dir_A = os.path.join(os.path.join(ROOT_PATH, 'front-end', 'Static_Files', 'uploads'))
#         else:
#             self.dir_A = os.path.join(self.root, 'Test_data')
#
#         self.A_paths = make_dataset(self.dir_A)
#
#         self.A_paths = sorted(self.A_paths)
#
#         self.transform = get_transform(opt)
#
#     def __getitem__(self, index):
#         A_path = self.A_paths[index]
#
#         A_img = Image.open(A_path).convert('RGB')
#         A_size = A_img.size
#         A_size = (A_size[0] // 16 * 16, A_size[1] // 16 * 16)
#         A_img = A_img.resize(A_size, Image.BICUBIC)
#
#         A_img = self.transform(A_img)
#
#         return {'A': A_img, 'A_paths': A_path}
#
#     def __len__(self):
#         return len(self.A_paths)
#
#     def name(self):
#         return 'SingleImageDataset'


class DataLoader:
    def __init__(self, isWeb=False):
        self.opt = opt
        self.dataset = UnalignedDataset(isWeb) # if self.opt.isTrain else SingleDataset(isWeb)
        # self.dataset = UnalignedDataset(isWeb)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batchSize,
            shuffle=not opt.serial_batches)

    def name(self):
        return 'DataLoader'

    def load_data(self):
        return self.dataloader

    def __len__(self):
        return min(len(self.dataset), opt.max_dataset_size)
