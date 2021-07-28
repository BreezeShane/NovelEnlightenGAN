from Config import *
import Networks.VGG as VGG
import Networks.FCN as FCN
from Networks.GAN_Definition import *


class Network:
    def __init__(self):
        pass

    def initialize(self, opt):
        self.opt = opt
        self.Tensor = torch.cuda.FloatTensor if self.opt.use_gpu else torch.Tensor
        self.save_dir = os.path.join(ROOT_PATH, 'Model/')

        nb = opt.batchSize
        size = opt.fineSize
        # nc means the number of channels
        self.input_A = self.Tensor(nb, opt.input_nc, size, size)
        self.input_B = self.Tensor(nb, opt.output_nc, size, size)
        self.input_img = self.Tensor(nb, opt.input_nc, size, size)
        self.input_A_gray = self.Tensor(nb, 1, size, size)

        if opt.vgg > 0:
            self.vgg_loss = VGG.PerceptualLoss(opt)
            if self.opt.IN_vgg:
                self.vgg_patch_loss = VGG.PerceptualLoss(opt)
                if opt.use_gpu:
                    self.vgg_patch_loss.cuda()
            if opt.use_gpu:
                self.vgg_loss.cuda()
            self.vgg = VGG.load_vgg16(opt, ROOT_PATH + "/Model/VGG/", GPU_IDs)
            self.vgg.eval()
            for param in self.vgg.parameters():
                param.requires_grad = False
        elif opt.fcn > 0:
            self.fcn_loss = FCN.SemanticLoss(opt)
            self.fcn_loss.cuda()
            self.fcn = FCN.load_fcn(opt, "Model/FCN/", GPU_IDs)
            self.fcn.eval()
            for param in self.fcn.parameters():
                param.requires_grad = False
        # load/define networks
        # The naming conversion is different from those used in the paper
        # Code (paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)

        skip = True if opt.skip > 0 else False
        self.netG_A = define_G(opt.input_nc, opt.output_nc,
                                        opt.ngf, opt.which_model_netG, opt, opt.norm, not opt.no_dropout, GPU_IDs, skip=skip)
        # self.netG_B = networks.define_G(opt.output_nc, opt.input_nc,
        #                                 opt.ngf, opt.which_model_netG, opt.norm, not opt.no_dropout, self.gpu_ids, skip=False, opt=opt)

        if self.opt.train:
            self.netD_A = define_D(opt.output_nc, opt.ndf,
                                            opt.which_model_netD,
                                            opt.n_layers_D, opt.norm, gpu_ids=GPU_IDs, patch=False)
            if self.opt.patchD:
                self.netD_P = define_D(opt.input_nc, opt.ndf,
                                            opt.which_model_netD,
                                            opt.n_layers_patchD, opt.norm, gpu_ids=GPU_IDs, patch=True)
        if not self.opt.train or self.opt.continue_train:
            which_epoch = opt.which_epoch
            self.load_network(self.netG_A, 'G_A', which_epoch)
            # self.load_network(self.netG_B, 'G_B', which_epoch)
            if self.opt.train or self.opt.continue_train:
                self.load_network(self.netD_A, 'D_A', which_epoch)
                if self.opt.patchD:
                    self.load_network(self.netD_P, 'D_P', which_epoch)

        if self.opt.train or self.opt.continue_train:
            self.old_lr = opt.lr
            # self.fake_A_pool = ImagePool(opt.pool_size)
            self.fake_B_pool = ImagePool(opt.pool_size)
            # define loss functions
            if opt.use_wgan:
                self.criterionGAN = DiscLossWGANGP()
            else:
                self.criterionGAN = GANLoss(use_lsgan=True, tensor=self.Tensor)
            self.criterionCycle = torch.nn.MSELoss()
            # self.criterionCycle = torch.nn.L1Loss()
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(self.netG_A.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_A = torch.optim.Adam(self.netD_A.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            if self.opt.patchD:
                self.optimizer_D_P = torch.optim.Adam(self.netD_P.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

        print('---------- Networks initialized -------------')
        print_network(self.netG_A)
        # networks.print_network(self.netG_B)
        if self.opt.train or self.opt.continue_train:
            print_network(self.netD_A)
            if self.opt.patchD:
                print_network(self.netD_P)
            # networks.print_network(self.netD_B)
        if self.opt.train or self.opt.continue_train:
            self.netG_A.train()
            # self.netG_B.train()
        else:
            self.netG_A.eval()
            # self.netG_B.eval()
        print('-----------------------------------------------')
