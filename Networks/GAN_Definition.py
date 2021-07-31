import numpy as np
from utils import *
from Config import *
from torch.autograd import Variable
from Networks.Unet_Resize_Conv import Unet_resize_conv


def define_G(input_nc, output_nc, ngf, opt, norm='batch', use_dropout=False, gpu_ids=[], skip=False):
    netG = None
    use_gpu = opt.use_gpu
    norm_layer = get_norm_layer(norm_type=norm)


    if use_gpu:
        assert (torch.cuda.is_available())
    if opt.which_model_netG == 'Unet_resize_conv':
        netG = Unet_resize_conv(opt, skip)
    elif opt.which_model_netG == 'unet_512':
        netG = UnetGenerator(input_nc, output_nc, 9, ngf, norm_layer=norm_layer, use_dropout=use_dropout,
                         gpu_ids=gpu_ids, skip=skip, opt=opt)
    elif opt.which_model_netG == 'sid_unet':
        netG = UnetGenerator(opt, skip)
    if len(gpu_ids) > 0 and opt.use_gpu:
        netG.cuda(device=gpu_ids[opt.gpu_id])
        netG = torch.nn.DataParallel(netG, gpu_ids)
    netG.apply(weights_init)
    return netG


def define_D(input_nc, ndf, which_model_netD, opt,
             n_layers_D=3, norm='batch', use_sigmoid=False, gpu_ids=[], patch=False):
    netD = None
    use_gpu = opt.use_gpu
    norm_layer = get_norm_layer(norm_type=norm)

    # todo: Check up which model is used.

    if use_gpu:
        assert (torch.cuda.is_available())
    if which_model_netD == 'n_layers':
        netD = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer, use_sigmoid=use_sigmoid,
                                   gpu_ids=gpu_ids)
    elif which_model_netD == 'no_norm':
        netD = NoNormDiscriminator(input_nc, ndf, n_layers_D, use_sigmoid=use_sigmoid, gpu_ids=gpu_ids)
    if use_gpu:
        netD.cuda(device=gpu_ids[opt.gpu_id])
        netD = torch.nn.DataParallel(netD, gpu_ids)
    netD.apply(weights_init)
    return netD


class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)


class DiscLossWGANGP():
    def __init__(self):
        self.LAMBDA = 10

    def name(self):
        return 'DiscLossWGAN-GP'

    def initialize(self, opt, tensor):
        # DiscLossLS.initialize(self, opt, tensor)
        self.LAMBDA = 10

    # def get_g_loss(self, net, realA, fakeB):
    #     # First, G(A) should fake the discriminator
    #     self.D_fake = net.forward(fakeB)
    #     return -self.D_fake.mean()

    def calc_gradient_penalty(self, netD, real_data, fake_data):
        alpha = torch.rand(1, 1)
        alpha = alpha.expand(real_data.size())
        alpha = alpha.cuda()

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)

        interpolates = interpolates.cuda()
        interpolates = Variable(interpolates, requires_grad=True)

        disc_interpolates = netD.forward(interpolates)

        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                        grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
                                        create_graph=True, retain_graph=True, only_inputs=True)[0]

        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.LAMBDA
        return gradient_penalty


# Defines the Unet generator.
# |num_downs|: number of downsamplings in UNet. For example,
# if |num_downs| == 7, image of size 128x128 will become of size 1x1
# at the bottleneck
class UnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False, gpu_ids=[], skip=False, opt=None):
        super(UnetGenerator, self).__init__()
        self.gpu_ids = gpu_ids
        self.opt = opt
        # currently support only input_nc == output_nc
        assert (input_nc == output_nc)

        # construct unet structure


        ### 此处循环为卷积滤波器
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, norm_layer=norm_layer, innermost=True, opt=opt)
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, unet_block, norm_layer=norm_layer, use_dropout=use_dropout, opt=opt)
        ### 此处循环为卷积滤波器
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, unet_block, norm_layer=norm_layer, opt=opt)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, unet_block, norm_layer=norm_layer, opt=opt)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, unet_block, norm_layer=norm_layer, opt=opt)
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, unet_block, outermost=True, norm_layer=norm_layer, opt=opt)


        if skip == True:
            skipmodule = SkipModule(unet_block, opt)
            self.model = skipmodule
        else:
            self.model = unet_block

    def forward(self, input):
        if GPU_IDs and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, GPU_IDs)
        else:
            return self.model(input)


class SkipModule(nn.Module):
    def __init__(self, submodule, opt):
        super(SkipModule, self).__init__()
        self.submodule = submodule
        self.opt = opt

    def forward(self, x):
        latent = self.submodule(x)
        return self.opt.skip * x + latent, latent


# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|
class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False,
                 opt=None):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost

        downconv = nn.Conv2d(outer_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if opt.use_norm == 0:
            if outermost:
                upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                            kernel_size=4, stride=2,
                                            padding=1)
                down = [downconv]
                up = [uprelu, upconv, nn.Tanh()]
                model = down + [submodule] + up
            elif innermost:
                upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                            kernel_size=4, stride=2,
                                            padding=1)
                down = [downrelu, downconv]
                up = [uprelu, upconv]
                model = down + up
            else:
                upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                            kernel_size=4, stride=2,
                                            padding=1)
                down = [downrelu, downconv]
                up = [uprelu, upconv]

                if use_dropout:
                    model = down + [submodule] + up + [nn.Dropout(0.5)]
                else:
                    model = down + [submodule] + up
        else:
            if outermost:
                upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                            kernel_size=4, stride=2,
                                            padding=1)
                down = [downconv]
                up = [uprelu, upconv, nn.Tanh()]
                model = down + [submodule] + up
            elif innermost:
                upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                            kernel_size=4, stride=2,
                                            padding=1)
                down = [downrelu, downconv]
                up = [uprelu, upconv, upnorm]
                model = down + up
            else:
                upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                            kernel_size=4, stride=2,
                                            padding=1)
                down = [downrelu, downconv, downnorm]
                up = [uprelu, upconv, upnorm]

                if use_dropout:
                    model = down + [submodule] + up + [nn.Dropout(0.5)]
                else:
                    model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([self.model(x), x], 1)


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, gpu_ids=[]):
        super(NLayerDiscriminator, self).__init__()
        self.gpu_ids = gpu_ids

        kw = 4
        padw = int(np.ceil((kw - 1) / 2))
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        # if len(self.gpu_ids) and isinstance(input.data, torch.cuda.FloatTensor):
        #     return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        # else:
        return self.model(input)


class NoNormDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, use_sigmoid=False, gpu_ids=[]):
        super(NoNormDiscriminator, self).__init__()
        self.gpu_ids = gpu_ids

        kw = 4
        padw = int(np.ceil((kw - 1) / 2))
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        # if len(self.gpu_ids) and isinstance(input.data, torch.cuda.FloatTensor):
        #     return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        # else:
        return self.model(input)
