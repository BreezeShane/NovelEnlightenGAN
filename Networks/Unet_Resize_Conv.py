import torch.nn.functional as F
from utils import *
from lib.nn import SynchronizedBatchNorm2d as SynBN2d


class Unet_resize_conv(nn.Module):

    def __init__(self, opt, skip):
        super(Unet_resize_conv, self).__init__()

        self.opt = opt
        self.skip = skip
        p = 1
        # self.conv1_1 = nn.Conv2d(4, 32, 3, padding=p)
        self.conv1_1 = nn.Conv2d(4, 32, 3, padding=p)
        # self.conv1_1 = nn.Conv2d(3, 32, 3, padding=p)
        self.downsample_1 = nn.MaxPool2d(2)
        self.downsample_2 = nn.MaxPool2d(2)
        self.downsample_3 = nn.MaxPool2d(2)
        self.downsample_4 = nn.MaxPool2d(2)
        self.LReLU1_1 = nn.LeakyReLU(0.2, inplace=True)
        if self.opt.use_norm == 1:
            self.bn1_1 = SynBN2d(32) if self.opt.syn_norm else nn.BatchNorm2d(32)
        self.conv1_2 = nn.Conv2d(32, 32, 3, padding=p)
        self.LReLU1_2 = nn.LeakyReLU(0.2, inplace=True)
        if self.opt.use_norm == 1:
            self.bn1_2 = SynBN2d(32) if self.opt.syn_norm else nn.BatchNorm2d(32)
        self.max_pool1 = nn.AvgPool2d(2) if self.opt.use_avgpool == 1 else nn.MaxPool2d(2)

        self.conv2_1 = nn.Conv2d(32, 64, 3, padding=p)
        self.LReLU2_1 = nn.LeakyReLU(0.2, inplace=True)
        if self.opt.use_norm == 1:
            self.bn2_1 = SynBN2d(64) if self.opt.syn_norm else nn.BatchNorm2d(64)
        self.conv2_2 = nn.Conv2d(64, 64, 3, padding=p)
        self.LReLU2_2 = nn.LeakyReLU(0.2, inplace=True)
        if self.opt.use_norm == 1:
            self.bn2_2 = SynBN2d(64) if self.opt.syn_norm else nn.BatchNorm2d(64)
        self.max_pool2 = nn.AvgPool2d(2) if self.opt.use_avgpool == 1 else nn.MaxPool2d(2)

        self.conv3_1 = nn.Conv2d(64, 128, 3, padding=p)
        self.LReLU3_1 = nn.LeakyReLU(0.2, inplace=True)
        if self.opt.use_norm == 1:
            self.bn3_1 = SynBN2d(128) if self.opt.syn_norm else nn.BatchNorm2d(128)
        self.conv3_2 = nn.Conv2d(128, 128, 3, padding=p)
        self.LReLU3_2 = nn.LeakyReLU(0.2, inplace=True)
        if self.opt.use_norm == 1:
            self.bn3_2 = SynBN2d(128) if self.opt.syn_norm else nn.BatchNorm2d(128)
        self.max_pool3 = nn.AvgPool2d(2) if self.opt.use_avgpool == 1 else nn.MaxPool2d(2)

        self.conv4_1 = nn.Conv2d(128, 256, 3, padding=p)
        self.LReLU4_1 = nn.LeakyReLU(0.2, inplace=True)
        if self.opt.use_norm == 1:
            self.bn4_1 = SynBN2d(256) if self.opt.syn_norm else nn.BatchNorm2d(256)
        self.conv4_2 = nn.Conv2d(256, 256, 3, padding=p)
        self.LReLU4_2 = nn.LeakyReLU(0.2, inplace=True)
        if self.opt.use_norm == 1:
            self.bn4_2 = SynBN2d(256) if self.opt.syn_norm else nn.BatchNorm2d(256)
        self.max_pool4 = nn.AvgPool2d(2) if self.opt.use_avgpool == 1 else nn.MaxPool2d(2)

        self.conv5_1 = nn.Conv2d(256, 512, 3, padding=p)
        self.LReLU5_1 = nn.LeakyReLU(0.2, inplace=True)
        if self.opt.use_norm == 1:
            self.bn5_1 = SynBN2d(512) if self.opt.syn_norm else nn.BatchNorm2d(512)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=p)
        self.LReLU5_2 = nn.LeakyReLU(0.2, inplace=True)
        if self.opt.use_norm == 1:
            self.bn5_2 = SynBN2d(512) if self.opt.syn_norm else nn.BatchNorm2d(512)

        # self.deconv5 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.deconv5 = nn.Conv2d(512, 256, 3, padding=p)
        self.conv6_1 = nn.Conv2d(512, 256, 3, padding=p)
        self.LReLU6_1 = nn.LeakyReLU(0.2, inplace=True)
        if self.opt.use_norm == 1:
            self.bn6_1 = SynBN2d(256) if self.opt.syn_norm else nn.BatchNorm2d(256)
        self.conv6_2 = nn.Conv2d(256, 256, 3, padding=p)
        self.LReLU6_2 = nn.LeakyReLU(0.2, inplace=True)
        if self.opt.use_norm == 1:
            self.bn6_2 = SynBN2d(256) if self.opt.syn_norm else nn.BatchNorm2d(256)

        # self.deconv6 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.deconv6 = nn.Conv2d(256, 128, 3, padding=p)
        self.conv7_1 = nn.Conv2d(256, 128, 3, padding=p)
        self.LReLU7_1 = nn.LeakyReLU(0.2, inplace=True)
        if self.opt.use_norm == 1:
            self.bn7_1 = SynBN2d(128) if self.opt.syn_norm else nn.BatchNorm2d(128)
        self.conv7_2 = nn.Conv2d(128, 128, 3, padding=p)
        self.LReLU7_2 = nn.LeakyReLU(0.2, inplace=True)
        if self.opt.use_norm == 1:
            self.bn7_2 = SynBN2d(128) if self.opt.syn_norm else nn.BatchNorm2d(128)

        # self.deconv7 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.deconv7 = nn.Conv2d(128, 64, 3, padding=p)
        self.conv8_1 = nn.Conv2d(128, 64, 3, padding=p)
        self.LReLU8_1 = nn.LeakyReLU(0.2, inplace=True)
        if self.opt.use_norm == 1:
            self.bn8_1 = SynBN2d(64) if self.opt.syn_norm else nn.BatchNorm2d(64)
        self.conv8_2 = nn.Conv2d(64, 64, 3, padding=p)
        self.LReLU8_2 = nn.LeakyReLU(0.2, inplace=True)
        if self.opt.use_norm == 1:
            self.bn8_2 = SynBN2d(64) if self.opt.syn_norm else nn.BatchNorm2d(64)

        # self.deconv8 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.deconv8 = nn.Conv2d(64, 32, 3, padding=p)
        self.conv9_1 = nn.Conv2d(64, 32, 3, padding=p)
        self.LReLU9_1 = nn.LeakyReLU(0.2, inplace=True)
        if self.opt.use_norm == 1:
            self.bn9_1 = SynBN2d(32) if self.opt.syn_norm else nn.BatchNorm2d(32)
        self.conv9_2 = nn.Conv2d(32, 32, 3, padding=p)
        self.LReLU9_2 = nn.LeakyReLU(0.2, inplace=True)

        self.conv10 = nn.Conv2d(32, 3, 1)
        if self.opt.tanh:
            self.tanh = nn.Tanh()

    def depth_to_space(self, input, block_size):
        block_size_sq = block_size * block_size
        output = input.permute(0, 2, 3, 1)
        (batch_size, d_height, d_width, d_depth) = output.size()
        s_depth = int(d_depth / block_size_sq)
        s_width = int(d_width * block_size)
        s_height = int(d_height * block_size)
        t_1 = output.resize(batch_size, d_height, d_width, block_size_sq, s_depth)
        spl = t_1.split(block_size, 3)
        stack = [t_t.resize(batch_size, d_height, s_width, s_depth) for t_t in spl]
        output = torch.stack(stack, 0).transpose(0, 1).permute(0, 2, 1, 3, 4).resize(batch_size, s_height, s_width,
                                                                                     s_depth)
        output = output.permute(0, 3, 1, 2)
        return output

    def forward(self, input, gray):
        flag = 0
        if input.size()[3] > 2200:
            avg = nn.AvgPool2d(2)
            input = avg(input)
            gray = avg(gray)
            flag = 1
            # pass
        input, pad_left, pad_right, pad_top, pad_bottom = pad_tensor(input)
        gray, pad_left, pad_right, pad_top, pad_bottom = pad_tensor(gray)
        gray_2 = self.downsample_1(gray)
        gray_3 = self.downsample_2(gray_2)
        gray_4 = self.downsample_3(gray_3)
        gray_5 = self.downsample_4(gray_4)
        # todo: Change F.upsample to nn.functional.interpolate &&&
        if self.opt.use_norm == 1:
            x = self.bn1_1(self.LReLU1_1(self.conv1_1(torch.cat((input, gray), 1))))
            # x = self.bn1_1(self.LReLU1_1(self.conv1_1(input)))
            conv1 = self.bn1_2(self.LReLU1_2(self.conv1_2(x)))
            x = self.max_pool1(conv1)

            x = self.bn2_1(self.LReLU2_1(self.conv2_1(x)))
            conv2 = self.bn2_2(self.LReLU2_2(self.conv2_2(x)))
            x = self.max_pool2(conv2)

            x = self.bn3_1(self.LReLU3_1(self.conv3_1(x)))
            conv3 = self.bn3_2(self.LReLU3_2(self.conv3_2(x)))
            x = self.max_pool3(conv3)

            x = self.bn4_1(self.LReLU4_1(self.conv4_1(x)))
            conv4 = self.bn4_2(self.LReLU4_2(self.conv4_2(x)))
            x = self.max_pool4(conv4)

            x = self.bn5_1(self.LReLU5_1(self.conv5_1(x)))
            x = x * gray_5
            conv5 = self.bn5_2(self.LReLU5_2(self.conv5_2(x)))

            conv5 = F.upsample(conv5, scale_factor=2, mode='bilinear')
            conv4 = conv4 * gray_4
            up6 = torch.cat([self.deconv5(conv5), conv4], 1)
            x = self.bn6_1(self.LReLU6_1(self.conv6_1(up6)))
            conv6 = self.bn6_2(self.LReLU6_2(self.conv6_2(x)))

            conv6 = F.upsample(conv6, scale_factor=2, mode='bilinear')
            conv3 = conv3 * gray_3
            up7 = torch.cat([self.deconv6(conv6), conv3], 1)
            x = self.bn7_1(self.LReLU7_1(self.conv7_1(up7)))
            conv7 = self.bn7_2(self.LReLU7_2(self.conv7_2(x)))

            conv7 = F.upsample(conv7, scale_factor=2, mode='bilinear')
            conv2 = conv2 * gray_2
            up8 = torch.cat([self.deconv7(conv7), conv2], 1)
            x = self.bn8_1(self.LReLU8_1(self.conv8_1(up8)))
            conv8 = self.bn8_2(self.LReLU8_2(self.conv8_2(x)))

            conv8 = F.upsample(conv8, scale_factor=2, mode='bilinear')
            conv1 = conv1 * gray
            up9 = torch.cat([self.deconv8(conv8), conv1], 1)
            x = self.bn9_1(self.LReLU9_1(self.conv9_1(up9)))
            conv9 = self.LReLU9_2(self.conv9_2(x))

            latent = self.conv10(conv9)

            if self.opt.times_residual:
                latent = latent * gray

            # output = self.depth_to_space(conv10, 2)
            if self.opt.tanh:
                latent = self.tanh(latent)
            if self.skip:
                if self.opt.linear_add:
                    if self.opt.latent_threshold:
                        latent = F.relu(latent)
                    elif self.opt.latent_norm:
                        latent = (latent - torch.min(latent)) / (torch.max(latent) - torch.min(latent))
                    input = (input - torch.min(input)) / (torch.max(input) - torch.min(input))
                    output = latent + input * self.opt.skip
                    output = output * 2 - 1
                else:
                    if self.opt.latent_threshold:
                        latent = F.relu(latent)
                    elif self.opt.latent_norm:
                        latent = (latent - torch.min(latent)) / (torch.max(latent) - torch.min(latent))
                    output = latent + input * self.opt.skip
            else:
                output = latent

            if self.opt.linear:
                output = output / torch.max(torch.abs(output))


        elif self.opt.use_norm == 0:
            x = self.LReLU1_1(self.conv1_1(torch.cat((input, gray), 1)))
            conv1 = self.LReLU1_2(self.conv1_2(x))
            x = self.max_pool1(conv1)

            x = self.LReLU2_1(self.conv2_1(x))
            conv2 = self.LReLU2_2(self.conv2_2(x))
            x = self.max_pool2(conv2)

            x = self.LReLU3_1(self.conv3_1(x))
            conv3 = self.LReLU3_2(self.conv3_2(x))
            x = self.max_pool3(conv3)

            x = self.LReLU4_1(self.conv4_1(x))
            conv4 = self.LReLU4_2(self.conv4_2(x))
            x = self.max_pool4(conv4)

            x = self.LReLU5_1(self.conv5_1(x))
            x = x * gray_5 if self.opt.self_attention else x
            conv5 = self.LReLU5_2(self.conv5_2(x))

            conv5 = F.upsample(conv5, scale_factor=2, mode='bilinear')
            conv4 = conv4 * gray_4 if self.opt.self_attention else conv4
            up6 = torch.cat([self.deconv5(conv5), conv4], 1)
            x = self.LReLU6_1(self.conv6_1(up6))
            conv6 = self.LReLU6_2(self.conv6_2(x))

            conv6 = F.upsample(conv6, scale_factor=2, mode='bilinear')
            conv3 = conv3 * gray_3 if self.opt.self_attention else conv3
            up7 = torch.cat([self.deconv6(conv6), conv3], 1)
            x = self.LReLU7_1(self.conv7_1(up7))
            conv7 = self.LReLU7_2(self.conv7_2(x))

            conv7 = F.upsample(conv7, scale_factor=2, mode='bilinear')
            conv2 = conv2 * gray_2 if self.opt.self_attention else conv2
            up8 = torch.cat([self.deconv7(conv7), conv2], 1)
            x = self.LReLU8_1(self.conv8_1(up8))
            conv8 = self.LReLU8_2(self.conv8_2(x))

            conv8 = F.upsample(conv8, scale_factor=2, mode='bilinear')
            conv1 = conv1 * gray if self.opt.self_attention else conv1
            up9 = torch.cat([self.deconv8(conv8), conv1], 1)
            x = self.LReLU9_1(self.conv9_1(up9))
            conv9 = self.LReLU9_2(self.conv9_2(x))

            latent = self.conv10(conv9)

            if self.opt.times_residual:
                latent = latent * gray

            if self.opt.tanh:
                latent = self.tanh(latent)
            if self.skip:
                if self.opt.linear_add:
                    if self.opt.latent_threshold:
                        latent = F.relu(latent)
                    elif self.opt.latent_norm:
                        latent = (latent - torch.min(latent)) / (torch.max(latent) - torch.min(latent))
                    input = (input - torch.min(input)) / (torch.max(input) - torch.min(input))
                    output = latent + input * self.opt.skip
                    output = output * 2 - 1
                else:
                    if self.opt.latent_threshold:
                        latent = F.relu(latent)
                    elif self.opt.latent_norm:
                        latent = (latent - torch.min(latent)) / (torch.max(latent) - torch.min(latent))
                    output = latent + input * self.opt.skip
            else:
                output = latent

            if self.opt.linear:
                output = output / torch.max(torch.abs(output))

        output = pad_tensor_back(output, pad_left, pad_right, pad_top, pad_bottom)
        latent = pad_tensor_back(latent, pad_left, pad_right, pad_top, pad_bottom)
        gray = pad_tensor_back(gray, pad_left, pad_right, pad_top, pad_bottom)
        if flag == 1:
            output = F.upsample(output, scale_factor=2, mode='bilinear')
            gray = F.upsample(gray, scale_factor=2, mode='bilinear')
        if self.skip:
            return output, latent
        else:
            return output
