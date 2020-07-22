import json
from collections import OrderedDict
from math import exp

from contextlib import contextmanager
from math import sqrt, log

import torch
import torch.nn as nn


# import warnings
# warnings.simplefilter('ignore')


class BaseModule(nn.Module):
    def __init__(self):
        self.act_fn = None
        super(BaseModule, self).__init__()

    def selu_init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) and m.weight.requires_grad:
                m.weight.data.normal_(0.0, 1.0 / sqrt(m.weight.numel()))
                if m.bias is not None:
                    m.bias.data.fill_(0)
            elif isinstance(m, nn.BatchNorm2d) and m.weight.requires_grad:
                m.weight.data.fill_(1)
                m.bias.data.zero_()

            elif isinstance(m, nn.Linear) and m.weight.requires_grad:
                m.weight.data.normal_(0, 1.0 / sqrt(m.weight.numel()))
                m.bias.data.zero_()

    def initialize_weights_xavier_uniform(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) and m.weight.requires_grad:
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d) and m.weight.requires_grad:
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def load_state_dict(self, state_dict, strict=True, self_state=False):
        own_state = self_state if self_state else self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                try:
                    own_state[name].copy_(param.data)
                except Exception as e:
                    print("Parameter {} fails to load.".format(name))
                    print("-----------------------------------------")
                    print(e)
            else:
                print("Parameter {} is not in the model. ".format(name))

    @contextmanager
    def set_activation_inplace(self):
        if hasattr(self, 'act_fn') and hasattr(self.act_fn, 'inplace'):
            # save memory
            self.act_fn.inplace = True
            yield
            self.act_fn.inplace = False
        else:
            yield

    def total_parameters(self):
        total = sum([i.numel() for i in self.parameters()])
        trainable = sum([i.numel() for i in self.parameters() if i.requires_grad])
        print("Total parameters : {}. Trainable parameters : {}".format(total, trainable))
        return total

    def forward(self, *x):
        raise NotImplementedError


class ResidualFixBlock(BaseModule):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, dilation=1,
                 groups=1, activation=nn.SELU(), conv=nn.Conv2d):
        super(ResidualFixBlock, self).__init__()
        self.act_fn = activation
        self.m = nn.Sequential(
            conv(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation, groups=groups),
            activation,
            # conv(out_channels, out_channels, kernel_size, padding=(kernel_size - 1) // 2, dilation=1, groups=groups),
            conv(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation, groups=groups),
        )

    def forward(self, x):
        out = self.m(x)
        return self.act_fn(out + x)


class ConvBlock(BaseModule):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, dilation=1, groups=1,
                 activation=nn.SELU(), conv=nn.Conv2d):
        super(ConvBlock, self).__init__()
        self.m = nn.Sequential(conv(in_channels, out_channels, kernel_size, padding=padding,
                                    dilation=dilation, groups=groups),
                               activation)

    def forward(self, x):
        return self.m(x)


class UpSampleBlock(BaseModule):
    def __init__(self, channels, scale, activation, atrous_rate=1, conv=nn.Conv2d):
        assert scale in [2, 4, 8], "Currently UpSampleBlock supports 2, 4, 8 scaling"
        super(UpSampleBlock, self).__init__()
        m = nn.Sequential(
            conv(channels, 4 * channels, kernel_size=3, padding=atrous_rate, dilation=atrous_rate),
            activation,
            nn.PixelShuffle(2)
        )
        self.m = nn.Sequential(*[m for _ in range(int(log(scale, 2)))])

    def forward(self, x):
        return self.m(x)


class SpatialChannelSqueezeExcitation(BaseModule):
    # https://arxiv.org/abs/1709.01507
    # https://arxiv.org/pdf/1803.02579v1.pdf
    def __init__(self, in_channel, reduction=16, activation=nn.ReLU()):
        super(SpatialChannelSqueezeExcitation, self).__init__()
        linear_nodes = max(in_channel // reduction, 4)  # avoid only 1 node case
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.channel_excite = nn.Sequential(
            # check the paper for the number 16 in reduction. It is selected by experiment.
            nn.Linear(in_channel, linear_nodes),
            activation,
            nn.Linear(linear_nodes, in_channel),
            nn.Sigmoid()
        )
        self.spatial_excite = nn.Sequential(
            nn.Conv2d(in_channel, 1, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, h, w = x.size()
        #
        channel = self.avg_pool(x).view(b, c)
        # channel = F.avg_pool2d(x, kernel_size=(h,w)).view(b,c) # used for porting to other frameworks
        cSE = self.channel_excite(channel).view(b, c, 1, 1)
        x_cSE = torch.mul(x, cSE)

        # spatial
        sSE = self.spatial_excite(x)
        x_sSE = torch.mul(x, sSE)
        # return x_sSE
        return torch.add(x_cSE, x_sSE)


class PartialConv(nn.Module):
    # reference:
    # Image Inpainting for Irregular Holes Using Partial Convolutions
    # http://masc.cs.gmu.edu/wiki/partialconv/show?time=2018-05-24+21%3A41%3A10
    # https://github.com/naoto0804/pytorch-inpainting-with-partial-conv/blob/master/net.py
    # https://github.com/SeitaroShinagawa/chainer-partial_convolution_image_inpainting/blob/master/common/net.py
    # partial based padding
    # https: // github.com / NVIDIA / partialconv / blob / master / models / pd_resnet.py
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):

        super(PartialConv, self).__init__()
        self.feature_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                                      padding, dilation, groups, bias)

        self.mask_conv = nn.Conv2d(1, 1, kernel_size, stride,
                                   padding, dilation, groups, bias=False)
        self.window_size = self.mask_conv.kernel_size[0] * self.mask_conv.kernel_size[1]
        torch.nn.init.constant_(self.mask_conv.weight, 1.0)

        for param in self.mask_conv.parameters():
            param.requires_grad = False

    def forward(self, x):
        output = self.feature_conv(x)
        if self.feature_conv.bias is not None:
            output_bias = self.feature_conv.bias.view(1, -1, 1, 1).expand_as(output)
        else:
            output_bias = torch.zeros_like(output, device=x.device)

        with torch.no_grad():
            ones = torch.ones(1, 1, x.size(2), x.size(3), device=x.device)
            output_mask = self.mask_conv(ones)
            output_mask = self.window_size / output_mask
        output = (output - output_bias) * output_mask + output_bias

        return output


# +++++++++++++++++++++++++++++++++++++
#           FP16 Training      
# -------------------------------------
#  Modified from Nvidia/Apex
# https://github.com/NVIDIA/apex/blob/master/apex/fp16_utils/fp16util.py
import torch
import torch.nn as nn

class tofp16(nn.Module):
    def __init__(self):
        super(tofp16, self).__init__()

    def forward(self, input):
        if input.is_cuda:
            return input.half()
        else:  # PyTorch 1.0 doesn't support fp16 in CPU
            return input.float()


def BN_convert_float(module):
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        module.float()
    for child in module.children():
        BN_convert_float(child)
    return module


def network_to_half(network):
    return nn.Sequential(tofp16(), BN_convert_float(network.half()))


# warnings.simplefilter('ignore')

# +++++++++++++++++++++++++++++++++++++
#           DCSCN
# -------------------------------------

class DCSCN(BaseModule):
    # https://github.com/jiny2001/dcscn-super-resolution
    def __init__(self,
                 color_channel=3,
                 up_scale=2,
                 feature_layers=12,
                 first_feature_filters=196,
                 last_feature_filters=48,
                 reconstruction_filters=128,
                 up_sampler_filters=32
                 ):
        super(DCSCN, self).__init__()
        self.total_feature_channels = 0
        self.total_reconstruct_filters = 0
        self.upscale = up_scale

        self.act_fn = nn.SELU(inplace=False)
        self.feature_block = self.make_feature_extraction_block(color_channel,
                                                                feature_layers,
                                                                first_feature_filters,
                                                                last_feature_filters)

        self.reconstruction_block = self.make_reconstruction_block(reconstruction_filters)
        self.up_sampler = self.make_upsampler(up_sampler_filters, color_channel)
        self.selu_init_params()

    def selu_init_params(self):
        for i in self.modules():
            if isinstance(i, nn.Conv2d):
                i.weight.data.normal_(0.0, 1.0 / sqrt(i.weight.numel()))
                if i.bias is not None:
                    i.bias.data.fill_(0)

    def conv_block(self, in_channel, out_channel, kernel_size):
        m = OrderedDict([
            # ("Padding", nn.ReplicationPad2d((kernel_size - 1) // 2)),
            ('Conv2d', nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)),
            ('Activation', self.act_fn)
        ])

        return nn.Sequential(m)

    def make_feature_extraction_block(self, color_channel, num_layers, first_filters, last_filters):
        # input layer
        feature_block = [("Feature 1", self.conv_block(color_channel, first_filters, 3))]
        # exponential decay
        # rest layers
        alpha_rate = log(first_filters / last_filters) / (num_layers - 1)
        filter_nums = [round(first_filters * exp(-alpha_rate * i)) for i in range(num_layers)]

        self.total_feature_channels = sum(filter_nums)

        layer_filters = [[filter_nums[i], filter_nums[i + 1], 3] for i in range(num_layers - 1)]

        feature_block.extend([("Feature {}".format(index + 2), self.conv_block(*x))
                              for index, x in enumerate(layer_filters)])
        return nn.Sequential(OrderedDict(feature_block))

    def make_reconstruction_block(self, num_filters):
        B1 = self.conv_block(self.total_feature_channels, num_filters // 2, 1)
        B2 = self.conv_block(num_filters // 2, num_filters, 3)
        m = OrderedDict([
            ("A", self.conv_block(self.total_feature_channels, num_filters, 1)),
            ("B", nn.Sequential(*[B1, B2]))
        ])
        self.total_reconstruct_filters = num_filters * 2
        return nn.Sequential(m)

    def make_upsampler(self, out_channel, color_channel):
        out = out_channel * self.upscale ** 2
        m = OrderedDict([
            ('Conv2d_block', self.conv_block(self.total_reconstruct_filters, out, kernel_size=3)),
            ('PixelShuffle', nn.PixelShuffle(self.upscale)),
            ("Conv2d", nn.Conv2d(out_channel, color_channel, kernel_size=3, padding=1, bias=False))
        ])

        return nn.Sequential(m)

    def forward(self, x):
        # residual learning
        lr, lr_up = x
        feature = []
        for layer in self.feature_block.children():
            lr = layer(lr)
            feature.append(lr)
        feature = torch.cat(feature, dim=1)

        reconstruction = [layer(feature) for layer in self.reconstruction_block.children()]
        reconstruction = torch.cat(reconstruction, dim=1)

        lr = self.up_sampler(reconstruction)
        return lr + lr_up


# +++++++++++++++++++++++++++++++++++++
#           CARN      
# -------------------------------------

class CARN_Block(BaseModule):
    def __init__(self, channels, kernel_size=3, padding=1, dilation=1,
                 groups=1, activation=nn.SELU(), repeat=3,
                 SEBlock=False, conv=nn.Conv2d,
                 single_conv_size=1, single_conv_group=1):
        super(CARN_Block, self).__init__()
        m = []
        for i in range(repeat):
            m.append(ResidualFixBlock(channels, channels, kernel_size=kernel_size, padding=padding, dilation=dilation,
                                      groups=groups, activation=activation, conv=conv))
            if SEBlock:
                m.append(SpatialChannelSqueezeExcitation(channels, reduction=channels))
        self.blocks = nn.Sequential(*m)
        self.singles = nn.Sequential(
            *[ConvBlock(channels * (i + 2), channels, kernel_size=single_conv_size,
                        padding=(single_conv_size - 1) // 2, groups=single_conv_group,
                        activation=activation, conv=conv)
              for i in range(repeat)])

    def forward(self, x):
        c0 = x
        for block, single in zip(self.blocks, self.singles):
            b = block(x)
            c0 = c = torch.cat([c0, b], dim=1)
            x = single(c)

        return x


class CARN(BaseModule):
    # Fast, Accurate, and Lightweight Super-Resolution with Cascading Residual Network
    # https://github.com/nmhkahn/CARN-pytorch
    def __init__(self,
                 color_channels=3,
                 mid_channels=64,
                 scale=2,
                 activation=nn.SELU(),
                 num_blocks=3,
                 conv=nn.Conv2d):
        super(CARN, self).__init__()

        self.color_channels = color_channels
        self.mid_channels = mid_channels
        self.scale = scale

        self.entry_block = ConvBlock(color_channels, mid_channels, kernel_size=3, padding=1, activation=activation,
                                     conv=conv)
        self.blocks = nn.Sequential(
            *[CARN_Block(mid_channels, kernel_size=3, padding=1, activation=activation, conv=conv,
                         single_conv_size=1, single_conv_group=1)
              for _ in range(num_blocks)])
        self.singles = nn.Sequential(
            *[ConvBlock(mid_channels * (i + 2), mid_channels, kernel_size=1, padding=0,
                        activation=activation, conv=conv)
              for i in range(num_blocks)])

        self.upsampler = UpSampleBlock(mid_channels, scale=scale, activation=activation, conv=conv)
        self.exit_conv = conv(mid_channels, color_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.entry_block(x)
        c0 = x
        for block, single in zip(self.blocks, self.singles):
            b = block(x)
            c0 = c = torch.cat([c0, b], dim=1)
            x = single(c)
        x = self.upsampler(x)
        out = self.exit_conv(x)
        return out


class CARN_V2(CARN):
    def __init__(self, color_channels=3, mid_channels=64,
                 scale=2, activation=nn.LeakyReLU(0.1),
                 SEBlock=True, conv=nn.Conv2d,
                 atrous=(1, 1, 1), repeat_blocks=3,
                 single_conv_size=3, single_conv_group=1):
        super(CARN_V2, self).__init__(color_channels=color_channels, mid_channels=mid_channels, scale=scale,
                                      activation=activation, conv=conv)

        num_blocks = len(atrous)
        m = []
        for i in range(num_blocks):
            m.append(CARN_Block(mid_channels, kernel_size=3, padding=1, dilation=1,
                                activation=activation, SEBlock=SEBlock, conv=conv, repeat=repeat_blocks,
                                single_conv_size=single_conv_size, single_conv_group=single_conv_group))

        self.blocks = nn.Sequential(*m)

        self.singles = nn.Sequential(
            *[ConvBlock(mid_channels * (i + 2), mid_channels, kernel_size=single_conv_size,
                        padding=(single_conv_size - 1) // 2, groups=single_conv_group,
                        activation=activation, conv=conv)
              for i in range(num_blocks)])

    def forward(self, x):
        x = self.entry_block(x)
        c0 = x
        res = x
        for block, single in zip(self.blocks, self.singles):
            b = block(x)
            c0 = c = torch.cat([c0, b], dim=1)
            x = single(c)
        x = x + res
        x = self.upsampler(x)
        out = self.exit_conv(x)
        return out


# +++++++++++++++++++++++++++++++++++++
#           original Waifu2x model
# -------------------------------------


class UpConv_7(BaseModule):
    # https://github.com/nagadomi/waifu2x/blob/3c46906cb78895dbd5a25c3705994a1b2e873199/lib/srcnn.lua#L311
    def __init__(self):
        super(UpConv_7, self).__init__()
        self.act_fn = nn.LeakyReLU(0.1, inplace=False)
        self.offset = 7  # because of 0 padding
        from torch.nn import ZeroPad2d
        self.pad = ZeroPad2d(self.offset)
        m = [nn.Conv2d(3, 16, 3, 1, 0),
             self.act_fn,
             nn.Conv2d(16, 32, 3, 1, 0),
             self.act_fn,
             nn.Conv2d(32, 64, 3, 1, 0),
             self.act_fn,
             nn.Conv2d(64, 128, 3, 1, 0),
             self.act_fn,
             nn.Conv2d(128, 128, 3, 1, 0),
             self.act_fn,
             nn.Conv2d(128, 256, 3, 1, 0),
             self.act_fn,
             # in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=
             nn.ConvTranspose2d(256, 3, kernel_size=4, stride=2, padding=3, bias=False)
             ]
        self.Sequential = nn.Sequential(*m)

    def load_pre_train_weights(self, json_file):
        with open(json_file) as f:
            weights = json.load(f)
        box = []
        for i in weights:
            box.append(i['weight'])
            box.append(i['bias'])
        own_state = self.state_dict()
        for index, (name, param) in enumerate(own_state.items()):
            own_state[name].copy_(torch.FloatTensor(box[index]))

    def forward(self, x):
        x = self.pad(x)
        return self.Sequential.forward(x)



class Vgg_7(UpConv_7):
    def __init__(self):
        super(Vgg_7, self).__init__()
        self.act_fn = nn.LeakyReLU(0.1, inplace=False)
        self.offset = 7
        m = [nn.Conv2d(3, 32, 3, 1, 0),
             self.act_fn,
             nn.Conv2d(32, 32, 3, 1, 0),
             self.act_fn,
             nn.Conv2d(32, 64, 3, 1, 0),
             self.act_fn,
             nn.Conv2d(64, 64, 3, 1, 0),
             self.act_fn,
             nn.Conv2d(64, 128, 3, 1, 0),
             self.act_fn,
             nn.Conv2d(128, 128, 3, 1, 0),
             self.act_fn,
             nn.Conv2d(128, 3, 3, 1, 0)
             ]
        self.Sequential = nn.Sequential(*m)
