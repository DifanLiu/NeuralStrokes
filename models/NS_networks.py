"""
Copyright (C) 2021 Adobe. All rights reserved.
"""

import torch.nn as nn
import functools
import torch
import torch.nn.functional as F


class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out


class ResnetGeneratorWith1DConv(nn.Module):

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False,
                 n_blocks=6, padding_type='reflect', no_antialias=False,
                 no_antialias_up=False, opt=None, activation_type='tanh'):
        # default ngf = 10

        assert(n_blocks >= 0)
        super(ResnetGeneratorWith1DConv, self).__init__()
        self.opt = opt
        if type(norm_layer) == functools.partial:  # instance -> use_bias
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        model = []
        model += [nn.ConstantPad2d(3, 1.0)]
        model += [nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias), norm_layer(ngf), nn.ReLU(True)]  # 1, 2, 3    input_nc -> 10

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            if(no_antialias):  # True
                model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                          norm_layer(ngf * mult * 2),
                          nn.ReLU(True)]  
            else:
                assert 0

        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks ---- default 4 blocks

            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]  # 40 -> 40

        mult = 2 ** n_downsampling
        for i in range(n_downsampling):  # add upsampling layers
            if no_antialias_up:
                model += [nn.ConvTranspose2d(ngf * mult, ngf * mult,
                                             kernel_size=3, stride=2,
                                             padding=1, output_padding=1,
                                             bias=use_bias),
                          norm_layer(ngf * mult),
                          nn.ReLU(True)]  # 40 -> 40 -> 40
            else:
                assert 0

        model += [nn.Conv2d(ngf * mult, ngf * mult, kernel_size=1, padding=0)]  # 40 -> 40

        if activation_type == 'tanh':
            model += [nn.Tanh()]
        elif activation_type == 'none':  # default
            pass
        elif activation_type == 'sigmoid':
            model += [nn.Sigmoid()]
        else:
            assert 0

        self.model = nn.Sequential(*model)

        # ----- add Path Geometry Module
        self.n_1Dconv = opt.NS_n_1Dconv
        assert self.n_1Dconv > 0
        self.conv_1d = nn.Conv1d(ngf * mult + opt.NS_1d_cc, ngf * mult, kernel_size=opt.NS_ks_f_1Dconv,
                                 padding=int((opt.NS_ks_f_1Dconv - 1) / 2))  # 40 -> 40
        last_nc = ngf * mult

        model_1d = [nn.ReLU(True)]

        for temp_id in range(self.n_1Dconv):
            if temp_id == self.n_1Dconv - 1:  # last layer
                model_1d += [nn.Conv1d(last_nc, output_nc, kernel_size=3, padding=1)]  # 40 -> output_nc
            else:
                model_1d += [nn.Conv1d(last_nc, last_nc, kernel_size=3, padding=1), nn.ReLU(True)]  # 40 -> 40

        self.model_1d = nn.Sequential(*model_1d)

    def forward(self, input, list_pm_dict, layers=[], encode_only=False, use_flip=False):
        # input   bs, C, h, w     input_pm  a list of dict

        if len(layers) > 0:
            assert 0
        else:  # True
            """Standard forward"""
            assert 'ODfeat_list' in list_pm_dict[0].keys()
            fake = self.model(input)  #  feature map F

            ans_list = []
            bs = fake.shape[0]
            for bid in range(bs):
                # compose 1D feature  ns, 40, nv

                feat_1D_signal = compose_1D_signal(list_pm_dict[bid], fake[bid, :, :, :])  # ns, nc, max_nv
                if use_flip:
                    feat_OD_1 = compose_ODfeat(list_pm_dict[bid], feat_1D_signal, self.opt.NS_1d_cc)  # ns, nc2, max_nv
                    temp_out_feat_1 = self.model_1d(self.conv_1d(torch.cat((feat_1D_signal, feat_OD_1), dim=1)))  # ns, out, nv
                    feat_OD_2 = compose_ODfeat(list_pm_dict[bid], feat_1D_signal, self.opt.NS_1d_cc, invert=True)  # ns, nc2, max_nv
                    temp_out_feat_2 = self.model_1d(self.conv_1d(torch.cat((feat_1D_signal, feat_OD_2), dim=1)))  # ns, out, nv
                    temp_out_feat = (temp_out_feat_1 + temp_out_feat_2) * 0.5
                else:
                    feat_OD = compose_ODfeat(list_pm_dict[bid], feat_1D_signal, self.opt.NS_1d_cc)  # ns, nc2, max_nv
                    temp_out_feat = self.model_1d(self.conv_1d(torch.cat((feat_1D_signal, feat_OD), dim=1)))  # ns, out, nv
                ans_list.append(temp_out_feat)
            return fake, ans_list


def compose_ODfeat(pm_dict, feat_1D_signal, NS_1d_cc, invert=False):
    ns = feat_1D_signal.shape[0]
    max_nv = feat_1D_signal.shape[2]
    ans_ODfeat = torch.zeros(ns, NS_1d_cc, max_nv, dtype=torch.float32).to(feat_1D_signal.device)

    for path_id, ODfeat_single in enumerate(pm_dict['ODfeat_list']):
        this_nv = ODfeat_single.shape[0]
        ans_ODfeat[path_id, :, 0:this_nv] = ODfeat_single.permute(1, 0)
    if invert:
        ans_ODfeat[:, 1:, :] = -ans_ODfeat[:, 1:, :]

    return ans_ODfeat


def compose_1D_signal(pm_dict, feat_2D):
    # feat_2D  40xhxw
    ns = len(pm_dict['ans_list'])
    nc = feat_2D.shape[0]
    max_nv = -1
    for a_tensor in pm_dict['ans_list']:
        if a_tensor.shape[0] > max_nv:
            max_nv = a_tensor.shape[0]

    ans_1D_signal = torch.zeros(ns, nc, max_nv, dtype=torch.float32).to(feat_2D.device)

    for path_id, indices_tensor in enumerate(pm_dict['indices_list']):
        this_nv = indices_tensor.shape[0]
        ans_1D_signal[path_id, :, 0:this_nv] = feat_2D[:, indices_tensor[:, 0], indices_tensor[:, 1]] # 40, this_nv

    return ans_1D_signal


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = planes
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
