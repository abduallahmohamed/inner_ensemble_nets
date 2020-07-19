import torch
import torch.nn.init as init
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Conv2d_maxout(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros', m=4):
        super().__init__()

        self.mms = nn.ModuleList(
            [nn.Conv2d(in_channels, out_channels, kernel_size,
                       stride=stride, padding=padding, dilation=dilation,
                       groups=groups, bias=bias, padding_mode=padding_mode)
             for i in range(m)]
        )

        self.m = m
        self.minv = nn.ParameterList(
            [nn.Parameter(torch.rand(1).fill_(1.0 / self.m), requires_grad=False) for i in range(self.m)])

        self.bias = bias

        self.domms = True
        self.subcnn = nn.Conv2d(in_channels, out_channels, kernel_size,
                                stride=stride, padding=padding, dilation=dilation,
                                groups=groups, bias=bias, padding_mode=padding_mode)

    def apply_weights_pruning(self):

        var_inv_sum = 0
        for bkey in range(self.m):
            var_inv_sum += 1 / self.mms[bkey].weight[:, :, :, :].var()

        self.subcnn.weight.data.fill_(0)
        if self.bias:
            self.subcnn.bias.data.fill_(0)
        for bkey in range(self.m):
            vrinv = (1 / self.mms[bkey].weight[:, :, :, :].var()) / var_inv_sum
            self.subcnn.weight[:, :, :, :] += \
                vrinv * self.mms[bkey].weight[:, :, :, :]
            if self.bias:
                self.subcnn.bias[:] += vrinv * self.mms[bkey].bias[:]

    def forward(self, x):

        if self.domms:
            x_0 = self.mms[0](x)
            for i in range(1, self.m):
                x_0 = torch.max(x_0, self.mms[i](x))
            return x_0

        else:
            return self.subcnn(x)


class Linear_maxout(nn.Module):
    def __init__(self, in_features, out_features, bias=True, m=4):
        super().__init__()
        self.mms = nn.ModuleList(
            [nn.Linear(in_features, out_features, bias=bias)
             for i in range(m)]
        )
        self.m = m

        self.sublin = nn.Linear(in_features, out_features, bias=bias)
        self.domms = True
        self.bias = bias

    def forward(self, x):
        if self.domms:
            x_0 = self.mms[0](x)
            for i in range(1, self.m):
                x_0 = torch.max(x_0, self.mms[i](x))

            return x_0
        else:
            return self.sublin(x)


