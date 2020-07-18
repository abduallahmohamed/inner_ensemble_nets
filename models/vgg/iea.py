import torch
import torch.nn.init as init
import math
import torch 
import torch.nn as nn 
import torch.nn.functional as F
import numpy as np
# from aggutils import *

def iea_normal_(tensor, a=0,m=3, mode='fan_in', nonlinearity='relu'):
    r"""Fills the input `Tensor` with values according to the method
    described in `Delving deep into rectifiers: Surpassing human-level
    performance on ImageNet classification` - He, K. et al. (2015), using a
    normal distribution. The resulting tensor will have values sampled from
    :math:`\mathcal{N}(0, \text{std}^2)` where

    .. math::
        \text{std} = \frac{\text{gain}}{\sqrt{\text{fan\_mode}}}

    Also known as Abdu initialization.

    Args:
        tensor: an n-dimensional `torch.Tensor`
        a: the negative slope of the rectifier used after this layer (only 
        used with ``'leaky_relu'``)
        mode: either ``'fan_in'`` (default) or ``'fan_out'``. Choosing ``'fan_in'``
            preserves the magnitude of the variance of the weights in the
            forward pass. Choosing ``'fan_out'`` preserves the magnitudes in the
            backwards pass.
        nonlinearity: the non-linear function (`nn.functional` name),
            recommended to use only with ``'relu'`` or ``'leaky_relu'`` (default).

    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.kaiming_normal_(w, mode='fan_out', nonlinearity='relu')
    """
    
    # fan = _calculate_correct_fan(tensor, mode)
    # gain = calculate_gain(nonlinearity, a)
    # std = gain / math.sqrt(fan)
    # with torch.no_grad():
        # return tensor.normal_(0, std)
    fan = init._calculate_correct_fan(tensor, mode)
    gain = init.calculate_gain(nonlinearity, a)
    gain = gain *  math.sqrt(m)
    std = gain / math.sqrt(fan)
    with torch.no_grad():
        return tensor.normal_(0, std)

class Conv2d_iea(nn.Module):
    #m = number of ensembles 
    def __init__(self, in_channels, out_channels, kernel_size,
                        stride=1, padding=0, dilation=1, groups=1, 
                         bias=True, padding_mode='zeros', m=2):
        super().__init__()
        self.mms = nn.ModuleList(
        [nn.Conv2d(in_channels,out_channels,kernel_size,
                 stride=stride, padding=padding, dilation=dilation,
                  groups=groups,  bias=bias, padding_mode=padding_mode)
         for i in range(m)])
        
        self.m = m
        #print(m)
        #if self.m is None:
        #    print("ZERO")
        self.bias = bias
        self.minv = nn.ParameterList([ nn.Parameter(torch.rand(1).fill_(1.0/self.m), requires_grad=False) for i in range(self.m) ])
        #print(self.minv)
        self.initmm = False
        self.domms = True
        self.subcnn = nn.Conv2d(in_channels,out_channels,kernel_size,
                 stride=stride, padding=padding, dilation=dilation,
                  groups=groups,  bias=bias, padding_mode=padding_mode)
                  
        self.apply_layers_init()
       
    def apply_layers_init(self):
        for bkey in range(self.m):
            iea_normal_(self.mms[bkey].weight,m=self.m)  

            
    def apply_weights_pruning(self):

        var_inv_sum = 0
        for bkey in range(self.m):
            var_inv_sum +=  1/self.mms[bkey].weight[:,:,:,:].var()

        self.subcnn.weight.data.fill_(0)
        if self.bias:
            self.subcnn.bias.data.fill_(0)
        for bkey in range(self.m):
            vrinv = (1/self.mms[bkey].weight[:,:,:,:].var())/var_inv_sum
            self.subcnn.weight[:,:,:,:] += \
            vrinv*self.mms[bkey].weight[:,:,:,:]
            if self.bias:
                self.subcnn.bias[:] += vrinv* self.mms[bkey].bias[:] 
      
    def forward(self,x):
        if self.domms:
            x_0 = self.minv[0]*self.mms[0](x)    
            for i in range(1,self.m):
                x_0 = x_0 +self.minv[i]*self.mms[i](x)
            return x_0
        else:
            return self.subcnn(x)
            
            
class Conv2d_maxout(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                        stride=1, padding=0, dilation=1, groups=1, 
                         bias=True, padding_mode='zeros', m=2):
        super().__init__()

        self.mms = nn.ModuleList(
        [nn.Conv2d(in_channels,out_channels,kernel_size,
                 stride=stride, padding=padding, dilation=dilation,
                  groups=groups,  bias=bias, padding_mode=padding_mode)
         for i in range(m)]
        )

        self.m = m
        self.minv = nn.ParameterList([ nn.Parameter(torch.rand(1).fill_(1.0/self.m), requires_grad=False) for i in range(self.m) ])
        
        self.bias = bias

        self.domms = True
        self.subcnn = nn.Conv2d(in_channels,out_channels,kernel_size,
                 stride=stride, padding=padding, dilation=dilation,
                  groups=groups,  bias=bias, padding_mode=padding_mode)
    def apply_weights_pruning(self):

        var_inv_sum = 0
        for bkey in range(self.m):
            var_inv_sum +=  1/self.mms[bkey].weight[:,:,:,:].var()

        self.subcnn.weight.data.fill_(0)
        if self.bias:
            self.subcnn.bias.data.fill_(0)
        for bkey in range(self.m):
            vrinv = (1/self.mms[bkey].weight[:,:,:,:].var())/var_inv_sum
            self.subcnn.weight[:,:,:,:] += \
            vrinv*self.mms[bkey].weight[:,:,:,:]
            if self.bias:
                self.subcnn.bias[:] += vrinv* self.mms[bkey].bias[:] 
    def forward(self,x):

        if self.domms:
            x_0 = self.mms[0](x)
            for i in range(1,self.m):
                x_0 = torch.max(x_0,self.mms[i](x))
            return x_0

        else:
            return self.subcnn(x)
            
      
class Linear_iea(nn.Module):
    def __init__(self, in_features, out_features,bias=True, m=3,nonlinea=F.tanh):
        super().__init__()
        self.mms = nn.ModuleList(
        [nn.Linear(in_features,out_features, bias=bias)
         for i in range(m)]
        )
        self.nonline = nonlinea
        self.m = m
        self.minv = nn.ParameterList([ nn.Parameter(torch.rand(1).fill_(1.0/self.m), requires_grad=False) for i in range(self.m) ])
        
        self.sublin = nn.Linear(in_features,out_features, bias=bias)
        self.domms = True
        self.bias = bias
        self.apply_layers_init()
        

    def apply_layers_init(self):
        for bkey in range(self.m):
            iea_normal_(self.mms[bkey].weight,m=self.m)   


    def apply_weights_pruning(self):

        var_inv_sum = 0
        for bkey in range(self.m):
            var_inv_sum +=  1/self.mms[bkey].weight[:,:].var()

        self.sublin.weight.data.fill_(0)
        if self.bias:
            self.sublin.bias.data.fill_(0)

        for bkey in range(self.m):
            vrinv = (1/self.mms[bkey].weight[:,:].var())/var_inv_sum
            self.sublin.weight[:,:] += \
            vrinv*self.mms[bkey].weight[:,:]
            if self.bias:
                self.sublin.bias[:] += vrinv* self.mms[bkey].bias[:]    

    def forward(self,x):
        if self.domms:
            x_0 = self.minv[0]*self.mms[0](x)
            for i in range(1,self.m):
                x_0 = x_0 +self.minv[i]*self.mms[i](x)

            return x_0
        else:
            return self.sublin(x)
            
class Linear_maxout(nn.Module):
    def __init__(self, in_features, out_features,bias=True, m=3):
        super().__init__()
        self.mms = nn.ModuleList(
        [nn.Linear(in_features,out_features, bias=bias)
         for i in range(m)]
        )
        self.m = m
        
        self.sublin = nn.Linear(in_features,out_features, bias=bias)
        self.domms = True
        self.bias = bias

        
    def forward(self,x):
        if self.domms:
            x_0 = self.mms[0](x)
            for i in range(1,self.m):
                x_0 = torch.max(x_0,self.mms[i](x))

            return x_0
        else:
            return self.sublin(x)
