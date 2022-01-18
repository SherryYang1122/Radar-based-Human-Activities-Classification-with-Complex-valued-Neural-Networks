import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from complexPyTorch.complexLayers import ComplexBatchNorm2d

"""## Complex Blocks"""

## Complex Blocks
class ComplexConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=(1,1), padding=1, dilation=1, groups=1, bias=True):
        super(ComplexConv,self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.padding = padding

        ## Model components
        self.conv_re = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.conv_im = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        
    def forward(self, x): # shpae of x : [batch,2,channel,axis1,axis2]
        real = self.conv_re(x[:,0]) - self.conv_im(x[:,1])
        imaginary = self.conv_re(x[:,1]) + self.conv_im(x[:,0])
        output = torch.stack((real,imaginary),dim=1)
        return output

class CReLU(nn.Module):
    def __init__(self):
        super(CReLU,self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ## Model components
        self.relu_re = nn.ReLU()
        self.relu_im = nn.ReLU()

    def forward(self, x): # shpae of x : [batch,2,channel,axis1,axis2]
        real = self.relu_re(x[:,0])
        imaginary = self.relu_im(x[:,1])
        output = torch.stack((real,imaginary),dim=1)
        return output

class ComplexPool(nn.Module):
    def __init__(self, kernel_size, stride=(1,1), padding=0, dilation=1):
        super(ComplexPool,self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.padding = padding
        self.pool_re = nn.MaxPool2d(kernel_size,stride, padding=padding)
        self.pool_im = nn.MaxPool2d(kernel_size,stride,  padding=padding)
        
    def forward(self, x): # shpae of x : [batch,2,channel,axis1,axis2]
        real = self.pool_re(x[:,0]) 
        imaginary =self.pool_im(x[:,1]) 
        output = torch.stack((real,imaginary),dim=1)
        return output

class ComplexAdaptiveAvgPool2d(nn.Module):
    def __init__(self, output_size = (1,1)):
        super(ComplexAdaptiveAvgPool2d,self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pool_re = nn.AdaptiveAvgPool2d(output_size)
        self.pool_im = nn.AdaptiveAvgPool2d(output_size)
        
    def forward(self, x): # shpae of x : [batch,2,channel,axis1,axis2]
        real = self.pool_re(x[:,0]) 
        imaginary = self.pool_im(x[:,1]) 
        output = torch.stack((real,imaginary),dim=1)
        return output

class ComplexBN(nn.Module):
    def __init__(self, num_features):
        super(ComplexBN,self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.bn = ComplexBatchNorm2d(num_features)
        
    def forward(self, x): # shpae of x : [batch,2,channel,axis1,axis2]
        data_x = np.zeros([x.shape[0],x.shape[2],x.shape[3],x.shape[4]])
        data_x = x[:,0,:,:,:].type(torch.complex64) + 1j*x[:,1,:,:,:].type(torch.complex64)
        x = self.bn(data_x)
        real = x.real
        imaginary = x.imag
        output = torch.stack((real,imaginary),dim=1)
        return output
