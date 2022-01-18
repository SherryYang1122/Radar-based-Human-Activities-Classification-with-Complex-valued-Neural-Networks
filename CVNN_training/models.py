import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from complexnn import ComplexConv
from complexnn import CReLU
from complexnn import ComplexPool
from complexnn import ComplexAdaptiveAvgPool2d
from complexnn import ComplexBN

class shallow_Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 64, kernel_size=(4,4), stride = (2,2), padding=1)
        self.bn_1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size= (2, 2), stride = (2,2))
        self.relu = nn.ReLU()

        self.fc1 = nn.Linear(in_features = 64*60*30, out_features = 2048)
        self.fc2 = nn.Linear(2048, 9)
    def forward(self, x):
        x = self.pool1(self.relu(self.bn_1(self.conv1(x)))) #before flatten 
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = torch.sigmoid(x)
        return x


class shallow_ch2_Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels = 2, out_channels = 64, kernel_size=(4,4), stride = (2,2), padding=1)
        self.bn_1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size= (2, 2), stride = (2,2))
        self.relu = nn.ReLU()

        self.fc1 = nn.Linear(in_features = 64*60*30, out_features = 2048)
        self.fc2 = nn.Linear(2048, 9)

    def forward(self, x):
        x = self.pool1(self.relu(self.bn_1(self.conv1(x)))) #before flatten 

        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = torch.sigmoid(x)
        return x

class deep_Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 64, kernel_size=(4,4), stride = (2,2), padding=1)
        self.bn_1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size= (2, 2), stride = (2,2))
        self.relu = nn.ReLU()
        
        self.conv2 = nn.Conv2d(64, 64, (3,3),(1,1), padding=1)
        self.bn_2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size= (2, 2), stride = (2,2))
        
        self.conv3 = nn.Conv2d(64, 128, (3,3),(1,1), padding=1)
        self.bn_3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size= (2, 2), stride = (2,2))
        
        self.conv4 = nn.Conv2d(128, 128, (2,2),(1,1), padding=1)
        self.bn_4 = nn.BatchNorm2d(128)
        self.pool4 = nn.MaxPool2d(kernel_size= (2, 2), stride = (2,1))
        
        self.conv5 = nn.Conv2d(128, 128, (2,2),(1,1), padding=1)
        self.bn_5 = nn.BatchNorm2d(128)
        self.pool5 = nn.MaxPool2d(kernel_size= (2, 2), stride = (1,1))      
        '''
        self.conv6 = nn.Conv2d(128, 128, (2,2),(1,1), padding=1)
        self.bn_6 = nn.BatchNorm2d(128)
        self.pool6 = nn.MaxPool2d(kernel_size= (2, 2), stride = (1,1))
        
        self.conv7 = nn.Conv2d(128, 128, (2,2),(1,1), padding=1)
        self.bn_7 = nn.BatchNorm2d(128)
        self.pool7 = nn.MaxPool2d(kernel_size= (2, 2), stride = (1,1))
        '''
        self.fc1 = nn.Linear(in_features = 128*8*7, out_features = 2048)
        self.fc2 = nn.Linear(2048, 9)

    def forward(self, x):
        x = self.pool1(self.relu(self.bn_1(self.conv1(x)))) #before flatten 
        x = self.pool2(self.relu(self.bn_2(self.conv2(x)))) 
        x = self.pool3(self.relu(self.bn_3(self.conv3(x)))) 
        x = self.pool4(self.relu(self.bn_4(self.conv4(x))))
        x = self.pool5(self.relu(self.bn_5(self.conv5(x)))) 
        #x = self.pool6(self.relu(self.bn_6(self.conv6(x)))) 
        #x = self.pool7(self.relu(self.bn_6(self.conv6(x)))) 

        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = torch.sigmoid(x)
        return x

class deep_ch2_Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels = 2, out_channels = 64, kernel_size=(4,4), stride = (2,2), padding=1)
        self.bn_1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size= (2, 2), stride = (2,2))
        self.relu = nn.ReLU()
          
        self.conv2 = nn.Conv2d(64, 64, (3,3),(1,1), padding=1)
        self.bn_2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size= (2, 2), stride = (2,2))
    
        self.conv3 = nn.Conv2d(64, 128, (3,3),(1,1), padding=1)
        self.bn_3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size= (2, 2), stride = (2,2))
   
        self.conv4 = nn.Conv2d(128, 128, (2,2),(1,1), padding=1)
        self.bn_4 = nn.BatchNorm2d(128)
        self.pool4 = nn.MaxPool2d(kernel_size= (2, 2), stride = (2,1))
        
        self.conv5 = nn.Conv2d(128, 128, (2,2),(1,1), padding=1)
        self.bn_5 = nn.BatchNorm2d(128)
        self.pool5 = nn.MaxPool2d(kernel_size= (2, 2), stride = (1,1))
        
        '''
        self.conv6 = nn.Conv2d(128, 128, (2,2),(1,1), padding=1)
        self.bn_6 = nn.BatchNorm2d(128)
        self.pool6 = nn.MaxPool2d(kernel_size= (2, 2), stride = (1,1))
        '''

        self.fc1 = nn.Linear(in_features = 128*8*7, out_features = 2048)
        self.fc2 = nn.Linear(2048, 9)

    def forward(self, x):
        x = self.pool1(self.relu(self.bn_1(self.conv1(x)))) #before flatten 
        x = self.pool2(self.relu(self.bn_2(self.conv2(x)))) 
        x = self.pool3(self.relu(self.bn_3(self.conv3(x)))) 
        x = self.pool4(self.relu(self.bn_4(self.conv4(x)))) 
        x = self.pool5(self.relu(self.bn_5(self.conv5(x)))) 
        #x = self.pool6(self.relu(self.bn_6(self.conv6(x)))) 

        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = torch.sigmoid(x)
        return x

class shallow_Complex_Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = ComplexConv(in_channels = 1, out_channels = 64, kernel_size=(4,4), stride = (2,2))
        self.bn_1 = ComplexBN(64)
        self.pool1 = ComplexPool(kernel_size= (2, 2), stride = (2,2))
        self.relu = CReLU()
        
        self.fc1 = nn.Linear(in_features = 2*64*60*30, out_features = 2048)
        self.fc2 = nn.Linear(2048, 9)

    def forward(self, x):
        x = self.pool1(self.relu(self.bn_1(self.conv1(x)))) #before flatten 2, 64, 60, 30  
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = torch.sigmoid(x)
        return x

class deep_Complex_Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = ComplexConv(in_channels = 1, out_channels = 64, kernel_size=(4,4), stride = (2,2))
        self.bn_1 = ComplexBN(64)
        self.pool1 = ComplexPool(kernel_size= (2, 2), stride = (2,2))
        self.relu = CReLU()
        
        self.conv2 = ComplexConv(64, 64, (3,3),(1,1))
        self.bn_2 = ComplexBN(64)
        self.pool2 = ComplexPool(kernel_size= (2, 2), stride = (2,2))
        
        self.conv3 = ComplexConv(64, 128, (3,3),(1,1))
        self.bn_3 = ComplexBN(128)
        self.pool3 = ComplexPool(kernel_size= (2, 2), stride = (2,2))
        
        self.conv4 = ComplexConv(128, 128, (2,2),(1,1))
        self.bn_4 = ComplexBN(128)
        self.pool4 = ComplexPool(kernel_size= (2, 2), stride = (2,1))
        
        self.conv5 = ComplexConv(128, 128, (2,2),(1,1))
        self.bn_5 = ComplexBN(128)
        self.pool5 = ComplexPool(kernel_size= (2, 2), stride = (1,1))
        '''
        self.conv6 = ComplexConv(128, 128, (2,2),(1,1))
        self.bn_6 = ComplexBN(128)
        self.pool6 = ComplexPool(kernel_size= (2, 2), stride = (1,1))
        
        self.conv7 = ComplexConv(128, 128, (2,2),(1,1))
        self.bn_7 = ComplexBN(128)
        self.pool7 = ComplexPool(kernel_size= (2, 2), stride = (1,1))
        '''
        self.fc1 = nn.Linear(in_features = 2*128*8*7, out_features = 2048)
        self.fc2 = nn.Linear(2048, 9)

    def forward(self, x):
        x = self.pool1(self.relu(self.bn_1(self.conv1(x)))) #before flatten 
        x = self.pool2(self.relu(self.bn_2(self.conv2(x)))) 
        x = self.pool3(self.relu(self.bn_3(self.conv3(x)))) 
        x = self.pool4(self.relu(self.bn_4(self.conv4(x)))) 
        x = self.pool5(self.relu(self.bn_5(self.conv5(x)))) 
        #x = self.pool6(self.relu(self.bn_6(self.conv6(x))))
        #x = self.pool7(self.relu(self.bn_6(self.conv6(x)))) 
        #print(x.shape)
  
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = torch.sigmoid(x)
        return x

"""## ResNet"""

class ResNet_complex(nn.Module):
  def __init__(
        self, num_classes: int = 9
        ):
        super(ResNet_complex, self).__init__()
        self.inplanes = 64
        self.conv1 = ComplexConv(in_channels = 1, out_channels = 64, kernel_size=(7,7), stride = (2,2), padding=3)
        self.bn1 = ComplexBN(64)
        self.relu = CReLU()
        self.pool1 = ComplexPool(kernel_size= (3, 3), stride = (2,2), padding=1)

        self.res11 = nn.Sequential(*self.make_res_block(64, 64, stride=(1,1)))
        self.res12 = nn.Sequential(*self.make_res_block(64, 64, stride=(1,1)))

        self.res21 = nn.Sequential(*self.make_res_block(64, 128, stride=(2,2)))
        self.res22 = nn.Sequential(*self.make_res_block(128, 128, stride=(1,1)))
        self.downsample2 = ComplexConv(64, 128, (1, 1), stride = (2,2), padding=0)

        self.res31 = nn.Sequential(*self.make_res_block(128, 256, stride=(2,2)))
        self.res32 = nn.Sequential(*self.make_res_block(256, 256, stride=(1,1)))
        self.downsample3 = ComplexConv(128, 256, (1, 1), stride = (2,2), padding=0)

        self.res41 = nn.Sequential(*self.make_res_block(256, 512, stride=(2,2)))
        self.res42 = nn.Sequential(*self.make_res_block(512, 512, stride=(1,1)))
        self.downsample4 = ComplexConv(256, 512, (1, 1), stride = (2,2), padding=0)

        self.avgpool = ComplexAdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512*2, num_classes) #
        
  def make_res_block(self, in_channel, out_channel, stride):  
      res_block = []
      res_block.append(ComplexConv(in_channel, out_channel, kernel_size=(3,3), stride=stride, padding=1, groups=1, bias=False, dilation=1))
      res_block.append(ComplexBN(out_channel))
      res_block.append(CReLU())
      res_block.append(ComplexConv(out_channel, out_channel, kernel_size=(3,3), padding=1, groups=1, bias=False, dilation=1))
      res_block.append(ComplexBN(out_channel))
      return res_block
      
  def forward(self, x):
     x = self.pool1(self.relu(self.bn1(self.conv1(x))))
     # building block 1
     x = x + self.res11(x)
     x = self.relu(x)
     x = x + self.res12(x)
     x = self.relu(x)
     # building block 2
     x = self.downsample2(x) + self.res21(x)
     x = self.relu(x)
     x = x + self.res22(x)
     x = self.relu(x)
     # building block 3
     x = self.downsample3(x) + self.res31(x)
     x = self.relu(x)
     x = x + self.res32(x)
     x = self.relu(x)
     # building block 4
     x = self.downsample4(x) + self.res41(x)
     x = self.relu(x)
     x = x + self.res42(x)
     x = self.relu(x)
     
     x = self.avgpool(x)
     #print(x.shape)
     x = torch.flatten(x, 1)
     x = self.fc(x)
     return x

class ResNet18(nn.Module):
  def __init__(
        self,
        in_channels = 1,
        num_classes: int = 9
        ):
        super(ResNet18, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(in_channels = in_channels, out_channels = 64, kernel_size=(7,7), stride = (2,2), padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size= (3, 3), stride = (2,2), padding=1)

        self.res11 = nn.Sequential(*self.make_res_block(64, 64, stride=(1,1)))
        self.res12 = nn.Sequential(*self.make_res_block(64, 64, stride=(1,1)))

        self.res21 = nn.Sequential(*self.make_res_block(64, 128, stride=(2,2)))
        self.res22 = nn.Sequential(*self.make_res_block(128, 128, stride=(1,1)))
        self.downsample2 = nn.Conv2d(64, 128, (1, 1), stride = (2,2))

        self.res31 = nn.Sequential(*self.make_res_block(128, 256, stride=(2,2)))
        self.res32 = nn.Sequential(*self.make_res_block(256, 256, stride=(1,1)))
        self.downsample3 = nn.Conv2d(128, 256, (1, 1), stride = (2,2))

        self.res41 = nn.Sequential(*self.make_res_block(256, 512, stride=(2,2)))
        self.res42 = nn.Sequential(*self.make_res_block(512, 512, stride=(1,1)))
        self.downsample4 = nn.Conv2d(256, 512, (1, 1), stride = (2,2))

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes) #

  def make_res_block(self, in_channel, out_channel, stride):
        res_block = []
        res_block.append(nn.Conv2d(in_channel, out_channel, kernel_size=(3,3), stride=stride, padding=1, groups=1, bias=False, dilation=1))
        res_block.append(nn.BatchNorm2d(out_channel))
        res_block.append(nn.ReLU(inplace=True))
        res_block.append(nn.Conv2d(out_channel, out_channel, kernel_size=(3,3), padding=1, groups=1, bias=False, dilation=1))
        res_block.append(nn.BatchNorm2d(out_channel))
        return res_block
    
  def forward(self, x):
     x = self.pool1(self.relu(self.bn1(self.conv1(x))))
     # building block 1
     x = x + self.res11(x)
     x = self.relu(x)
     x = x + self.res12(x)
     x = self.relu(x)
     # building block 2
     x = self.downsample2(x) + self.res21(x)
     x = self.relu(x)
     x = x + self.res22(x)
     x = self.relu(x)
     # building block 3
     x = self.downsample3(x) + self.res31(x)
     x = self.relu(x)
     x = x + self.res32(x)
     x = self.relu(x)
     # building block 4
     x = self.downsample4(x) + self.res41(x)
     x = self.relu(x)
     x = x + self.res42(x)
     x = self.relu(x)
     
     x = self.avgpool(x)
     #print(x.shape)
     x = torch.flatten(x, 1)
     x = self.fc(x)
     return x
