########## HDNet ##############################
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
import random

def weight_init(module):
    for n, m in module.named_children():
        print('initialize: '+n)
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Dropout2d):
            weight_init(m)
        elif isinstance(m, nn.Sequential):
            weight_init(m)
        elif isinstance(m, nn.Sigmoid):
            weight_init(m)
        elif isinstance(m, nn.ConvTranspose2d):
            weight_init(m)
        elif isinstance(m, nn.AdaptiveAvgPool2d):
            weight_init(m)
        elif isinstance(m, nn.Upsample):
            weight_init(m)    
        elif isinstance(m, nn.ReLU):
            pass
        elif isinstance(m, HNet):
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                m.weight.data.normal_(0.0, 0.02)
            elif classname.find('BatchNorm') != -1:
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)
        else:
            m.initialize()
            
################  Foreground Attention Moudle (FAM) ###################################
class FAM(nn.Module):
    def __init__(self):
        super(FAM, self).__init__()
        
        self.conv_fam1 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn_fam1   = nn.BatchNorm2d(32)
        
        self.conv_fam2 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1)
        self.bn_fam2   = nn.BatchNorm2d(1)    
         
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        fam1_out = F.relu(self.bn_fam1(self.conv_fam1(x)), inplace=True)
        maxout, _ = torch.max(fam1_out, dim=1, keepdim=True)
        fam2_out = F.relu(self.bn_fam2(self.conv_fam2(maxout)), inplace=True)

        return self.sigmoid(fam2_out)
#############################################################################  

class HNet(nn.Module):

	def __init__(self):
		super(HNet, self).__init__()

		self.relu = nn.ReLU(inplace=True)

		number_f = 32
		self.e_conv1 = nn.Conv2d(3,number_f,3,1,1,bias=True) 
		self.e_conv2 = nn.Conv2d(number_f,number_f,3,1,1,bias=True) 
		self.e_conv3 = nn.Conv2d(number_f,number_f,3,1,1,bias=True) 
		self.e_conv4 = nn.Conv2d(number_f,number_f,3,1,1,bias=True) 
		self.e_conv5 = nn.Conv2d(number_f*2,number_f,3,1,1,bias=True) 
		self.e_conv6 = nn.Conv2d(number_f*2,number_f,3,1,1,bias=True) 
		self.e_conv7 = nn.Conv2d(number_f*2,3,3,1,1,bias=True)  
		self.fam = FAM()
        
	def forward(self, x):
		x1 = self.relu(self.e_conv1(x))
		# print("x1.shape:", x1.shape)
		x2 = self.relu(self.e_conv2(x1))
		# print("x2.shape:", x2.shape)
		x3 = self.relu(self.e_conv3(x2))
		# print("x3.shape:", x3.shape)
		x4 = self.relu(self.e_conv4(x3))
		# print("x4.shape:", x4.shape)
		x4_fam = self.fam(x4)
		# print("x4_fam.shape:", x4_fam.shape)
		x4 = x4 * x4_fam
		# print("x4_2.shape:", x4.shape)
		x5 = self.relu(self.e_conv5(torch.cat([x3,x4],1)))
		# print("x5.shape:", x5.shape)
		x6 = self.relu(self.e_conv6(torch.cat([x2,x5],1)))
		# print("x6.shape:", x6.shape)
		r = F.tanh(self.e_conv7(torch.cat([x1,x6],1)))
		# print("r.shape:", r.shape)
		enhance_image = x + r*(torch.pow(x,2)-x)

		return enhance_image, r, x4_fam


class Bottleneck(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(Bottleneck, self).__init__()
        self.conv1      = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1        = nn.BatchNorm2d(planes)
        self.conv2      = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=(3*dilation-1)//2, bias=False, dilation=dilation)
        self.bn2        = nn.BatchNorm2d(planes)
        self.conv3      = nn.Conv2d(planes, planes*4, kernel_size=1, bias=False)
        self.bn3        = nn.BatchNorm2d(planes*4)
        self.downsample = downsample

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            x = self.downsample(x)
        return F.relu(out+x, inplace=True)


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1    = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1      = nn.BatchNorm2d(64)
        self.layer1   = self.make_layer( 64, 3, stride=1, dilation=1)
        self.layer2   = self.make_layer(128, 4, stride=2, dilation=1)
        self.layer3   = self.make_layer(256, 6, stride=2, dilation=1)
        self.layer4   = self.make_layer(512, 3, stride=2, dilation=1)

    def make_layer(self, planes, blocks, stride, dilation):
        downsample    = nn.Sequential(nn.Conv2d(self.inplanes, planes*4, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes*4))
        layers        = [Bottleneck(self.inplanes, planes, stride, downsample, dilation=dilation)]
        self.inplanes = planes*4
        for _ in range(1, blocks):
            layers.append(Bottleneck(self.inplanes, planes, dilation=dilation))
        return nn.Sequential(*layers)

    def forward(self, x):
        out1 = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out1 = F.max_pool2d(out1, kernel_size=3, stride=2, padding=1)
        out2 = self.layer1(out1)
        out3 = self.layer2(out2)
        out4 = self.layer3(out3)
        out5 = self.layer4(out4)
        return out2, out3, out4, out5     # 加一个返回out1

    def initialize(self):
        self.load_state_dict(torch.load('/data/Wang/lx/Enhancement/Zero-DCE-master/Zero-DCE_code/resnet50-19c8e357.pth'), strict=False)
    
 
###################################################################
# ###################### MFR  ##########################
###################################################################
class MFR(nn.Module):
    def __init__(self):
        super(MFR, self).__init__()
        self.conv1e = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn1e   = nn.BatchNorm2d(64)
        self.conv2e = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn2e   = nn.BatchNorm2d(64)
        
    def forward(self, x):
        out1e = F.relu(self.bn1e(self.conv1e(x)), inplace=True)
        out2e = F.relu(self.bn2e(self.conv2e(out1e)), inplace=True)
        out3e = out2e + x 
        return out3e
    
    def initialize(self):
        weight_init(self)

###################################################################
# ###################### Context feature fusion ##########################
###################################################################
class CFF(nn.Module):
    def __init__(self):
        super(CFF, self).__init__()
        self.conv1c = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn1c   = nn.BatchNorm2d(64)
        self.conv2c = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn2c   = nn.BatchNorm2d(64)
        self.conv3c = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn3c   = nn.BatchNorm2d(64)
        
        self.conv4c = nn.Conv2d(128, 64, kernel_size=1, bias=False)
        self.bn4c   = nn.BatchNorm2d(64)

        
    def forward(self, x, y):
        out1c = F.relu(self.bn1c(self.conv1c(x)), inplace=True)
        print("x.shape:", x.shape)
        print("y.shape:", y.shape)
        print("out1c.shape:", out1c.shape)
        out2c = F.relu(self.bn2c(self.conv2c(y)), inplace=True)
        print("out2c_1.shape:", out2c.shape)
        out2c = F.interpolate(out2c, size=out1c.size()[2:], mode='bilinear')
        print("out2c_2.shape:", out2c.shape)
        out_fuse = out1c * out2c
        print("out_fuse.shape:", out_fuse.shape)
        out3c = F.relu(self.bn3c(self.conv3c(out_fuse)), inplace=True)      
        print("out3c.shape:", out3c.shape)
        out_concat = torch.cat((out2c, out3c),1)      
        print("out_concat.shape:", out_concat.shape)
        out4c = F.relu(self.bn4c(self.conv4c(out_concat)), inplace=True)
        print("out4c.shape:", out4c.shape)
        out5c = out3c + out4c
        print("out5c.shape:", out5c.shape)
        
        return out5c, out4c
    
    def initialize(self):
        weight_init(self)
 
class DecoderBlock(nn.Module):
    def __init__(self):
        super(DecoderBlock, self).__init__()
        self.cff45  = CFF()
        self.cff34  = CFF()
        self.cff23  = CFF()
        
        self.mfr5 = MFR()
        self.mfr4 = MFR()
        self.mfr3 = MFR()
        self.mfr2 = MFR()

        self.dila_conv5 = nn.Sequential(nn.Conv2d(64, 64, 3, 1, 1, dilation=1), nn.BatchNorm2d(64), nn.ReLU())
        self.dila_conv4 = nn.Sequential(nn.Conv2d(64, 64, 3, 1, 2, dilation=2), nn.BatchNorm2d(64), nn.ReLU())   
        self.dila_conv3 = nn.Sequential(nn.Conv2d(64, 64, 3, 1, 5, dilation=5), nn.BatchNorm2d(64), nn.ReLU())    
        self.dila_conv2 = nn.Sequential(nn.Conv2d(64, 64, 3, 1, 7, dilation=7), nn.BatchNorm2d(64), nn.ReLU())
        
        self.dila_conv = nn.Sequential(nn.Conv2d(64, 64, 3, 1, 5, dilation=5), nn.BatchNorm2d(64), nn.ReLU())

    def forward(self, out2h, out3h, out4h, out5v):
                  
        out5v = self.dila_conv(out5v)
        # print("out4h1.shape:", out4h.shape)
        out45, out4h = self.cff45(out4h, out5v)
        # print("out4h2.shape:", out4h.shape)
        # print("out5v.shape:", out5v.shape)
        # print("out45.shape:", out45.shape)
        
        # print("out3h1.shape:", out3h.shape)
        out34, out3h = self.cff34(out3h, out45)
        # print("out3h2.shape:", out3h.shape)
        # print("out34.shape:", out34.shape)
        
        
        out23, out2h = self.cff23(out2h, out34)
        
        out5b = self.mfr5(out5v)
        out4b = self.mfr4(out4h)
        out3b = self.mfr3(out3h)
        out2b = self.mfr2(out2h)

        out5v = self.dila_conv5(out5b)
        out5v = F.interpolate(out5v, size=out4b.size()[2:], mode='bilinear')
        fuse45 = out4b + out5v
        out4h = self.dila_conv4(fuse45)
        
        out4h = F.interpolate(out4h, size=out3b.size()[2:], mode='bilinear')
        fuse34 = out3b + out4h
        out3h = self.dila_conv3(fuse34)
        
        out3h = F.interpolate(out3h, size=out2b.size()[2:], mode='bilinear')
        fuse23 = out2b + out3h
        out2h = self.dila_conv2(fuse23)

        return out2h, out3h, out4h, out5v
    
    def initialize(self):
        weight_init(self)

#####################################################################
class HDNet(nn.Module):
    def __init__(self, cfg):
        
        super(HDNet, self).__init__()
        self.DCE = HNet()
        
        self.drop = nn.Dropout2d(0.5)
        
        self.decoder = DecoderBlock()
        
        self.cfg      = cfg
        self.bkbone   = ResNet()
        self.squeeze5 = nn.Sequential(nn.Conv2d(2048, 64, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.squeeze4 = nn.Sequential(nn.Conv2d(1024, 64, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.squeeze3 = nn.Sequential(nn.Conv2d( 512, 64, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.squeeze2 = nn.Sequential(nn.Conv2d( 256, 64, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        
        self.linearp1 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)

        self.linearr2 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.linearr3 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.linearr4 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.linearr5 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.initialize()
        
    def forward(self, x, epoch=None, shape=None):
        enhance_image, r, x4_fam   = self.DCE(x)
            
        out2h, out3h, out4h, out5v = self.bkbone(enhance_image)
        out2h, out3h, out4h, out5v = self.squeeze2(out2h), self.squeeze3(out3h), self.squeeze4(out4h), self.squeeze5(out5v)
        out2h, out3h, out4h, out5v= self.decoder(out2h, out3h, out4h, out5v)
        
        shape = enhance_image.size()[2:] if shape is None else shape
        
        # pred = F.interpolate(self.linearp1(pred), size=shape, mode='bilinear')

        out2h = F.interpolate(self.linearr2(out2h), size=shape, mode='bilinear')
        out3h = F.interpolate(self.linearr3(out3h), size=shape, mode='bilinear')
        out4h = F.interpolate(self.linearr4(out4h), size=shape, mode='bilinear')
        out5v = F.interpolate(self.linearr5(out5v), size=shape, mode='bilinear')
        return  out2h, out3h, out4h, out5v, enhance_image, r, x4_fam


    def initialize(self):
        if self.cfg.snapshot:
            self.load_state_dict(torch.load(self.cfg.snapshot))
        else:
            weight_init(self)