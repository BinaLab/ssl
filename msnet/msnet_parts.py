""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pywt ## added by @dv
import math ## added by @dv

### added by @dv    
# class waveT(object):
#     def __init__(self, level):
#         self.level = level
#     def __call__(self, x):
#         wt=pywt.wavedec2(x.cpu().detach().numpy(),'haar', mode='periodic', level=self.level)
#         #### return only high frequency outputs below
#         return torch.from_numpy(wt[1][0]),torch.from_numpy(wt[1][1]),torch.from_numpy(wt[1][2])
    

### added by @dv ## for haar   
# class waveT(object):
#     def __init__(self, level):
#         self.level = level
#     def __call__(self, x):
#         w=pywt.wavedec2(x.cpu().detach().numpy(),'haar', mode='periodic', level=self.level)
#         # wt={}
#         wt={f'cA{self.level}': prepare_w(w[0])}
#         for i in range(1,self.level+1): # @dv
#             wt.update({f'cH{i}': prepare_w(w[-i][0])})
#             wt.update({f'cV{i}': prepare_w(w[-i][1])})
#             wt.update({f'cD{i}': prepare_w(w[-i][2])})
        
#         return wt


### added by @dv from another file. originally by Dr. Yari
# def crop(variable, th, tw):
#         # h, w = variable.shape[2], variable.shape[3] ## original, commented by @dv
#         h, w = variable.shape[1], variable.shape[2] ## added by @dv
#         x1 = int(round((w - tw) / 2.))
#         y1 = int(round((h - th) / 2.))
#         # return variable[:, :, y1 : y1 + th, x1 : x1 + tw] ## original, commented by @dv
#         return variable[:, y1 : y1 + th, x1 : x1 + tw] ## by @dv

## @dv
def tensorToNumpy(x):
    return x.cpu().detach().numpy()

def numpyToTensor(img):
    return torch.from_numpy(img).cuda()

### added by @dv ## for db2 etc
class waveT(object):
    def __init__(self, level, wavelet):
        self.level = level
        self.wavelet = wavelet
        
    def __call__(self, x):
        im = tensorToNumpy(x)
        _,_,rows, cols = im.shape
        w=pywt.wavedec2(im,self.wavelet, mode='periodic', level=self.level)
        # wt={}
        wt={f'cA{self.level}': prepare_w(w[0])}
        for i in range(1,self.level+1): # @dv
            w_row = math.ceil(rows/(2**i))
            w_col = math.ceil(cols/(2**i))
            
            wt.update({f'cH{i}': crop(prepare_w(w[-i][0]),w_row,w_col)})
            wt.update({f'cV{i}': crop(prepare_w(w[-i][1]),w_row,w_col)})
            wt.update({f'cD{i}': crop(prepare_w(w[-i][2]),w_row,w_col)})
        
        return wt
## by @dv  
class invWaveT(object):
    def __init__(self, wavelet, mode='periodic'):
        self.wavelet = wavelet
        self.mode = mode
        
    def __call__(self, coeffs):
        ### assuming we get (cA2, (cH2, cV2, cD2)) as coeffs
        coeffs[0] = tensorToNumpy(coeffs[0])
        for idx, coeff in enumerate(coeffs[1]):
            coeffs[1][idx] = tensorToNumpy(coeff)
            
        img = pywt.waverec2(coeffs, self.wavelet, mode=self.mode)   
        return numpyToTensor(img)

#### @dv
def prepare_w(img):
        img=np.array(img, dtype=np.float32)
        # img=np.expand_dims(img,axis=0) ## commented by @dv
        return numpyToTensor(img)
    

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            # nn.BatchNorm2d(mid_channels), ## commented by yari
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            # nn.BatchNorm2d(out_channels), ## commented by yari
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class TripleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 3"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.triple_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            # nn.BatchNorm2d(mid_channels), ## commented by @dv
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            # nn.BatchNorm2d(out_channels), ## commented by @dv
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            # nn.BatchNorm2d(out_channels), ## commented by @dv
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.triple_conv(x)


### @dv
class QuadConv(nn.Module):
    """(convolution => [BN] => ReLU) * 4"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.quad_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            # nn.BatchNorm2d(mid_channels), ## commented by @dv
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            # nn.BatchNorm2d(out_channels), ## commented by @dv
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            # nn.BatchNorm2d(out_channels), ## commented by @dv
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            # nn.BatchNorm2d(out_channels), ## commented by @dv
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.quad_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)
    

class Down3(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_3conv = nn.Sequential(
            # nn.MaxPool2d(2), # original, commented by @dv
            nn.MaxPool2d(2, stride=2, ceil_mode=True), # added by @dv from Down()
            TripleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_3conv(x)

### @dv
class Down4(nn.Module):
    """Downscaling with maxpool then four conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_4conv = nn.Sequential(
            # nn.MaxPool2d(2), # original, commented by @dv
            nn.MaxPool2d(2, stride=2, ceil_mode=True), # added by @dv from Down()
            QuadConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_4conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)



# class Up2(nn.Module):
#     """Upscaling then double conv: not working!"""

#     def __init__(self, stride=2):
#         super().__init__()
#         self.stride=stride
#         self.num_classes=1
#         if torch.cuda.is_available():
#             self.weight_deconv =  make_bilinear_weights(size=stride*2, num_channels=self.num_classes).cuda()
#         else:
#             self.weight_deconv =  make_bilinear_weights(size=stride*2,  num_channels=self.num_classes)

        
#         def forward(self, H, W, x):
#             upsample=F.conv_transpose2d(x, self.weight_deconv, stride=self.stride)
#             return crop(upsample, H, W)
            




# class Up3(nn.Module):
#     """Upscaling then double conv: not working!"""

#     def __init__(self, in_channels, out_channels, bilinear=True):
#         super().__init__()

#         # if bilinear, use the normal convolutions to reduce the number of channels
#         if bilinear:
#             self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
#         else:
#             self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)


#     def forward(self, x1, x2):
#         x1 = self.up(x1)
#         # input is CHW
#         diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
#         diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

#         x = F.pad(x1, [diffX // 2, diffX - diffX // 2,
#                         diffY // 2, diffY - diffY // 2])
#         # if you have padding issues, see
#         # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
#         # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
#         #x = torch.cat([x2, x1], dim=1)
#         return x


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels=1):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

def make_bilinear_weights(size, num_channels):
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    # print(filt)
    filt = torch.from_numpy(filt)
    w = torch.zeros(num_channels, num_channels, size, size)
    w.requires_grad = False
    for i in range(num_channels):
        for j in range(num_channels):
            if i == j:
                w[i, j] = filt
    return w

def crop(variable, th, tw):
        h, w = variable.shape[2], variable.shape[3]
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        return variable[:, :, y1 : y1 + th, x1 : x1 + tw]