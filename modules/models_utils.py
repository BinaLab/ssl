import torch
import torch.nn as nn
#import torchvision.models as models
import numpy as np
import torch.nn.functional as F


from os.path import join, isfile
#import torch

#import torch.nn.init as init
import torch.cuda
import cv2

#from pytorch_wavelets import DWTForward
# from torchvision import transforms



def convert_vgg(vgg16):
	net = vgg()
	vgg_items = list(net.state_dict().items())
	vgg16_items = list(vgg16.items())
	pretrain_model = {}
	j = 0
	for k, v in net.state_dict().items():
	    v = vgg16_items[j][1]
	    k = vgg_items[j][0]
	    pretrain_model[k] = v
	    j += 1
	return pretrain_model

def vgg_pretrain(model, pretrained_path):
    pretrained_dict = torch.load(pretrained_path)
    pretrained_dict = convert_vgg(pretrained_dict)

    model_dict = model.state_dict()
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

def resume(model, resume_path):
    if isfile(resume_path):
        print("=> loading checkpoint '{}'".format(resume_path))
        checkpoint = torch.load(resume_path)
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}'".format(resume_path))
    else:
        print("=> no checkpoint found at '{}'".format(resume_path))

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        # xavier(m.weight.data)
        #m.weight.data.normal_(0, 0.01)
        if m.weight.data.shape == torch.Size([1, 5, 1, 1]):
            # for new_score_weight
            torch.nn.init.constant_(m.weight, 0.2)
        if m.weight.data.shape == torch.Size([1, 4, 1, 1]):
            # for new_score_weight
            torch.nn.init.constant_(m.weight, 0.25)
        if m.bias is not None:
            m.bias.data.zero_()


def convert_vgg(vgg16):
	net = vgg()
	vgg_items = list(net.state_dict().items())
	vgg16_items = list(vgg16.items())
	pretrain_model = {}
	j = 0
	for k, v in net.state_dict().items():
	    v = vgg16_items[j][1]
	    k = vgg_items[j][0]
	    pretrain_model[k] = v
	    j += 1
	return pretrain_model




# make a bilinear interpolation kernel
# def upsample_filt(size):
#     factor = (size + 1) // 2
#     if size % 2 == 1:
#         center = factor - 1
#     else:
#         center = factor - 0.5
#     og = np.ogrid[:size, :size]
#     return (1 - abs(og[0] - center) / factor) * \
#            (1 - abs(og[1] - center) / factor)

# # set parameters s.t. deconvolutional layers compute bilinear interpolation
# # N.B. this is for deconvolution without groups
# def interp_surgery(in_channels, out_channels, h, w):
#     weights = np.zeros([in_channels, out_channels, h, w])
#     if in_channels != out_channels:
#         raise ValueError("Input Output channel!")
#     if h != w:
#         raise ValueError("filters need to be square!")
#     filt = upsample_filt(h)
#     weights[range(in_channels), range(out_channels), :, :] = filt
#     return np.float32(weights)



# def make_wlets(X):
#     xfm = DWTForward(J=4, wave='db1').cuda()

#     cA,(c1,c2,c3,c4)=xfm(X)
#     # cH1,cV1,cD1=c1[:,:,0,:,:], c1[:,:,1,:,:], c1[:,:,2,:,:]
#     # cH2,cV2,cD2=c2[:,:,0,:,:], c2[:,:,1,:,:], c2[:,:,2,:,:]
#     # cH3,cV3,cD3=c3[:,:,0,:,:], c3[:,:,1,:,:], c3[:,:,2,:,:]
#     # cH4,cV4,cD4=c4[:,:,0,:,:], c4[:,:,1,:,:], c4[:,:,2,:,:]

#     cH1,cV1,cD1=c1[:,0,0,:,:], c1[:,0,1,:,:], c1[:,0,2,:,:]
#     cH2,cV2,cD2=c2[:,0,0,:,:], c2[:,0,1,:,:], c2[:,0,2,:,:]
#     cH3,cV3,cD3=c3[:,0,0,:,:], c3[:,0,1,:,:], c3[:,0,2,:,:]
#     cH4,cV4,cD4=c4[:,0,0,:,:], c4[:,0,1,:,:], c4[:,0,2,:,:]

#     return cH4,cV4,cD4,cH3,cV3,cD3,cH2,cV2,cD2,cH1,cV1,cD1#cA,c3H,c2H,c1H #w2,w3,w4,w5
