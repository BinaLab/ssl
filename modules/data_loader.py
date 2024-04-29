

from torch.utils import data as D
from os.path import join, abspath #,split, abspath, splitext, split, isdir, isfile
import numpy as np
import cv2
import os

import pandas as pd

import glob

from scipy.io import loadmat

import pywt
import math # added by @dv
#S3
import boto3
from PIL import Image
from io import BytesIO
from botocore.client import Config # added by @dv
from msnet.msnet_parts import crop as crop_img #@dv
from msnet.msnet_parts import make_bilinear_weights #@dv
import torch.nn.functional as F #@dv
import torch #@dv
import matplotlib.pyplot as plt
from scipy.signal import find_peaks_cwt

class Dataset_s3(D.Dataset):
    """
    dataset from list
    returns data after preperation
    """
    # def __init__(self, bucket, keys, s3Get, prepare=False, transform=None): # original commented by @dv
    def __init__(self, keys, prepare=True, transform=None): # @dv
        self.df=keys
        self.transform=transform
        self.prepare=prepare

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):

        key=self.df[index]
        assert os.path.isfile(key), "file  {}. doesn't exist.".format(key)
        
        img = cv2.imread(key, cv2.IMREAD_GRAYSCALE)
        ctour=np.array(ctour, dtype=np.float32)
        img=np.array(img, dtype=np.float32)

 
        
        if self.transform:
            img=self.transform(img)
            ctour=self.transform(ctour)

        if self.prepare: ## original, commented by @dv
            img, ctour= prepare_img(img), prepare_ctour(ctour) ## original, commented by @dv
            # img, ctour= prepare_img_mat(img), prepare_ctour(ctour) ## added for tiff processing by @dv
        (data_id, _) = os.path.splitext(os.path.basename(key))



        return {'image': img, 'mask' : ctour , 'id': data_id}



class BasicDataset(D.Dataset):
    """
    dataset from directory
    returns img after preperation, no label
    """
    def __init__(self, root, ext):
        self.root=root
        self.ext=ext
        self.rel_paths=glob.glob(join(root, '*.{}'.format(ext)))

    def __len__(self):
        return len(self.rel_paths)

    def __getitem__(self, index):

        # get image
        img_abspath= abspath(self.rel_paths[index])
        assert os.path.isfile(img_abspath), "file  {}. doesn't exist.".format(img_abspath)

        img=cv2.imread(img_abspath) ### original, commented by @dv

        img= prepare_img_3c(img) ### original, commented by @dv

        # img = Image.open(img_abspath) ## added by @dv
        # img=np.array(img, dtype=np.float32) ## added by @dv
        
        (data_id, _) = os.path.splitext(os.path.basename(img_abspath))


        return {'image': img, 'id':data_id}



def prepare_img_mat(img):
        img=np.array(img, dtype=np.float32)
        #img=(img-np.min(img))/(np.max(img)-np.min(img))
        img=img*255
        img=np.expand_dims(img, axis=2)
        img=np.repeat(img,3,axis=2)
        img -= np.array((104.00698793,116.66876762,122.67891434))
        img=img.transpose(2,0,1)
        return img


"""assuming img is made up of 3 channels"""
def prepare_img_mat_2(img):
        img=np.array(img, dtype=np.float32)
        #img=(img-np.min(img))/(np.max(img)-np.min(img))
        img=img*255
        img=img.transpose(1,2,0)
        img -= np.array((104.00698793,116.66876762,122.67891434))
        img=img.transpose(2,0,1)
        return img
    
def prepare_img_mat_tiff(img):
        img=np.array(img, dtype=np.float32)
        #img=(img-np.min(img))/(np.max(img)-np.min(img))
        img=img*255

        img -= np.array((104.00698793,116.66876762,122.67891434))

        return img

def prepare_img(img):
        img=np.expand_dims(img,axis=2)
        img=np.repeat(img,3,axis=2)
        img -= np.array((104.00698793,116.66876762,122.67891434))
        img=img.transpose(2,0,1)
        return img

def prepare_ctour(ctour):
        #ctour=np.array(ctour, dtype=np.float32)
        ctour = (ctour > 0 ).astype(np.float32)
        ctour=np.expand_dims(ctour,axis=0)
        return ctour

def prepare_img_3c(img):
        img=np.array(img, dtype=np.float32)
        img -= np.array((104.00698793,116.66876762,122.67891434))
        img=img.transpose(2,0,1)
        return img

def prepare_w(img):
        img=np.array(img, dtype=np.float32)
        img=np.expand_dims(img,axis=0)
        return img


def wt_scale(wt):
    return 255*(wt-np.min(wt))/(np.max(wt)-np.min(wt))

def get_wt(im, wname, mode, level,scaleit=False):
    w=pywt.wavedec2(im,wname, mode=mode, level=level)
    if scaleit:
        wt={f'cA{level}': prepare_w(wt_scale(w[0]))}
        for i in range(1,level):
            wt.update({f'cH{i}': prepare_w(wt_scale(w[-i][0]))})
            wt.update({f'cV{i}': prepare_w(wt_scale(w[-i][1]))})
            wt.update({f'cD{i}': prepare_w(wt_scale(w[-i][2]))})
    else:
        wt={f'cA{level}': prepare_w(w[0])}
        # for i in range(1,level): 
        for i in range(1,level+1): # @dv
            wt.update({f'cH{i}': prepare_w(w[-i][0])})
            wt.update({f'cV{i}': prepare_w(w[-i][1])})
            wt.update({f'cD{i}': prepare_w(w[-i][2])})
    return wt

def crop(variable, th, tw):
        # h, w = variable.shape[2], variable.shape[3] ## original, commented by @dv
        h, w = variable.shape[1], variable.shape[2] ## added by @dv
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        # return variable[:, :, y1 : y1 + th, x1 : x1 + tw] ## original, commented by @dv
        return variable[:, y1 : y1 + th, x1 : x1 + tw] ## by @dv
    
    
### added by @dv. Crop out wavelet transforms so that they do not create a concatenation issue
### especially for non-haar wavelets ## @dv
def get_wt_dv(im, wname, mode, level,scaleit=False):
    rows, cols = im.shape
    w=pywt.wavedec2(im,wname, mode=mode, level=level)
    wt={f'cA{level}': prepare_w(w[0])}
    for i in range(1,level+1):
        w_row = math.ceil(rows/(2**i))
        w_col = math.ceil(cols/(2**i))
        
        wt.update({f'cH{i}': crop(prepare_w(w[-i][0]),w_row,w_col)})
        wt.update({f'cV{i}': crop(prepare_w(w[-i][1]),w_row,w_col)})
        wt.update({f'cD{i}': crop(prepare_w(w[-i][2]),w_row,w_col)})
    return wt
# def make_wlets(original, wname='db1', level=4):
#    # original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
#     original = np.array(original, dtype=np.float32)
#     coeffs=pywt.wavedec2(original, wname, level=level)
#     #LL, (LH, HL, HH) = pywt.dwt2(original, 'db1')
#     cA4, (cH4,cV4,cD4),(cH3,cV3,cD3),(cH2,cV2,cD2),(cH1,cV1,cD1)=coeffs
#     return cH4,cV4,cD4, cH3,cV3,cD3,cH2,cV2,cD2, cH1,cV1,cD1

# enum ImreadModes
# {
#     IMREAD_UNCHANGED           = -1,
#     IMREAD_GRAYSCALE           = 0,
#     IMREAD_COLOR               = 1,
#     IMREAD_ANYDEPTH            = 2,
#     IMREAD_ANYCOLOR            = 4,
#     IMREAD_LOAD_GDAL           = 8,
#     IMREAD_REDUCED_GRAYSCALE_2 = 16,
#     IMREAD_REDUCED_COLOR_2     = 17,
#     IMREAD_REDUCED_GRAYSCALE_4 = 32,
#     IMREAD_REDUCED_COLOR_4     = 33,
#     IMREAD_REDUCED_GRAYSCALE_8 = 64,
#     IMREAD_REDUCED_COLOR_8     = 65,
#     IMREAD_IGNORE_ORIENTATION  = 128,}
