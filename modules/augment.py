# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 09:54:00 2020

@author: yari
"""
import cv2
import numpy as np
import os
osp=os.path

from mymodules.data_loader import datasetd
from mymodules.transforms import Rescale_cv2, Fliplr

folder='image'

#root=osp.join('..','..','Datasets''Cresis/2012_main')
root=osp.join("..","..","Datasets","Cresis","macgregor","train",folder)

scales=[0.25,0.5,0.75,1.0]
ext='png'
#%% Resize
for scale in scales:
    if scale==1.0:
        ##flip only
        new_root= root.replace(folder, '{folder}_flip')
        ds=datasetd(root,ext, mode=-1, transform= Fliplr())
        print(len(ds))
        dir_name=folder+"_flip"
        new_dir=root.replace(folder, dir_name)
        if not osp.isdir(new_dir):
            os.makedirs(new_dir)
        for image , name in ds:
            new_name=osp.join(new_dir, osp.basename(name))
            cv2.imwrite(new_name, image)
    else:
        ds=datasetd(root,ext, mode=-1, transform=Rescale_cv2((scale,scale)))
        print(len(ds))
        dir_name=f"{folder}_{scale}_scale"
        new_dir=root.replace(folder, dir_name)
        if not osp.isdir(new_dir):
            os.makedirs(new_dir)
        for image , name in ds:
            new_name=osp.join(new_dir, osp.basename(name))
            cv2.imwrite(new_name, image)
        # flip the resized
        fstr=f"{folder}_{scale}_scale"
        new_root= root.replace(folder, fstr)
        ds=datasetd(root,ext, mode=-1, transform= Fliplr())
        print(len(ds))
        new_dir=new_root+"_flip"
        if not osp.isdir(new_dir):
            os.makedirs(new_dir)
        for image , name in ds:
            new_name=osp.join(new_dir, osp.basename(name))
            cv2.imwrite(new_name, image)

#%%

folder='image'

#root=osp.join('..','..','Datasets''Cresis/2012_main')
root=osp.join("..","..","Datasets","Cresis","macgregor","train",folder)

scale=0.25
ext='tiff'
fstr=f"{folder}_{scale}_scale"
new_root= root.replace(folder, fstr)
ds=datasetd(root,ext, mode=-1, transform= Fliplr())
print(len(ds))
new_dir=new_root+"_flip"
if not osp.isdir(new_dir):
    os.makedirs(new_dir)
for image , name in ds:
    new_name=osp.join(new_dir, osp.basename(name))
    cv2.imwrite(new_name, image)
for image , name in ds:
    new_name=osp.join(new_dir, osp.basename(name))
    cv2.imwrite(new_name, image)
