# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 15:21:01 2020

@author: yari
"""
img_wavelets = False ## by @dv if you need WT of input image
swt = False
import os
import numpy as np
#from PIL import Image
import cv2
import torch

import torchvision



from os.path import join,  isfile #isdir, split, splitext, abspath

#from torchvision import datasets, transforms

# from modules.models import HED ## commented by @dv
from torch.utils.data import DataLoader
#from modules.models_RCF import RCF
from msnet import msNet, msNet_4so # added by @dv


from scipy.io import savemat

from pathlib import Path
#from datetime import datetime
#import glob
#from torch.utils.data import DataLoader
#from torch.utils import data as D

from pandas import read_csv ## added by @dv

if (swt):
    from modules.data_loader import Dataset_s3_swt as Dataset_s3
elif (img_wavelets):
    from modules.data_loader import Dataset_s3_w as Dataset_s3 ## added by @dv for network with wavelets
else:
    from modules.data_loader import Dataset_s3 ## added by @dv for network without wavelets

# from modules.data_loader import  S3Images ## added by @dv


def test(model_type, restore_path, save_dir, test_loader , output_ext, input_prefix , output_prefix):
    # model
    model = model_type
    #model = nn.DataParallel(model)
   # model.cuda()
    if torch.cuda.is_available():
        model.cuda()

    if isfile(restore_path):
        print("=> loading checkpoint '{}'".format(restore_path))
    else:
        raise('Restore path error!')

    if torch.cuda.is_available():
        checkpoint=torch.load(restore_path)
    else:
        checkpoint = torch.load(restore_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['state_dict'])
    print("=> loaded checkpoint")
    

    # model
    model.eval()
    # setup the save directories
    dirs = ['side_1', 'side_2', 'side_3', 'side_4', 'side_5', 'fuse', 'merge', 'jpg_out']
    for idx, dir_path in enumerate(dirs):
        os.makedirs(join(save_dir, dir_path), exist_ok=True)
        if(idx < 6): os.makedirs(join(save_dir, 'mat' , dir_path), exist_ok=True)
    # run test
    #print(len(test_loader))
    for idx, data in enumerate(test_loader):

        print("\rRunning test [%d/%d]" % (idx + 1, len(test_loader)), end='')
        image=data['image']

        filename=data['id'][0].replace(input_prefix, output_prefix)
        
        w = data.get('w',None) ## added by @dv

        if torch.cuda.is_available():
            image = image.cuda()
            if w is not None:
                for key in w: ## added by @dv
                    w[key]=w[key].cuda() ## added by @dv
                
        _,_ , H, W = image.shape
        with torch.no_grad():
            # results = model(image) ## original, commented by @dv 
            results = model(image,w) ## added by @dv
            # results = torch.round(results) ## by @dv for binarizing outptus TGRS
            # results = model(image,None) ## added by @dv. network without wavelets
        results_all = torch.zeros((len(results), 1, H, W))
       ## make our result array
        for i in range(len(results)):
          results_all[i, 0, :, :] = results[i]
        #filename = splitext(split(test_list[idx])[1])[0]
        torchvision.utils.save_image(results_all, join(save_dir, dirs[-1], f'{filename}.{output_ext}'))

        # now go through and save all the results
        for i, r in enumerate(results):

            img= torch.squeeze(r.detach()).cpu().numpy()
            savemat(join(save_dir,'mat',dirs[i],'{}.mat'.format(filename)), {'img': img})
            #img = Image.fromarray((img * 255).astype(np.uint8))

            #img.save(join(save_dir,dirs[i],'{}.jpg'.format(filename)))
            cv2.imwrite(join(save_dir,dirs[i],f'{filename}.{output_ext}'), (img*255).astype(np.uint8))


        merge = sum(results) / 5
        torchvision.utils.save_image(torch.squeeze(merge), join(save_dir, dirs[-2], f'{filename}.{output_ext}'))
        torchvision.transforms.transforms.ToPILImage(torch.squeeze(results[i]))
    print('')

#root=join('..','..','atasets','HED-BSDS')
#test_loader=get_dataLoaders_eval(root=root, image_dir=None , eval_image_list='test.lst' , ext='png' )
#%%

input_ext='jpg'
input_ext='tiff' ## @dv
output_ext='jpg'
output_ext='png' ## @dv
input_prefix='layer_binary'
output_prefix='layer_binary'
batch_size=1

root=Path(".") ## added by @dv
root = 'C:/Research/Datasets/SR_Dataset_v1/'
# root = 'G:/My Drive/Research/Dataset/MacGregor_200824_AWS/'

# df_test=read_csv(root/'lst_files/2012_dry_test.lst', header=None) ## added by @dv
# images=['greenland_picks_final_2009_2012_reformat/2012/'+item for item in df_test[0].values] ## added by @dv
# images=['Debvrat/2012_dry/'+item for item in df_test[0].values] ## added by @dv

# df_test=read_csv(root/'lst_files/big_frames.lst', header=None)
# images=['Debvrat/big_frames/'+item for item in df_test[0].values]

"""
2012_Greenland_P3
"""
# df_test=read_csv(root/'lst_files/frames_001_243_20120330_04_dv_test.lst', header=None)
# images=['Debvrat/frames_001_243_20120330_04_dv/'+item for item in df_test[0].values]


"""
SR_Dataset_v1
"""
# df_test=read_csv(root+'lst_files_dv/test_L3.lst', header=None)
# images=[root+'final_test_data_dv/'+item for item in df_test[0].values]

df_test=read_csv(root+'lst_files_dv/test.lst', header=None)
images=[root+'final_test_data/'+item for item in df_test[0].values]


"""
RDS
"""
# df_test=read_csv(root+'/frames_001_018_20120508_04.lst', header=None)
# images=[root+'frames_001_018_20120508_04/image/'+item for item in df_test[0].values]



test_dataset=Dataset_s3(keys=images)
test_loader= DataLoader(test_dataset, batch_size=batch_size, shuffle=True) ## added by @dv

tags = ['SRD_v1-CWT-HED-batch4']



for tag in tags:
    print('processing tag: ' + tag) ### @dv
    #test_root= Path('G:/My Drive/BinaLab/Datasets/Cresis/2012_main/test/image')
    # test_root= Path(f'C:/Users/yari/Documents/testsON12_trainedON12/{tag}/fuse') ## original, commented by @dv
    # test_root= Path('G:/My Drive/Research_Debvrat/Dataset/Snow Radar/2012_main_dv/test/image') ## @dv

    # test_dataset=BasicDataset(test_root, ext=input_ext) ## original, commented by @dv
    # test_loader= DataLoader(test_dataset, batch_size=batch_size) ## original, commented by @dv
        
    
    restore_path=Path(f'./model-outputs/tmp/{tag}/checkpoint_epoch15.pth')
    save_dir=Path(f'G:/My Drive/Research/Outputs/RDS/HED/tmp/{tag}')
    save_dir=Path(f'G:/My Drive/Research/Outputs/SRD_v1/HED/tmp/{tag}')
    
    # model_type= HED() ## original, commented by @dv
    model_type= msNet() ## added by @dv
    # model_type= msNet_4so() ## added by @dv
    
    test(model_type= model_type,
        restore_path=restore_path,
        save_dir=save_dir,      
        test_loader=test_loader,
        output_ext=output_ext,
        input_prefix=input_prefix,
        output_prefix=output_prefix
        )
# =============================================================================
