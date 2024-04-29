img_wavelets = False ## by @dv if you need WT
swt = False
#!/user/bin/python
# coding=utf-8
import os
from os.path import join,  isdir, dirname,  abspath  #split, splitext, isfile
from pathlib import Path

#from utils import Logger, arg_parser
if (swt):
    from modules.data_loader import Dataset_s3_swt as Dataset_s3
elif (img_wavelets):
    from modules.data_loader import Dataset_s3_w as Dataset_s3 ## original, commented by @dv ## use when wavelet of input image is required
else:
    from modules.data_loader import Dataset_s3 ## added by @dv ## use when WT is not required
# from modules.data_loader import  S3Images
#from modules.models import HED
from msnet import msNet
from modules.trainer import Network, Trainer
from modules.utils import struct
from modules.transforms import Fliplr, Rescale_byrate, Rescale_size
#from modules.options import arg_parser

import torch.cuda
from torch.utils.data import DataLoader, ConcatDataset
from datetime import datetime

from pandas import read_csv


# root='G:/My Drive/Research/Dataset/SR_Dataset_v1/'
root='C:/Research/Datasets/SR_Dataset_v1/'
startTime = datetime.now() ## @dv
# tag = 'HED-CWT-7epch-continue-'+startTime.strftime("%y%m%d-%H%M%S")
tag = 'SRD_v1-CWT-HED-exp2trybatchsize4'


params={
     'root': root,
     'tmp': Path(f'./model-outputs/tmp/{tag}'), ##os.getcwd()
     'log_dir': Path(f'./model-outputs/logs/{tag}'),
     # 'dev_dir':root/'sample36',
     'val_percent': 0,
      'start_epoch' : 3, # by @dv
     # 'start_epoch' : 0,
     'max_epoch' : 15,
     'batch_size': 4,
     'itersize': 10,
     'stepsize': 3,
     'lr': 1e-06,
     'momentum': 0.9,
     'weight_decay': 0.0002,
     'gamma': 0.1,
     'pretrained_path': None,
      'resume_path': f'C:/Research/Codes/HED/model-outputs/tmp/{tag}/checkpoint_epoch3.pth', # by @dv
      # 'resume_path': None,
     'use_cuda': torch.cuda.is_available()
     }

args= struct(**params)


#%% def main():

if not isdir(args.tmp):
    os.makedirs(args.tmp)
#%% define network
net=Network(args, model=msNet())


#%% train dataset S3


# df_train=read_csv(root/'lst_files/2012_dry_train.lst', header=None)
df_train=read_csv(root+'lst_files_dv/train.lst', header=None)

# images=['greenland_picks_final_2009_2012_reformat/2012/'+item for item in df_train[0].values] # for lora's 2012 data
images=[root+'/train_data/'+item for item in df_train[0].values]
# images=['Debvrat/2012_dry/'+item for item in df_train[0].values]

# ds=Dataset_s3(bucket='cresis', keys=images, s3Get=si.from_s3)
# train_loader= DataLoader(ds, batch_size=1, shuffle=True) ## by @dv


ds=[Dataset_s3(keys=images, transform=Rescale_size(224))]
# ds=[Dataset_s3(keys=images),
# Dataset_s3(keys=images, transform=Rescale_byrate(.75)),
# Dataset_s3(keys=images, transform=Rescale_byrate(.5)),
# Dataset_s3(keys=images,transform=Rescale_byrate(.25)),
# Dataset_s3(keys=images, transform=Fliplr())
# ] ### commented by @dv
# Loader
train_dataset=ConcatDataset(ds) ## original, commented by @dv
train_loader= DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True) ## original, commented by @dv


# df_dev=read_csv(root/'dev_s3.lst', header=None)
# images_dev=['greenland_picks_final_2009_2012_reformat/2012/'+item for item in df_dev[0].values]

# dev_dataset=Dataset_s3(bucket='cresis', keys=images_dev, s3Get=si.from_s3) ## commented by @dv
# dev_loader= DataLoader(dev_dataset, batch_size=1) ## commented by @dv

#%% train dataset
# ds=[Dataset_ls(root=root,lst='train_pair.lst'),
# Dataset_ls(root=root,lst='train_pair.lst', transform=Rescale_byrate(.75)),
# Dataset_ls(root=root,lst='train_pair.lst',transform=Rescale_byrate(.5)),
# Dataset_ls(root=root,lst='train_pair.lst',transform=Rescale_byrate(.25)),
# Dataset_ls(root=root,lst='train_pair.lst', transform=Fliplr())
# ]
#train_dataset=Dataset_ls(root=root,lst='train_pair.lst')


# dev dataset optional
#dev_dataset=Dataset_ls(root=root,lst='dev.lst')
#dev_loader= DataLoader(dev_dataset, batch_size=1)

#%% define trainer
trainer=Trainer(args,net, train_loader=train_loader)
#%%


# switch to train mode: not needed!  model.train()
for epoch in range(args.start_epoch, args.max_epoch):

    ## initial log (optional:sample36)
    # if epoch == 0: ## commented by @dv
    #     print("Performing initial testing...") ## commented by @dv
    #     trainer.dev(dev_loader=dev_loader,save_dir = join(args.tmp, 'testing-record-0-initial'), epoch=epoch) ## commented by @dv

    ## training
    trainer.train(save_dir = args.tmp, epoch=epoch)

    ## dev check (optional:sample36)
    # trainer.dev(dev_loader=dev_loader,save_dir = join(args.tmp, 'testing-record-epoch-%d' % epoch), epoch=epoch) ## commented by @dv

print('\n\n Training complete. Output directory: ' + tag)

endTime = datetime.now() ## @dv
elapsedTime = endTime-startTime
# Calculate days, hours, minutes, and seconds
days, seconds = divmod(elapsedTime.total_seconds(), 86400)
hours, seconds = divmod(seconds, 3600)
minutes, seconds = divmod(seconds, 60)
# Print the formatted result
print(f"{int(days)} days, {int(hours)} hours, {int(minutes)} minutes, {int(seconds)} seconds")
# if __name__ == '__main__':
#     main()
