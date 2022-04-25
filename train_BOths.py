# py libs
import os
import sys
import yaml
import argparse
import numpy as np
import math
from PIL import Image
# pytorch libs
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision.transforms as transforms
# local libs
from nets.commons import Weights_Normal, VGG19_PercepLoss
from nets.BOths import BOths
from utils.data_utils import GetTrainingPairs
from utils.wavelet_mse_loss import WMSEloss

## get configs and training options
parser = argparse.ArgumentParser()
parser.add_argument("--cfg_file", type=str, default="configs/train.yaml")
parser.add_argument("--epoch", type=int, default=0, help="which epoch to start from")
parser.add_argument("--num_epochs", type=int, default=100, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=6, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0001, help="adam: learning rate")
args = parser.parse_args()

## training params
epoch = args.epoch
num_epochs = args.num_epochs
batch_size = args.batch_size
lr_rate = args.lr

# load the data config file
with open(args.cfg_file) as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)
# get info from config file
dataset_name = cfg["dataset_name"]
dataset_path = cfg["dataset_path"]
channels = cfg["chans"]
img_width = cfg["im_width"]
img_height = cfg["im_height"]
ckpt_interval = cfg["ckpt_interval"]

## create dir for model and validation data
checkpoint_dir = os.path.join("checkpoints/BOths/", dataset_name)
os.makedirs(checkpoint_dir, exist_ok=True)

## loss
L_WMSE = WMSEloss()
L_1 = nn.SmoothL1Loss()
L_VGG = VGG19_PercepLoss()

# Initialize generator
generator = BOths()

# see if cuda is available
if torch.cuda.is_available():
    generator.cuda()
    L_WMSE.cuda()
    L_1.cuda()
    L_VGG.cuda()
    Tensor = torch.cuda.FloatTensor
else:
    Tensor = torch.FloatTensor

# Initialize weights or load pretrained models
if args.epoch == 0:
    generator.apply(Weights_Normal)
else:
    generator.load_state_dict(torch.load("checkpoints/BOths/%s/generator_%d.pth" % (dataset_name, args.epoch)))
    print("Loaded model from epoch %d" % (epoch))

# Optimizers
optimizer_G = torch.optim.RMSprop(generator.parameters(), lr=lr_rate)

## Data pipeline
transforms_ = [
    transforms.Resize((img_height, img_width), Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
]

dataloader = DataLoader(
    GetTrainingPairs(dataset_path, dataset_name, transforms_=transforms_),
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,
)

## Training pipeline
for epoch in range(epoch, num_epochs):
    for i, batch in enumerate(dataloader):

        # Learning rate changing
        if epoch > 30:
            lr_rate = 0.0000001

        # Model inputs
        imgs_distorted = Variable(batch["A"].type(Tensor))
        imgs_good_gt = Variable(batch["B"].type(Tensor))

        ## Train Generator
        for flag in range(2):
            optimizer_G.zero_grad()
            imgs_fake = generator(imgs_distorted)
            if flag == 0:
                WMSE_loss = L_WMSE(imgs_fake, imgs_good_gt)
                WMSE_loss.backward()
            if flag == 1:
                L1_loss = L_1(imgs_fake, imgs_good_gt)
                VGG_loss = L_VGG(imgs_fake, imgs_good_gt)
                TOT_loss = L1_loss + 0.1 * VGG_loss
                TOT_loss.backward()
        optimizer_G.step()

        ## Print log
        if not i % 50:
            sys.stdout.write(
                "\r[Epoch %d/%d: batch %d/%d] [WMSE_loss: %.3f, L1_loss: %.3f, VGG_loss: %.3f, TOT_loss: %.3f, lr: %.7f]"
                % (
                    epoch, num_epochs, i, len(dataloader),
                    WMSE_loss.item(), L1_loss.item(), VGG_loss.item(), TOT_loss.item(), lr_rate
                )
                )

    ## Save model checkpoints
    if (epoch % ckpt_interval == 0):
        torch.save(generator.state_dict(), "checkpoints/BOths/%s/generator_%d.pth" % (dataset_name, epoch),
                   _use_new_zipfile_serialization=False)


