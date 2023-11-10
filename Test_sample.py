import argparse
import yaml
from tqdm import tqdm
import torch.nn as nn
from torchvision.transforms import ToTensor, ToPILImage
import torchvision.transforms as transforms
from cluster_vae import cluster_vae
import os
import torch
import torchvision.utils as vutils

import numpy as np
import random
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam
from torchvision.utils import save_image


torch.cuda.empty_cache()

parser = argparse.ArgumentParser(description='runner for cluster_vae models')
parser.add_argument('--config', '-c',
                    dest="filename",
                    metavar='FILE',
                    help='path to the config file',
                    default='configs/cluster_vae.yaml')

args = parser.parse_args()
with open(args.filename, 'r', encoding='utf-8') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)

device_ids = config['trainer_params']['gpus']

vae_model = cluster_vae(**config['model_params'])
vae_model.load_state_dict(torch.load("./logs/cluster_vae/model.pth"))
vae_model = vae_model.to('cuda:0')  # 将模型发送到GPU
vae_model.eval()

samples = vae_model.sample(144,0)
samples = samples.squeeze(1)
vutils.save_image(samples.cpu().data,
                  os.path.join("./samples","sample2.png"),
                  normalize=True,
                  nrow=12)
