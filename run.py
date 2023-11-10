import argparse
import yaml
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from cluster_vae import cluster_vae
from custom_celeba import MyDataset
import os
import torch
import numpy as np
import random
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam

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

log_dir = os.path.join(config['logging_params']['save_dir'], config['model_params']['name'])
tb_logger = SummaryWriter(log_dir=log_dir)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


seed = config['exp_params']['manual_seed']
seed_everything(seed)

device_ids = config['trainer_params']['gpus']

model = cluster_vae(**config['model_params'])
model = model.to('cuda:1')
# # 如果有多个GPU，使用DataParallel
# if len(device_ids) > 1:
#     model = nn.DataParallel(model, device_ids=device_ids)

optimizer = Adam(model.parameters(), lr=config['exp_params']['LR'])
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=config['exp_params']['scheduler_gamma'])

current_data = MyDataset(**config["data_params"], pin_memory=len(config['trainer_params']['gpus']) != 0)
data = current_data.train_dataset
dataloader = current_data.train_loader

# Training loop
num_epochs = config['trainer_params']['max_epochs']
loss_history = {'train': [], 'eval': []}
model.train()
for epoch in range(num_epochs):
    train_loss = 0
    train_nsample = 0
    t = tqdm(dataloader, desc=f'[train]epoch:{epoch}')
    printloss = 0
    outputs = []
    P = config['data_params']['num_samples']
    recons_loss_show = kl_loss_show = loss_show = 0

    for transformed_image, combined_images, _ in t:
        transformed_image = transformed_image.to('cuda:1')
        combined_images = combined_images.to('cuda:1')

        mu, logvar = model.encode(transformed_image)

        # Vectorized reparameterization and decoding
        samples = model.reparameterize(mu.unsqueeze(1).expand(-1, P, -1), logvar.unsqueeze(1).expand(-1, P, -1))
        samples_to_train = model.decode(samples)
        total, C, H, W = samples_to_train.size()
        batch_size = total // P
        samples_to_train = samples_to_train.view(batch_size, P, C, H, W)
        loss, recons_loss, kl_loss = model.loss_function(samples_to_train, combined_images, mu, logvar,
                                                         M_N=config['exp_params']['kld_weight'])
        loss_show, recons_loss_show, kl_loss_show, = loss, recons_loss, kl_loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        tb_logger.add_scalar('Loss/Train', loss.item(), epoch)

    current_lr = optimizer.param_groups[0]['lr']
    tb_logger.add_scalar('Learning Rate', current_lr, epoch)
    scheduler.step()
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        # ... any other data you wish to save
    }, os.path.join(log_dir, 'checkpoint.pth'))

    print("loss------------", loss_show.item(), "recons----------", recons_loss_show.item(), "kl------------",
          kl_loss_show.item())

# Save model
torch.save(model.state_dict(), os.path.join(log_dir, 'model.pth'))
