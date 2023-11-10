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

log_dir = os.path.join(config['logging_params']['val_save_dir'], config['model_params']['name'])
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

device = torch.device("cuda:1")

model = cluster_vae(**config['model_params'])
optimizer = Adam(model.parameters(), lr=config['exp_params']['LR'])
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=config['exp_params']['scheduler_gamma'])

# Load checkpoint if exists
checkpoint_path = os.path.join(log_dir, 'checkpoint.pth')
start_epoch = 0
if os.path.exists(checkpoint_path):
    print("---------------yes---------------------")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    new_lr = 1e-6  # Set your desired new learning rate here
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr

# After loading the checkpoint
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

# Move optimizer's state to the correct device
for state in optimizer.state.values():
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            state[k] = v.to(device)

model.to(device)

current_data = MyDataset(**config["eval_params"], pin_memory=len(config['trainer_params']['gpus']) != 0)
data = current_data.val_dataset
dataloader = current_data.val_loader

# Training loop
num_epochs = config['trainer_params']['max_epochs']
model.train()
for epoch in range(start_epoch, num_epochs):
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