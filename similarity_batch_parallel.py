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

# 设置可见的CUDA设备
device_ids = [0, 1]  # 选择要使用的GPU设备索引
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, device_ids))

# 解析命令行参数
parser = argparse.ArgumentParser(description='Runner for cluster_vae models')
parser.add_argument('--config', '-c',
                    dest="filename",
                    metavar='FILE',
                    help='path to the config file',
                    default='configs/cluster_vae.yaml')

args = parser.parse_args()

# 读取配置文件
with open(args.filename, 'r', encoding='utf-8') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)

# 设置日志目录
log_dir = os.path.join(config['logging_params']['save_dir'], config['model_params']['name'])
tb_logger = SummaryWriter(log_dir=log_dir)

# 设置随机种子
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

# 获取可用的GPU设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 初始化模型
def initialize_model():
    model = cluster_vae(**config['model_params']).to(device)
    if len(device_ids) > 1:
        print("yes-----------")
        model = nn.DataParallel(model, device_ids=device_ids)
    return model

# 创建数据集和数据加载器
current_data = MyDataset(**config["data_params"], pin_memory=len(device_ids) != 0)
data = current_data.train_dataset

# 使用 DataParallel 包装数据加载器
dataloader = nn.DataParallel(current_data.train_loader, device_ids=device_ids)

# 初始化模型
model = initialize_model()

# 初始化优化器和学习率调度器
optimizer = Adam(model.parameters(), lr=config['exp_params']['LR'])
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=config['exp_params']['scheduler_gamma'])

# 训练循环
num_epochs = config['trainer_params']['max_epochs']
loss_history = {'train': [], 'eval': []}
model.train()
# 训练循环
for epoch in range(num_epochs):
    train_loss = 0
    train_nsample = 0
    t = tqdm(dataloader.module, desc=f'[train]epoch:{epoch}')  # 使用dataloader.module获取原始数据加载器
    printloss = 0
    outputs = []
    P = config['data_params']['num_samples']
    recons_loss_show = kl_loss_show = loss_show = 0

    for transformed_image, combined_images, _ in t:

        transformed_image = transformed_image.to(device)
        combined_images = combined_images.to(device)

        mu, logvar = model.module.encode(transformed_image)

        # Vectorized reparameterization and decoding
        samples = model.module.reparameterize(mu.unsqueeze(1).expand(-1, P, -1), logvar.unsqueeze(1).expand(-1, P, -1))
        samples_to_train = model.module.decode(samples)
        total, C, H, W = samples_to_train.size()
        batch_size = total // P
        samples_to_train = samples_to_train.view(batch_size, P, C, H, W)
        loss, recons_loss, kl_loss = model.module.loss_function(samples_to_train, combined_images, mu, logvar,
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
        'model_state_dict': model.module.state_dict() if len(device_ids) > 1 else model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        # ... any other data you wish to save
    }, os.path.join(log_dir, 'checkpoint.pth'))

    print("loss------------", loss_show.item(), "recons----------", recons_loss_show.item(), "kl------------",
          kl_loss_show.item())

# 保存模型
if len(device_ids) > 1:
    torch.save(model.module.state_dict(), os.path.join(log_dir, 'model_200.pth'))
else:
    torch.save(model.state_dict(), os.path.join(log_dir, 'model_200.pth'))
