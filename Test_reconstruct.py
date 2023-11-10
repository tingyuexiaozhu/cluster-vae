import argparse
import yaml
import torchvision.transforms as transforms
from cluster_vae import cluster_vae
from custom_celeba import MyDataset
import os
import torch
import torchvision.utils as vutils
import torch.nn.functional as F

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

vae_model = cluster_vae(**config['model_params'])
vae_model.load_state_dict(torch.load("./logs/cluster_vae/model_200.pth"))
vae_model = vae_model.to('cuda:0')  # 将模型发送到GPU
vae_model.eval()

current_data = MyDataset(**config["test_params"], pin_memory=len(config['trainer_params']['gpus']) != 0)
data = current_data.test_dataset
data_loader = current_data.test_loader

ori_images = []
related_images = []

# 定义一个将PIL图像转换为张量的转换
to_tensor = transforms.ToTensor()

transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                transforms.CenterCrop(148),
                                transforms.Resize(64),
                                transforms.ToTensor(), ])
allimgs = []
num_reconstruct = config['reconstruct_params']['num_reconstruct']
# 重构
recon_imgs = []
total_num = config['reconstruct_params']['total_num_reconstruct']

for _, _, img, _ in data_loader:
    if torch.rand(1).item() < 1 / 16:
        img = img.to('cuda:0')
        recon_imgs.append(img[0])

        # # 从所有里挑最近的
        # mu, logvar = vae_model.encode(img[0].unsqueeze(0))
        # # 生成total_num个重构的图片
        # recon_candidates = []
        # for i in range(total_num):
        #     re_img = vae_model.decode(vae_model.reparameterize(mu, logvar)).squeeze(0)
        #     recon_candidates.append(re_img)
        #
        # # 计算每个重构图片与原图的MSELoss
        # diffs = [(F.mse_loss(recon, img[0]), recon) for recon in recon_candidates]
        #
        # # 根据MSELoss对重构图片进行排序并选择num_reconstruct个最接近的图片
        # closest_recons = sorted(diffs, key=lambda x: x[0])[:num_reconstruct]
        # for _, recon in closest_recons:
        #     recon_imgs.append(recon)
        for i in range(num_reconstruct):
            re_img=vae_model(img[0].unsqueeze(0))[0].squeeze(0)
            recon_imgs.append(re_img)

recon_imgs = torch.stack(recon_imgs)
vutils.save_image(recon_imgs.data,
                  os.path.join("./reconstruct_200", "re5_direct.png"),
                  normalize=True,
                  nrow=6)
