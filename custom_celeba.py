import argparse
import os
from typing import Any

import PIL
import torch
import matplotlib.pyplot as plt
import yaml
from torchvision import transforms
from torchvision.datasets import CelebA
from torch.utils.data import DataLoader
from functools import partial
from PIL import Image
import numpy as np
from torchvision.utils import make_grid
import matplotlib.pyplot as plt


class MyCelebA(CelebA):
    def _check_integrity(self) -> bool:
        return True

    def __getitem__(self, index: int):
        X = PIL.Image.open(os.path.join(self.root, self.base_folder, "img_align_celeba", self.filename[index]))
        self.abso_dir = os.path.join(self.root, self.base_folder, "img_align_celeba")
        target: Any = []
        for t in self.target_type:
            if t == "attr":
                target.append(self.attr[index, :])
            elif t == "identity":
                target.append(self.identity[index, 0])
            elif t == "bbox":
                target.append(self.bbox[index, :])
            elif t == "landmarks":
                target.append(self.landmarks_align[index, :])
            else:
                # TODO: refactor with utils.verify_str_arg
                raise ValueError("Target type \"{}\" is not recognized.".format(t))

        if self.transform is not None:
            X = self.transform(X)

        if target:
            target = tuple(target) if (len
                                       (target) > 1) else target[0]

            if self.target_transform is not None:
                target = self.target_transform(target)
        else:
            target = None
        return self.abso_dir, int(self.filename[index].split('.')[0]) - 1, X, target


class MyDataset:

    def __init__(self, data_dir, related_data, num_samples, patch_size, train_batch_size, val_batch_size, num_workers,
                 pin_memory):
        super(MyDataset, self).__init__()
        self.num_samples = num_samples
        if related_data != "":
            self.related_data = torch.load(related_data)
        else:
            self.related_data = ""
        self.patch_size = patch_size
        self.train_transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                                   transforms.CenterCrop(148),
                                                   transforms.Resize(self.patch_size),
                                                   transforms.ToTensor(), ])
        self.val_transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                                 transforms.CenterCrop(148),
                                                 transforms.Resize(self.patch_size),
                                                 transforms.ToTensor(), ])
        # Datasets
        self.train_dataset = MyCelebA(root=data_dir, split='train', download=False)
        self.val_dataset = MyCelebA(root=data_dir, split='valid', download=False)
        self.test_dataset = MyCelebA(root=data_dir, split='test', download=False, transform=self.val_transform)

        # Datasets
        self.train_loader = DataLoader(self.train_dataset, batch_size=train_batch_size, num_workers=num_workers,
                                       shuffle=True, pin_memory=pin_memory,
                                       collate_fn=self.batch_custom_collate_fn)
        self.val_loader = DataLoader(self.val_dataset, batch_size=val_batch_size, num_workers=num_workers,
                                     shuffle=True, pin_memory=pin_memory,
                                     collate_fn=self.batch_custom_collate_fn_eval)
        self.test_loader = DataLoader(self.test_dataset, batch_size=val_batch_size, num_workers=num_workers,
                                      shuffle=True, pin_memory=pin_memory
                                      )

    def batch_custom_collate_fn(self, batch):
        combined_images = []
        transformed_images = []
        targets = []

        # 加载所有的embeddings
        all_embeddings = np.load('./embeddings.npy')

        # 计算批次中每张图像的embeddings
        batch_embeddings = [all_embeddings[filename] for _, (_, filename, _, _) in enumerate(batch)]
        batch_embeddings_tensor = torch.tensor(np.array(batch_embeddings)).float()

        # 使用广播机制计算所有图像与批次中其他图像的MSE距离
        distances_matrix = ((batch_embeddings_tensor.unsqueeze(1) - batch_embeddings_tensor.unsqueeze(0)) ** 2).sum(-1)

        for idx, (abso, filename, image, _) in enumerate(batch):
            # 获取当前图像与其他图像的距离，并排除与自己的距离
            distances = distances_matrix[idx]
            nearest_indices = distances.argsort()[1:6]

            # 合并原图与最接近的图像
            combined = [self.train_transform(image)] + [self.train_transform(batch[i][2]) for i in nearest_indices]
            combined_tensor = torch.stack(combined)

            combined_images.append(combined_tensor)
            transformed_images.append(self.train_transform(image))

        return torch.stack(transformed_images), torch.stack(combined_images), targets

    def batch_custom_collate_fn_eval(self, batch):
        combined_images = []
        transformed_images = []
        targets = []

        # 加载所有的embeddings
        all_embeddings = np.load('./embeddings_eval.npy')

        # 计算批次中每张图像的embeddings
        batch_embeddings = [all_embeddings[filename - 162770] for _, (_, filename, _, _) in enumerate(batch)]
        batch_embeddings_tensor = torch.tensor(np.array(batch_embeddings)).float()

        # 使用广播机制计算所有图像与批次中其他图像的MSE距离
        distances_matrix = ((batch_embeddings_tensor.unsqueeze(1) - batch_embeddings_tensor.unsqueeze(0)) ** 2).sum(-1)

        for idx, (abso, filename, image, _) in enumerate(batch):
            # 获取当前图像与其他图像的距离，并排除与自己的距离
            distances = distances_matrix[idx]
            nearest_indices = distances.argsort()[1:6]

            # 合并原图与最接近的图像
            combined = [self.train_transform(image)] + [self.train_transform(batch[i][2]) for i in nearest_indices]
            combined_tensor = torch.stack(combined)

            combined_images.append(combined_tensor)
            transformed_images.append(self.train_transform(image))

        return torch.stack(transformed_images), torch.stack(combined_images), targets

    def custom_collate_fn(self, batch, data):
        combined_images = []
        transformed_images = []
        targets = []
        for _, (abso, filename, image, target) in enumerate(batch):
            # print("filename--------------", filename, "related data--------------------", data[filename])
            if filename >= 182637:
                filename -= 182637
            if filename >= 162769:
                filename -= 162769
            if filename >= len(data): print("filename--------------", filename)
            related_imgs_indices = data[filename]
            related_imgs_indices = [x + 1 for x in related_imgs_indices]

            related_imgs = [self.train_transform(Image.open(os.path.join(abso, f"{i:06}.jpg"))) for i in
                            related_imgs_indices]
            # related_imgs = [transforms.ToTensor()(Image.open(os.path.join(abso, f"{i:06}.jpg"))) for i in
            #                 related_imgs_indices]
            # Combine the main image with the related images
            transformed_image = self.train_transform(image)
            combined = torch.stack([transformed_image] + related_imgs)
            combined_images.append(combined)
            targets.append(target)
            transformed_images.append(transformed_image)

        return torch.stack(transformed_images), torch.stack(combined_images), targets

    def test_custom_collate_fn(self, batch, data):
        ori_images = []
        combined_images = []
        targets = []
        filenames = []
        i = 0
        for _, (abso, filename, image, target) in enumerate(batch):
            # print("filename--------------", filename, "related data--------------------", data[filename])
            if filename >= 182637:
                filename -= 182637
            if filename >= 162769:
                filename -= 162769
            if filename >= len(data): print("filename--------------", filename)
            related_imgs_indices = data[filename]
            # print("filename-----------------------",filename,"rela_indices----------------------------",related_imgs_indices)
            related_imgs_indices = [x + 1 for x in related_imgs_indices]
            # print("filename------------------------", filename,"related--------------------------", related_imgs_indices)
            related_imgs = [self.train_transform(Image.open(os.path.join(abso, f"{i:06}.jpg"))) for i in
                            related_imgs_indices]
            # related_imgs = [transforms.ToTensor()(Image.open(os.path.join(abso, f"{i:06}.jpg"))) for i in
            #                 related_imgs_indices]
            # Combine the main image with the related images
            combined = torch.stack([self.train_transform(image)] + related_imgs)
            combined_images.append(combined)
            # #画出来看一下先
            # if i==0:
            #     # 假设 images_tensor 是您的图片张量，形状为 [N x C x H x W]
            #
            #     # 使用make_grid创建一个图像网格
            #     grid_img = make_grid(combined, nrow=int(combined.size(0) ** 0.5))  # 使用图片数量的平方根作为行数
            #
            #     # 使用matplotlib展示图像网格
            #     plt.figure(figsize=(10, 10))  # 设置图像大小
            #     plt.imshow(grid_img.permute(1, 2, 0))
            #     plt.axis('off')  # 关闭坐标轴
            #     plt.show()
            #     i+=1
            targets.append(target)
            ori_images.append(image)
            filenames.append(filename)

        return torch.stack(combined_images), ori_images, filenames, targets

    def test_custom_collate_fn2(self, batch):
        ori_images = []
        filenames = []
        imgs_tensor = []
        for _, (abso, filename, image, target) in enumerate(batch):
            imgs_tensor.append(self.val_transform(image).unsqueeze(1))
            ori_images.append(image)
            filenames.append(filename)
        return torch.stack(imgs_tensor), ori_images, filenames


import argparse
import yaml
from tqdm import tqdm
import torch.nn as nn

from cluster_vae import cluster_vae
from custom_celeba import MyDataset
import os
import torch
import numpy as np
import random
import torchvision.transforms as transforms

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='runner for cluster_vae models')
    parser.add_argument('--config', '-c',
                        dest="filename",
                        metavar='FILE',
                        help='path to the config file',
                        default='configs/cluster_vae.yaml')

    args = parser.parse_args()
    with open(args.filename, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    current_data = MyDataset(**config["data_params"], pin_memory=len(config['trainer_params']['gpus']) != 0)
    data = current_data.val_dataset
    data_loader = current_data.val_loader

    transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                    transforms.CenterCrop(148),
                                    transforms.Resize(64),
                                    transforms.ToTensor(), ])

    transformed_images, combined_images, _ = next(iter(data_loader))


    def display_images(images, num_images=5):
        for idx in range(num_images):
            fig, axes = plt.subplots(1, len(images[idx]), figsize=(15, 5))

            for ax, img in zip(axes, images[idx]):
                # Assuming the images are in the format (C, H, W)
                ax.imshow(img.permute(1, 2, 0).numpy())  # Convert to (H, W, C) for displaying
                ax.axis('off')

            plt.show()


    print(combined_images.shape)
    display_images(combined_images)
