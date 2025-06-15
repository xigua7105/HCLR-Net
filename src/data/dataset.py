import os
import numpy as np
import random

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets.folder import ImageFolder, IMG_EXTENSIONS
import torchvision.transforms.functional as tf


def pil_loader(path):
    return Image.open(path).convert('RGB')


class UIETestDataset(Dataset):
    def __init__(self, root, cfg, is_train: bool = False, transform=None, target_transform=None):
        super().__init__()
        self.root = root
        self.dataset_name = os.path.basename(root)
        self.output_dir = cfg.tester.save_dir
        self.save_output = cfg.tester.save_output
        self.save_comparison = cfg.tester.save_comparison

        self.loader = pil_loader
        self.samples = self.make_dataset()

    def make_dataset(self):
        instance = []
        for root, dirs, files in os.walk(self.root):
            for file in files:
                if file.lower().endswith(IMG_EXTENSIONS):
                    input_type = root.split('/')[-1]
                    assert input_type in ['input', 'target']
                    if input_type == 'input':
                        input_path = os.path.join(root, file)
                        target_path = os.path.join(os.path.dirname(root), 'target', file)
                        assert os.path.exists(target_path)
                        item = input_path, target_path
                        instance.append(item)
        return instance

    def pad(self, x):
        _, height, width = x.shape
        target_height = (height + 31) // 32 * 32
        target_width = (width + 31) // 32 * 32
        pad_left = max(0, (target_width - width) // 2)
        pad_right = max(0, target_width - width - pad_left)
        pad_top = max(0, (target_height - height) // 2)
        pad_bottom = max(0, target_height - height - pad_top)
        padding = [pad_left, pad_top, pad_right, pad_bottom]
        x = tf.pad(x, padding, padding_mode='reflect')
        return x, padding

    def input_transforms(self, x):
        x = self.loader(x)
        x = tf.to_tensor(x)
        x, padding = self.pad(x)
        return x, padding

    def target_transforms(self, x):
        x = self.loader(x)
        x = tf.to_tensor(x)
        return x

    def get_save_path(self, path, mode='output'):
        basename = os.path.basename(path).split('.')[0] + '.png'
        return os.path.join(self.output_dir, mode, self.dataset_name, basename)

    def __getitem__(self, index: int) -> dict:
        path, target = self.samples[index]

        sample, input_padding = self.input_transforms(path)
        target_img = self.target_transforms(target)

        output_path, comparison_path = 'none', 'none'
        if self.save_output:
            output_path = self.get_save_path(path, mode='output')
        if self.save_comparison:
            comparison_path = self.get_save_path(target, mode='comparison')

        return {'img': sample, 'target': target_img, 'input_path': path, 'output_path': output_path, 'comparison_path': comparison_path, 'input_padding': input_padding}

    def __len__(self) -> int:
        return len(self.samples)


class HCLRDataset(Dataset):
    def __init__(self, root, cfg, is_train: bool = False, transform=None, target_transform=None):
        super().__init__()

        self.root = root
        self.is_train = is_train
        self.loader = pil_loader
        self.crop_size = cfg.data.crop_size
        self.samples = self.make_dataset()
        self.lens = self.__len__()

    def make_dataset(self):
        instance = []
        for root, _, files in os.walk(self.root):
            for file in files:
                if file.lower().endswith(IMG_EXTENSIONS):
                    input_type = root.split('/')[-2] if self.is_train else root.split('/')[-1]
                    assert input_type in ['input', 'target']
                    if input_type == 'input':
                        input_path = os.path.join(root, file)
                        target_path = input_path.replace('input', 'target')
                        assert os.path.exists(target_path)
                        item = input_path, target_path
                        instance.append(item)
        return instance

    def pad(self, x):
        _, height, width = x.shape
        target_height, target_width = self.crop_size, self.crop_size
        pad_left = max(0, (target_width - width) // 2)
        pad_right = max(0, target_width - width - pad_left)
        pad_top = max(0, (target_height - height) // 2)
        pad_bottom = max(0, target_height - height - pad_top)
        padding = [pad_left, pad_top, pad_right, pad_bottom]
        x = tf.pad(x, padding, padding_mode='reflect')
        return x

    def train_transform(self, x, y, z):
        x = tf.to_tensor(x)
        y = tf.to_tensor(y)
        z = tf.to_tensor(z)

        _, h, w = x.shape

        if h < self.crop_size or w < self.crop_size:
            x = self.pad(x)
            y = self.pad(y)
            z = self.pad(z)

        _, h, w = x.shape

        top = random.randint(0, h - self.crop_size)
        left = random.randint(0, w -self.crop_size)

        x = tf.crop(x, top, left, self.crop_size, self.crop_size)
        y = tf.crop(y, top, left, self.crop_size, self.crop_size)
        z = tf.crop(z, top, left, self.crop_size, self.crop_size)

        if random.random() > 0.5:
            x = tf.hflip(x)
            y = tf.hflip(y)
            z = tf.hflip(z)

        rand_rot = random.randint(0, 3)
        angle = 90 * rand_rot
        x = tf.rotate(x, angle)
        y = tf.rotate(y, angle)
        z = tf.rotate(z, angle)

        return x, y, z

    def test_transforms(self, x):
        x = self.loader(x)
        x = tf.to_tensor(x)
        x = tf.resize(x, size=[self.crop_size, self.crop_size])
        return x

    def __getitem__(self, index: int) -> dict:
        if self.is_train:
            path_1, target = self.samples[index]

            index_2 = random.randint(0, self.lens - 1)
            path_2, _ = self.samples[index_2]

            input_1 = self.loader(path_1)
            input_2 = self.loader(path_1)
            target = self.loader(target)

            sample, input_2, target = self.train_transform(input_1, input_2, target)
        else:
            sample, target = self.samples[index]

            sample = self.test_transforms(sample)
            target = self.test_transforms(target)
            input_2 = 0

        return {'img': sample, 'target': target, 'unpair_img': input_2}

    def __len__(self):
        return len(self.samples)
