import os
import torch
import numpy as np
from glob import glob

from PIL import Image
from torch.utils.data import Dataset
import itertools
from torch.utils.data.sampler import Sampler
from skimage import transform

class data_process(Dataset):
    """ BraTS2019 Dataset """

    def __init__(self, base_dir=None, split='train', num=None, transform=None):
        self._base_dir = base_dir
        self.transform = transform
        self.sample_list = []

        train_path = self._base_dir+'/train.txt'
        test_path = self._base_dir+'/val.txt'

        if split == 'train':
            with open(train_path, 'r') as f:
                self.image_list = f.readlines()
        elif split == 'test':
            with open(test_path, 'r') as f:
                self.image_list = f.readlines()

        self.image_list = [item.replace('\n', '').split(",")[0] for item in self.image_list]
        if num is not None:
            self.image_list = self.image_list[:num]
        # print(self.image_list)
        print("total {} samples".format(len(self.image_list)))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        img_3T_path = self._base_dir + "/data/{}.png".format(image_name)
        img_7T_path = img_3T_path.replace('3T','7T')
        image_3T = Image.open(img_3T_path,'r')
        image_7T = Image.open(img_7T_path,'r')
        paired_3T = (np.array(image_3T)).astype(float)
        paired_7T = (np.array(image_7T)).astype(float)
        print("paired_3T:", type(paired_3T),paired_3T.shape)
        print("paired_7T", type(paired_7T),paired_7T.shape)
        sample = {'paired_3T': paired_3T, 'paired_7T': paired_7T}
        if self.transform:
            sample = self.transform(sample)
        return sample

class Resize_2d(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        print('image.shape bf resize',image.shape)
        image = transform.resize(image, output_shape = self.output_size)
        label = transform.resize(label, output_shape = self.output_size)


        return {'image': image, 'label': label}




class RandomRotFlip(object):
    """
    Crop randomly flip the dataset in a sample
    Args:
    output_size (int): Desired output size
    """

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        k = np.random.randint(0, 4)
        image = np.rot90(image, k)
        label = np.rot90(label, k)
        axis = np.random.randint(0, 2)
        image = np.flip(image, axis=axis).copy()
        label = np.flip(label, axis=axis).copy()

        return {'image': image, 'label': label}


class RandomNoise(object):
    def __init__(self, mu=0, sigma=0.1):
        self.mu = mu
        self.sigma = sigma

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        noise = np.clip(self.sigma * np.random.randn(
            image.shape[0], image.shape[1], image.shape[2]), -2*self.sigma, 2*self.sigma)
        noise = noise + self.mu
        image = image + noise
        return {'image': image, 'label': label}


# class CreateOnehotLabel(object):
#     def __init__(self, num_classes):
#         self.num_classes = num_classes
#
#     def __call__(self, sample):
#         image, label = sample['image'], sample['label']
#         onehot_label = np.zeros(
#             (self.num_classes, label.shape[0], label.shape[1], label.shape[2]), dtype=np.float32)
#         for i in range(self.num_classes):
#             onehot_label[i, :, :, :] = (label == i).astype(np.float32)
#         return {'image': image, 'label': label, 'onehot_label': onehot_label}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image = sample['image']
        # image = image.reshape(
        #     1, image.shape[0], image.shape[1], image.shape[2]).astype(np.float32)
        image = image.reshape(
            1, image.shape[0], image.shape[1]).astype(np.float32)
        if 'onehot_label' in sample:
            #print('onehot_label')
            return {'image': torch.from_numpy(image), 'label': torch.from_numpy(sample['label']).long(),
                    'onehot_label': torch.from_numpy(sample['onehot_label']).long()}
        else:
            #print('no onehot_label')
            return {'image': torch.from_numpy(image), 'label': torch.from_numpy(sample['label'])}


class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """

    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in zip(grouper(primary_iter, self.primary_batch_size),
                   grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size


def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)