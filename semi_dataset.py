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

    def __init__(self, base_dir=None, split='train',  transform=None):
        self._base_dir = base_dir
        self.transform = transform
        self.sample_list = []
        self.split = split

        paired_train_path = self._base_dir+'/train.txt'
        unpaired_train_path = self._base_dir+'/unpaired_train.txt'
        test_path = self._base_dir+'/val.txt'

        if split == 'train':
            with open(paired_train_path, 'r') as f1:
                self.paired_image_list = f1.readlines()
            with open(unpaired_train_path, 'r') as f2:
                self.unpaired_image_list = f2.readlines()

            self.paired_image_list = [item.replace('\n', '').split(",")[0] for item in self.paired_image_list]
            self.unpaired_image_list = [item.replace('\n', '').split(",")[0] for item in self.unpaired_image_list]

            print("total {} paired samples".format(len(self.paired_image_list)))
            print("total {} unpaired samples".format(len(self.unpaired_image_list)))
        elif split == 'test':
            with open(test_path, 'r') as f:
                self.image_list = f.readlines()
            self.image_list = [item.replace('\n', '').split(",")[0] for item in self.image_list]
            print("paired_total {} samples".format(len(self.image_list)))
        #if num is not None:
        #    self.image_list = self.image_list[:num]
        # print(self.image_list)



    def __len__(self):
        if self.split == 'train':
            return len(self.paired_image_list)
        elif self.split == 'test':
            return len(self.image_list)

    def __getitem__(self, idx):
        if self.split == 'train':
            image_name = self.paired_image_list[idx]
            unp_image_name = self.unpaired_image_list[idx]
            img_3T_path = self._base_dir + "/paired_images/{}.png".format(image_name)
            unp_img_3T_path = self._base_dir + "/unpaired_images/{}.png".format(unp_image_name)
            img_7T_path = img_3T_path.replace('3T','7T')

            image_3T = Image.open(img_3T_path,'r')
            image_7T = Image.open(img_7T_path, 'r')
            unp_image_3T = Image.open(unp_img_3T_path,'r')


            paired_3T = (np.array(image_3T)).astype(float)
            paired_7T = (np.array(image_7T)).astype(float)
            unpaired_3T = (np.array(unp_image_3T)).astype(float)


            print("paired_3T:", type(paired_3T),paired_3T.shape)
            print("paired_7T", type(paired_7T), paired_7T.shape)
            print("unpaired_3T:", type(unpaired_3T),unpaired_3T.shape)


            sample = {'paired_3T': paired_3T, 'paired_7T': paired_7T, 'unpaired_3T':unpaired_3T}
            if self.transform:
                sample = self.transform(sample)
        elif self.split == 'test':
            image_name = self.image_list[idx]
            img_3T_path = self._base_dir + "/paired_images/{}.png".format(image_name)
            img_7T_path = img_3T_path.replace('3T', '7T')
            image_3T = Image.open(img_3T_path, 'r')
            image_7T = Image.open(img_7T_path, 'r')
            paired_3T = (np.array(image_3T)).astype(float)
            paired_7T = (np.array(image_7T)).astype(float)
            print("paired_3T:", type(paired_3T), paired_3T.shape)
            print("paired_7T", type(paired_7T), paired_7T.shape)
            sample = {'paired_3T': paired_3T, 'paired_7T': paired_7T}
            if self.transform:
                sample = self.transform(sample)
        return sample

class Resize_2d(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        paired_3T_image, paired_7T_image = sample['paired_3T'], sample['paired_7T']
        unpaired_3T_image = sample['unpaired_3T']

        paired_3T_image = transform.resize(paired_3T_image, output_shape = self.output_size)
        paired_7T_image = transform.resize(paired_7T_image, output_shape = self.output_size)
        unpaired_3T_image = transform.resize(unpaired_3T_image, output_shape = self.output_size)


        return {'paired_3T': paired_3T_image, 'paired_7T': paired_7T_image, 'unpaired_3T':unpaired_3T_image}




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
        paired_3T_image = sample['paired_3T']
        unpaired_3T_image = sample['unpaired_3T']
        # image = image.reshape(
        #     1, image.shape[0], image.shape[1], image.shape[2]).astype(np.float32)
        paired_3T_image = paired_3T_image.reshape(
            1, paired_3T_image.shape[0], paired_3T_image.shape[1]).astype(np.float32)
        unpaired_3T_image = unpaired_3T_image.reshape(
            1, unpaired_3T_image.shape[0], unpaired_3T_image.shape[1]).astype(np.float32)

            #print('no onehot_label')
        return {'paired_3T': torch.from_numpy(paired_3T_image), 'paired_7T': torch.from_numpy(sample['paired_7T']),
                'unpaired_3T': torch.from_numpy(unpaired_3T_image)}


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