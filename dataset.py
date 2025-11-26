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
    """ Dataset """

    def __init__(self, base_dir=None, split='train', transform=None):
        self._base_dir = base_dir
        self.transform = transform
        self.sample_list = []

        train_path = self._base_dir+'/train.txt'
        test_path = self._base_dir+'/val.txt'
        unp_train_path = self._base_dir+'/unpaired_train.txt'
        if split == 'train':
            with open(train_path, 'r') as f1:
                self.paired_image_list = f1.readlines()
            with open(unp_train_path, 'r') as f2:
                self.unp_image_list = f2.readlines()

            self.paired_image_list = [item.replace('\n', '').split(",")[0] for item in self.paired_image_list]
            self.unpaired_image_list = [item.replace('\n', '').split(",")[0] for item in self.unp_image_list]

            print("total {} paired samples".format(len(self.paired_image_list)))
            print("total {} unpaired samples".format(len(self.unpaired_image_list)))
            self.image_list = self.paired_image_list + self.unpaired_image_list
        elif split == 'test':
            with open(test_path, 'r') as f:
                self.image_list = f.readlines()

    def __len__(self):
        return len(self.image_list)
    def get_data_lenth(self):
        return len(self.paired_image_list), len(self.unpaired_image_list)
    def get_image_list(self):
        return self.image_list
    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        img_3T_path = self._base_dir + "/images/{}.png".format(image_name)
        img_7T_path = img_3T_path.replace('3T','7T')
        image_3T = Image.open(img_3T_path,'r')
        paired_3T = (np.array(image_3T)).astype(float)
        paired_3T = (paired_3T-np.min(paired_3T)) / (np.max(paired_3T)-np.min(paired_3T))
        print("paired_3T:", type(paired_3T), paired_3T.shape)
        if os.path.exists(img_7T_path):
            image_7T = Image.open(img_7T_path,'r')
            paired_7T = (np.array(image_7T)).astype(float)
            paired_7T = (paired_7T - np.min(paired_7T)) / (np.max(paired_7T) - np.min(paired_7T))
            print("paired_7T", type(paired_7T), paired_7T.shape)
        else:
            paired_7T = np.zeros(paired_3T.shape, dtype=float)
            paired_7T = (paired_7T - np.min(paired_7T)) / (np.max(paired_7T) - np.min(paired_7T))
        sample = {'paired_3T': paired_3T, 'paired_7T': paired_7T}
        if self.transform:
            sample = self.transform(sample)
        return sample
class Resize_2d(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image_3T, image_7T = sample['paired_3T'], sample['paired_7T']
        #print('image.shape bf resize',image_3T.shape)
        image_3T = transform.resize(image_3T, output_shape = self.output_size)
        #image_7T_copy = image_7T.astype(int)
        #if not np.all(image_7T_copy==0):
        image_7T = transform.resize(image_7T, output_shape = self.output_size)


        return {'paired_3T': image_3T, 'paired_7T': image_7T}




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
        image_3T = sample['paired_3T']
        image_7T = sample['paired_7T']
        # image = image.reshape(
        #     1, image.shape[0], image.shape[1], image.shape[2]).astype(np.float32)
        image_3T = image_3T.reshape(
            1, image_3T.shape[0], image_3T.shape[1]).astype(np.float32)
        if image_7T is not None:
            image_7T = torch.from_numpy(image_7T)

        return {'paired_3T': torch.from_numpy(image_3T), 'paired_7T': image_7T}


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
        print('primary_iter', primary_iter)
        print('max in primary_iter', max(primary_iter))
        print('secondary_iter', secondary_iter)
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