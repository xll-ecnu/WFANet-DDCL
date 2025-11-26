# Standard Libraries
import os
import numpy as np
import nibabel
import SimpleITK as sitk
import imageio
# PyTorch
import torch
import torch.nn as nn
from PIL import Image
from matplotlib import pyplot as plt
import cv2

def generate_mask(image_shape):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = image_shape[0]
    img_height = image_shape[2]
    img_width = image_shape[3]
    radius = image_shape[2] / 6
    y, x = np.ogrid[0:img_height, 0:img_width]
    mask = (x - img_height / 2) ** 2 + (y - img_width / 2) ** 2 <= radius ** 2
    mask = mask[np.newaxis, np.newaxis, :, :]
    mask = torch.tensor(mask)
    mask = mask.repeat(batch_size, 1, 1, 1).to(device)
    mask = mask.to(device)
    print('mask.shape in ptb_f', mask.shape)
    return mask
def ptb_f(image):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_shape = image.shape
    mask = generate_mask(image_shape)
    print('x.shape in ptb_f', image.shape)
    fft = torch.fft.fftn(image, dim = (2,3), norm='ortho')
    fft = torch.fft.fftshift(fft)
    fft = fft * mask
    fft = torch.fft.ifftshift(fft)
    xx = torch.fft.ifftn(fft, dim = (2,3), norm='ortho')
    image = torch.abs(xx)
    return image.to(device)


if __name__ == '__main__':
    image_path = '/img_1.png'
    img = Image.open(image_path).convert('L')
    # img = cv2.imread(image_path)
    # image = Image.convert('L')
    array_img = np.array(img)
    print('array_img.shape',array_img.shape)
    array_img = array_img[np.newaxis, np.newaxis,:, :]
    tensor = torch.tensor(array_img)
    tensor = tensor.repeat(2, 1, 1, 1)
    print('tensor.shape',tensor.shape)
    # tensor = torch.unsqueeze(tensor, 0)
    # tensor = torch.unsqueeze(tensor, 0)
    # print('tensor.shape',tensor.shape)
    img1,mask,res = ptb_f(tensor)


    print(img1.shape)
    print(res.shape)

