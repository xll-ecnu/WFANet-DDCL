import math
from glob import glob

import h5py
import nibabel as nib
import numpy as np

from PIL import Image
from medpy import metric
from tqdm import tqdm

from skimage.metrics.simple_metrics import peak_signal_noise_ratio as PSNR
from utils.utils import *
from skimage import transform
from torch.nn import L1Loss
criterion_L1 = nn.L1Loss()
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr


def cal_metric(gt, pred):
    gt = gt.detach().cpu().numpy().squeeze()
    pred = pred.squeeze()
    print('gt',type(gt),gt.shape)
    print('pred',type(pred),pred.shape)
    PSNR_value = compare_psnr(gt, pred, data_range=255)
    SSIM_value = compare_ssim(gt, pred, data_range=255)
    #print(PSNR)
    return np.array([PSNR_value, SSIM_value])


def test_all_case(net, base_dir, snapshot_path, test_list="full_test.list", output_size = [224, 224]):
    with open(base_dir + '/{}'.format(test_list), 'r') as f:
        image_list = f.readlines()
    image_list = [base_dir + "/images/{}.png".format(
        item.replace('\n', '').split(",")[0]) for item in image_list]
    total_metric = np.zeros((1, 2))
    print("Validation begin")
    for image_path in tqdm(image_list):
        #nii_file_path = (image_path.replace('mm_trainMR3Tto7T_', '')).replace('.h5','_3T.nii.gz')
        img_3T_path = image_path
        img_7T_path = img_3T_path.replace('3T','7T')
        image_3T = Image.open(img_3T_path, 'r')
        image_7T = Image.open(img_7T_path, 'r')
        image = (np.array(image_3T)).astype(float)
        label = (np.array(image_7T)).astype(float)


        image = transform.resize(image, output_shape=output_size)
        label = transform.resize(label, output_shape=output_size)

        image = torch.unsqueeze(torch.unsqueeze(torch.from_numpy(image), 0), 0)
        label = torch.unsqueeze(torch.unsqueeze(torch.from_numpy(label), 0), 0)
        image = image.float()
        image = Variable(image.cuda())
        #label = Variable(label.cuda())


        net.cuda()
        net.eval()
        prediction = net(image)
        prediction = prediction.detach().cpu().numpy().squeeze().astype(np.uint8)
        pil_image = Image.fromarray(prediction)

        prediction_path = snapshot_path + '/prediction'
        if not os.path.exists(prediction_path):
            os.makedirs(prediction_path)

        pred_file_id = (img_3T_path.split('/')[-1]).replace('3T','val')
        pil_image.save(prediction_path + '/{}'.format(pred_file_id) + '.png')

        #print(image_path.split('/')[-1])

        print('pred: ', prediction.dtype, ' shape: ', prediction.shape)
        print('gt: ', label.dtype, ' shape: ', label.shape)

        #for i in range(1, 2):# [SSIM,PSNR]
        total_metric[0, :] += cal_metric(label, prediction)
    print("Validation end")
    return total_metric / len(image_list)