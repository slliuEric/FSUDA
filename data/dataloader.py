from torch.utils.data import Dataset, DataLoader
import glob
import matplotlib.pyplot as plt
from math import sqrt
import argparse
import torchvision.transforms as transforms
try:  # SciPy >= 0.19
    from scipy.special import comb
except ImportError:
    from scipy.misc import comb
import numpy as np
import os
from PIL import Image, ImageFile, ImageFilter
ImageFile.LOAD_TRUNCATED_IMAGES = True
import random
import torch
from imgaug import augmenters as iaa
from NSCT import  NSCT_filter
sometimes = lambda aug: iaa.Sometimes(0.5, aug)
np.random.seed(20)
def colorful_spectrum_mix(img1, img2, alpha, ratio=1.0):
    """Input image size: ndarray of [H, W, C]"""
    lam = np.random.uniform(0, alpha)

    assert img1.shape == img2.shape
    h, w = img1.shape
    h_crop = int(h * sqrt(ratio))
    w_crop = int(w * sqrt(ratio))
    h_start = h // 2 - h_crop // 2
    w_start = w // 2 - w_crop // 2

    img1_fft = np.fft.fft2(img1, axes=(0, 1))
    img2_fft = np.fft.fft2(img2, axes=(0, 1))
    img1_abs, img1_pha = np.abs(img1_fft), np.angle(img1_fft)
    img2_abs, img2_pha = np.abs(img2_fft), np.angle(img2_fft)

    img1_abs = np.fft.fftshift(img1_abs, axes=(0, 1))
    img2_abs = np.fft.fftshift(img2_abs, axes=(0, 1))

    img1_abs_ = np.copy(img1_abs)
    img2_abs_ = np.copy(img2_abs)
    img1_abs[h_start:h_start + h_crop, w_start:w_start + w_crop] = \
        lam * img2_abs_[h_start:h_start + h_crop, w_start:w_start + w_crop] + (1 - lam) * img1_abs_[
                                                                                          h_start:h_start + h_crop,
                                                                                          w_start:w_start + w_crop]
    img2_abs[h_start:h_start + h_crop, w_start:w_start + w_crop] = \
        lam * img1_abs_[h_start:h_start + h_crop, w_start:w_start + w_crop] + (1 - lam) * img2_abs_[
                                                                                          h_start:h_start + h_crop,
                                                                                          w_start:w_start + w_crop]

    img1_abs = np.fft.ifftshift(img1_abs, axes=(0, 1))
    img2_abs = np.fft.ifftshift(img2_abs, axes=(0, 1))

    img21 = img1_abs * (np.e ** (1j * img1_pha))
    img12 = img2_abs * (np.e ** (1j * img2_pha))
    img21 = np.real(np.fft.ifft2(img21, axes=(0, 1)))
    img12 = np.real(np.fft.ifft2(img12, axes=(0, 1)))
    img21 = np.uint8(np.clip(img21, 0, 255))
    img12 = np.uint8(np.clip(img12, 0, 255))

    return img21, img12

def brightness_aug(x, gamma_high, gamma_low, prob=0.5):
    if random.random() >= prob:
        return x
    aug_brightness = iaa.Sequential(
        [
            sometimes(iaa.GammaContrast(gamma=gamma_high)),
            sometimes(iaa.GammaContrast(gamma=gamma_low)),
        ],
        random_order=True
    )
    aug_image = aug_brightness(images=x)
    return aug_image

def select_region_auto(img, k = None, nums = None):
    size = img.shape
    region = []
    box = []

    for i in range(nums):
        # plt.imshow(img, cmap=plt.cm.gray)
        box_L = np.random.randint(0, size[0] - k[i])
        box_H = np.random.randint(0, size[1] - k[i])
        region.append([box_L,box_H])
        # print(box1_L, box1_H)
        box.append(img[box_L:box_L + k[i], box_H:box_H + k[i]])

    return box, region

class datareader(Dataset):
    def __init__(self, args, dataroot, dataset, partition='train', domain='target'):
        self.args = args
        self.partition = partition
        self.domain = domain
        self.dataset = dataset
        self.rotation = self.args.rotate
        self.reconstruction = self.args.reconstruction
        self._tensor = transforms.ToTensor()
        self.data = []
        folders = os.path.join(dataroot, dataset, '*.npy')
        data_files = sorted(glob.glob(folders), key=os.path.getmtime)
        for file in data_files:
            self.data.append(np.load(file, allow_pickle=True))

    def __getitem__(self, item):
        data = self.data[item][:,:,0]
        label = self.data[item][:,:,1:]
        label_orig = label.copy()
        for i in range(label.shape[2]):
            a = label[:,:,i]
            a[a == 1] = i
        label = label[:, :, 0] + label[:, :, 1] + label[:, :, 2] + label[:, :, 3] + label[:, :, 4]
        img = self._tensor(data)
        image = data.copy()
        if self.rotation:
            rotated_imgs = [
                    img,
                    transforms.functional.rotate(img, 90, expand=True)
                ]
            # transforms.functional.rotate(img, 180, expand=True),
            # transforms.functional.rotate(img, 270, expand=True)
            rotation_labels = torch.LongTensor([0, 1])

        if self.reconstruction:
            index = np.random.randint(0, 4, size=self.args.region_nums)
            k = np.array([4, 8, 16, 32, 64])[index]
            img_region, region = select_region_auto(image, k=k, nums=self.args.region_nums)
            for i in range(self.args.region_nums):
                box_L, box_H = region[i][0], region[i][1]
                # image = img_region[i]
                image[box_L:box_L + k[i], box_H:box_H + k[i]] = 0

            return data, label, image,label_orig

        # if self.rotation:
        #     return data, label,torch.stack(rotated_imgs,dim=0),rotation_labels
        else:
            return data, label, label_orig

    def __len__(self):
        return len(self.data)


class SSLdatareader(Dataset):
    def __init__(self, args, dataroot, dataset, partition='train', domain='target'):
        self.args = args
        self.partition = partition
        self.domain = domain
        self.dataset = dataset
        self.rotation = self.args.rotate
        self.reconstruction = self.args.reconstruction
        self._tensor = transforms.ToTensor()
        self.data = []
        folders = os.path.join(dataroot, dataset, '*.npy')
        data_files = sorted(glob.glob(folders), key=os.path.getmtime)
        for file in data_files:
            self.data.append(np.load(file, allow_pickle=True))
        self.img_ids = data_files
    def __getitem__(self, item):
        name = self.img_ids[item]
        data = self.data[item][:,:,0]
        label = self.data[item][:,:,1:]
        label_orig = label.copy()
        for i in range(label.shape[2]):
            a = label[:,:,i]
            a[a == 1] = i
        label = label[:, :, 0] + label[:, :, 1] + label[:, :, 2] + label[:, :, 3] + label[:, :, 4]

        return data, label, name ,label_orig
    def __len__(self):
        return len(self.data)
class PSUdatareader(Dataset):
    def __init__(self, args, dataroot, dataset, partition='train', domain='target'):
        self.args = args
        self.partition = partition
        self.domain = domain
        self.dataset = dataset
        self.rotation = self.args.rotate
        self.reconstruction = self.args.reconstruction
        self._tensor = transforms.ToTensor()
        self.data = []
        folders = os.path.join(dataroot, dataset, '*')
        data_files = sorted(glob.glob(folders), key=os.path.getmtime)
        for file in data_files:
            self.data.append(np.load(file, allow_pickle=True))
        self.img_ids = data_files

    def __getitem__(self, item):
        name = self.img_ids[item]
        data = self.data[item][0, :, :]
        label = self.data[item][1, :, :]

        # for i in range(label.shape[2]):
        #     a = label[:, :, i]
        #     a[a == 1] = i
        # label = label[:, :, 0] + label[:, :, 1] + label[:, :, 2] + label[:, :, 3] + label[:, :, 4]

        return data, label, name

    def __len__(self):
        return len(self.data)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--region_nums', type=int, default=15, help='number of episode to train')
    args = parser.parse_args()
    dataroot =  '/media/eric/DATA/Githubcode/Few_MDA/PointSegDA/data_npy'
    src_trainset = datareader(args,dataroot, dataset='ct_train', partition='train', domain='source')
    src_valset = datareader(args, dataroot, dataset='ct_val', partition='val', domain='source')
    tgt_trainset = datareader(args,dataroot, dataset='mr_train', partition='train', domain='target')
    tgt_valset = datareader(args,dataroot, dataset='mr_val', partition='val', domain='target')
    src_train_loader = DataLoader(src_trainset, num_workers=0, batch_size=8,
                                  shuffle=False, drop_last=True)
    src_val_loader = DataLoader(src_valset, num_workers=0, batch_size=8,
                                  shuffle=False, drop_last=True)
    tgt_train_loader = DataLoader(tgt_trainset, num_workers=0, batch_size=8,
                                  shuffle=False, drop_last=True)
    tgt_val_loader = DataLoader(tgt_valset, num_workers=0, batch_size=8,
                                  shuffle=False, drop_last=True)
    for k,data in enumerate(zip(src_val_loader, tgt_val_loader)):
        src_data, src_label = data[0][0], data[0][1]
        tgt_data, tgt_label = data[1][0], data[1][1]