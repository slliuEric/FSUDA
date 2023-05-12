import glob
import matplotlib.pyplot as plt
from math import sqrt
import argparse
import torchvision.transforms as transforms
import numpy as np
import os
from PIL import Image, ImageFile, ImageFilter
import random
import torch
from NSCT import  *
levels=[1,2,3]
pname = 'pyr'
dname = 'pkva'
type = 'NSCT'
mask = 0
data = []

folders = os.path.join('/media/eric/DATA/Githubcode/FSUDA/heart/mr_val', '*.npy')
trgt_folders = os.path.join('/media/eric/DATA/Githubcode/FSUDA/heart/ct_val', '*.npy')

data_files = sorted(glob.glob(folders), key=os.path.getmtime)
trgt_files = sorted(glob.glob(trgt_folders), key=os.path.getmtime)
index = 0
trgt = True
trgt_data = []
random.seed(1)
alpha = 1.0
begin = time.time()

for file in trgt_files:
    data = np.load(file,allow_pickle=True)
    trgt_data.append(data[:,:,0])
for file in data_files:
    start = time.time()
    flag = np.random.randint(0,len(trgt_files))
    data = np.load(file, allow_pickle=True)
    img = data[:, :, 0]
    if not (data[:,:,1].max()  == 1.0  and data[:,:,1].min()  == 1.0) : #make sure the image to be transferred is not the ineffective image containing no segmented parts.
        img_trgt = trgt_data[flag]
        H, W = img.shape
        [Insp, Insct] = myNSCTd(img, levels, pname, dname, type)
        [Insp_trgt, Insct_trgt] = myNSCTd(img_trgt, levels, pname, dname, type)
        nsct = []
        nsct_filter_hm = np.zeros([15, H, W])
        nsct_filter = np.zeros([15, H, W])
        nsct_trgt = []
        for i in range(len(levels) + 1): #flatten the nsct result
            if i == 0:
                nsct.append(Insct[0])
                nsct_trgt.append(Insct_trgt[0])
            else:
                for j in range(2 ** levels[i - 1]):
                    nsct.append(Insct[i][j])
                    nsct_trgt.append(Insct_trgt[i][j])
        nsct_filter[0] = nsct[0]
        # nsct_filter[1:7] = nsct[1:7]
        nsct_filter[7:15] = nsct[7:15]

        nsct_filter_hm[0] = nsct[0]
        nsct_filter_hm[1:7] = nsct[1:7]
        lam = np.random.beta(1.0, 2.0)
        nsct_filter[1:7] = (0.8 * np.array(nsct[1:7]) + 0.2 * np.array(nsct_trgt[1:7])).tolist() #mixup based on the random Beta distribution.

        src = (np.array(nsct[1:15]) + 1) * 128
        src[src > 255] = 255
        trgt = (np.array(nsct_trgt[1:15]) + 1) * 128
        trgt[trgt > 255] = 255
        from skimage.exposure import match_histograms
        src_in_trg = []
        for i in range(src.shape[0]):
            st = match_histograms(np.array(src[i]), np.array(trgt[i]))
            src_in_trg.append(st/255.0)
        nsct_filter_hm[1:15] = src_in_trg

        for i in range(len(levels) + 1):
            if i == 0:
                Insct[0] = nsct_filter[0]
                m = 1
            else:
                for j in range(2 ** levels[i - 1]):
                    Insct[i][j] = nsct_filter[m]
                    m += 1
        img_filter = myNSCTr(Insct, levels, pname, dname, type).astype('float32')
        for i in range(len(levels) + 1):
            if i == 0:
                Insct[0] = nsct_filter_hm[0]
                m = 1
            else:
                for j in range(2 ** levels[i - 1]):
                    Insct[i][j] = nsct_filter_hm[m]
                    m += 1
        img_filter_hm = myNSCTr(Insct, levels, pname, dname, type).astype('float32')
        data[:, :, 0] = img_filter
        # plt.imshow(img_filter, cmap='gray')
        # plt.axis('off')
        # plt.savefig('/media/eric/DATA/Githubcode/FSUDA//data_npy/mr_filter_trgt/mr_train%d.jpg' % index, bbox_inches='tight', pad_inches=0, dpi=256)
        # plt.close()
        # np.save('/media/eric/DATA/Githubcode/FSUDA//data_npy/mr_filter_trgt/mr_train%d' % index, data)
        print("%d s used" % (time.time() - start))
        print("%d files have been processed!" % index)
        index += 1
    else:
        print("No.%d file is not what we need to tansfer" % index)
        # plt.imshow(data[:,:,0],cmap='gray')
        # plt.axis('off')
        # plt.savefig('/media/eric/DATA/Githubcode/FSUDA//data_npy/mr_filter_trgt_no/mr_train%d.jpg' % index, bbox_inches='tight', pad_inches=0, dpi=256)
        # plt.close()
        # np.save('/media/eric/DATA/Githubcode/FSUDA//data_npy/mr_filter_trgt/mr_train%d' % index, data)
        print("%d s used" % (time.time() - start))
        index += 1
end = time.time()
print("%d mins used" % ((end - begin)/60.0))


plt.figure()
plt.imshow(img, cmap='gray')
plt.show()
plt.figure()
plt.imshow(img_trgt, cmap='gray')
plt.show()
plt.figure()
plt.imshow(img_filter, cmap='gray')
plt.show()

plt.imshow(img, cmap='gray')
plt.axis('off')
plt.savefig('./chaos_ct%d.jpg' % index, bbox_inches='tight', pad_inches=0, dpi=256)
plt.imshow(img_trgt, cmap='gray')
plt.axis('off')
plt.savefig('./chaos_mr%d.jpg' % index, bbox_inches='tight', pad_inches=0, dpi=256)
plt.imshow(img_filter, cmap='gray')
plt.axis('off')
plt.savefig('./chaos_ct_mr_nsct%d.jpg' % index, bbox_inches='tight', pad_inches=0, dpi=256)

src = (img + 1) * 128
src[src > 255] = 255
trgt = (img_trgt + 1) * 128
trgt[trgt > 255] = 255
from skimage.exposure import match_histograms
import numpy as np
st = match_histograms(np.array(src), np.array(trgt))
plt.imshow(st,cmap='gray')
plt.axis('off')
plt.savefig('./chaos_ct_mr_hm%d.jpg' % index, bbox_inches='tight', pad_inches=0, dpi=256)

