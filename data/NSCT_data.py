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
folders = os.path.join('/media/eric/DATA/Githubcode/OAL_DA/data_npy/mr_train', '*')
data_files = sorted(glob.glob(folders), key=os.path.getmtime)
index = 8414

for file in data_files:
    start = time.time()
    data = np.load(file, allow_pickle=True)
    img = data[:, :, 0]
    H,W = img.shape
    [Insp, Insct] = myNSCTd(img, levels, pname, dname, type)

    nsct = []
    nsct_filter = np.zeros([15, H, W])
    for i in range(len(levels) + 1):
        if i == 0:
            nsct.append(Insct[0])
        else:
            for j in range(2 ** levels[i - 1]):
                nsct.append(Insct[i][j])
    nsct_filter[mask] = nsct[mask]
    nsct_filter[3:7] = nsct[3:7]

    for i in range(len(levels) + 1):
        if i == 0:
            Insct[0] = nsct_filter[0]
            m = 1
        else:
            for j in range(2 ** levels[i - 1]):
                Insct[i][j] = nsct_filter[m]
                m += 1

    img_filter = myNSCTr(Insct, levels, pname, dname, type).astype('float32')
    data[:,:,0] = img_filter
    np.save('/media/eric/DATA/Githubcode/FSUDA//data_npy/mr_filter2/mr_train%d'%index,data)
    print("%d s used"%(time.time() - start))
    index+=1
    print("%d files have been processed!"%index)
#     data.append(np.load(file, allow_pickle=True))
# for item in range(len(data)):
#     img = data[item][:,:,0]
#     label = data[item][:,:,1:]
