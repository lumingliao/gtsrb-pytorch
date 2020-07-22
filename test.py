# -*- coding: utf-8 -*-
# @Time : 2020/5/13 下午8:59
# @Author : lumingliao
# @File : tmp.py
# @Software: PyCharm

from __future__ import print_function
import argparse
from tqdm import tqdm

import torch
import os
import numpy as np
from torch.autograd import Variable
import torchvision.datasets as datasets
# from model import Net
# from model_ import Net
from model_global import model
# from model.model_cbam import model
# from model_global_ca import model
# from model.model_cbam_c import model
import multiprocessing
from imutils import paths
from time import time

parser = argparse.ArgumentParser(description='PyTorch GTSRB evaluation script')
parser.add_argument('--data', type=str, default='data/data0', metavar='D',
                    help="folder where data is located. train_data.zip and test_data.zip need to be found in the folder")
parser.add_argument('--model', type=str, default='model_b/90.pth', metavar='M',
                    help="the model file to be evaluated. Usually it is of the form model_X.pth")
parser.add_argument('--model_dir', type=str, default='model/model_global', metavar='M',
                    help="the model file to be evaluated. Usually it is of the form model_X.pth")
parser.add_argument('--outfile', type=str, default='pred.csv', metavar='D',
                    help="name of the output csv file")
parser.add_argument('--batch_size', type=int, default=16, metavar='N',
                    help='input batch size for training (default: 64)')

args = parser.parse_args()

if torch.cuda.is_available():
    use_gpu = True
    print("Using GPU")
else:
    use_gpu = False
    print("Using CPU")

for path_ in sorted(paths.list_images(args.model_dir), key = lambda x : x[14:-4]):
    # state_dict = torch.load(args.model)
    state_dict = torch.load(path_)
    # model = Net()
    model.load_state_dict(state_dict)
    model.eval()
    if use_gpu:
        model.cuda()

    from data import data_transforms,data_jitter_hue,data_jitter_brightness,data_jitter_saturation,data_jitter_contrast,data_rotate,data_hvflip,data_shear,data_translate,data_center,data_grayscale

    test_dataset = datasets.ImageFolder(args.data + '/test_images', transform=data_transforms)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=args.batch_size,
                                              shuffle=False,
                                              num_workers=multiprocessing.cpu_count(),
                                              pin_memory=use_gpu)


    correct = 0
    total = 0
    t1 = time()
    for data, target in test_loader:
        with torch.no_grad():
            data, target = Variable(data), Variable(target)
            if use_gpu:
                data = data.cuda()
                target = target.cuda()
            total += target.size(0)
            output = model(data)
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    t2 = time()
    print(f'spend time is {t2 - t1}')
    correct = correct.numpy()
    acc = correct / total
    print('number %d Test accuracy on %d images is %.4f'%(int(path_.split('/')[-1][6:-4]), total, acc))