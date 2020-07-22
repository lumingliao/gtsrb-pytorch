from __future__ import print_function
import zipfile      # zip文件操作
import os
import numpy as np
import torch
import cv2
from torch import nn
import torchvision.transforms as transforms


def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam, label_smoothing = False):
    if label_smoothing:
        return lam * CrossEntropyLoss_label_smooth(pred, y_a) + (1 - lam) * CrossEntropyLoss_label_smooth(pred, y_b)
    return lam * criterion.nll_loss(pred, y_a) + (1 - lam) * criterion.nll_loss(pred, y_b)

def CrossEntropyLoss_label_smooth(outputs, targets, num_classes=58, epsilon=0.1):
    N = targets.size(0)
    smoothed_labels = torch.full(size=(N, num_classes), fill_value=epsilon / (num_classes - 1))
    smoothed_labels = smoothed_labels.cuda()
    smoothed_labels.scatter_(dim=1, index=torch.unsqueeze(targets, dim=1), value=1-epsilon)
    log_prob = nn.functional.log_softmax(outputs, dim=1)
    loss = - torch.sum(log_prob * smoothed_labels) / N
    return loss

class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
        	# (x,y)表示方形补丁的中心位置
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img

class Histeq(object):
    def __call__(self, img):
        img = np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = np.concatenate([np.expand_dims(cv2.equalizeHist(img[:,:,i]), axis=2) for i in range(3)], axis=2)
        return img

# data augmentation for training and test time
# Resize all images to 32 * 32 and normalize them to mean = 0 and standard-deviation = 1 based on statistics collected from the training set
data_size = (43, 43)

data_transforms = transforms.Compose([
	transforms.Resize(data_size),
    transforms.ToTensor(),
    transforms.Normalize((0.3337, 0.3064, 0.3171), ( 0.2672, 0.2564, 0.2629)),
    # Cutout(n_holes=1, length=5)
])

# Resize, normalize and jitter image brightness      亮度
data_jitter_brightness = transforms.Compose([
	transforms.Resize(data_size),
    # transforms.ColorJitter(brightness=-5),
    transforms.ColorJitter(brightness=5),
    transforms.ToTensor(),
    transforms.Normalize((0.3337, 0.3064, 0.3171), ( 0.2672, 0.2564, 0.2629)),
    # Cutout(n_holes=1, length=5)
])

# Resize, normalize and jitter image saturation    饱和度
data_jitter_saturation = transforms.Compose([
	transforms.Resize(data_size),
    transforms.ColorJitter(saturation=5),
    # transforms.ColorJitter(saturation=-5),
    transforms.ToTensor(),
    transforms.Normalize((0.3337, 0.3064, 0.3171), ( 0.2672, 0.2564, 0.2629)),
    # Cutout(n_holes=1, length=5)
])

# Resize, normalize and jitter image contrast         对比度
data_jitter_contrast = transforms.Compose([
	transforms.Resize(data_size),
    transforms.ColorJitter(contrast=5),
    # transforms.ColorJitter(contrast=-5),
    transforms.ToTensor(),
    transforms.Normalize((0.3337, 0.3064, 0.3171), ( 0.2672, 0.2564, 0.2629)),
    # Cutout(n_holes=1, length=5)
])

# Resize, normalize and jitter image hues
data_jitter_hue = transforms.Compose([
	transforms.Resize(data_size),
    transforms.ColorJitter(hue=0.4),
    transforms.ToTensor(),
    transforms.Normalize((0.3337, 0.3064, 0.3171), ( 0.2672, 0.2564, 0.2629)),
    # Cutout(n_holes=1, length=5)
])

# Resize, normalize and rotate image
data_rotate = transforms.Compose([
	transforms.Resize(data_size),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize((0.3337, 0.3064, 0.3171), ( 0.2672, 0.2564, 0.2629)),
    # Cutout(n_holes=1, length=5)
])

# Resize, normalize and flip image horizontally and vertically
data_hvflip = transforms.Compose([
	transforms.Resize(data_size),
    transforms.RandomHorizontalFlip(1),
    transforms.RandomVerticalFlip(1),
    transforms.ToTensor(),
    transforms.Normalize((0.3337, 0.3064, 0.3171), ( 0.2672, 0.2564, 0.2629)),
    # Cutout(n_holes=1, length=5)
])

# Resize, normalize and flip image horizontally
data_hflip = transforms.Compose([
	transforms.Resize(data_size),
    transforms.RandomHorizontalFlip(1),
    transforms.ToTensor(),
    transforms.Normalize((0.3337, 0.3064, 0.3171), ( 0.2672, 0.2564, 0.2629)),
    # Cutout(n_holes=1, length=5)
])

# Resize, normalize and flip image vertically
data_vflip = transforms.Compose([
	transforms.Resize(data_size),
    transforms.RandomVerticalFlip(1),
    transforms.ToTensor(),
    transforms.Normalize((0.3337, 0.3064, 0.3171), ( 0.2672, 0.2564, 0.2629)),
    # Cutout(n_holes=1, length=5)
])

# Resize, normalize and shear image
data_shear = transforms.Compose([
	transforms.Resize(data_size),
    transforms.RandomAffine(degrees = 15,shear=2),
    transforms.ToTensor(),
    transforms.Normalize((0.3337, 0.3064, 0.3171), ( 0.2672, 0.2564, 0.2629)),
    # Cutout(n_holes=1, length=5)
])

# Resize, normalize and translate image
data_translate = transforms.Compose([
	transforms.Resize(data_size),
    transforms.RandomAffine(degrees = 15,translate=(0.1,0.1)),
    transforms.ToTensor(),
    transforms.Normalize((0.3337, 0.3064, 0.3171), ( 0.2672, 0.2564, 0.2629)),
    # Cutout(n_holes=1, length=5)
])

# Resize, normalize and crop image 
data_center = transforms.Compose([
	transforms.Resize((50, 50)),
    transforms.CenterCrop(data_size[0]),
    transforms.ToTensor(),
    transforms.Normalize((0.3337, 0.3064, 0.3171), ( 0.2672, 0.2564, 0.2629)),
    # Cutout(n_holes=1, length=5)
])

# Resize, normalize and convert image to grayscale
data_grayscale = transforms.Compose([
	transforms.Resize(data_size),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize((0.3337, 0.3064, 0.3171), ( 0.2672, 0.2564, 0.2629)),
    # Cutout(n_holes=1, length=5)
])

data_grayscale_1 = transforms.Compose([
	transforms.Resize(data_size),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.3337, 0.3064, 0.3171), ( 0.2672, 0.2564, 0.2629)),
    # Cutout(n_holes=1, length=5)
])

def initialize_data(folder):
    train_zip = folder + '/train_images.zip'
    test_zip = folder + '/test_images.zip'
    if not os.path.exists(train_zip) or not os.path.exists(test_zip):
        raise(RuntimeError("Could not find " + train_zip + " and " + test_zip
              + ', please download them from https://www.kaggle.com/c/nyu-cv-fall-2017/data '))
              
    # extract train_data.zip to train_data
    train_folder = folder + '/train_images'
    if not os.path.isdir(train_folder):
        print(train_folder + ' not found, extracting ' + train_zip)
        zip_ref = zipfile.ZipFile(train_zip, 'r')
        zip_ref.extractall(folder)       # 解压zip文档中的所有文件到当前目录
        zip_ref.close()
        
    # extract test_data.zip to test_data
    test_folder = folder + '/test_images'
    if not os.path.isdir(test_folder):
        print(test_folder + ' not found, extracting ' + test_zip)
        zip_ref = zipfile.ZipFile(test_zip, 'r')
        zip_ref.extractall(folder)
        zip_ref.close()
        
    # make validation_data by using images 00000*, 00001* and 00002* in each class
    val_folder = folder + '/val_images'
    if not os.path.isdir(val_folder):
        print(val_folder + ' not found, making a validation set')
        os.mkdir(val_folder)
        for dirs in os.listdir(train_folder):        # 从训练集中抽除测试机
            if dirs.startswith('000'):
                os.mkdir(val_folder + '/' + dirs)     # from 00000 -> 00042
                for f in os.listdir(train_folder + '/' + dirs):         # train_images/000000/*.png
                    if f.startswith('00000') or f.startswith('00001') or f.startswith('00002'):
                        # move file to validation folder
                        os.rename(train_folder + '/' + dirs + '/' + f, val_folder + '/' + dirs + '/' + f)
