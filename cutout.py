# -*- coding: utf-8 -*-
# @Time : 2020/5/15 下午5:05
# @Author : lumingliao
# @File : tes.py
# @Software: PyCharm

from imutils import paths
from tqdm import tqdm
import torch
import cv2
from torchvision import utils as vutils
import numpy as np
from torchvision import transforms
from PIL import Image

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
        # print(mask.shape)
        img = img * mask

        return img

transform = transforms.Compose([
	# transforms.CenterCrop((224,224)), # 只能对PIL图片进行裁剪
	transforms.ToTensor(),
	])

# save img
def save_image_tensor2pillow(input_tensor: torch.Tensor, filename, mod = 'default', return_ = False):
    """
    将tensor保存为pillow
    :param input_tensor: 要保存的tensor
    :param filename: 保存的文件名
    """
    assert (len(input_tensor.shape) == 4 and input_tensor.shape[0] == 1)
    # 复制一份   创建一个新的tensor,将其从当前的计算图中分离出来.新的tensor与之前的共享data,但是不具有梯度
    input_tensor = input_tensor.clone().detach()
    # 到cpu
    input_tensor = input_tensor.to(torch.device('cpu'))
    # 反归一化
    # input_tensor = unnormalize(input_tensor)
    if mod == 'default':
        vutils.save_image(input_tensor, filename)
        return
    # 去掉批次维度
    input_tensor = input_tensor.squeeze()
    # 从[0,1]转化为[0,255]，再从CHW转为HWC，最后转为numpy
    input_tensor = input_tensor.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).type(torch.uint8).numpy()
    if mod == 'pil':
        # 转成pillow
        input_tensor = Image.fromarray(input_tensor)
    elif mod == 'cv':
        # RGB转BRG
        input_tensor = cv2.cvtColor(input_tensor, cv2.COLOR_RGB2BGR)
    else:
        print('not recognize save mode')
    # input_tensor.save(filename)
    if return_:
        return input_tensor

if __name__ == "__main__":
    path = './footage'
    cut = Cutout(n_holes=3, length=25)
    for img_path in tqdm(paths.list_images(path)):
        img = Image.open(img_path)
        img = transform(img)
        img = cut(img)       # tensor
        img = img.unsqueeze(0)           # 增加一个维度
        save_path = img_path[:-4] + '_.png'
        save_image_tensor2pillow(img, save_path, pil)     # pil, cv or default


