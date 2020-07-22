import albumentations
import cv2
from PIL import Image, ImageDraw
import numpy as np

"""
Blur(blur_limit=7, always_apply=False, p=0.5) 使用随机大小的内核模糊输入图像。
VerticalFlip(always_apply=False, p=0.5) 围绕X轴垂直翻转输入。
HorizontalFlip(always_apply=False, p=0.5) 围绕y轴水平翻转输入。
Flip(always_apply=False, p=0.5) 水平，垂直或水平和垂直翻转输入。
Transpose(always_apply=False, p=0.5) 通过交换行和列来转置输入。
RandomCrop(height, width, always_apply=False, p=1.0) 裁剪输入的随机部分。
RandomGamma(gamma_limit=(80, 120), eps=None, always_apply=False, p=0.5)
RandomRotate90(always_apply=False, p=0.5) 将输入随机旋转90度，零次或多次。
ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, interpolation=1, border_mode=4, value=None, mask_value=None, always_apply=False, p=0.5) 随机应用仿射变换：平移，缩放和旋转输入。
CenterCrop(height, width, always_apply=False, p=1.0) 裁剪输入的中心部分。
GridDistortion(num_steps=5, distort_limit=0.3, interpolation=1, border_mode=4, value=None, mask_value=None, always_apply=False, p=0.5) 网格失真
ElasticTransform(alpha=1, sigma=50, alpha_affine=50, interpolation=1, border_mode=4, value=None, mask_value=None, always_apply=False, approximate=False, p=0.5) 弹性变换
RandomGridShuffle(grid=(3, 3), always_apply=False, p=0.5) 图像上随机排列的网格单元。
HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, always_apply=False, p=0.5) 随机更改输入图像的色相，饱和度和值。
PadIfNeeded(min_height=1024, min_width=1024, border_mode=4, value=None, mask_value=None, always_apply=False, p=1.0)[source] 垫图像的一面/如果一面小于所需数目，则为最大值。
RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, always_apply=False, p=0.5) 为输入RGB图像的每个通道随机移动值。
GaussianBlur(blur_limit=7, always_apply=False, p=0.5) 使用具有随机核大小的高斯滤波器对输入图像进行模糊处理。
CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), always_apply=False, p=0.5) 将对比度受限的自适应直方图均衡应用于输入图像。
ChannelShuffle(always_apply=False, p=0.5)[source])随机重新排列输入RGB图像的通道。
InvertImg(always_apply=False, p=0.5) 通过从255减去像素值来反转输入图像。
Cutout(num_holes=8, max_h_size=8, max_w_size=8, fill_value=0, always_apply=False, p=0.5) 随机擦处
RandomFog(fog_coef_lower=0.3, fog_coef_upper=1, alpha_coef=0.08, always_apply=False, p=0.5) 模拟图像雾
GridDropout(ratio: float = 0.5, unit_size_min: int = None, unit_size_max: int = None, holes_number_x: int = None, holes_number_y: int = None, shift_x: int = 0, shift_y: int = 0, random_offset: bool = False, fill_value: int = 0, mask_fill_value: int = None, always_apply: bool = False, p: float = 0.5) 以网格方式删除图像的矩形区域和相应的蒙版。
"""

from albumentations import (Blur,Flip,ShiftScaleRotate,GridDistortion,ElasticTransform,HorizontalFlip,CenterCrop,
                            HueSaturationValue,Transpose,RandomBrightnessContrast,CLAHE,RandomCrop,Cutout,CoarseDropout,
                            CoarseDropout,Normalize,ToFloat,OneOf,Compose,Resize,RandomRain,RandomFog,Lambda
                            ,ChannelDropout,ISONoise,VerticalFlip,RandomGamma,RandomRotate90)

def strong_aug(p=0.5):
    return Compose([
        RandomRotate90(),
        Flip(),
        Transpose(),
        OneOf([
            IAAAdditiveGaussianNoise(),
            GaussNoise(),
        ], p=0.2),
        OneOf([
            MotionBlur(p=0.2),
            MedianBlur(blur_limit=3, p=0.1),
            Blur(blur_limit=3, p=0.1),
        ], p=0.2),
        ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
        OneOf([
            OpticalDistortion(p=0.3),
            GridDistortion(p=0.1),
            IAAPiecewiseAffine(p=0.3),
        ], p=0.2),
        OneOf([
            CLAHE(clip_limit=2),
            IAASharpen(),
            IAAEmboss(),
            RandomBrightnessContrast(),
        ], p=0.3),
        HueSaturationValue(p=0.3),
    ], p=p)