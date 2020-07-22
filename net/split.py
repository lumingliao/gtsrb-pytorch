from glob import glob
import os
import random
from imutils import paths
from shutil import move
from tqdm import tqdm

# root_dir='/home/llm/data/BreaKHis_v1/histology_slides/data/'
# benign = '/home/llm/gtsrb-pytorch/data/breakhis/test_images/00001/'
# dst = '/home/llm/gtsrb-pytorch/data/breakhis/val_images/00001'

# res = []
# for imgs in paths.list_images(benign):
#     res.append(imgs)
# index = list(range(len(res)))
# random.shuffle(index)
# tmp = int(len(index) * 0.3)
# res = [res[i] for i in index]
# move_res = res[:tmp]
# # print(len(move_res))
# for path in tqdm(move_res):
#     move(path, dst)


# images = glob(benign + '*.png')
# print(len(images))
# for img in images:
#     os.(img, img[:-12])

# -*- coding: utf-8 -*-

###################################
#########  作者：行歌   ############
#########  时间：2018.5.11  ########
######  email:1013007057@qq.com  ##
###################################

import os
from PIL import Image


def get_files(file_dir):
    """
        Loads a data set and returns two lists:
        images: a list of Numpy arrays, each representing an image.
        labels: a list of numbers that represent the images labels.
    """
    directories = [file for file in os.listdir(file_dir) if os.path.isdir(os.path.join(file_dir, file))]
    # os.listdir()方法用于返回指定的文件夹包含的文件或文件夹的名字的列表
    # directories是一个列表，部分如下所示：
    # ['00000', '00001', '00002', '00003', '00004', '00005', '00006', '00007',
    # '00008', '00009', '00010', '00011', '00012', '00013', '00014', '00015',
    # ...

    images = []
    labels = []

    for files in directories:
        data_dir = os.path.join(file_dir, files)
        # data_dir列表部分展示如下所示：
        # E:\DataSet\BelgiumTS\BelgiumTSC_Training\Training\00000
        # E:\DataSet\BelgiumTS\BelgiumTSC_Training\Training\00001
        # E:\DataSet\BelgiumTS\BelgiumTSC_Training\Training\00002
        # E:\DataSet\BelgiumTS\BelgiumTSC_Training\Training\00003
        # E:\DataSet\BelgiumTS\BelgiumTSC_Training\Training\00004
        # E:\DataSet\BelgiumTS\BelgiumTSC_Training\Training\00005
        # ...

        file_names = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".ppm")]
        # filenames是一个列表，其部分展示如下所示：
        # ['E:\\DataSet\\BelgiumTS\\BelgiumTSC_Training\\Training\\00000\\01153_00000.ppm',
        # 'E:\\DataSet\\BelgiumTS\\BelgiumTSC_Training\\Training\\00000\\01153_00001.ppm',
        # 'E:\\DataSet\\BelgiumTS\\BelgiumTSC_Training\\Training\\00000\\01153_00002.ppm',
        # ...

        # For each label, load it's images and add them to the images list.
        # And add the label number (i.e. directory name) to the labels list.
        for f in file_names:
            images.append(f)
            labels.append(int(files))
    # print(images)
    # print(labels)

    # images是由每幅ppm格式图像的绝对路径组成的列表，如下所示：
    # ['E:\\DataSet\\BelgiumTS\\BelgiumTSC_Training\\Training\\00000\\01153_00000.ppm',
    #  'E:\\DataSet\\BelgiumTS\\BelgiumTSC_Training\\Training\\00000\\01153_00001.ppm',
    # 'E:\\DataSet\\BelgiumTS\\BelgiumTSC_Training\\Training\\00000\\01153_00002.ppm',
    # ...
    #  labels是由每幅图像的类别标签组成的列表，如下所示：
    # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
    # 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    # ...
    return images, labels


def change_jpg(images, flag):
    """
    这个程序批量将所有的ppm格式的图像文件转换成jpg的格式，然后批量保存在指定的路径下
    """
    for i in range(len(images)):
        # 遍历每一个ppm格式图像文件所在的绝对地址

        address_list = images[i].split("/")
        #  切分每一个绝对地址，获得一个列表,输出如下所示
        # ['E:', 'DataSet', 'BelgiumTS', 'BelgiumTSC_Training', 'Training', '00000', '01153_00000.ppm']
        # ['E:', 'DataSet', 'BelgiumTS', 'BelgiumTSC_Training', 'Training', '00000', '01153_00001.ppm']
        # ['E:', 'DataSet', 'BelgiumTS', 'BelgiumTSC_Training', 'Training', '00000', '01153_00002.ppm']
        # ['E:', 'DataSet', 'BelgiumTS', 'BelgiumTSC_Training', 'Training', '00000', '01160_00000.ppm']
        # ...

        img = Image.open(images[i])

        if flag == 0:
            root_dir = './data/BelgiumTS/BelgiumTSC_Training_jpg'
        elif flag == 1:
            root_dir = './data/BelgiumTS/BelgiumTSC_Testing_jpg'
        # flag ==0表示对训练集进行转换，flag ==1表示对测试集进行转换

        tail = address_list[-1].split(".")[-2] + ".jpg"
        # 每一幅图像重新保存的文件名，输出如下：
        #  01153_00000.jpg
        #  01153_00001.jpg
        #  01153_00002.jpg
        #  01160_00000.jpg
        #  ...

        new_dir = os.path.join(root_dir, address_list[-2], tail)
        # 获得用于每一幅图像另存的的新绝对路径，输出如下：
        # E:\DataSet\BelgiumTS\BelgiumTSC_Training_jpg\00000\01153_00000.jpg
        # E:\DataSet\BelgiumTS\BelgiumTSC_Training_jpg\00000\01153_00001.jpg
        # E:\DataSet\BelgiumTS\BelgiumTSC_Training_jpg\00000\01153_00002.jpg
        # E:\DataSet\BelgiumTS\BelgiumTSC_Training_jpg\00000\01160_00000.jpg
        # E:\DataSet\BelgiumTS\BelgiumTSC_Training_jpg\00000\01160_00001.jpg

        path = os.path.join(root_dir, address_list[-2])
        # 用于创建上级文件夹，输出如下：
        # E:\DataSet\BelgiumTS\BelgiumTSC_Training_jpg\00000
        # E:\DataSet\BelgiumTS\BelgiumTSC_Training_jpg\00000
        # E:\DataSet\BelgiumTS\BelgiumTSC_Training_jpg\00000
        # E:\DataSet\BelgiumTS\BelgiumTSC_Training_jpg\00000
        # E:\DataSet\BelgiumTS\BelgiumTSC_Training_jpg\00000

        if not os.path.exists(path):
            os.makedirs(path)
        # 判断路径是否存在，不存在就先创建路径

        img.save(new_dir, 'JPEG')


if __name__ == '__main__':
    train_data_dir = './data/belgiumTS/Training'
    test_data_dir = './data/belgiumTS/Testing'
    train_images, train_labels = get_files(train_data_dir)
    test_images, test_labels = get_files(test_data_dir)
    change_jpg(train_images, flag=0)
    change_jpg(test_images, flag=1)

    print("****训练集的信息：******")
    print("Unique Labels:%d \nTotal Images: %d" % (len(set(train_labels)), len(train_images)))
    print("****测试集的信息：******")
    print("Unique Labels:%d \nTotal Images: %d" % (len(set(test_labels)), len(test_images)))
