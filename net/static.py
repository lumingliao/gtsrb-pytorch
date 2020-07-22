import os
import shutil
from tqdm import tqdm

Path = '/home/llm/data/BreaKHis_v1/histology_slides/breast/'
det = '/home/llm/data/BreaKHis_v1/histology_slides/data/'
classes = ['benign', 'malignant']


def check_if_dir(file_path):
    temp_list = os.listdir(file_path)    #put file name from file_path in temp_list
    for temp_list_each in temp_list:
        temp_path = file_path + '/' + temp_list_each
        if os.path.isfile(temp_path):
            if os.path.splitext(temp_path)[-1] == '.png':    #自己需要处理的是.log文件所以在此加一个判断
                path_read.append(temp_path)
            else:
                continue
        else:
            check_if_dir(temp_path)    #loop traversal
    # print(f'Total images is {total}')

if __name__ == "__main__":
    for cls in classes:
        path_read = []  # path_read saves all executable files
        path = os.path.join(Path, cls)
        check_if_dir(path)
        # print(len(path_read))
        tmp = det + cls
        print(len(path_read))
        for i in tqdm(path_read):
            shutil.copy(i, tmp)