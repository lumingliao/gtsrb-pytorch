import torch
import matplotlib.pyplot as plt
from torchvision import datasets
import multiprocessing
from cutout import save_image_tensor2pillow

data_path = 'data/data0'
save_dir = './footage/'
from data import initialize_data, data_jitter_brightness
initialize_data(data_path) # extracts the zip files, makes a validation set


train_loader = torch.utils.data.DataLoader(
   torch.utils.data.ConcatDataset([datasets.ImageFolder(data_path + '/train_images',transform=data_jitter_brightness)]),
                                   batch_size=1,
                                   shuffle=True,
                                   num_workers=multiprocessing.cpu_count(),
                                   pin_memory=True)

for batch_idx, (data, target) in enumerate(train_loader):
    print(type(data), target)
    target = target.to(torch.device('cpu')).type(torch.uint8).numpy()
    save_path = save_dir + format(target[0], '05d') + '.png'
    img = save_image_tensor2pillow(data, save_path, 'pil', True)
    plt.figure("img")
    plt.imshow(img)
    plt.show()
    break