from __future__ import print_function
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from cnn_finetune import make_model


# model = make_model('resnet34', num_classes=2, pretrained=True, input_size=(320, 320))
# model = make_model('resnet50', num_classes=2, pretrained=True, input_size=(320, 320))
# model = torchvision.models.resnet18(pretrained=True)
# model = make_model('resnet18', num_classes=43, pretrained=True)
# model = make_model('resnet50', num_classes=2, pretrained=True, input_size=(320, 320), classifier_factory=make_classifier)

# =========================================================
nclasses = 43  # GTSRB as 43 classes

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # CNN layers
        self.conv1 = nn.Conv2d(3, 100, kernel_size=5)
        self.bn1 = nn.BatchNorm2d(100)
        self.conv2 = nn.Conv2d(100, 150, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(150)
        self.conv3 = nn.Conv2d(150, 250, kernel_size=3)
        self.bn3 = nn.BatchNorm2d(250)
        self.conv_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(2250, 350)
        self.fc2 = nn.Linear(350, nclasses)
        self.filters = 250
        self.glob = nn.AdaptiveAvgPool2d((1, 1))

        self.localization = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        # Regressor for the 3 * 2 affine matrix
        # 子网络（全连接或卷积网络，再加上一个回归层）用来生成空间变换的参数θ，θ的形式可以多样，
        # 如需实现2D仿射变换，θ 就是一个6维（2x3）向量的输出
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 7 * 7, 32),  # 160*32
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)  # 32*6
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

        # SENet
        # self.se = nn.Sequential(
        #     nn.AdaptiveAvgPool2d((1,1)),
        #     nn.Conv2d(self.filters, self.filters//16, kernel_size=1),
        #     nn.ReLU(),
        #     nn.Conv2d(self.filters//16, self.filters, kernel_size=1),
        #     nn.Sigmoid()
        # )


    # Spatial transformer network forward function
    # 整个空间变换器包含三个部分，本地网络(Localisation Network)、网格生成器(Grid Genator)和采样器(Sampler)
    def stn(self, x):
        xs = self.localization(x)  # torch.Size([1, 10, 4, 4])
        xs = xs.view(-1, 10 * 7 * 7)  # 361, 160
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)
        return x

    def forward(self, x):
        # transform the input
        x = self.stn(x)

        # Perform forward pass
        x = self.bn1(F.max_pool2d(F.leaky_relu(self.conv1(x)), 2))
        x = self.conv_drop(x)
        x = self.bn2(F.max_pool2d(F.leaky_relu(self.conv2(x)), 2))
        x = self.conv_drop(x)
        x = self.bn3(F.max_pool2d(F.leaky_relu(self.conv3(x)), 2))
        # x1 = self.se(x)
        # x = x * x1
        x = self.conv_drop(x)
        x = x.view(-1, 2250)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
