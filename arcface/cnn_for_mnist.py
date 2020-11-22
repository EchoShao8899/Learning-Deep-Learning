import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
from torchvision import datasets
import torch.nn as nn
import torchvision.transforms as transforms


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(), # activation function
            nn.MaxPool2d(kernel_size=2) # max pooling with 2*2 window
        )
        # conv -> (N-F+2*P)/S+1=28 -> pooling -> 28/2=14
        # shape[16, 14, 14]
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        # conv -> 14 -> pooling -> 7
        # shape [32, 7, 7]
        # 全连接层, flatten the vector to have the size 32*7*7
        self.prediction = nn.Linear(32*7*7, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0),-1) # flatten
        output = self.prediction(x)
        return output

