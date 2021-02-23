import torch
from torch import nn
import numpy as np
import torch.nn.functional as F


class MLP(nn.Module):
    """
    Module for a multi-layer perceptron (MLP).

    input: <2D tensor> [batch_size] * [input_dim]
    output: <2D tensor> [batch_size] * [classes]

    """
    def __init__(self, input_dim, classes):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.classes = classes
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 10)

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        result = F.softmax(self.fc3(x), dim=1)

        return result



