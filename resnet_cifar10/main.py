import torch
import torchvision
from torchvision import datasets
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import torch.nn as nn
import torch.optim as optim

from model import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152

device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")
print(device)

# define hyper parameters
n_epochs = 120  #训练样本训练次数
batch_size = 128  #mini-batch gradient descent
LR = 0.001

# data augmentation
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4), # 先四周填充0，再把图像随机裁减成32*32
    transforms.RandomHorizontalFlip(), # 图像一半概率翻转，一半概率不翻转
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)) #R,G,B每层的归一化用到的均值和方差
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

# load data
train_data = datasets.CIFAR10(root = './data', train=True, download=True, transform=transform_train)
train_loader = torch.utils.data.DataLoader(
    train_data, batch_size=batch_size, shuffle=True, num_workers=2) # num_workers是加载数据（batch）的线程数目
test_data = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=100, shuffle=False, num_workers=2)

# declare model
net = ResNet50().to(device)

def train():
    lossfunc = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=LR)

    for epoch in range(n_epochs):
        net.train()
        for batchidx, (data, label) in enumerate(train_loader):
            # data: [b, 3, 32, 32]
            # label: [b]
            data = data.to(device)
            label = label.to(device)
            optimizer.zero_grad()

            output = net(data)
            loss = lossfunc(output, label)
            # update the parameters
            loss.backward()
            optimizer.step()

        print("epoch: ", epoch, " training loss: ", loss.item())

        # 在每个epoch中都进行测试
        net.eval()
        with torch.no_grad():

            total_correct = 0
            total_num = 0
            for data, label in test_loader:
                data = data.to(device)
                label = label.to(device)
                output = net(data)
                _, predicted = torch.max(output.data, 1)

                total_correct += predicted.eq(label.data).cpu().sum()
                total_num += label.size(0)

            acc = 100.0 * total_correct / total_num
            print("epoch: ", epoch, " test acc: ", float(acc), "%")


def main():
    train()


if __name__ == '__main__':
    main()






