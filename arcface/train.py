'''
Windows: 10.0
python version: 3.7
pytorch version: 1.6.0+cpu
'''
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
from torchvision import datasets
import torch.nn as nn
import torchvision.transforms as transforms
from dataLoader import ClassFilterDataloader
from cnn_for_mnist import CNN
from random import randint
import pickle

# define hyper parameters
n_epochs = 5
batch_size = 64
LR = 0.001

# load data
train_data = datasets.MNIST(
    root='./data', train=True, download=True, transform=transforms.ToTensor())

# create the loader
train_loader = ClassFilterDataloader(train_data, range(10), 1, False, batch_size)

# declare the network
model = CNN()


# train the cnn
def train():
    # use the entropy loss function
     lossfunc = nn.CrossEntropyLoss()
    # 使用pytorch提供的优化器
    optimizer = torch.optim.Adam(params=model.parameters(), lr=LR)
    # 开始训练
    for epoch in range(n_epochs):
        train_loss = 0.0
        for data, target in train_loader:
            optimizer.zero_grad()  # 清空上一步的残余更新参数值
            output = model(data)  # 通过定义好的神经网络得到预测值
            loss = lossfunc(output, target)  # 和标签比较，计算lost function
            loss.backward()  # back propagation, 更新参数
            optimizer.step()  # 把新参数写入网络
            train_loss = loss.item() * data.size(0)
        # 输出本轮训练结果
        train_loss = train_loss / len(train_loader.dataset)
        print('Epoch:  {} \tTraining Loss: {:.6f}'.format(epoch + 1, train_loss))
    # 保存网络参数
    torch.save(model.state_dict(), 'CNN_net_params.pkl')


def save_vector(class_num, class_label):
    dict = {}
    cnt = 0
    loader = ClassFilterDataloader(train_data, range(10), 1, True, 1)
    with torch.no_grad():
        for data, target in loader:
            target = int(target)
            if target in class_label and target not in dict:
                dict[target] = model(data)
                cnt += 1
            if cnt == class_num:
                break
    output = open('vector_representation.pkl', 'wb')
    pickle.dump(dict, output)


if __name__ == '__main__':
    train()
    save_vector(10, range(10))
