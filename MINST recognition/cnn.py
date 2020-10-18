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

# define hyper parameters
n_epochs = 5  # 训练样本训练次数
batch_size = 64  # mini-batch gradient descent
LR = 0.001  # learning rate

# load data
train_data = datasets.MNIST(
    root = './data', train=True, download=True, transform=transforms.ToTensor())
test_data = datasets.MNIST(
    root = './data', train=False, download=True, transform=transforms.ToTensor())

# create the loader
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=0)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=0)

# define a CNN network with two convolution layers (plus pooling) and a fully connected layer
class CNN(nn.Module):  # 继承torch的module
    def __init__(self):
        super(CNN, self).__init__()  # 初始化基类
        # 第一个卷积层
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1, # 1代表灰度图
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(), # activation function
            nn.MaxPool2d(kernel_size=2) # max pooling with 2*2 window
        )
        # conv -> (N-F+2*P)/S+1=28 -> pooling -> 28/2=14
        # 这层结束后，shape为[16, 14, 14]
        # 第二个卷积层
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
        # 这层结束后，shape为[32, 7, 7]
        # 全连接层, flatten the vector to have the size 32*7*7
        self.prediction = nn.Linear(32*7*7, 10)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0),-1) # flatten
        output = self.prediction(x)
        return output

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

if __name__ == '__main__':
    train()
