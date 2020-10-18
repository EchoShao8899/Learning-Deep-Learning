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

# test the cnn
def test():
    correct = 0
    total = 0
    with torch.no_grad():  # 训练集不需要反向传播，只需要通过神经网络函数算出预测结果即可
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the test images: %d %%' %(100 * correct / total))
    return 100.0*correct / total

def main():
    model.load_state_dict(torch.load('CNN_net_params.pkl'))
    test()

if __name__=='__main__':
    main()
