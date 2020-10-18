'''
Windows: 10.0
python version: 3.7
pytorch version: 1.6.0+cpu
'''
import torch
from torchvision import datasets
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

# define hyper parameters
n_epochs = 10  #训练样本训练次数
batch_size = 20  #mini-batch gradient descent

# load data
train_data = datasets.MNIST(
    root = './data', train=True, download=True, transform=transforms.ToTensor())
test_data = datasets.MNIST(
    root = './data', train=False, download=True, transform=transforms.ToTensor())

# create the loader
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=0)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=0)

# test whether data have been loaded in
'''
def show_data():
    dataiter = iter(train_data)
    images, labels = next(dataiter)
    images = images.numpy()

    fig = plt.figure(figsize = (25,4))
    for idx in np.arange(batch_size):
        ax = fig.add_subplot(2,20/2,idx+1,xticks=[],yticks=[])
        ax.imshow(np.squeeze(images[idx]),cmap='gray')

        ax.set_title(str(labels[idx].item()))
'''

# 四层mlp:输入层，隐藏层，隐藏层，输出层
class MLP(torch.nn.Module):   # 继承torch的Module
    def __init__(self):
        super(MLP, self).__init__()
        # 初始化神经网络
        # 图片是28*28的，所以输入层有784个神经元
        # 可能的数字有10个，所以输出层有10个神经元，其值为预测为这个数字的概率
        hidden_1 = 512 # hyper parameter
        hidden_2 = 512
        self.fc1 = torch.nn.Linear(28*28,hidden_1)
        self.fc2 = torch.nn.Linear(hidden_1,hidden_2)
        self.fc3 = torch.nn.Linear(hidden_2,10)
    def forward(self,din):
        # input din, return dout
        din = din.view(-1, 28*28)  # flatten the tensor
        dout = F.relu(self.fc1(din))  #第一个隐藏层选用relu作为激活函数
        dout = F.relu(self.fc2(dout))  #第二个隐藏层用relu作为激活函数
        dout = F.softmax(self.fc3(dout), dim=1) #输出层用softmax激活，得到最终的分类结果
        return dout

# train the multi-layer network
def train():
    # use the entropy loss function
    lossfunc = torch.nn.CrossEntropyLoss()
    # 使用pytorch提供的优化器
    optimizer = torch.optim.SGD(params=model.parameters(), lr=0.01)
    # 开始训练
    for epoch in range(n_epochs):
        train_loss = 0.0
        for data, target in train_loader:
            optimizer.zero_grad()  # 清空上一步的残余更新参数值
            output = model(data)  # 通过定义好的神经网络得到预测值
            loss = lossfunc(output, target)  # 和标签比较，计算lost function
            loss.backward()  # back propagation, 更新参数
            optimizer.step()  # 把新参数写入网络
            train_loss = loss.item()*data.size(0)
        # 输出本轮训练结果
        train_loss = train_loss / len(train_loader.dataset)
        print('Epoch:  {} \tTraining Loss: {:.6f}'.format(epoch+1, train_loss))
        # 保存网络参数
    torch.save(model.state_dict(), 'net_params.pkl')

# test the multi-layer network
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

# declare the multi-layer network
model = MLP()

if __name__ == '__main__':
    train()

