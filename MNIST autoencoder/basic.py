'''
A basic AutoEncoder
Encoder: NN with two hidden layers
Bottleneck: 3 neurons
Decoder: NN with two hidden layers
'''

import torch
from torchvision import datasets
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

# define hyper parameters
n_epochs = 10
batch_size = 64
learning_rate = 0.001
N_TEST_IMG = 8

# load data
train_data = datasets.MNIST(
    root = './data', train=True, download=True, transform=transforms.ToTensor())
test_data = datasets.MNIST(
    root = './data', train=False, download=True, transform=transforms.ToTensor())

# create the loader
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=0)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=0)

# plot one example
'''
plt.imshow(train_data.train_data[2].numpy(), cmap='gray')
plt.title('%i' % train_data.train_labels[2])
plt.show()
'''

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()

        # encoder（压缩信息）
        self.encoder = nn.Sequential(
            nn.Linear(28*28,128),
            nn.Tanh(),
            nn.Linear(128,64), # hidden layer1
            nn.Tanh(),
            nn.Linear(64,12), # hidden layer2
            nn.Tanh(),
            nn.Linear(12,3), # compress to 3 features in the bottleneck layer
        )

        # decoder（重建信息）
        # asymmetric to the encoder network
        self.decoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.Tanh(),
            nn.Linear(12,64),
            nn.Tanh(),
            nn.Linear(64,128),
            nn.Tanh(),
            nn.Linear(128, 28*28),
            nn.Sigmoid(), # compress to range(0,1)
        )

    def forward(self, x):
        encoded_info = self.encoder(x)
        decoded_result = self.decoder(encoded_info)
        return encoded_info, decoded_result

model = AutoEncoder()

# train the basic AutoEncoder
def train():
    # use the L2-loss function
    loss_func = nn.MSELoss()
    # 使用pytorch提供的优化器
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
    # 开始训练
    for epoch in range(n_epochs):
        train_loss = 0.0
        for data, target in train_loader:
            x = data.view(-1, 28*28)
            y = data.view(-1, 28*28) # give artificial label y=x
            optimizer.zero_grad()  # 清空上一步的残余更新参数值
            encoded, decoded = model(x)  # 通过定义好的神经网络得到预测值
            loss = loss_func(decoded, y)  # 和标签比较，计算lost function
            loss.backward()  # back propagation, 更新参数
            optimizer.step()  # 把新参数写入网络
            train_loss = loss.item()*data.size(0)
        # 输出本轮训练结果
        train_loss = train_loss / len(train_loader.dataset)
        print('Epoch:  {} \tTraining Loss: {:.6f}'.format(epoch+1, train_loss))
    # 保存网络参数
    torch.save(model.state_dict(), 'basic_encoder_params.pkl')

# 比较原数据和经过编码解码后的数据
def test():
    # view results dynamically
    '''
    plt.ion()
    plt.show()

    for i in range(10):
        #index = np.random.randint(0, 100, 1)
        input = train_data.train_data[i].view(-1, 28 * 28).type(torch.FloatTensor) / 255.
        _, result = model(input)

        im_result = result.view(28, 28)
        plt.figure(1, figsize=(10, 3))
        plt.subplot(121)
        plt.title('train_data')
        plt.imshow(train_data.train_data[i].numpy(), cmap='Greys')

        plt.figure(1, figsize=(10, 3))
        plt.subplot(122)
        plt.title('result_data')
        plt.imshow(im_result.detach().numpy(), cmap='Greys')
        plt.show()
        plt.pause(0.5)

    plt.ioff()
    '''
    f, a = plt.subplots(2, N_TEST_IMG, figsize=(5, 2))
    plt.ion()

    view_data = train_data.train_data[:N_TEST_IMG].view(-1, 28 * 28).type(torch.FloatTensor) / 255.
    for i in range(N_TEST_IMG):
        a[0][i].imshow(np.reshape(view_data.data.numpy()[i], (28, 28)), cmap='gray')
        a[0][i].set_xticks(())
        a[0][i].set_yticks(())

    _, decoded = model(view_data)
    for i in range(N_TEST_IMG):
        a[1][i].imshow(np.reshape(decoded.data.numpy()[i], (28, 28)), cmap='gray')
        a[1][i].set_xticks(())
        a[1][i].set_yticks(())
        plt.draw()
    plt.pause(0)

def main():
    train_flag = False
    test_flag = True
    if train_flag == True:
        train()
    if test_flag == True:
        model.load_state_dict(torch.load('basic_encoder_params.pkl'))
        test()

if __name__=='__main__':
    main()
