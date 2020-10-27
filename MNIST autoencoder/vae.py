'''
A Variational AutoEncoder
'''

import torch
from torchvision import datasets
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

# define hyper parameters
n_epochs = 50
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


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(28*28, 400)  # input layer
        self.fc2_m = nn.Linear(400, 20)  # get mean
        self.fc2_v = nn.Linear(400, 20)  # get variance
        self.fc3 = nn.Linear(20, 400)  # asymmetrically connect (input the learned distribution to the decoder)
        self.fc4 = nn.Linear(400, 28*28)  # ouput layer

    #  learn the Gaussian distribution
    def encoder(self, x):
        h = F.relu(self.fc1(x))
        mean = self.fc2_m(h)
        variance = self.fc2_v(h)
        return mean, variance

    # use learned mean and variance to generate code with noise
    def gen_code_with_noise(self, mu, log_var):
        temp = torch.exp(log_var / 2)
        e = torch.randn(temp.size())
        return temp * e + mu

    def decoder(self, z):
        h = F.relu(self.fc3(z))
        output = torch.sigmoid(self.fc4(h))  # normalization
        return output

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.gen_code_with_noise(mu, log_var)
        output = self.decoder(z)
        return output, mu, log_var  # notice that it generate by-product

model = VAE()

# train the VAE
def train():
    # 使用pytorch提供的优化器
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
    # 开始训练
    for epoch in range(n_epochs):
        train_loss = 0.0
        for data, target in train_loader:
            x = data.view(-1, 28*28)
            y = data.view(-1, 28*28) # give artificial label y=x
            optimizer.zero_grad()  # 清空上一步的残余更新参数值
            reconstruct, mu, log_var = model(x)  # 通过定义好的神经网络得到预测值
            # reconstruct loss
            reconstruct_loss = F.binary_cross_entropy(reconstruct, y, size_average=False)
            # KL divergence
            kl_divergence = -0.5 * torch.sum(1+log_var-torch.exp(log_var) -mu**2)
            # loss function
            loss = reconstruct_loss + kl_divergence
            loss.backward()  # back propagation, 更新参数
            optimizer.step()  # 把新参数写入网络
            train_loss = loss.item()*data.size(0)
        # 输出本轮训练结果
        train_loss = train_loss / len(train_loader.dataset)
        print('Epoch:  {} \tTraining Loss: {:.6f}'.format(epoch+1, train_loss))
    # 保存网络参数
    torch.save(model.state_dict(), 'VAE_params.pkl')

# 比较原数据和经过编码解码后的数据
def test():
    f, a = plt.subplots(2, N_TEST_IMG, figsize=(5, 2))
    plt.ion()

    view_data = train_data.train_data[:N_TEST_IMG].view(-1, 28 * 28).type(torch.FloatTensor) / 255.
    for i in range(N_TEST_IMG):
        a[0][i].imshow(np.reshape(view_data.data.numpy()[i], (28, 28)), cmap='gray')
        a[0][i].set_xticks(())
        a[0][i].set_yticks(())

    reconstruct, _, _ = model(view_data)
    for i in range(N_TEST_IMG):
        a[1][i].imshow(np.reshape(reconstruct.data.numpy()[i], (28, 28)), cmap='gray')
        a[1][i].set_xticks(())
        a[1][i].set_yticks(())
        plt.draw()
    plt.pause(0)

def generate():
    scale = 1
    f, a = plt.subplots(N_TEST_IMG, N_TEST_IMG, figsize=(5, 2))
    plt.ion()
    x = (torch.randn(N_TEST_IMG*N_TEST_IMG, 20)) * scale
    decoded_result = model.decoder(x)
    for i in range(N_TEST_IMG):
        for j in range(N_TEST_IMG):
            a[i][j].imshow(np.reshape(decoded_result.data.numpy()[i*N_TEST_IMG+j], (28, 28)), cmap='gray')
            a[i][j].set_xticks(())
            a[i][j].set_yticks(())
    plt.pause(0)


def main():
    train_flag = False
    test_flag = True
    if train_flag == True:
        train()
    if test_flag == True:
        model.load_state_dict(torch.load('VAE_params.pkl'))
        test()

if __name__=='__main__':
    # main()
    model.load_state_dict(torch.load('VAE_params.pkl'))
    generate()
