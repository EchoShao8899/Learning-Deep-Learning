'''
textCNN for Chinese text classification
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import title2vec


class Config(object):
    """配置参数"""
    def __init__(self, word2index):
        self.vocab_size = len(word2index)
        self.embed_dim = 300  # 指定embedding的维度
        self.class_num = 4  # 0,1,2,3
        self.kernel_num = 100
        self.kernel_size = [3, 4, 5]  # 使用不同大小的kernel以提取出不同关联度的信息
        self.dropout = 0.5  # 随机失活概率，0.5是一个惯常的取法


class textCNN(nn.Module):
    def __init__(self, config):
        super(textCNN, self).__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.embed_dim, padding_idx=1)  # padding_idx 填充id embedding层起到利用矩阵乘法降维的作用
        #  convolutional layer with multiple filter widths and feature maps
        self.conv11 = nn.Conv2d(
            in_channels=1,
            out_channels=config.kernel_num,
            kernel_size=(config.kernel_size[0], config.embed_dim),  # kernel_size的第二维由词向量决定（相当于把一个词向量当作一个整体）
            stride=1,  # default
            padding=0  # default
        )
        self.conv12 = nn.Conv2d(
            in_channels=1,
            out_channels=config.kernel_num,
            kernel_size=(config.kernel_size[1], config.embed_dim)
        )
        self.conv13 = nn.Conv2d(
            in_channels=1,
            out_channels=config.kernel_num,
            kernel_size=(config.kernel_size[2], config.embed_dim)
        )
        # employ dropout on the penultimate layer
        self.dropout = nn.Dropout(config.dropout)
        # fully connected layer
        self.fc1 = nn.Linear(len(config.kernel_size) * config.kernel_num, config.class_num)

    def conv_and_pool(self, x, conv):
        # x: (batch, 1, sentence_length, )
        x = conv(x)
        # x: (batch, kernel_num, output_height, output_width=1)
        x = F.relu(x.squeeze(3))  # squeeze(n)去掉值为1的维度
        # x: (batch, kernel_num, output_height)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        # x: (batch, kernel_num)
        return x

    def forward(self, x):
        # x: (batch, sentence_length)
        x = self.embedding(x)
        # x: (batch, sentence_length, embedding_dim)
        x = x.unsqueeze(1)  # 增加一个维度
        # x: (batch, 1, sentence_length, embedding_dim)
        x1 = self.conv_and_pool(x, self.conv11)
        x2 = self.conv_and_pool(x, self.conv12)
        x3 = self.conv_and_pool(x, self.conv13)
        x = torch.cat((x1, x2, x3), 1)  # concatenate x: (batch, 3*kernel_num)
        x = self.dropout(x)
        out = self.fc1(x)
        return out

