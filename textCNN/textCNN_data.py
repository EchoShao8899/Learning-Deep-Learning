from torch.utils.data import Dataset, DataLoader  # Dataset类可以传入DataLoader从而进行进一步操作
import torch
import random
import numpy as np


trainDataFile = 'train_vec.txt'


class textCNN_data(Dataset):  # 继承Dataset类后，关键就是重写len和getitem方法
    def __init__(self):
        trainData = open(trainDataFile, 'r').read().split('\n')
        random.shuffle(trainData)
        self.trainData = trainData

    def __len__(self):
        return len(self.trainData)

    def __getitem__(self, idx):
        data = self.trainData[idx]
        data = list(filter(None, data.split(',')))  # 构造的词向量各个维度之间是用逗号分隔的
        data = [int(x) for x in data]  # 字符串转整型
        class_label = data[0]  # 构造的数据中第一维存的是类别
        sentence = np.array(data[1:])

        return class_label, sentence


def textCNN_dataLoader(param):  # DataLoader的常用操作：batch_size, (bool) shuffle, num_workers (加载数据时的子进程数）
    dataset = textCNN_data()
    batch_size = param['batch_size']
    shuffle = param['shuffle']
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


# testing
if __name__ == "__main__":
    dataset = textCNN_data()
    class_lable, sentence = dataset.__getitem__(0)

    print(class_lable)
    print(sentence)