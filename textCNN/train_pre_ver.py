import torch
import torch.nn as nn
import numpy as np
import time

from textcnn import textCNN
from textcnn import Config
import title2vec
import textCNN_data

word2index = title2vec.get_word_dict('wordList.txt')
config = Config(word2index)
model = textCNN(config)
dataLoader_param = {
    'batch_size': 128,
    'shuffle': True
}
n_epoch = 20


def predict_result(out):
    score = max(out)
    label = np.where(out == score)[0][0]

    return label


def test():
    test_data = open('train_vec.txt', 'r').read().split('\n')
    cnt_all = 0
    cnt_correct = 0
    with torch.no_grad():
        for data in test_data:
            cnt_all += 1
            data = list(filter(None, data.split(',')))
            data = [int(x) for x in data]
            label = data[0]
            sentence = np.array(data[1:])
            sentence = torch.from_numpy(sentence)  # 类型转换
            sentence = sentence.unsqueeze(0).type(torch.LongTensor)
            predict = model(sentence)
            # print(predict)
            pre_label = predict_result(predict)
            if pre_label == label:
                cnt_correct += 1
    print('Accuracy of the network on the test sentence: %d %%' % (100 * cnt_correct / cnt_all))

def main():
    dataLoader = textCNN_data.textCNN_dataLoader(dataLoader_param)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    lossfunc = nn.CrossEntropyLoss()
    for epoch in range(n_epoch):
        for i, (class_label, sentence) in enumerate(dataLoader):  # 将迭代器传入枚举函数从而实现遍历
            optimizer.zero_grad()  # 清空上一步的残余更新参数值
            sentence = sentence.type(torch.LongTensor)
            class_label = class_label.type(torch.LongTensor)
            output = model(sentence)  # 通过定义好的神经网络得到预测值
            loss = lossfunc(output, class_label)  # 和标签比较，计算lost function
            loss.backward()  # back propagation, 更新参数
            optimizer.step()  # 把新参数写入网络
            train_loss = loss.item() * sentence.size(0)
            # 输出本轮训练结果
        train_loss = train_loss / len(dataLoader.dataset)
        print('Epoch:  {} \tTraining Loss: {:.6f}'.format(epoch + 1, train_loss))
        # 保存网络参数
    torch.save(model.state_dict(), 'textCNN_params.pkl')
    test()

if __name__ == "__main__":
    main()