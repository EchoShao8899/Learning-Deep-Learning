import torch
import torch.nn as nn
import numpy as np
import time

from textcnn import textCNN
from textcnn import Config
import title2vec

word2index = title2vec.get_word_dict('wordList.txt')
config = Config(word2index)
model = textCNN(config)
model.load_state_dict(torch.load('textCNN_params.pkl'))


def predict_result(out):
    score = max(out)
    label = np.where(out == score)[0][0]

    return label


def main():
    test_data = open('dev_vec.txt', 'r').read().split('\n')
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


if __name__ == "__main__":
    main()
