import sys, io
import jieba
import random
import numpy as np

trainFile = 'text_classification_data/train.txt'
devFile = 'text_classification_data/dev.txt'
pad_size = 20 # 设定句子长度为20，多割少补
trainVecFile = 'train_vec.txt'
devVecFile = 'dev_vec.txt'

def get_word_dict(file):
    data = open(file, 'r', encoding='utf_8').read().split('\n')
    word2index = {}
    for line in data:
        line_list = line.split(' ')
        word2index[line_list[0]] = int(line_list[1])
    return word2index


def main():
    word2index = get_word_dict('wordList.txt')
    # 对标题分词，分词后在词表中查找每个单词的id，从而将标题转化为向量
    # data = open(trainFile, 'r', encoding='utf_8').read().split('\n')
    data = open(devFile, 'r', encoding='utf_8').read().split('\n')
    stoplist = open('stopword.txt', 'r', encoding='utf_8').read().split('\n')
    # f = open('train_vec.txt', 'w', encoding='utf_8')
    f = open(devVecFile, 'w', encoding='utf_8')
    # maxlen = 0
    begin = 1
    for line in data:
        line = line.split('\t')  # 把每一行数据中的标题和label分割开
        title = line[0]
        # print(title)
        title_seg = jieba.cut(title, cut_all=False)  # 分词，采用精确模式，返回类型是generator
        title_vec = []
        title_vec.append(line[1])  # 把标签放在向量的第一维
        for w in title_seg:
            if w in stoplist:  # 去掉无意义的词
                continue
            if w in word2index:
                title_vec.append(word2index[w])
            else:
                title_vec.append(np.random.randint(0, 3000))
        length = len(title_vec)
        # maxlen = max(maxlen, length)
        if length > pad_size + 1:
            title_vec = title_vec[0:21]  # cut
        if length < pad_size + 1:
            title_vec.extend([0] * (pad_size + 1 -length))  # padding
        if begin == 1:
            begin = 0
        else:
            f.write('\n')
        for n in title_vec:
            f.write(str(n) + ',')
    # print(maxlen)


if __name__ == "__main__":
    main()