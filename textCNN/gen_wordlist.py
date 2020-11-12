'''
词表生成：
读语料 -> 分词 -> 词频统计
'''
import jieba  # 中文分词库
from tqdm import tqdm  # 进度条

trainFile = 'text_classification_data/train.txt'


def main():
    word_dict = {} # 词频统计
    stoplist = open('stopword.txt', 'r', encoding='utf_8').read().split('\n')
    data = open(trainFile, 'r', encoding='utf_8').read().split('\n')
    # print(data)
    data_num = len(data)
    # title = data[0].split('\t')[0]
    # title_seg = jieba.lcut(title, cut_all=False)
    # print(title_seg)

    # 分词并统计词频
    for line in data:
        title = line.split('\t')[0]  # 去掉一行数据中的标签
        title_seg = jieba.cut(title, cut_all=False)  # 分词，采用精确模式，返回类型是generator
        for w in title_seg:
            if w in stoplist:  # 去掉无意义的词
                continue
            if w in word_dict:
                word_dict[w] += 1
            else:
                word_dict[w] = 1

    word_list = sorted(word_dict.items(), key=lambda item:item[1], reverse=True)
    f = open('wordList.txt', 'w', encoding='utf_8')
    _id = 0
    # 词表格式：词 id 词频
    begin = 1
    for t in word_list:
        if begin == 1:
            begin = 0
            d = t[0] + ' ' + str(_id) + ' ' + str(t[1])
        else:
            d = '\n' + t[0] + ' ' + str(_id) + ' ' + str(t[1])
        _id += 1
        f.write(d)



if __name__ == "__main__":
    main()