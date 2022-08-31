import random

from torch import optim
from torch.utils.data import Dataset, DataLoader
import csv
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import torch


def DataInit(): #数据初始化
    with open('train.tsv') as f:
        tsvreader = csv.reader(f, delimiter='\t')
        raw_list = list(tsvreader)
        data_list = [[row[2], row[3]] for row in raw_list]
        train = list()
        test = list()
        for row in data_list:
            if random.random() > 0.3:
                train.append(row)
            else:
                test.append(row)
    return train, test # 分别返回训练集和测试集的列表

def DictInit(train, test): # 字典初始化
    dict_words = dict()  # 单词到单词编号的映射
    total_data = train+test
    sentences = [row[0] for row in total_data]
    for sentence in sentences:
        sentence = sentence.upper()
        words = sentence.split()
        for word in words:  # 一个一个单词寻找
            if word not in dict_words:
                dict_words[word] = len(dict_words)
    return dict_words

def Word2Tensor(sentences, dict_words): # 将列表转换为tensor
    max_length = 0
    for sentence in sentences:
        max_length = len(sentence.split()) if len(sentence.split()) > max_length else max_length
    sentences_tensor = torch.zeros(len(sentences), max_length).long() # 生成num_sentences行，max_length列的全0矩阵
    for idx, sentence in enumerate(sentences):
        IntSentence = list()
        sentence = sentence.upper()
        words = sentence.split()
        for word in words:
            IntSentence.append(dict_words[word])
        sentence_len = len(IntSentence)
        sentences_tensor[idx, :sentence_len] = torch.LongTensor(IntSentence)
    return sentences_tensor



class Dataset(Dataset):
    # 自定义数据集的结构
    def __init__(self, datalist, dict_words):
        self.sentences = [row[0] for row in datalist]
        self.emotions = [row[1] for row in datalist]
        self.sentences_tensor = Word2Tensor(self.sentences, dict_words)
        self.emotions_tensor = torch.LongTensor([int(char_emotion) for char_emotion in self.emotions])

    def __getitem__(self, index):
        return self.sentences_tensor[index], self.emotions_tensor[index]

    def __len__(self):
        return len(self.emotions)

