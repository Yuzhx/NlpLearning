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
    return train, test

def DictInit(train, test): #字典初始化
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

def Word2Tensor(sentences, dict_words):
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

train, test = DataInit()
dict_words = DictInit(train,test)
sentences = [row[0] for row in train]
sentences_tensor = Word2Tensor(sentences, dict_words)

class Embedding(nn.Module):
    def __init__(self, d_model, vocab): #d_model为词嵌入的维度，vocab为词表大小
        super(Embedding, self).__init__()
        # 定义Embedding层
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model
    def forward(self, x): #x为输入模型的文本通过此映射之后的数字张良
        return self.lut(x)



class Dataset(Dataset):
    # 自定义数据集的结构
    def __init__(self, datalist):
        self.sentences = [row[0] for row in datalist]
        self.emotions = [row[1] for row in datalist]
        self.sentences_tensor = Word2Tensor(self.sentences,dict_words)
        self.emotions_tensor = torch.LongTensor([int(char_emotion) for char_emotion in self.emotions])

    def __getitem__(self, index):
        return self.sentences_tensor[index], self.emotions_tensor[index]

    def __len__(self):
        return len(self.emotions)

class RNN(nn.Module):
    def __init__(self,len_feature, len_hidden, layer=1, drop_out=0.5, nonlinearity='tanh', batch_first=True, type_num=5):
        super(RNN, self).__init__()
        self.len_feature = len_feature # 特征向量长度
        self.len_hidden = len_hidden # 隐层宽度
        self.layer = layer # 隐层层数
        self.embedding = Embedding(len_feature,len(dict_words))
        self.dropout = nn.Dropout(drop_out) # drop_out层
        self.rnn = nn.RNN(input_size=len_feature, hidden_size=len_hidden, num_layers=layer, nonlinearity=nonlinearity, batch_first=batch_first)
        self.fc = nn.Linear(len_hidden, type_num)
    def forward(self, x):
        batch_size = x.size(0)
        # x: [batch_size, max_length]
        embedded = self.embedding(x)
        # embedded: [batch_size, max_length, d_model]
        dropouted = self.dropout(embedded)
        # dropouted: [batch_size, max_length, d_model]

        h0 = torch.autograd.Variable(torch.zeros(self.layer, batch_size, self.len_hidden)) # 初始化隐层参数
        _, hn = self.rnn(dropouted, h0)
        # hn: [1, batch_size, len_hidden]
        output = self.fc(hn).squeeze(0)
        # output: [1, batch_size, 5] -> [batch_size, 5]
        return output

trainset = Dataset(train)
trainloader = DataLoader(trainset, batch_size=128, shuffle=True)
loss_func = torch.nn.CrossEntropyLoss()
rnn = RNN(50, 50)
optimizer = optim.Adam(rnn.parameters(), lr=0.01)
def Train():
    for epoch in range(30):
        epoch_loss = 0
        for i, (sentences_batch, emotions_batch) in enumerate(trainloader,1):
            # sentences_batch: [batch_size,max_length]
            # emotions_batch: [batch_size,]
            pred = rnn(sentences_batch)
            optimizer.zero_grad()
            loss = loss_func(pred, emotions_batch)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
        print(epoch,":",epoch_loss)
Train()