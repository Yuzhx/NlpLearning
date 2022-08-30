import pandas as pd
from torch import optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
from collections import Counter
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from gensim.models import keyedvectors
import matplotlib.pyplot as plt

train_data = pd.read_csv("train.txt",names = ["label","comment"],sep = "\t")

# ----------------第一次已经处理过，不需要再处理----------------
"""
# 首先提取所有的词
words=[]
for i in range(len(train_data)):
    com=train_data["comment"][i].split()
    words=words+com

# 统计词频，去除词频比较小的词
with open("word_freq.txt",'w',encoding='utf-8') as fout:
    for word,freq in Counter(words).most_common():
        if freq > 30:
            fout.write(word + "\n")"""
# ---------------------------------------------------------


with open("word_freq.txt", encoding='utf-8') as fin:
    vocab = [i.strip() for i in fin]
vocab=set(vocab) # 将进行词频处理过后的文件重新读取
word2idx = {i:index for index, i in enumerate(vocab)} #词-id的索引
idx2word = {index:i for index, i in enumerate(vocab)} #id-词的索引
vocab_size = len(vocab)
print(vocab_size)
#对输入数据进行预处理,主要是对句子用索引表示且对句子进行截断与padding，将填充使用”把“来。
max_length = 62
Embedding_size = 50 # 词嵌入维度
Batch_Size = 256
Kernel = 3
Filter_num = 10 # 卷积核数目
Dropout = 0.5
Learning_rate = 0.0001

pad_id=word2idx["把"]
def tokenizer():
    inputs = []
    sentence_char = [i.split() for i in train_data["comment"]]
    # 将输入文本转为数字化
    for index, raw_sentence in enumerate(sentence_char):
        sentence = [word2idx.get(j, pad_id) for j in raw_sentence] # 表示如果词表中没有这个稀有词，无法获得，那么就默认返回pad_id。
        if len(raw_sentence) < max_length:
            # padding。
            for _ in range(max_length-len(raw_sentence)):
                sentence.append(pad_id)
        else:
            sentence = sentence[:max_length]
        inputs.append(sentence)
    return inputs
data_input = tokenizer()
print("loadover")




class TextCNNDataSet(Dataset):
    def __init__(self, data_inputs, data_targets): # 输入的是列表
        self.inputs = torch.LongTensor(data_inputs)
        self.label = torch.LongTensor(data_targets)

    def __getitem__(self, index):
        return self.inputs[index], self.label[index]

    def __len__(self):
        return len(self.inputs)

w2v=keyedvectors.load_word2vec_format("wiki_word2vec_50.bin",binary=True)
def word2vec(x):
    # 输入x：batch_size,sequence_length
    # 输出 ：batch_size,sequence_length,embedding_size
    # x是以编号的形式来反映的，所以需要将其翻译一下。
    x2v=np.ones((x.shape[0],x.shape[1],Embedding_size))
    for i in range(len(x)):
        for j,x_i in enumerate(x[i], 0):
            now_word = idx2word[x_i.item()]
            x2v[i][j]=w2v[now_word] if w2v.has_index_for(now_word) else np.random.randn(50,)
    return torch.tensor(x2v).to(torch.float32)

class TextCNN(nn.Module):
    def __init__(self):
        super(TextCNN, self).__init__()
        out_channel = Filter_num # 通道数目
        self.conv = nn.Sequential(
            nn.Conv2d(1, out_channel, (3, Embedding_size)), # 卷积核大小为2*Embedding_size
            nn.ReLU(),
            nn.MaxPool2d((max_length-2,1)),
        )
        self.dropout = nn.Dropout(Dropout)
        self.fc = nn.Linear(out_channel, 2) # 二分类

    def forward(self, x):
        # x:batch_size, seq_len
        batch_size = x.shape[0]
        embedding_x = word2vec(x) # 进行word2vec词向量嵌入
        # embedding_x:batch_size, sequence_length, embedding_size
        embedding_x = embedding_x.unsqueeze(1)
        # embedding_x:batch_size, 1, sequence_length, embedding_size
        conved = self.conv(embedding_x)
        # conved:batch_size,10,seq_len-1,1 -> batch_size,10,seq_len-1,1 -> batch_size,10,1,1
        conved = self.dropout(conved)
        flatten = conved.view(batch_size, -1)
        # flatten:batch_size, 10
        output = self.fc(flatten)
        return output

textCNN = TextCNN()
optimizer = optim.Adam(textCNN.parameters(),lr=Learning_rate)
criterion = torch.nn.CrossEntropyLoss()

TrainDataSet = TextCNNDataSet(data_input, list(train_data["label"]))
TrainDataLoader = DataLoader(TrainDataSet, batch_size=Batch_Size, shuffle=True)

def train():
    total_loss = []
    for epoch in range(20):
        epoch_loss = 0
        for index, (batch_x, batch_y) in enumerate(TrainDataLoader):
            pred = textCNN(batch_x)
            loss = criterion(pred, batch_y)
            total_loss.append(loss.data)
            epoch_loss+=loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print("epoch",epoch,":",epoch_loss)
    plt.plot(total_loss,"r-")
    plt.show()
train()

