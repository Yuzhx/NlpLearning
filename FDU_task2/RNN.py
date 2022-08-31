import GetFeature
import random
from torch import optim
from torch.utils.data import Dataset, DataLoader
import csv
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import torch

train, test = GetFeature.DataInit()
dict_words = GetFeature.DictInit(train,test)
sentences = [row[0] for row in train]
sentences_tensor = GetFeature.Word2Tensor(sentences, dict_words)
class Embedding(nn.Module):
    def __init__(self, d_model, vocab): #d_model为词嵌入的维度，vocab为词表大小
        super(Embedding, self).__init__()
        # 定义Embedding层
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model
    def forward(self, x): #x为输入模型的文本通过此映射之后的数字张良
        return self.lut(x)
class RNN(nn.Module):
    def __init__(self,len_feature, len_hidden, layer=1, drop_out=0.5, nonlinearity='tanh', batch_first=True, type_num=5):
        super(RNN, self).__init__()
        self.len_feature = len_feature # 特征向量长度
        self.len_hidden = len_hidden # 隐层宽度
        self.layer = layer # 隐层层数
        self.embedding = Embedding(len_feature,len(dict_words))
        self.dropout = nn.Dropout(drop_out) # drop_out层
        self.rnn = nn.RNN(input_size=len_feature, hidden_size=len_hidden, num_layers=layer, nonlinearity=nonlinearity, batch_first=batch_first,dropout=drop_out)
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

trainset = GetFeature.Dataset(train, dict_words)
trainloader = DataLoader(trainset, batch_size=500, shuffle=True)
loss_func = torch.nn.CrossEntropyLoss()
rnn = RNN(50, 50)
optimizer = optim.Adam(rnn.parameters(), lr=0.001)
def Train():
    for epoch in range(30):
        epoch_loss = 0
        for i, (sentences_batch, emotions_batch) in enumerate(trainloader,1):
            # sentences_batch: [batch_size,max_length]
            # emotions_batch: [batch_size,]
            pred = rnn(sentences_batch)

            loss = loss_func(pred, emotions_batch)
            optimizer.zero_grad()
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
        print(epoch,":",epoch_loss)
Train()