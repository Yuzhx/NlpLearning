import torch.nn as nn
import torch
from torch import optim

import GetFeature
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

train, test = GetFeature.DataInit()
dict_words = GetFeature.DictInit(train,test)
sentences = [row[0] for row in train]
sentences_tensor = GetFeature.Word2Tensor(sentences, dict_words)

def get_longest(train, test):
    longest = 0
    total_data = train + test
    for sentence in total_data:
        words = sentence[0].split()
        longest = max(longest, len(words))
    return  longest

longest = get_longest(train, test)

class Embedding(nn.Module):
    def __init__(self, d_model, vocab): #d_model为词嵌入的维度，vocab为词表大小
        super(Embedding, self).__init__()
        # 定义Embedding层
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model
    def forward(self, x): #x为输入模型的文本通过此映射之后的数字张良
        return self.lut(x)
class CNN(nn.Module):
    def __init__(self, len_feature, max_len, typenum=5, weight=None, drop_out=0.5):
        super(CNN, self).__init__()
        self.len_feature = len_feature
        self.max_len = max_len
        self.dropout = nn.Dropout(drop_out)
        self.embedding = Embedding(len_feature, len(dict_words))

        self.conv1 = nn.Sequential(nn.Conv2d(1, max_len, (2, len_feature), padding=(1, 0)), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(1, max_len, (3, len_feature), padding=(1, 0)), nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(1, max_len, (4, len_feature), padding=(2, 0)), nn.ReLU())
        self.conv4 = nn.Sequential(nn.Conv2d(1, max_len, (5, len_feature), padding=(2, 0)), nn.ReLU())

        self.fc = nn.Linear(4*max_len, typenum)

    def forward(self, x):
        # x: [batch_size, max_length]
        embedded = self.embedding(x)
        # embedded: [batch_size, max_length, d_model]
        dropouted = self.dropout(embedded)
        # dropouted: [batch_size, max_length, d_model]
        expended = dropouted.unsqueeze(1)
        # expended: [batch_size, 1, max_length, d_model]
        conv1 = self.conv1(expended).squeeze(3)
        conv2 = self.conv2(expended).squeeze(3)
        conv3 = self.conv3(expended).squeeze(3)
        conv4 = self.conv4(expended).squeeze(3)

        pool1 = F.max_pool1d(conv1, conv1.shape[2])
        pool2 = F.max_pool1d(conv2, conv2.shape[2])
        pool3 = F.max_pool1d(conv3, conv3.shape[2])
        pool4 = F.max_pool1d(conv4, conv4.shape[2])
        pool = torch.cat([pool1, pool2, pool3, pool4], 1).squeeze(2)
        out_put = self.fc(pool)  # 全连接层
        # out_put = self.act(out_put)  #softmax层可以不加
        return out_put

trainset = GetFeature.Dataset(train, dict_words)
trainloader = DataLoader(trainset, batch_size=500, shuffle=True)
loss_func = torch.nn.CrossEntropyLoss()
cnn = CNN(50, longest)
optimizer = optim.Adam(cnn.parameters(), lr=0.001)

def Train():
    for epoch in range(30):
        epoch_loss = 0
        for i, (sentences_batch, emotions_batch) in enumerate(trainloader,1):
            # sentences_batch: [batch_size,max_length]
            # emotions_batch: [batch_size,]
            pred = cnn(sentences_batch)

            loss = loss_func(pred, emotions_batch)
            optimizer.zero_grad()
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
        print(epoch,":",epoch_loss)
Train()