from GetFeature import get_batch, get_feature
from torch import optim
import random,numpy,torch
from LSTM import LSTMModule
import torch.nn.functional as F
import matplotlib.pyplot as plt

random.seed(2021)
numpy.random.seed(2021)
torch.manual_seed(2021)

with open('poetryFromTang.txt', 'rb') as f: # 读取文件
    raw_data = f.readlines()

Datalist = get_feature(raw_data)
Datalist.get_digitpoems()
train = get_batch(Datalist.digitpoems,1)

lstm_model = LSTMModule(50, len(Datalist.word2index), 50, Datalist.index2word, Datalist.word2index)
optimizer = optim.Adam(lstm_model.parameters(), lr=0.004)
loss_func = F.cross_entropy

total_loss = []
for epoch in range(200):
    epoch_loss = 0
    i_num =0
    for i, batch in enumerate(train):
        x, x_next = batch[:,:-1], batch[:,1:] # 分别为诗的第1~倒数第2个字以及第2~倒数第1个字
        pred = lstm_model(x).transpose(1,2)
        optimizer.zero_grad()
        loss = loss_func(pred, x_next)
        epoch_loss += loss.item()
        i_num += 1
        loss.backward()
        optimizer.step()
    total_loss.append(epoch_loss/i_num)
    print(epoch_loss/i_num)

plt.plot(total_loss,"r-")
plt.show()

while True:
    heads = list()
    print("input heads, end with 。")
    while True:
        head = input()
        if (head != "。"):
            heads.append(head)
            print(head)
        else:
            break
    poem = lstm_model.make_poem(heads,max_len=20,random=True)
    print(poem)




