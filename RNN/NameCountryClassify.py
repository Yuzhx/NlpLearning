# 使用RNN实现人名-国籍的分类
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pack_padded_sequence
import gzip
import csv
import matplotlib.pyplot as plt

N_CHAR = 128
HIDDEN_SIZE = 100
BATCH_SIZE = 256


class Dataset(Dataset):
    def __init__(self,is_train = True):
        filename = 'names_train.csv.gz' if is_train else 'names_test.csv.gz'
        with gzip.open(filename,'rt') as file:
            reader = csv.reader(file)
            rows = list(reader)
            # 读取数据集之中的人名以及对应的国家，转换为对应的列表
            self.names = [row[0].lower() for row in rows] #将所有人名转化为小写，因为整个数据集之中全部都是只有第一个字母为大写
            self.countries = [row[1] for row in rows]
            self.len = len(rows)
            # 生成国家字典：set去重，sort排序，list生成列表
            self.country_list = list(sorted(set(self.countries)))
            self.country_dict = dict()
            for idx, country_name in enumerate(self.country_list,0):
                self.country_dict[country_name] = idx
            self.country_num = len(self.country_list)
            # 生成国家名数字列表
            self.countries = [self.country_dict[country] for country in self.countries]  # 将每个国的国名转化为数字列表
            self.countries = torch.LongTensor(self.countries)  # 将数字列表转为Tensor
            print("数据集初始化完毕，is_train =",is_train)

    def __getitem__(self, index):
        return self.names[index], self.countries[index]

    def __len__(self):
        return self.len

    def getCountriesNum(self):
        return self.country_num





class Classifier(torch.nn.Module):
    def __init__(self,inputsize,hiddensize,outputsize,n_layers=1,bidirectional=True): # 输入向量的规模，隐层规模，输出向量的规模，隐层数目，是否为双向循环网络
        super(Classifier, self).__init__()
        self.hiddensize = hiddensize
        self.n_layers = n_layers
        self.n_directions = 2 if bidirectional else 1

        self.embedding = torch.nn.Embedding(inputsize, hiddensize) #embedding层
        self.gru = torch.nn.GRU(hiddensize, hiddensize, n_layers, bidirectional = bidirectional) # 循环层
        self.linearlayer = torch.nn.Linear(hiddensize * self.n_directions, outputsize) # 线性输出层

    def _init_hidden(self, batch_size):
        hidden = torch.zeros((self.n_layers * self.n_directions,
                              batch_size, self.hiddensize))
        return hidden

    def forward(self, input, namelens):
        # 输入的input矩阵的规模为B*S，其中B为batch的长度，S为这个一个batch之中的最长的单词的长度
        # 需要将其转换为S*B的规模
        input = input.t()
        batch_size = input.size(1)
        hidden = self._init_hidden(batch_size) # 初始化循环层的hidden矩阵

        embedding = self.embedding(input)
        gru_input = pack_padded_sequence(embedding, namelens) # 将padding过程之中所填充的0去掉，提供指导的是namelens数组
        output, hidden = self.gru(gru_input,hidden)
        if self.n_directions == 2:
            hidden_cat = torch.cat([hidden[-1], hidden[-2]], dim=1)
        else:
            hidden_cat = hidden[-1]
        classify_output = self.linearlayer(hidden_cat)
        return classify_output

def Name2arr(name): # 将一个名字转化为一个数组
    arr = [ord(c) for c in name]
    return arr, len(arr)

def MakeTensors(names,countries):
    # name提取
    namearr_and_namelens = [Name2arr(name) for name in names]  # 首先取得每一个name的数组形式以及长度
    # namearrs存放每个name的数组形式，即每个name是一个数组namearr，数组之中存放的是其每个字符减去a的差，多个数组构成列表namearrs
    namearrs = [nans[0] for nans in namearr_and_namelens]
    # namelens 存放每个name的长度，即数组长
    namelens = torch.LongTensor([nans[1] for nans in namearr_and_namelens])
    # 矩阵生成
    name_tensor = torch.zeros(len(names), namelens.max()).long()  # 生成num_names行，seq_names列的全0矩阵
    for idx, (namearr, namelen) in enumerate(zip(namearrs, namelens), 0):  # 将第idx的前namelens[idx]列填充
        name_tensor[idx, :namelen] = torch.LongTensor(namearr)
    # 矩阵排序，按照namelens降序，对namelens、names、countries三个变量同步排序
    namelens, perm_idx = namelens.sort(dim=0, descending=True)  # perm_idx可以用于排序另外两个tensor
    names = name_tensor[perm_idx]
    countries = countries[perm_idx]
    return names,countries,namelens


trainset = Dataset(is_train=True)
trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
testset = Dataset(is_train=False)
testloader = DataLoader(testset, batch_size=500, shuffle=True)
N_COUNTRIES = trainset.getCountriesNum()
criterion = torch.nn.CrossEntropyLoss()
RNNclassifier = Classifier(N_CHAR, HIDDEN_SIZE, N_COUNTRIES)
optimizer = torch.optim.Adam(RNNclassifier.parameters(), lr=0.01)
total_loss = []
total_test = []
def Train():
    for epoch in range(30):
        epoch_loss=0
        for i, (names, countries) in enumerate(trainloader,1):
            names, countries, namelens = MakeTensors(names,countries)
            output = RNNclassifier(names,namelens)
            loss = criterion(output, countries)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss+=loss.data
        total_loss.append(epoch_loss)
        Test()
        print(epoch,'+',epoch_loss)
def Test():
    total = 0
    correct = 0
    for i, (names, countries) in enumerate(testloader,1):
        names, countries, namelens = MakeTensors(names, countries)
        output = RNNclassifier(names, namelens)
        pred = output.max(dim=1, keepdim=True)[1]
        correct += pred.eq(countries.view_as(pred)).sum().item()
        total += countries.size(0)
    AccuracyRate = correct/total
    print('test',AccuracyRate)
    total_test.append(AccuracyRate)


Test()
Train()
plt.title("loss")
plt.plot(total_loss,"r-")
plt.show()
plt.title("test")
plt.plot(total_test,"r-")
plt.show()


