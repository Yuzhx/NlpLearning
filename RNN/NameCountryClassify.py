# 使用RNN实现人名-国籍的分类
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import gzip
import csv
import matplotlib.pyplot as plt

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
            self.MakeNameTensor()
            print("数据集初始化完毕，is_train=",is_train)

    def __getitem__(self, index):
        return self.names[index], self.countries[index]

    def MakeNameTensor(self):
        # 首先处理countries
        self.countries = [self.country_dict[country] for country in self.countries] # 将每个国的国名转化为数字列表
        self.countries = torch.LongTensor(self.countries) # 将数字列表转为Tensor

        # 其次处理names
        namearr_and_namelens = [self.Name2arr(name) for name in self.names] # 首先取得每一个name的数组形式以及长度
        # namearrs存放每个name的数组形式，即每个name是一个数组namearr，数组之中存放的是其每个字符减去a的差，多个数组构成列表namearrs
        namearrs = [nans[0] for nans in namearr_and_namelens]
        # namelens 存放每个name的长度，即数组长
        self.namelens = torch.LongTensor([nans[1] for nans in namearr_and_namelens])
        name_tensor = torch.zeros(len(self.names),self.namelens.max()).long() # 生成num_names行，seq_names列的全0矩阵
        for idx,(namearr,namelen) in enumerate(zip(namearrs,self.namelens),0): # 将第idx的前namelens[idx]列填充
            name_tensor[idx, :namelen] = torch.LongTensor(namearr)

        # 最后进行排序，按照namelens降序，对namelens、names、countries三个变量同步排序
        self.namelens, perm_idx = self.namelens.sort(dim=0, descending=True) # perm_idx可以用于排序另外两个tensor
        self.names = name_tensor[perm_idx]
        self.countries = self.countries[perm_idx]

        return


    def Name2arr(self,name): # 将一个名字转化为一个数组
        arr = [(ord(c)-ord('a')+1) for c in name]
        return arr, len(arr)

dataset = Dataset()




