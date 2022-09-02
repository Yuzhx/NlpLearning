from torch.utils.data import Dataset, DataLoader
import torch
from torch.nn.utils.rnn import pad_sequence

class get_feature():
    def __init__(self, data):
        self.data = data
        self.poemlist = []
        self.word2index = {'<pad>': 0, '<begin>': 1, '<end>': 2}
        self.index2word = {0:'<pad>',1:'<begin>',2:'<end>'}
        self.digitpoems = list()

    def poem2list(self): # 将从文件读出的诗句转化为列表
        data_utf8 = list(map(lambda x, y: str(x, encoding=y), self.data, ['utf-8'] * len(self.data)))
        poems = list()
        new_poem = ""
        for item in data_utf8:
            if item == '\n':
                if new_poem:
                    poems.append(new_poem)
                new_poem = ""
            else:
                if item[-2] == ' ':
                    position = -2
                else:
                    position = -1
                new_poem = ''.join([new_poem, item[:position]])
        self.poemlist = poems

    def get_dict(self): # 生成双向字典
        for poem in self.poemlist:
            for word in poem:
                if word not in self.word2index:
                    self.index2word[len(self.word2index)]=word
                    self.word2index[word] = len(self.word2index)

    def get_digitlist(self): # 生成数码形式的诗句列表
        for poem in self.poemlist:
            self.digitpoems.append([self.word2index[word] for word in poem])

    def get_digitpoems(self):
        self.poem2list()
        self.poemlist.sort(key=lambda x: len(x))
        self.get_dict()
        self.get_digitlist()


# 以下使用了https://github.com/0oTedo0/NLP-Beginner中的dataset方案，最终的dataloader每次可以返回一个tensor，对应的是一首诗
class ClsDataset(Dataset):
    def __init__(self, poem):
        self.poem = poem

    def __getitem__(self, item):
        return self.poem[item]

    def __len__(self):
        return len(self.poem)


def collate_fn(batch_data):
    poems = batch_data
    poems = [torch.LongTensor([1, *poem]) for poem in poems]

    padded_poems = pad_sequence(poems, batch_first=True, padding_value=0)
    padded_poems = [torch.cat([poem, torch.LongTensor([2])]) for poem in padded_poems]
    padded_poems = list(map(list,padded_poems))
    return torch.LongTensor(padded_poems)


def get_batch(x, batch_size):
    dataset = ClsDataset(x)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True, collate_fn=collate_fn)
    return dataloader