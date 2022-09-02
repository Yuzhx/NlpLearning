import torch.nn as nn
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset


class LSTMModule(nn.Module):
    def __init__(self, len_feature, num_words, len_hidden, num_to_word, word_to_num, strategy='lstm', pad_id=0, start_id=1, end_id=2, drop_out=0.5):
        # 特征长度，句子长度，隐层宽度，
        super(LSTMModule, self).__init__()
        # 初始化基本操作
        self.pad_id = pad_id
        self.start_id = start_id
        self.end_id = end_id
        self.num_to_word = num_to_word
        self.word_to_num = word_to_num
        self.len_feature = len_feature
        self.len_words = num_words
        self.len_hidden = len_hidden

        self.dropout =nn.Dropout(drop_out) # dropout层
        init_weight = nn.init.xavier_normal_(torch.Tensor(num_words, len_feature)) # 词嵌入向量初始化
        self.embedding = nn.Embedding(num_embeddings=num_words, embedding_dim=len_feature, _weight=init_weight) # 词嵌入层
        self.lstm = nn.LSTM(input_size=len_feature, hidden_size=len_hidden, batch_first=True) # LSTM层
        self.fc = nn.Linear(len_hidden, num_words) # 全连接层

    def forward(self, x):
        embedded = self.embedding(x)
        dropouted = self.dropout(embedded)
        self.lstm.flatten_parameters()
        lstmed, _ = self.lstm(dropouted)
        output = self.fc(lstmed)

        return output

    def make_poem(self, heads, max_len=50, random=False):
        for head in heads:
            if head not in self.word_to_num:
                raise Exception("Word: " + head + " is not in the dictionary, please try another word")
            poem = list()
            if random:
                initialize = torch.randn
            else:
                initialize = torch.zeros
            for i in range(len(heads)):
                head_digit = self.word_to_num[heads[i]]
                sentence = [heads[i]]
                hn = initialize((1, 1, self.len_hidden))
                cn = initialize((1, 1, self.len_hidden))
                for j in range(max_len-1): # 开始作诗
                    head_digit = torch.LongTensor([head_digit])
                    embedded = self.embedding(head_digit).view(1,1,-1)
                    lstmed, (hn, cn) = self.lstm(embedded, (hn,cn))
                    output_digit = self.fc(lstmed)
                    best_digit = output_digit.topk(1)[1][0].item() #选择最有可能的
                    word = self.num_to_word[best_digit]
                    sentence.append(word)
                    if '。' == word:
                        break
                poem.append(sentence)
        return poem





