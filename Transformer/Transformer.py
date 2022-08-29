import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import matplotlib.pyplot as plt
import numpy as np
import copy

d_model = 512
dropout = 0.1
max_len = 60
# 文本嵌入层
class Embedding(nn.Module):
    def __init__(self, d_model, vocab): #d_model为词嵌入的维度，vocab为词表大小
        super(Embedding, self).__init__()
        # 定义Embedding层
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x): #x为输入模型的文本通过此映射之后的数字张良
        return self.lut(x) * math.sqrt(self.d_model)
# 位置编码器
class PositionalEncoding(nn.Module):
    def __init__(self,d_model, dropout, max_len=5000):
        # d_model：词嵌入的维度
        # dropout：Dropout层的置零比率
        # max_len：句子的长度
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe = torch.zeros(max_len,d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0,d_model,2)*-(math.log(10000.0)/d_model))

        self.pe[:,0::2] = torch.sin(position*div_term)
        self.pe[:,1::2] = torch.cos(position*div_term)
        self.pe = self.pe.unsqueeze(0)

        # 注册为buffer，再模型保存后将这个位置编码器和其参数读取进来
        self.register_buffer('pe',self.pe)

    def forward(self, x):
        x = x + Variable(self.pe[:,:x.size(1)], requirez_grad=False)
        return self.dropout(x)

# 构建掩码张量
def subsequent_mask(size):
    # size为掩码张量的后两个维度，形成一个方阵
    attn_shape = (1, size, size)
    # 形成一个上三角矩阵
    subsequent_mask = np.triu(np.ones(attn_shape),k = 1).astype('uint8')
    # 便为下三角矩阵
    return torch.from_numpy(1 - subsequent_mask)

size = 5
sm = subsequent_mask(size)
print(sm)
print((sm.shape))

plt.figure(figsize=(5,5))
plt.imshow(subsequent_mask(20)[0])
plt.show()