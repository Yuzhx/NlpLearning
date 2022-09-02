import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import transpose
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

def attention(query, key, value, mask=None, dropout=None): # 注意力机制
    # 三个输入张量：query, key, value
    # mask：掩码张量
    # dropout：已经实例化的dropout层

    # 提取query的最后一个维度，即词嵌入维度
    d_k = query.size(-1)

    # 将query和key的转置进行矩阵乘法，然后除以缩放系数
    scores = torch.matmul(query, key.transpose(-2,-1))/math.sqrt(d_k)

    # mask
    if mask is not None:
        scores = scores.mask_fill(mask == 0, -1e9)
    # 对scores的最后一个维度进行softmax操作
    p_attn = F.softmax(scores, dim=-1)
    # dropout
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

def clone(module, N): # 克隆函数，在多头注意力机制下，需要获得多个结构相同的先行曾，该函数可以将其同时初始化，返回一个网络层列表对象
    # module：实例化的目标网络层对象
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class MultiHeadedAttention(nn.Module): # 多头注意力机制
    def __init__(self, head, embedding_dim, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert embedding_dim % head == 0
        # 得到每个头的词向量的维度数目
        self.d_k = embedding_dim//head
        self.head = head
        self.embedding_dim = embedding_dim

        # 获得4个线性层，对于Q、K、V以及最后的输出层
        self.linears = clone(nn.Linear(embedding_dim, embedding_dim), 4)

        # 初始化注意力张量
        self.attn = None

        # 初始化dropout对象
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)

        batch_size = query.size(0)
        query, key, value = \
          [model(x).view(batch_size, -1, self.head, self.d_k).transpose(1,2)
            for model, x in zip(self.linears, (query, key, value))]

        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        # 每个头的计算结果是4维张量
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.head * self.d_k)
        return  self.linears[-1](x)

class PositionwiseFeedForward(nn.Module): # 前馈全连接层
    def __init__(self, d_model, d_ff, dropout=0.1):
        # d_model：词嵌入维度，同时也是输入和输出维度
        # d_ff：第一个先行曾的输入，第二个线性层的输出
        super(PositionwiseFeedForward, self).__init__()

        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

class LayerNorm(nn.Module): # 规范化层
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a2 = nn.Parameter(torch.ones(features))
        self.b2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a2 * (x-mean)/(std+self.eps)+self.b2

class SublayerConnection(nn.Module): # 子层
    def __init__(self,size, dropout=0.1):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(p=dropout)
    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clone(SublayerConnection(size, dropout),2)
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)

# 以下对编码器进行实现
class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clone(layer, N)
        self.norm =  LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        # size：词嵌入唯独大小
        # self_attn：多头自注意力对象
        # src_attn：多头注意力对象
        # feed_forward：前馈全连接层对象
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clone(SublayerConnection(size,dropout),3)

    def forward(self, x, mermony, source_mask, target_mask):
        x = self.sublayer[0](x, lambda  x: self.self_attn(x, x, x, target_mask))
        x = self.sublayer[1](x, lambda  x: self.src_attn(x, mermony, mermony, source_mask))
        return  self.sublayer[2](x, self.feed_forward)

# 以下对解码器进行实现
class Decoder(nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clone(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, source_mask, target_mask):
        for layer in self.layers:
            x = layer(x, memory, source_mask, target_mask)
        return  self.norm(x)

# 以下对生成器进行实现
class Genrator(nn.Module):
    def __init__(self, d_model, vocab_size):
        super(Genrator, self).__init__()
        self.project = nn.Linear(d_model, vocab_size)
    def forward(self, x):
        return  F.log_softmax(self.project(x), dim=-1)



