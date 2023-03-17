# 任务二：基于深度学习的文本分类

> 任务要求：熟悉Pytorch，用Pytorch重写《任务一》，实现CNN、RNN的文本分类

本任务的重点包含CNN、RNN进行特征提取以及词向量的初始化两个方面。

## embedding

embedding方式分为两种，包括随机初始化和预训练的词向量初始化：

1. 预训练的初始化：预训练的初始化是指将预训练的 word embedding 加载到模型中作为初始化。例如，可以使用已经训练好的 word2vec、GloVe 或 BERT 等预训练模型，将它们的参数直接作为神经网络的初始参数，然后继续进行训练。这种方法通常能够提高模型的效果，特别是在数据较少的情况下。
2. 随机初始化：随机初始化是指将 word embedding 的参数随机初始化为一组较小的随机数。这种方法通常用于在没有预训练数据的情况下训练神经网络，其优点是可以在训练过程中更新参数以更好地适应任务。通常使用高斯分布或均匀分布来初始化参数，以确保随机性。

在pytorch之中，实现预训练的初始化，实际上就是用预训练的权重向量来给embedding层进行赋值，如下所示：

```python
nn.Embedding.from_pretrained(weights)
```

其中weights是预训练的权重向量

```python
model = Word2Vec.load(path)
weights = torch.FloatTensor(model.wv.vectors)
```

其中path为预训练模型的路径。

在近期的科研之中，我一般使用的都是基于bert的embedding：

```python
nodes_bert = []
maxlength = 0
tokenizer = BertTokenizer.from_pretrained("bert-large-cased", cache_dir="/root/autodl-tmp/GCN/bert_cache", do_lower_case=False)
for sent in nodes_str:
    # 将source和replies的句子转换为可以送入bert之中的格式
    bert_tokens = ['[CLS]'] + tokenizer.tokenize(sent) + ['[SEP]']
    bert_tokens = tokenizer.convert_tokens_to_ids(bert_tokens)
    if len(bert_tokens) > maxlength:
        maxlength = len(bert_tokens)
        nodes_bert.append(bert_tokens)
        berts = []
        masks = []
for bert_tokens in nodes_bert:
    # 将bert的id进行padding得到mask
    bert_tokens, mask = padding(bert_tokens, maxlength)
    berts.append(bert_tokens)
    masks.append(mask)
    berts = berts  # [num_sents, max_len]
    masks = masks  # [num_sents, max_len]
    berts = torch.tensor(berts)
    masks = torch.tensor(masks)
```

## CNN

CNN来应用于序列信息（自然语言）的特征提取最早在[[1408.5882\] Convolutional Neural Networks for Sentence Classification (arxiv.org)](https://arxiv.org/abs/1408.5882)一文中提出，如下图所示

<img src="https://yzx-drawing-bed.oss-cn-hangzhou.aliyuncs.com/img/image-20230317091850340.png" alt="image-20230317091850340" style="zoom: 67%;" />

对于一个句子，将其embedding为规模为[max_len, embedding_dim]的向量之后，使用cnn沿着第一个维度进行卷积，其中可以设置不同卷积核的宽度，卷积核的宽度形象地来说，可以认为是一次卷积最多可以看到多少个相邻的单词，例如上图中的黄色卷积核，便一次看到了"video and do"三个单词的信息。

在代码中我设置了宽度分别为2、3、4、5的四个卷积核，如下所示：

```python
# 不同卷积核宽度的四个卷积层
self.conv1 = nn.Sequential(nn.Conv2d(1, max_len, (2, len_feature), padding=(1, 0)), nn.ReLU())
self.conv2 = nn.Sequential(nn.Conv2d(1, max_len, (3, len_feature), padding=(1, 0)), nn.ReLU())
self.conv3 = nn.Sequential(nn.Conv2d(1, max_len, (4, len_feature), padding=(2, 0)), nn.ReLU())
self.conv4 = nn.Sequential(nn.Conv2d(1, max_len, (5, len_feature), padding=(2, 0)), nn.ReLU())
```

该网络的架构十分简单：

```python
# 1表示的是通道数目为1
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
```

分别将embedding之后的句子向量送进四个不同卷积核的卷积层，分别池化之后，将其进行拼接得到最终的表示句子特征的向量。

## RNN

相比于CNN，RNN可以更好地捕捉序列中的时间依赖性。由于自然语言具有序列性质，因此在处理文本时，序列中的前后关系很重要。而RNN可以通过将前一时刻的隐藏状态作为当前时刻的输入，捕捉到时间依赖性，从而更好地处理序列数据。

我所设置的基于RNN的网络如下所示：

```python
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
```

其中值得注意的是rnn的输出包括hn和out，其中前者是网络的最终隐藏状态，后者是每个时间步的输出，此时所选择的是hn，这一输出也常用于表示整个句子的特征。

