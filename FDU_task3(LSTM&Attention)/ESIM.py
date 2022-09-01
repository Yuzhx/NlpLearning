import torch.nn as nn
import torch
import torch.nn.functional as F
class BiLSTM(nn.Module): # BiLSTM单元
    def __init__(self, input_size, hidden_size=128, dropout_rate=0.1, layer_num=1):
        super(BiLSTM, self).__init__()
        self.hidden_size = hidden_size
        # input_size:输入数据的向量维数 hidden_size:lstm隐藏层的特征维度 layer_num:隐藏层的层数
        if layer_num == 1:
            self.bilstm = nn.LSTM(input_size, hidden_size//2, layer_num, batch_first=True, bidirectional=True)
        else:
            self.bilstm = nn.LSTM(input_size, hidden_size//2, layer_num, batch_first=True, dropout=dropout_rate, bidirectional=True)
        self.init_weights()

    def init_weights(self): # 初始化参数
        for p in self.bilstm.parameters():
            if p.dim() > 1:
                nn.init.normal_(p)
                p.data.mul_(0.01)
            else:
                p.data.zero_()
                p.data[self.hidden_size//2: self.hidden_size] = 1

    def forward(self, x, lens):
        # x: [batch, len_seq, d_model]
        # lens: [barch] (存放batch之中每个句子的长度的数组，用于指导pack_padded_sequence)
        ordered_lens, pram = lens.sort(descending=True) # 按照句子长度从大到小排序
        ordered_x = x[pram]
        # 执行pack_padded_sequence
        packed_x = nn.utils.rnn.pack_padded_sequence(ordered_x, ordered_lens, batch_first=True)
        packed_output, _ = self.bilstm(packed_x)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

        recover_pram = pram.argsort()
        recover_output = output[recover_pram]
        return recover_output

class Input_Encoding(nn.Module): # 输入encoding层，其中包括进行embedding，执行dropout，送入LSTM单元
    def __init__(self, num_features, hidden_size, embedding_size, vectors,
                 num_layers=1,  drop_out=0.5):
        super(Input_Encoding, self).__init__()
        self.num_features = num_features
        self.num_hidden = hidden_size
        self.embedding_size = embedding_size
        self.num_layers = num_layers
        self.dropout = nn.Dropout(drop_out)
        self.embedding = nn.Embedding.from_pretrained(vectors)
        self.bilstm = BiLSTM(embedding_size, hidden_size, drop_out, num_layers)

    def forward(self, x, lens):
        # x = torch.LongTensor(x)
        embedded = self.embedding(x)
        dropouted = self.dropout(embedded)
        output = self.bilstm(dropouted, lens)
        return output
        # output: [batch, len_seq, hidden_size]

class Local_Inference_Modeling(nn.Module):
    def __init__(self):
        super(Local_Inference_Modeling, self).__init__()
        self.softmax_1 = nn.Softmax(dim=1)
        self.softmax_2 = nn.Softmax(dim=2)

    def forward(self, sentences_a, sentences_b):
        # 一个玄学的矩阵乘法，I don't understand it
        # sentences_a: [batch, seq_len_A, hidden_size]
        # sentences_b: [batch, seq_len_B, hidden_size]

        e = torch.matmul(sentences_a, sentences_b.transpose(1, 2))

        a_tilde = (self.softmax_2(e)).bmm(sentences_b)
        b_tilde = (self.softmax_1(e).transpose(1, 2)).bmm(sentences_a)

        output_a = torch.cat([sentences_a, a_tilde, sentences_a - a_tilde, sentences_a * a_tilde], dim=-1)
        output_b = torch.cat([sentences_b, b_tilde, sentences_b - b_tilde, sentences_b * b_tilde], dim=-1)

        return output_a, output_b
        # return: [batch, seq_len_A, hidden_size * 4], [batch, seq_len_B, hidden_size * 4]

class Inference_Composition(nn.Module):
    def __init__(self, hidden_size, num_layers, drop_out=0.5):
        super(Inference_Composition,self).__init__()
        self.linear = nn.Linear(4 * hidden_size, hidden_size)
        self.bilstm = BiLSTM(hidden_size, hidden_size, drop_out, num_layers)
        self.drop_out = nn.Dropout(drop_out)

    def forward(self, x, lens):
        # x: [batch, seq_len, hidden_size * 4]
        # lens: [seq_len, ]
        lineared = self.linear(x)
        dropouted = self.drop_out(lineared)
        output = self.bilstm(dropouted, lens)

        return output
        # output: [batch, seq_len, hidden_size]

class Prediction(nn.Module):
    def __init__(self, v_size, mid_size, num_classes=4, drop_out=0.5):
        super(Prediction, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(v_size, mid_size), nn.Tanh(),
            nn.Linear(mid_size, num_classes)
        )

    def forward(self, a, b):
        # a: [batch, seq_len_A, hidden_size]
        # b: [batch, seq_len_B, hidden_size]
        v_a_avg = F.avg_pool1d(a, a.size(2)).squeeze(-1)
        v_a_max = F.max_pool1d(a, a.size(2)).squeeze(-1)

        v_b_avg = F.avg_pool1d(b, b.size(2)).squeeze(-1)
        v_b_max = F.max_pool1d(b, b.size(2)).squeeze(-1)

        out_put = torch.cat((v_a_avg, v_a_max, v_b_avg, v_b_max), dim=-1)

        return self.mlp(out_put)
        # return: [batch, num_classes]

class ESIM(nn.Module):
    def __init__(self, num_features, hidden_size, embedding_size, num_classes=4, vectors=None,
                 num_layers=1,  drop_out=0.5):
        super(ESIM, self).__init__()
        self.embedding_size = embedding_size
        self.input_encoding = Input_Encoding(num_features, hidden_size, embedding_size, vectors, num_layers=1, batch_first=True, drop_out=0.5)
        self.local_inference = Local_Inference_Modeling()
        self.inference_composition = Inference_Composition(num_features, 4 * hidden_size, hidden_size, num_layers, embedding_size=embedding_size, batch_first=True, drop_out=0.5)
        self.prediction = Prediction(4 * hidden_size, hidden_size, num_classes, drop_out)

        def forward(self, sentences_a, len_a, sentences_b, len_b):
            a_encoded = self.input_encoding(sentences_a, len_a)
            b_encoded = self.input_encoding(sentences_b, len_b)
            # encoded: [batch, len_seq, hidden_size]
            a_local_inferenced, b_local_inferenced = self.local_inference(a_encoded, b_encoded)
            # local_inferenced: [batch, seq_len, hidden_size * 4]
            v_a = self.inference_composition(a_local_inferenced, len_a)
            v_b = self.inference_composition(b_local_inferenced, len_b)
            # v: [batch, seq_len, hidden_size]
            out_put = self.prediction(v_a, v_b)
            # [batch, num_classes]
            return out_put
