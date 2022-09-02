import math

import numpy as np
import torch
import torch.nn as nn

def gelu(x):
    return x*0.5*(1.0+torch.erf(x/math.sqrt(2.0)))

def get_attn_pad_mask(seq_q, seq_k):
    batch_size, seq_len = seq_q.size()
    pad_attn_mask = seq_len.data.eq(0).unsqueeze(1)
    # pad_attn_mask: [batch_size, 1, seq_len]
    return pad_attn_mask.expand(batch_size, seq_len, seq_len)
    # return: [batch_size, seq_len, seq_len]
class Embedding(nn.Module):
    def __init__(self, vocab_size, maxlen, n_segments, d_model):
        super(Embedding, self).__init__()
        self.tok_embed = nn.Embedding(vocab_size, d_model)  # token嵌入
        self.pos_embed = nn.Embedding(maxlen, d_model)  #  位置信息编码的嵌入
        self.seg_embed = nn.Embedding(n_segments, d_model)  # segment嵌入
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, seg):
        seq_len = x.size(1)
        pos = torch.arange(seq_len, dtype=torch.long)
        pos = pos.unsqueeze(0).expand_as(x)  # [len_seq] -> [batch_size, len_seq]
        embedding = self.tok_embed(x) + self.pos_embed(pos) + self.seg_embed(seg)
        return self.norm(embedding)

class ScaledDotProductAttention(nn.Module): # DotProduct函数
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, d_k, attn_mask):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k) # scores : [batch_size, n_heads, seq_len, seq_len]
        scores.masked_fill_(attn_mask, -1e9) # Fills elements of self tensor with value where mask is one.
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v, n_heads):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.W_Q = nn.Linear(d_model, d_k * n_heads)
        self.W_K = nn.Linear(d_model, d_k * n_heads)
        self.W_V = nn.Linear(d_model, d_v * n_heads)
    def forward(self, Q, K, V, attn_mask):
        # q: [batch_size, seq_len, d_model], k: [batch_size, seq_len, d_model], v: [batch_size, seq_len, d_model]
        residual, batch_size = Q, Q.size(0)
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        q_s = self.W_Q(Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,2)  # q_s: [batch_size, n_heads, seq_len, d_k]
        k_s = self.W_K(K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,2)  # k_s: [batch_size, n_heads, seq_len, d_k]
        v_s = self.W_V(V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1,2)  # v_s: [batch_size, n_heads, seq_len, d_v]

        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1) # attn_mask : [batch_size, n_heads, seq_len, seq_len]

        # context: [batch_size, n_heads, seq_len, d_v], attn: [batch_size, n_heads, seq_len, seq_len]
        context = ScaledDotProductAttention()(q_s, k_s, v_s, attn_mask)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_v) # context: [batch_size, seq_len, n_heads, d_v]
        output = nn.Linear(self.n_heads * self.d_v, self.d_model)(context)
        return nn.LayerNorm(self.d_model)(output + residual) # output: [batch_size, seq_len, d_model]

class PositionwiseFeedForward(nn.Module): # 前馈全连接层
    def __init__(self, d_model, d_ff):
        super(PositionwiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_ff) -> (batch_size, seq_len, d_model)
        return self.fc2(gelu(self.fc1(x)))

class EncoderLayer(nn.Module): # 编码器层
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = PositionwiseFeedForward()

    def forward(self, enc_inputs, enc_self_attn_mask):
        enc_self_attned = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask) # enc_inputs to same Q,K,V
        enc_outputs = self.pos_ffn(enc_self_attned) # enc_outputs: [batch_size, seq_len, d_model]
        return enc_outputs

class BERT(nn.Module):
    def __init__(self, d_model, n_layers, vocab_size):
        super(BERT, self).__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.vocab_size = vocab_size
        self.embedding = Embedding()
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Dropout(0.5),
            nn.Tanh(),
        )
        self.classifier = nn.Linear(d_model, 2)
        self.linear = nn.Linear(d_model, d_model)
        self.activ2 = gelu
        # fc2 is shared with embedding layer
        embed_weight = self.embedding.tok_embed.weight
        self.fc2 = nn.Linear(d_model, vocab_size, bias=False)
        self.fc2.weight = embed_weight

    def forward(self, input_ids, segment_ids, masked_pos):
        embedded = self.embedding(input_ids, segment_ids)
        # embedded: [bach_size, seq_len, d_model]
        enc_self_attn_mask = get_attn_pad_mask(input_ids, input_ids)
        # enc_self_attn_mask: [batch_size, maxlen, maxlen]
        for layer in self.layers:
            # output: [batch_size, max_len, d_model]
            layer_output = layer(embedded, enc_self_attn_mask)
        h_pooled = self.fc(layer_output[:, 0])
        # h_pooled: [batch_size, d_model]
        logits_clsf = self.classifier(h_pooled)
        # logits_clsf: [batch_size, 2]

        masked_pos = masked_pos[:, :, None].expand(-1, -1, self.d_model)
        # masked_pos: [batch_size, max_pred, d_model]
        h_masked = torch.gather(layer_output, 1, masked_pos)
        # h_masked: masking position [batch_size, max_pred, d_model]
        actived = self.activ2(self.linear(h_masked))
        # actived: [batch_size, max_pred, d_model]
        logits_lm = self.fc2(actived)
        # logits_lm: [batch_size, max_pred, vocab_size]
        return logits_lm, logits_clsf