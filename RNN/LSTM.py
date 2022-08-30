from torch import nn

# rnn LSTM模型
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, hidden_layers, output_size):
        super(LSTM, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, hidden_layers)
        # 定义回归层网络，输入的特征维度等于LSTM的输出，输出维度为1
        self.reg = nn.Sequential(
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        x, (ht,ct) = self.rnn(x)
        seq_len, batch_size, hidden_size= x.shape
        x = x.view(-1, hidden_size)
        x = self.reg(x)
        x = x.view(seq_len, batch_size, -1)
        return x