import torch
import matplotlib.pyplot as plt

x = torch.unsqueeze(torch.linspace(-2, 2, 300), dim=1)  # 生成x序列
y = 0.4 * x.pow(4) - 0.8 * x.pow(3) - x.pow(2) + x + torch.randn(x.size())  # 生成y序列


class Net(torch.nn.Module):  # 定义神经网络
    def __init__(self):
        super(Net, self).__init__()
        self.hidden1 = torch.nn.Linear(1, 40)
        self.hidden2 = torch.nn.Linear(40, 10)
        self.regression = torch.nn.Linear(10, 1)

    def forward(self, x):
        x = torch.relu(self.hidden1(x))
        x = torch.sigmoid(self.hidden2(x))
        output = self.regression(x)
        return output


net = Net()  # 实例化神经网络
optimizer = torch.optim.Adam(net.parameters(), lr=0.05)  # 实例化优化器：Adam
loss_func = torch.nn.MSELoss()  # 损失函数：Mean Square Error
loss_all = []
for t in range(300):  # 不用batch的方式
    pred = net(x)
    loss = loss_func(pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    loss_all.append(loss.data)
plt.plot(loss_all, "r-")
plt.show()
