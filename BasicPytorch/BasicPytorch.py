import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


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


class MyDataset(Dataset):  # 定义数据集
    def __init__(self, num_point): #初始化
        self.x = torch.unsqueeze(torch.linspace(-2, 2, num_point), dim=1)  # 生成x序列
        self.y = 0.4 * self.x.pow(4) - 0.8 * self.x.pow(3) - self.x.pow(2) + self.x + torch.randn(
            self.x.size())  # 生成y序列
        self.len = num_point

    def __getitem__(self, index): # 返回一条数据和对应的标签y
        return self.x[index], self.y[index]

    def __len__(self): # 返回数据集的长度
        return self.len


dataset = MyDataset(30000) # 实例化数据集
train_loader = DataLoader(dataset=dataset, batch_size=2000, shuffle=True, num_workers=3) # 实例化DataLoader
net = Net()  # 实例化神经网络
optimizer = torch.optim.Adam(net.parameters(), lr=0.03)  # 实例化优化器：Adam
loss_func = torch.nn.MSELoss()  # 损失函数：Mean Square Error
loss_all = []
if __name__ == '__main__':
    for epoch in range(20):  # 进行50个epoch的训练
        for i, data in enumerate(train_loader, 0):
            x, y = data
            pred = net(x)
            loss = loss_func(pred, y) # 计算loss
            optimizer.zero_grad() # 清零原有的梯度
            loss.backward() # 将loss梯度进行反向传播
            optimizer.step() # 更新参数
            loss_all.append(loss.data)
    plt.plot(loss_all, "r-")
    plt.show()
