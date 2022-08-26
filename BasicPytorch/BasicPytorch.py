import torch
import torch.nn.functional as f
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset

x = torch.unsqueeze(torch.linspace(-2,2,300),dim=1)
y = 0.4*x.pow(4)-0.8*x.pow(3)-x.pow(2)+x

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.hidden1 = nn.Linear(1, 40)
        self.hidden2 = nn.Linear(40, 10)
        self.regression = nn.Linear(10, 1)

    def forward(self, x):
        x = f.relu(self.hidden1(x))
        x = f.sigmoid(self.hidden2(x))
        output = self.regression(x)
        return output

net =Net()
optimizer = torch.optim.Adam(net.parameters(),lr=0.05)
loss_func = torch.nn.MSELoss()
loss_all=[]
for t in range(300):
    pred = net(x)
    loss = loss_func(pred,y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    loss_all.append(loss.data)
plt.plot(loss_all,"r-")
plt.show()

