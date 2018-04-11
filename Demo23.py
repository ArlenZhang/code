# -*- coding: utf-8 -*-
import torch
from torch.autograd import Variable
N, D_in, H, D_out = 64, 1000, 100, 10
# 创建样本数据Tensor变量
x = Variable(torch.randn(N, D_in))
y = Variable(torch.randn(N, D_out), requires_grad=False)

# 自定义模型
model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, D_out),
)
loss_fn = torch.nn.MSELoss(size_average=False)
learning_rate = 1e-4

# Here we will use Adam; The first argument to the Adam constructor tells the
# optimizer which Variables it should update.
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
for t in range(500):
    y_pred = model(x)
    loss = loss_fn(y_pred, y)
    print(t, loss.data[0])
    # 在学习之前清空梯度缓冲区
    optimizer.zero_grad()
    # 根据loss值计算loss函数中的模型参数的梯度
    loss.backward()
    # 调用optimizer对计算的梯度进行同步的梯度更新步骤
    optimizer.step()
