# 编写模型
import torch
import torch.nn.functional as F


class MyModel(torch.nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.Conv1 = torch.nn.Conv2d(1, 10, kernel_size=(5, 5))
        self.Conv2 = torch.nn.Conv2d(10, 20, kernel_size=(5, 5))
        self.pool = torch.nn.MaxPool2d(2)
        self.fl = torch.nn.Linear(320, 10)

    def forward(self, x):
        bs = x.size(0)
        x = F.relu(self.pool(self.Conv1(x)))
        x = F.relu(self.pool(self.Conv2(x)))
        x = x.view(bs, -1)
        x = self.fl(x)
        return x
