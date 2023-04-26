# 数据预处理 统一将图像（或矩阵）返回成torch能处理的[original_iamges.tensor，label.tensor]
# encoding:utf-8
import torch.utils.data as data
import torch
import torch.nn.functional as F
from PIL import Image

import os
import os.path


class MyDataset(torch.utils.data.Dataset):  # 子类化
    def __init__(self, root, transform=None):  # 第一步初始化各个变量

        self.root = root
        self.data = os.listdir(self.root)
        self.transform = transform
        self.len = len(self.data)

    def __getitem__(self, index):  # 第二步装载数据，返回[img,label]
        image_index = self.data[index]
        img_path = os.path.join(self.root, image_index)
        img = Image.open(img_path)
        if self.transform:
            img = self.transform(img)

        label = int(image_index[-5])  # 图片名中已带有标签

        '''
        要将标签转成可用于学习的tensor类型。标签可分为两种，
        一种是常用于多分类的整型数字标签（类似0,1,2,3）
        另一种是one-hot类型，假设共有三个类别，那么1，2，3对应的one-hot 分别是0 0 0, 0 1 0, 0 0 1
        '''
        label = self.oneHot(label, 10)
        return img, label

    def __len__(self):
        return self.len

    # 将标签转为onehot编码
    def oneHot(self, label, classes):
        a = torch.tensor([label])
        b = F.one_hot(a, num_classes=classes)
        return b
