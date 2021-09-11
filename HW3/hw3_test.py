# -*- coding: utf-8 -*-

import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import time

def readfile(path, label):
    # label 是一個 boolean variable，代表需不需要回傳 y 值
    image_dir = sorted(os.listdir(path))
    x = np.zeros((len(image_dir), 48, 48, 3), dtype=np.uint8)
    y = np.zeros((len(image_dir)), dtype=np.uint8)
    for i, file in enumerate(image_dir):
        img = cv2.imread(os.path.join(path, file))
        x[i, :, :] = cv2.resize(img,(48, 48))
        if label:
          y[i] = int(file.split("_")[0])
    if label:
      return x, y
    else:
      return x

# testing 時不需做 data augmentation
test_transform = transforms.Compose([transforms.ToPILImage(),transforms.ToTensor(),])
class ImgDataset(Dataset):
    def __init__(self, x, y=None, transform=None):
        self.x = x
        # label is required to be a LongTensor
        self.y = y
        if y is not None:
            self.y = torch.LongTensor(y)
        self.transform = transform
    def __len__(self):
        return len(self.x)
    def __getitem__(self, index):
        X = self.x[index]
        if self.transform is not None:
            X = self.transform(X)
        if self.y is not None:
            Y = self.y[index]
            return X, Y
        else:
            return X

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        # torch.nn.MaxPool2d(kernel_size, stride, padding)
        # input 維度 [3, 48, 48]
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 24, 3, 1, 1), # [24, 48, 48]
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),    # [24, 24, 24]

            nn.Conv2d(24, 48, 3, 1, 1), # [48, 24, 24]
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),      # [48, 12, 12]

            nn.Conv2d(48, 96, 3, 1, 1), # [96, 12, 12]
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),      # [96, 6, 6]

            nn.Conv2d(96, 96, 3, 1, 1), # [192, 6, 6]
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),       # [192, 3, 3]
        )
        self.fc = nn.Sequential(
            nn.Linear(96*3*3, 192),
            nn.ReLU(),
            nn.Linear(192, 96),
            nn.ReLU(),
            nn.Linear(96, 7)
        )

    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.size()[0], -1)
        return self.fc(out)

import sys
test_x = readfile(sys.argv[1], False)
test_set = ImgDataset(test_x, transform=test_transform)
batch_size = 48
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

qq = Classifier().cuda()
qq.load_state_dict(torch.load('hw3_test.pth'))
qq.eval()

prediction = []
with torch.no_grad():
    for i, data in enumerate(test_loader):
        test_pred = qq(data.cuda())
        test_label = np.argmax(test_pred.cpu().data.numpy(), axis=1)
        for y in test_label:
            prediction.append(y)

#將結果寫入 csv 檔
with open(sys.argv[2], 'w') as f:
    f.write('Id,label\n')
    for i, y in  enumerate(prediction):
        f.write('{},{}\n'.format(i, y))

