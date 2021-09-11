# -*- coding: utf-8 -*-

import wget
DATA_URL = 'https://www.dropbox.com/s/li911szvfb3x5rh/last_checkpoint.pth?dl=1'
out_fname = 'last_checkpoint.pth'
wget.download(DATA_URL, out=out_fname)


import numpy as np
import sys

def preprocess(image_list):
    image_list = np.array(image_list)
    image_list = np.transpose(image_list, (0, 3, 1, 2))
    image_list = (image_list / 255.0) * 2 - 1
    image_list = image_list.astype(np.float32)
    return image_list

from torch.utils.data import Dataset

class Image_Dataset(Dataset):
    def __init__(self, image_list):
        self.image_list = image_list
    def __len__(self):
        return len(self.image_list)
    def __getitem__(self, idx):
        images = self.image_list[idx]
        return images

import torch.nn as nn

class AE(nn.Module):
    def __init__(self):
        super(AE, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 128, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 512, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(2),
            nn.Conv2d(512, 1024, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(1024),
            nn.MaxPool2d(2), # 1024 * 2 * 2
        )
 
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, 5, stride=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 7, stride=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 10, stride=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 3, 12, stride=1),
            nn.Tanh()
        )

    def forward(self, x):
        x1 = self.encoder(x)
        x  = self.decoder(x1)
        return x1, x


import torch
from sklearn.decomposition import KernelPCA
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from sklearn.cluster import MiniBatchKMeans

def inference(X, model, batch_size=1024):
    X = preprocess(X)
    dataset = Image_Dataset(X)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    latents = []
    for i, x in enumerate(dataloader):
        x = torch.FloatTensor(x)
        vec, img = model(x.cuda())
        if i == 0:
            latents = vec.view(img.size()[0], -1).cpu().detach().numpy()
        else:
            latents = np.concatenate((latents, vec.view(img.size()[0], -1).cpu().detach().numpy()), axis = 0)
    print('Latents Shape:', latents.shape)
    return latents

def predict(latents):
    # First Dimension Reduction
    transformer = KernelPCA(n_components=200, kernel='rbf', n_jobs=-1)
    kpca = transformer.fit_transform(latents)
    print('First Reduction Shape:', kpca.shape)

    # # Second Dimesnion Reduction
    X_embedded = TSNE(n_components=2).fit_transform(kpca)
    print('Second Reduction Shape:', X_embedded.shape)

    # Clustering
    pred = MiniBatchKMeans(n_clusters=2, random_state=0).fit(X_embedded)
    pred = [int(i) for i in pred.labels_]
    pred = np.array(pred)
    return pred, X_embedded

def invert(pred):
    return np.abs(1-pred)

def save_prediction(pred, out_csv):
    with open(out_csv, 'w') as f:
        f.write('id,label\n')
        for i, p in enumerate(pred):
            f.write(f'{i},{p}\n')
    print(f'Save prediction to {out_csv}.')

# load model
model = AE().cuda()
model.load_state_dict(torch.load('last_checkpoint.pth'))
model.eval()

# 準備 data
trainX = np.load(sys.argv[1])

# 預測答案
latents = inference(X=trainX, model=model)
pred, X_embedded = predict(latents)
save_prediction(invert(pred), sys.argv[2])
