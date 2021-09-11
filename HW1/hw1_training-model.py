#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd 
import numpy as np
import sys
import copy

data0 = pd.read_csv('/home/jeffreyhsu/桌面/ML2020FALL/hw1/train_datas_0.csv')   #補空值為0，將 "-" 補成前一小時的值
data1 = pd.read_csv('/home/jeffreyhsu/桌面/ML2020FALL/hw1/train_datas_1.csv')   #中間有一串完全是亂碼  
test = pd.read_csv('/home/jeffreyhsu/桌面/ML2020FALL/hw1/test_datas.csv')   #非常乾淨的資料

data1 = data1.drop([i for i in range(2162,2206)])

data0 = data0.dropna(axis=0,how='all')
data1 = data1.dropna(axis=0,how='all')
test = test.to_numpy()

def pre_remove_dash(df):
    df = np.array(df)
    for j in range(len(df[0])):
        for i in range(len(df)):
            if df[i,j] == "-":
                df[i,j] = df[i-1,j]
    return df

data0 = pre_remove_dash(data0)
data1 = pre_remove_dash(data1)

data = np.vstack((data0,data1))
data = data.astype(float)

train_y = np.array([data[i+9,10] for i in range(len(data)-9)])
train_y = np.array([train_y]).T

d=[]
for i in range(len(train_y)):
    if train_y[i]>50:
        d.append(i)
train_y = np.delete(train_y, d, axis=0)


# In[ ]:


def train(train_x, train_y, iteration,lda):
    w = np.ones([len(train_x[0])+1, 1])
    # w[[9,18,27,36,45,54,63,72,81,90,99,108,117,126,135]]=1
    # w[[8,9]]=1
    # w[[7,8]]=1
    # w[[9,18,27,36]]=1
    eps = 0.0000000001   # adagrad 避免分母為零
    x = np.concatenate((np.ones([len(train_x), 1]), train_x), axis = 1).astype(float)
    learning_rate = 100
    adagrad = np.zeros([len(train_x[0])+1, 1])
    for t in range(iteration):
        loss = np.sqrt(np.sum(np.power(np.dot(x, w) - train_y, 2))/len(train_x))#rmse
        if(t%5000==0):
            print(str(t) + ":" + str(loss))
        grad=copy.deepcopy(w)
        grad[[0,len(grad)-1]]=0
        gradient = 2 * (np.dot(x.T, np.dot(x, w) - train_y) + lda*grad)
        adagrad += gradient ** 2
        w = w - learning_rate * gradient / np.sqrt(adagrad+eps)
    return w


# In[ ]:


# 用全部變數的model
train_x=[]
for i in range(len(data)-9):
    a=[]
    for j in range(len(data[0,:])):
        a = np.hstack((a,data[i:i+9,j]))
    train_x.append(a)

train_x = np.array(train_x)
#q = np.load('var_selection_full.npy')
#train_x = np.delete(train_x, q, axis=1)
train_x = np.delete(train_x,d,axis=0)
w = train(train_x, train_y, 10000,0.05)
np.save('model_report.npy', w)


# In[ ]:


test_x = np.empty([500, 135], dtype = float)
for i in range(500):
    test_x[i, :] = test[9 * i: 9* (i + 1), :].T.reshape(1, -1)
# test_x = np.delete(test_x,q,axis=1)
test_x = np.concatenate((np.ones([500, 1]), test_x), axis = 1).astype(float)

w = np.load('model_report.npy')
ans_y = np.dot(test_x, w)

for i in range(len(ans_y)):
    if ans_y[i]<0:
        ans_y[i]=0


# In[ ]:


import csv
with open('ans.csv', mode='w', newline='') as submit_file:
    csv_writer = csv.writer(submit_file)
    header = ['id', 'value']
    csv_writer.writerow(header)
    for i in range(500):
        row = ['id_' + str(i), ans_y[i][0]]
        csv_writer.writerow(row)

test_y =[]
for i in range(500):
    test_y.append((test[9*(i),10]))
print(np.sqrt(np.sum((np.array(test_y)-ans_y.T)**2)/500))

