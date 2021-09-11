#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np
import sys
import copy
import csv


# In[2]:


def _normalize(X, train = True, specified_column = None, X_mean = None, X_std = None):
    if specified_column == None:
        specified_column = np.arange(X.shape[1])
    if train:
        X_mean = np.mean(X[:, specified_column] ,0).reshape(1, -1)
        X_std  = np.std(X[:, specified_column], 0).reshape(1, -1)
    X[:,specified_column] = (X[:, specified_column] - X_mean) / (X_std + 1e-8)
    return X, X_mean, X_std

def _train_valid(X, Y, dev_ratio):
    randomize = np.arange(len(X))
    np.random.shuffle(randomize)
    X = X[randomize] 
    Y = Y[randomize]
    train_size = int(len(X) * (1 - dev_ratio))
    return X[:train_size], Y[:train_size], X[train_size:], Y[train_size:]
    
def _sigmoid(z): # 強至設定最大最小值
    return np.clip(1 / (1.0 + np.exp(-z)), 1e-8, 1 - (1e-8))

def _f(X, w, b):
    return _sigmoid(np.dot(X, w) + b)

def _cross_entropy_loss(y_pred, Y_label):
    cross_entropy = -np.dot(Y_label, np.log(y_pred)) - np.dot((1 - Y_label), np.log(1 - y_pred))
    return cross_entropy

def _gradient(X, Y_label, w, b):
    y_pred = _f(X, w, b) #column vector
    pred_error = Y_label - y_pred # column vector
    w_grad = -np.dot(pred_error.T,X).T
    b_grad = -np.sum(pred_error)
    return w_grad, b_grad

def _accuracy(Y_pred, Y_label):
    acc = 1 - np.mean(np.abs(Y_pred - Y_label))
    return acc


# In[3]:


# capital_gain/loss不是0的數量太少，直接轉成 0 or 1
def _preprocess(X):
    X["capital_gain"][(X["capital_gain"]!=0)]=1
    X["capital_loss"][(X["capital_loss"]!=0)]=1

    # White/ Black/ Other 比例是 85%, 10%, 5%，所以我改成 White/ Non_white
    X = X.drop([' Black',' Other'],axis = 1,inplace = False)

    # 把married_status分類中，married開頭的分成一類
    # f1 = X[" Married-AF-spouse"]==1
    f2 = X[" Married-civ-spouse"]==1
    f3 = X[" Married-spouse-absent"]==1
    d = X[(f2 | f3)].index
    X[" Married-AF-spouse"][d]=1
    X.rename(columns = {" Married-AF-spouse":'Married'},inplace = True)
    X = X.drop([' Married-civ-spouse',' Married-spouse-absent'],axis = 1,inplace = False)
    
    # education  Preschool
    e2 = X[" 5th-6th"]==1
    e1 = X[" Preschool"]==1
    d = X[(e1|e2)].index
    X[" 1st-4th"][d]=1
    X.rename(columns = {" 1st-4th":'Grade-school'},inplace = True)
    X = X.drop([' 5th-6th',' Preschool'],axis = 1,inplace = False)
    
    e3 = X[" 7th-8th"]==1
    e4 = X[" 10th"]==1
    e5 = X[" 11th"]==1
    e6 = X[" 12th"]==1
    d = X[(e3|e4|e5|e6)].index
    X[" 9th"][d]=1
    X.rename(columns = {" 9th":'High-school'},inplace = True)
    X = X.drop([' 7th-8th'," 10th"," 11th"," 12th"],axis = 1,inplace = False)
    
    e7 = X[" Masters"]==1
    d = X[(e7)].index
    X[" Doctorate"][d]=1
    X.rename(columns = {" Doctorate":'Graduate'},inplace = True)
    X = X.drop([' Masters'],axis = 1,inplace = False)
    
    # workclass
    w2 = X[" Federal-gov"]==1
    w3 = X[" Local-gov"]==1
    d = X[(w2 | w3)].index
    X[" State-gov"][d]=1
    X.rename(columns = {" State-gov":'Gov'},inplace = True)
    X = X.drop([' Federal-gov',' Local-gov'],axis = 1,inplace = False)
    
    w1 = X[' Self-emp-not-inc']==1
    d = X[(w1)].index
    X[" Self-emp-inc"][d]=1
    X.rename(columns = {" Self-emp-inc":'Self-emp'},inplace = True)
    X = X.drop([' Self-emp-not-inc'],axis = 1,inplace = False)
    
    w0 = X[' Never-worked']==1
    d = X[(w1)].index
    X[" Without-pay"][d]=1
    X = X.drop([' Never-worked'],axis = 1,inplace = False)

    # age
    X['age'] = np.log10(X['age'])
    # fnlwgt
    X['fnlwgt'] = np.log10(X['fnlwgt'])
    # hours_per_week
    X['hours_per_week'] = np.log10(X['hours_per_week'])
    # native_country
    nonUS = [' Cuba', ' Jamaica', ' India', '?_native_country', ' Mexico',
       ' South', ' Puerto-Rico', ' Honduras', ' England', ' Canada',
       ' Germany', ' Iran', ' Philippines', ' Italy', ' Poland',
       ' Columbia', ' Cambodia', ' Thailand', ' Ecuador', ' Laos',
       ' Taiwan', ' Haiti', ' Portugal', ' Dominican-Republic',
       ' El-Salvador', ' France', ' Guatemala', ' China', ' Japan',
       ' Yugoslavia', ' Peru', ' Outlying-US(Guam-USVI-etc)', ' Scotland',
       ' Trinadad&Tobago', ' Greece', ' Nicaragua', ' Vietnam', ' Hong',
       ' Ireland', ' Hungary', ' Holand-Netherlands']
    X = X.drop(nonUS,axis = 1,inplace = False)
    return X


# In[4]:


train_x = pd.read_table('/home/jeffreyhsu/桌面/ML2020FALL/hw2/X_train',sep=',')
# train_y = pd.read_table('/home/jeffreyhsu/桌面/ML2020FALL/hw2/Y_train',sep=',')
test_x = pd.read_table('/home/jeffreyhsu/桌面/ML2020FALL/hw2/X_test',sep=',')
# 只有workclass, occupation, native country 裡面有？的值
data = pd.read_csv('/home/jeffreyhsu/桌面/ML2020FALL/hw2/train.csv')
test = pd.read_csv('/home/jeffreyhsu/桌面/ML2020FALL/hw2/test.csv')
# kaggel 給的 train_y 有錯，所以我重新做一個
train_y = []
for i in range(len(data)):
    if data["income"][i]==" <=50K":
        train_y.append(0)
    else:
        train_y.append(1)


# In[5]:


# 新增train_x沒有的 'education_num' 這個欄位，刪除本來 education 的 dummy variable
train_x['education_num']=data['education_num']
train_x = train_x.drop(data['education'].unique(),axis = 1,inplace=False)

# train_x = _preprocess(train_x)

# ['workclass', 'occupation', 'native_country']有缺失值 " ?" 總共有2399筆 之後看要不要把他們拿掉
# 但是testing 也有" ?"缺失值

f1 = train_x["?_workclass"]==1
f2 = train_x["?_occupation"]==1
f3 = train_x["?_native_country"]==1
d = train_x[(f1|f2|f3)].index

#去掉 '?_workclass','?_occupation'
train_x = train_x.drop(['?_workclass','?_occupation'],axis = 1,inplace=False)

# num = ['age','fnlwgt',"capital_gain", 'capital_loss','hours_per_week', 'education_num']
# t = [train_x.columns.get_loc(col) for col in num]

train_x = train_x.to_numpy().astype('float')

train_y = np.array([train_y]).T.astype('float')
#train_y = np.delete(train_y, d, axis=0)
#train_x = np.delete(train_x, d, axis=0)


# In[6]:


# Normalize training and testing data
#X_train, X_mean, X_std = _normalize(train_x, train = True)

# test_x = _preprocess(test_x)

# 新增test_x沒有的 'education_num' 這個欄位，刪除本來 education 的 dummy variable
test_x['education_num'] = test['education_num']
test_x = test_x.drop(test['education'].unique(),axis = 1,inplace=False)
test_x = test_x.drop(['?_workclass','?_occupation'],axis = 1,inplace=False)
test_x = test_x.to_numpy().astype('float')
#test_x, _, _= _normalize(test_x, train = False, specified_column = None, X_mean = X_mean, X_std = X_std)


# In[7]:


# Compute in-class mean
X_train_0 = np.array([x for x, y in zip(train_x, train_y) if y == 0])
X_train_1 = np.array([x for x, y in zip(train_x, train_y) if y == 1])


# In[8]:


mean_0 = np.mean(X_train_0, axis = 0)
mean_1 = np.mean(X_train_1, axis = 0)  


# In[9]:


cov_0 = np.zeros((train_x.shape[1], train_x.shape[1]))
cov_1 = np.zeros((train_x.shape[1], train_x.shape[1]))

for x in X_train_0:
    cov_0 += np.dot(np.transpose([x - mean_0]), [x - mean_0]) / X_train_0.shape[0]
for x in X_train_1:
    cov_1 += np.dot(np.transpose([x - mean_1]), [x - mean_1]) / X_train_1.shape[0]

cov = (cov_0 * X_train_0.shape[0] + cov_1 * X_train_1.shape[0]) / (X_train_0.shape[0] + X_train_1.shape[0])


# In[10]:


u, s, v = np.linalg.svd(cov, full_matrices=False)
inv = np.matmul(v.T * 1 / s, u.T)


# In[11]:


w = np.dot(inv, mean_0 - mean_1)


# In[12]:


b =  (-0.5) * np.dot(mean_0, np.dot(inv, mean_0)) + 0.5 * np.dot(mean_1, np.dot(inv, mean_1))+ np.log(float(X_train_0.shape[0]) / X_train_1.shape[0]) 


# In[13]:


Y_train_pred = np.transpose([1 - np.round(_f(train_x, w, b)).astype('int')])


# In[14]:


print('Training accuracy: {}'.format(_accuracy(Y_train_pred, train_y)))


# In[15]:


np.save('model_prob_w.npy', w)
np.save('model_prob_b.npy', b)

w = np.load('model_prob_w.npy')
b = np.load('model_prob_b.npy')


# In[16]:


y_test_pred = _f(test_x, w, b)

Y_test_pred = 1 - np.round(y_test_pred).astype('int')

with open('ans_prob.csv', mode='w', newline='') as submit_file:
    csv_writer = csv.writer(submit_file)
    header = ['id', 'label']
    csv_writer.writerow(header)
    for i,label in enumerate(Y_test_pred):
        submit_file.write('{},{}\n'.format(i+1, label))

