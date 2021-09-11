#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np
import sys
import copy
import csv


# In[14]:


def _preprocess(X):
    # capital_gain/loss不是0的數量太少，直接轉成 0 or 1
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


# In[4]:


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
train_y = np.delete(train_y, d, axis=0)
train_x = np.delete(train_x, d, axis=0)


# In[59]:


# 有一些資料重複把它刪除
# train = np.hstack((train_x,train_y))
# train = np.unique(train,axis=0)
# train_y = np.transpose([train[:,len(train[0])-1]])
# train_x = train[:,:len(train[0])-1]


# In[5]:


# Normalize training and testing data
train_x, X_mean, X_std = _normalize(train_x, train = True)

w = np.ones((len(train_x[0]),1))  # column vector
b = np.ones((1,1))
# w_grad_tmp = w**2
# b_grad_tmp = b**2


dev_ratio = 0.25
X_train, Y_train, X_dev, Y_dev = _train_valid(train_x, train_y, dev_ratio = dev_ratio)


# In[49]:


# 因為資料是 unbalance 的，所以把比較少的部份加一點擾動生成資料
#train_x_1 = np.array([x for x, y in zip(train_x, train_y) if y == 1])
#train_y_1 = np.array([y for x, y in zip(train_x, train_y) if y == 1])

# s = np.random.normal(loc=0, scale=np.exp(0.1), size=len(train_x_1))
#train_x = np.vstack((train_x,train_x))
#train_y = np.vstack((train_y,train_y))


# In[6]:


#test_x = _preprocess(test_x)

# 新增test_x沒有的 'education_num' 這個欄位，刪除本來 education 的 dummy variable
test_x['education_num'] = test['education_num']
test_x = test_x.drop(test['education'].unique(),axis = 1,inplace=False)
test_x = test_x.drop(['?_workclass','?_occupation'],axis = 1,inplace=False)
test_x = test_x.to_numpy().astype('float')
test_x, _, _= _normalize(test_x, train = False, specified_column = None, X_mean = X_mean, X_std = X_std)


# In[73]:


# train_x_0 = np.array([x for x, y in zip(train_x, train_y) if y == 0])
# train_y_0 = np.array([y for x, y in zip(train_x, train_y) if y == 0])
# train_x_1 = np.array([x for x, y in zip(train_x, train_y) if y == 1])
# train_y_1 = np.array([y for x, y in zip(train_x, train_y) if y == 1])
# train_x_1 = np.tile(train_x_1, (2,1))
# train_y_1 = np.tile(train_y_1, (2,1))
# train_x = np.vstack((train_x_0,train_x_1))
# train_y = np.vstack((train_y_0,train_y_1))


# In[7]:


# 雖然前面有切 training 跟 validaiton但是這邊還是用所有資料去train
w = np.zeros((len(train_x[0]),1))  # column vector
b = np.zeros((1,1))
step = 1
learning_rate=0.02
adagrad = np.zeros([len(train_x[0]), 1])
adagrad_b = np.zeros([1, 1])
for itr in range(10000):
    X = train_x
    Y = train_y
    w_grad, b_grad = _gradient(X, Y, w, b)
    adagrad += w_grad ** 2
    adagrad_b += b_grad ** 2
    w = w - (learning_rate/np.sqrt(adagrad + 0.00000000001)) * w_grad
    b = b - (learning_rate/np.sqrt(adagrad_b + 0.00000000001)) * b_grad
    step = step + 1


# In[143]:


np.save('model_log_w.npy', w)
np.save('model_log_b.npy', b)
w = np.load('model_log_w.npy')
b = np.load('model_log_b.npy')


# In[8]:


# Compute loss and accuracy of training set and development set
y_train_pred = _f(X_dev, w, b)
Y_train_pred = np.round(y_train_pred)


# In[9]:


train_acc = _accuracy(Y_train_pred, Y_dev)
train_acc


# In[10]:


y_test_pred = _f(test_x, w, b)
Y_test_pred = np.round(y_test_pred).astype('int')

with open('ans_logistic.csv', mode='w', newline='') as submit_file:
    csv_writer = csv.writer(submit_file)
    header = ['id', 'label']
    csv_writer.writerow(header)
    for i,label in enumerate(Y_test_pred):
        submit_file.write('{},{}\n'.format(i+1, label[0]))

