#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingClassifier  #GBM algorithm
from sklearn import metrics   #Additional scklearn functions
from joblib import dump, load
import seaborn as sns  
import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np
import sys
import copy
import csv


# In[2]:


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


# def _preprocess(df):
#     # 補值
#     df = df.replace(' ?',np.nan)
#     df['workclass'] = df['workclass'].fillna(' Private')
#     df['occupation'] = df['occupation'].fillna(' Prof-specialty')
#     df['native_country'] = df['native_country'].fillna(' United-States')
#     df = df.drop(['fnlwgt'], axis = 1)    # 因為不知道這個變數是殺小，就去掉
#     df['capital_diff'] = df['capital_gain'] - df['capital_loss']
#     df = df.drop(['capital_gain',"capital_loss"], axis = 1)    # 做了新的 capital diff 所以去掉 gain 跟 loss
#     # 對age去做切割
#     df['age'] = pd.cut(df['age'], bins = [0, 25, 65, 100], labels = ['Young', 'Adult', 'Old'])
#     # 去除 education
#    df = df.drop(['education'], axis = 1) 
#    # 處理race
#    df['race'] = df['race'].replace([' Black', ' Asian-Pac-Islander', ' Amer-Indian-Eskimo', ' Other'],' Other')
#    # 除了美國其他歸類為others
#    countries = np.array(df['native_country'].unique())
#    countries = np.delete(countries, 0)
#    df['native_country'] = df['native_country'].replace(countries, ' Other')
#    return df


# In[5]:


# train_x = _preprocess(data)
# train_x = train_x.drop(['income'],axis=1)
# test_x = _preprocess(test)


# In[6]:


# categorical = ['age','workclass', 'marital_status', 'occupation', 'relationship','race', 'sex','native_country']
# for feature in categorical:
#         le = preprocessing.LabelEncoder()
#         train_x[feature] = le.fit_transform(train_x[feature])
#         test_x[feature] = le.transform(test_x[feature]) # test data的處理要跟 training一樣


# In[3]:


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
d = train_x[(f1 | f2 | f3)].index


#去掉 '?_workclass','?_occupation'
train_x = train_x.drop(['?_workclass','?_occupation','?_native_country'],axis = 1,inplace=False)

# num = ['age','fnlwgt',"capital_gain", 'capital_loss','hours_per_week', 'education_num']
# t = [train_x.columns.get_loc(col) for col in num]

train_x = train_x.to_numpy().astype('float')

train_y = np.array([train_y]).T.astype('float')
train_y = np.delete(train_y, d, axis=0)
train_x = np.delete(train_x, d, axis=0)


# In[ ]:





# In[5]:


train_x, X_mean, X_std = _normalize(train_x, train = True)
test_x['education_num'] = test['education_num']
test_x = test_x.drop(test['education'].unique(),axis = 1,inplace=False)
test_x = test_x.drop(['?_workclass','?_occupation','?_native_country'],axis = 1,inplace=False)
test_x = test_x.to_numpy().astype('float')
test_x, _, _= _normalize(test_x, train = False, specified_column = None, X_mean = X_mean, X_std = X_std)


# In[6]:


# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# train_x = pd.DataFrame(scaler.fit_transform(train_x), columns = train_x.columns)
# test_x = pd.DataFrame(scaler.transform(test_x), columns = train_x.columns)


# In[7]:


train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size=0.25, random_state=24)


# In[69]:


scores1 = []
scores2 = [] 
scores3 = []

dtree1 = GradientBoostingClassifier(random_state = 42, max_depth=7, n_estimators=100, verbose=1) 
dtree1.fit(train_x, train_y)
scores1.append(dtree1.score(valid_x, valid_y))

dtree2 = GradientBoostingClassifier(random_state = 42, max_depth=7, n_estimators=95, verbose=1)
dtree2.fit(train_x, train_y)
scores2.append(dtree2.score(valid_x, valid_y))

dtree3 = GradientBoostingClassifier(random_state = 42, max_depth=7, n_estimators=103, verbose=1)
dtree3.fit(train_x, train_y)
scores3.append(dtree3.score(valid_x, valid_y))


# In[70]:


sns.pointplot(data=[scores1, scores2, scores3], notch=True)
plt.xticks([0,1,2], ['GB_Clf1', 'GB_Clf2', 'GB_Clf3'])
plt.ylabel('Test Accuracy')


# In[8]:


# dump(dtree1,'gb.m')
dtree1 = load('gb.m')

Y_prediction = dtree1.predict(test_x).astype('int')

with open('ans_gb.csv', mode='w', newline='') as submit_file:
    csv_writer = csv.writer(submit_file)
    header = ['id', 'label']
    csv_writer.writerow(header)
    for i,label in enumerate(Y_prediction):
        submit_file.write('{},{}\n'.format(i+1, label))

