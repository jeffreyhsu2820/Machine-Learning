from sklearn import preprocessing
from sklearn.ensemble import GradientBoostingClassifier  #GBM algorithm
from sklearn import metrics   #Additional scklearn functions
from joblib import dump, load
import pandas as pd 
import numpy as np
import sys
import copy
import csv

train_x = pd.read_table(sys.argv[3],sep=',')
test_x = pd.read_table(sys.argv[5],sep=',')
train_y = pd.read_table(sys.argv[4],sep=',')
data = pd.read_csv(sys.argv[1])
test = pd.read_csv(sys.argv[2])

def _normalize(X, train = True, specified_column = None, X_mean = None, X_std = None):
    if specified_column == None:
        specified_column = np.arange(X.shape[1])
    if train:
        X_mean = np.mean(X[:, specified_column] ,0).reshape(1, -1)
        X_std  = np.std(X[:, specified_column], 0).reshape(1, -1)
    X[:,specified_column] = (X[:, specified_column] - X_mean) / (X_std + 1e-8)
    return X, X_mean, X_std

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

# preprocess train_x to get normalization term
train_x['education_num']=data['education_num']
train_x = train_x.drop(data['education'].unique(),axis = 1,inplace=False)
f1 = train_x["?_workclass"]==1
f2 = train_x["?_occupation"]==1
f3 = train_x["?_native_country"]==1
d = train_x[(f1|f2|f3)].index
train_x = train_x.drop(['?_workclass','?_occupation','?_native_country'],axis = 1,inplace=False)
train_x = train_x.to_numpy().astype('float')
train_x = np.delete(train_x, d, axis=0)

# Normalize training and testing data
train_x, X_mean, X_std = _normalize(train_x, train = True)
test_x['education_num'] = test['education_num']
test_x = test_x.drop(test['education'].unique(),axis = 1,inplace=False)
test_x = test_x.drop(['?_workclass','?_occupation','?_native_country'],axis = 1,inplace=False)
test_x = test_x.to_numpy().astype('float')
test_x, _, _= _normalize(test_x, train = False, specified_column = None, X_mean = X_mean, X_std = X_std)


dtree1 = load('gb.m')
Y_prediction = dtree1.predict(test_x).astype('int')
with open(sys.argv[6], mode='w', newline='') as submit_file:
    csv_writer = csv.writer(submit_file)
    header = ['id', 'label']
    csv_writer.writerow(header)
    for i,label in enumerate(Y_prediction):
        submit_file.write('{},{}\n'.format(i+1, label))
