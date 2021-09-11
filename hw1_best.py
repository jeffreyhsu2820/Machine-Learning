#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd 
import numpy as np
import sys


# In[7]:


test = pd.read_csv(sys.argv[1])
test = test.to_numpy()

N = int(len(test)/9)

test_x = np.empty([N, 135], dtype = float)
for i in range(N):
    test_x[i, :] = test[9 * i: 9* (i + 1), :].T.reshape(1, -1)
# test_x = np.delete(test_x,q,axis=1)
test_x = np.concatenate((np.ones([N, 1]), test_x), axis = 1).astype(float)
w = np.load('model.npy')
ans_y = np.dot(test_x, w)
for i in range(len(ans_y)):
    if ans_y[i]<0:
        ans_y[i]=0



import csv
with open(sys.argv[2], mode='w', newline='') as submit_file:
    csv_writer = csv.writer(submit_file)
    header = ['id', 'value']
    csv_writer.writerow(header)
    for i in range(N):
        row = ['id_' + str(i), ans_y[i][0]]
        csv_writer.writerow(row)

