#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 15:13:00 2020

@author: luca Kiener,paul Grandjean

"""

from sklearn.linear_model import LinearRegression
from numpy import genfromtxt
import numpy as np

my_data = genfromtxt('train.csv', delimiter=',')
my_data=my_data[1:,:]

Y_train=my_data[:,1]
X_train=my_data[:,2:12]

my_data = genfromtxt('test.csv', delimiter=',')
my_data=my_data[1:,:]

X_ID=my_data[:,0]
X_test=my_data[:,1:11]

regr=LinearRegression()

regr.fit(X_train,Y_train)

Y_predict= regr.predict(X_test)

output1=Y_predict[:]
output2=X_ID[:]
output1=np.ndarray.tolist(output1[:])
output2=np.ndarray.tolist(X_ID[:])


new=np.zeros((2000, 2))

new[:,1]= output1[:]
new[:,0]= output2[:]


np.savetxt('output.csv', new, fmt=['%i','%.1f'],delimiter=',') 
