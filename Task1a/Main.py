#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 14:46:59 2020

@author: luca,paul
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

#read data
data= pd.read_csv("train.csv")

#X,Y
Train = data.values
X = Train[:,2:]
Y =Train[:,1]

#lambdas
lambdas=[0.1,1,10,100,1000] #double check

kf = KFold(n_splits=10, shuffle=False)
validation_error_mean = []

for i in lambdas:
    validation_error=[]
    for train_index, validation_index in kf.split(X,Y):
        
        X_train, X_validation = X[train_index], X[validation_index]
        Y_train, Y_validation = Y[train_index], Y[validation_index]
        
        reg = Ridge(alpha=i, fit_intercept=False, normalize=False)
        
        reg.fit(X_train,Y_train)
        predict = reg.predict(X_validation)
        val_err= mean_squared_error(Y_validation,predict)**0.5
        
        validation_error.append(val_err)
        
    validation_error_mean.append(np.mean(validation_error))
    
    
print(validation_error_mean)

output = pd.DataFrame(validation_error_mean)
output.to_csv("Output.csv", sep=",", index=False, header=False)