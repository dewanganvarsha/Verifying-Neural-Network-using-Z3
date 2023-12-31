# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 13:16:14 2023

@author: varsh
"""

import numpy as np

#Sigmoid Activation Function
def sigmoid(x):
    return (1/(1 + np.exp(-x)))

#Tanh Activation Function
def Tanh(x):
    return (np.exp(x)-np.exp(-x))/(np.exp(x) + np.exp(-x))

#Linear Approximation
def LinearApprox(limits, n, inputFun):
    n = 20
    m = np.zeros(n)
    c = np.zeros(n)
    x = np.linspace(limits[0], limits[1], n)
    for i in range(n-1):
        m[i] = (inputFun(x[i+1])-inputFun(x[i]))/(x[i+1]-x[i]) 
        c[i] = (inputFun(x[i])-m[i]*x[i])
    return m,c

#Approximation of Function
def ApproxFun(x,limits, limits_y, m, c, n):
    y = np.zeros(x.size)
    for i in range(len(x)):
        if x[i]<=limits[0]:
            y[i]= limits_y[0]
        elif x[i]>=limits[1]:
            y[i]= limits_y[1]
        else:
            interval = np.linspace(limits[0],limits[1],n)
            for j in range(n-1):
                if (interval[j] <=x[i] and x[i] <= interval[j+1])== True:
                    y[i]= m[j]*x[i]+c[j]
                    break
    return y

#Mean Square Error
def mse (actual, pred, limits,limits_y, m, c, var):
    n=100
    x = np.linspace(limits[0],limits[1], n)
    actual_Val = actual(x)
    pred_Val = pred(x, limits, limits_y, m, c, var)
    return np.square(np.subtract(actual_Val,pred_Val)).mean() 

def mseNew (actual, pred):
    n=100
    x = np.linspace(limits[0],limits[1], n)
    actual_Val = actual(x)
    pred_Val = pred(x)
    return np.square(np.subtract(actual_Val,pred_Val)).mean() 

limits = [-6,6]
limits_tan = [-2,2]
out1 = LinearApprox(limits,20,sigmoid)
out2 = LinearApprox(limits_tan,20,Tanh)

sigmoidApprox = lambda x: ApproxFun(x, limits, [0,1], out1[0], out1[1], 20);

# out2 = ApproxFun(limits, np.linspace(limits[0],  limits[1], 20), out1[0], out1[1], 20 )
#diff = mse(sigmoid, ApproxFun, limits,[0,1], out1[0], out1[1], 20)
#diff2 = mse(Tanh, ApproxFun, limits_tan, [-1,1], out2[0], out2[1], 20)
diff = mseNew(sigmoid, sigmoidApprox);

   