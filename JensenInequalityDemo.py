# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 11:07:44 2023

@author: YZ60069
"""
# this file expolores the relationship between the derivative of the function and the Jensen's inequality
# I compare the Jensen's inequality effect under different derivatives of the function f(x)

import random
import numpy as np
import matplotlib.pyplot as plt


X = np.random.randint(0,100, size = 100, dtype = np.int64)

#%%
Diff = []
for i in range(1,8):
    Y1 = np.mean(np.power(X, i))
    Y2 = np.power(np.mean(X),i)
    Diff.append((Y1 - Y2)/Y1)

plt.plot(np.arange(1,8), np.array(Diff),'ko')
plt.title('y = x^n')
plt.xlabel('n')
plt.ylabel('%difference')


#%%
Diff = []
for i in range(1,8):
    Y1 = i * np.mean(np.power(X, 2))
    Y2 = i * np.power(np.mean(X),2)
    Diff.append((Y1 - Y2)/1)

plt.plot(np.arange(1,8), np.array(Diff),'ko')
plt.title('y = ax^2')
plt.xlabel('a')
plt.ylabel('absolute difference')

#%%
plt.plot(np.arange(1,8), np.power(np.arange(1,8),7))


#%%
X = [1,10]
Y = np.power(X, 2)
plt.plot(X,Y,'b-')

Xc = np.arange(0,15,0.1)
Yc = np.power(Xc, 2)
plt.plot(Xc,Yc, 'b-', label = 'y = x^2')


X = [1,10]
Y = np.power(X, 3)
plt.plot(X,Y,'r-')

Xc = np.arange(0,11,0.1)
Yc = np.power(Xc, 3)
plt.plot(Xc,Yc, 'r-', label = 'y = x^3')


X = [3,8]
Y = np.power(X, 3)
plt.plot(X,Y,'k-')

plt.legend(loc = 0)