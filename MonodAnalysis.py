# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 12:37:53 2023

@author: YZ60069
"""

X = np.arange(1,1000,1)
Cth = 900
Y = (X - Cth) / (X - Cth + 1e2)
plt.plot(X,Y,'b-')

#%%
X = np.arange(0,100,1)
Y = np.arctan(X)
plt.plot(X,Y,'b-')