# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 16:40:52 2023

@author: YZ60069
"""
import numpy as np


# threshold concentration for methane at different depths, based on Jiaze's code (deltamarsh_profile_ebl.py Ln 401 - 418)
H = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])      #m
g = 9.8
rho = 1e3
Press_atm = 1e5   #Pa
T = 298
T0 = 298
Tstd = 273
R = 8.314

Press = Press_atm + rho * g * H
#Hcp = 1.4e-5 * np.exp(-1900*(1/T - 1/T0))  #CH4
Hcp = 1e-3 * np.exp(-2100*(1/T - 1/T0))    #sulfide
alpha = Hcp * Tstd * R
Eclim = Press * alpha / (R * T *1000)

