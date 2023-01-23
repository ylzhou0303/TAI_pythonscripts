# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 12:16:00 2022

@author: YZ60069
"""



#%% This file calculates the reaction rates along the modeling period
from contextlib import AsyncExitStack
#from socket import AF_X25
from functools import reduce
import numpy as np
import pandas as pd
import pickle
import statistics as st
import matplotlib
import matplotlib.pyplot as plt
import os
os.chdir('C:/Users/yz60069/TAI/TAI_fresh')
#os.chdir('C:/MBL/Research/PFLOTRAN DATA/pflotran outputs/OxyHet/Marsh Interior/All inundated/ root')
import scipy.io
#mat = scipy.io.loadmat('WaterTableData.mat')


#%% create a list to store the reaction rates and monod terms


# 1. data frame for maximum reaction rate constants
K_df = {'DOMAer': [5e-8], 'Met': [1e-9], 'MetOxi': [5e-9], 'SulRed': [1e-8],'SulOxi': [5e-9]}
K_df = pd.DataFrame(K_df)
K = np.array(K_df)



# 2. data frame for monod half saturation constants
HSC_df = {'DOMAer': [1e-4, np.nan, 2e-3, np.nan, np.nan],
          'Met': [np.nan, np.nan, 2e-3, np.nan, np.nan],
          'MetOxi':[1e-4, 3e-4, np.nan, np.nan, np.nan],
          'SulRed': [np.nan, np.nan, 2e-3, 5e-3, np.nan],
          'SulOxi': [1e-4, np.nan, np.nan, np.nan, 3e-4]}   #HSC: half saturation constant

HSC_df = pd.DataFrame(HSC_df)
HSC_df.index = ['Monod_o2', 'Monod_ch4', 'Monod_dom', 'Monod_so4', 'Monod_h2s']
# There are six numbers for each term, the first number is the maximum reaction rate, the next five numbers are the Monod
# terms for O2, CH4, DOM, SO4, H2S, respectively. If a certain molecule is not in the Monod expression of that
# reaction, I set it to be 0
HSC = np.array(HSC_df)


#%% This function calculates the actual reaction rates based on the maximum reaction rates, monod constants, and concentrations
nreac = 5
nspecies = 5

def rate_calc(K, HSC, Conc):
    Rates = np.zeros(shape = (ngrids, ntimepoint, nreac), dtype = np.float32)
    
    for n in range(0,nreac):                      #calculate the rates for each reaction one by one                
        Monod = np.ones(shape = (ngrids, ntimepoint))  #init a Monod matrix for accumulative multiplication of the monod terms 
        
        for j in range(0,nspecies):
            hsc = HSC[j,n]
            if not np.isnan(hsc):             #if the hsc is not NAN, it means that the reaction rate is dependent on this species
                monod_temp = Conc[:,:,j] / (Conc[:,:,j] + hsc)   #use the concentration matrix of this species times the half saturation constant
                Monod = Monod * monod_temp
        
        Rates[:,:,n] = K[0,n] * Monod             #save the rate for this timepoint and this reaction, then go into the next loop, i.e. next timepoint    
    
    return Rates
        
            
            
#%% calculate reaction rates for all reactions
Conc = Full_Data[:,:,1:6]  #concentration of the five species investigated
Rates = rate_calc(K, HSC, Conc)


#%% verify the rates calculation
reac = 0
grid = 178
t = 6

rate = K[0,reac] * Conc[grid,t,0] / (HSC[0,reac] + Conc[grid,t,0])  *  Conc[grid, t, 2] / (HSC[2,reac] + Conc[grid, t, 2])

print(rate - Rates[grid,t,reac])

#%% plot depth profiles of the rates for all columns
reac = 0
for i in range(0, ncols):
    rate = Rates[i:ngrids:ncols, t, reac]
    plt.plot(rate, depths)
    
plt.title(K_df.columns[reac] + ' (mol L-1 s-1)')    
plt.xlabel('rate (mol L-1 s-1)')
plt.ylabel('Depth(m)')

#%% for a certain layer
layer = 6
i_start = (nz - layer) * nx * ny
i_end = i_start + nx * ny

for g in range(i_start, i_end):
    plt.plot(np.arange(0,ntimepoint), Rates[g, :, 2])


#%% plot heat maps for reaction rates
t = 30       #specify the time point
reac = 4     #specify which reaction to plot
layer = 6        #specify which layer, layer 1 is the top layer
i_start = (nz - layer) * nx * ny
i_end = i_start + nx * ny

M = Rates[i_start:i_end, t, reac]  #extract the data to be inevestigated, by specifying the layer, timepoint, and reaction id
A = M.reshape(nx, ny)      #this is a 7*7matrix, representing the view from top of the soil grids
B = np.flipud(A)   #flip upside down the matrix so that the grids with smaller y coordinates are at the bottom
                   #same as in the field
plt.imshow(B, cmap ="Reds")
plt.colorbar()
plt.title(K_df.columns[reac] + ' (mol L-1 s-1)')



#%% relationship between O2 and reaction rates
t = 30       #specify the point time
var = 1      #specify the species
reac = 1     #specify the reaction 

layer = 2        #specify which layer, layer 1 is the top layer, layer 2 is where the ROL is
i_start = (nz - layer) * nx * ny
i_end = i_start + nx * ny

X = Full_Data[:,t,var] / 2.5e-4 * 100
Y = Rates[:,t,reac]

plt.plot(X, Y, 'ro')
plt.xlabel('O2 %sat')
plt.ylabel('Reaction rate (mol L-1 s-1)')
plt.title(K_df.columns[reac])



#%% plot concentration vs reaction rate
reac = 1

x = np.arange(2e-5, 2e-2, 1e-6)
monod = x / (x + 2e-3)
y = K[0, reac] * monod
plt.plot(x, y, 'b-')


#real reaction rate based on concentrations
t = 30
x = Full_Data[:,t,3]   #concentration of DOM
y = Rates[:,t,reac]
plt.plot(x, y, 'ko',label = 'rate inside each grid')

plt.xlabel('DOM mol L-1')
plt.ylabel('Methanogenesis rate (mol L-1 s-1)')
plt.legend(loc = 0)

#%% Methane production, I keep this section to verify the results calculated above
k = 9e-10
dom1 = Full_Data[Grid_id, :, 3]
dom1_monod = 5e-2

monod = (dom1 / (dom1 + dom1_monod))

rate_2 = k * monod

plt.plot(rate_2)
ax=plt.gca()
ax.set_xlabel('Time (hr)')
ax.set_ylabel('Reaction Rate (mol L-1 s-1)')
plt.rcParams.update({'font.size': 12})


# #%% Methane Oxidation
# k = 1.6e-9

# o2 = Full_Data[Grid_id, :, 1]
# o2_monod = 1e-4
# ch4 = Full_Data[Grid_id, :, 2]
# ch4_monod = 6e-3

# monod = (ch4 / (ch4 + ch4_monod)) * (o2 / (o2 + o2_monod))

# rate_3 = k * monod

# plt.plot(rate_3)
# ax=plt.gca()
# ax.set_xlabel('Time (hr)')
# ax.set_ylabel('Reaction Rate (mol L-1 s-1)')
# plt.rcParams.update({'font.size': 12})


# #%% sulfate reduction
# k = 2.26e-8    #reaction constant
# so4 = Full_Data[Grid_id, :, 4]
# dom = Full_Data[Grid_id, :, 3]

# so4_monod = 5e-2
# dom_monod = 5e-2



# monod = (so4 / (so4 + so4_monod)) * (dom / (dom + dom_monod))

# rate_4 = k * monod

# plt.plot(rate_4)
# ax=plt.gca()
# ax.set_xlabel('Time (hr)')
# ax.set_ylabel('Reaction Rate (mol L-1 s-1)')
# plt.rcParams.update({'font.size': 12})


# #%% sulfide oxidation
# k = 1.6e-9    #reaction constant, i.e. maximum reaction rate
# o2 = Full_Data[Grid_id, :, 1]
# h2s = Full_Data[Grid_id, :, 5]


# o2_monod = 1e-4
# h2s_monod = 6e-3

# monod = (h2s / (h2s + h2s_monod)) * (o2 / (o2 +o2_monod)) 

# rate_5 = k * monod

# plt.plot(rate_5)
# ax=plt.gca()
# ax.set_xlabel('Time (hr)')
# ax.set_ylabel('Reaction Rate (mol L-1 s-1)')
# plt.rcParams.update({'font.size': 12})


#%% predict the model output
#methane
# time = 3600 *24 * 7
# c_0 = 6e-4
# c_t = c_0 + (rate_2[75]) * time