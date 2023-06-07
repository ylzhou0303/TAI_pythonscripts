# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 12:16:00 2022

@author: YZ60069
"""



#%% This file calculates the reaction rates for each grid along the modeling period
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
from scipy.stats import pearsonr
#mat = scipy.io.loadmat('WaterTableData.mat')


#%% create a list to store the reaction rates and monod terms


# 1. data frame for maximum reaction rate constants
# standard runs: K_df = {'DOMAer': [2e-8], 'Met': [5e-10], 'MetOxi': [4e-9], 'SulRed': [5e-9],'SulOxi': [1e-7]}

K_df = {'DOMAer': [1e-8], 'Met': [4e-10], 'MetOxi': [5e-10], 'SulRed': [6.24e-9],'SulOxi': [5e-7], 'MetOxiSul': [2e-10]}
K_df = pd.DataFrame(K_df)
K = np.array(K_df)


K_O2 = 8e-6   # Monod Half saturation constant for O2

# 2. data frame for monod half saturation constants
HSC_df = {'DOMAer': [K_O2, np.nan, 4e-3, np.nan, np.nan],
          'Met': [np.nan, np.nan, 4e-3, np.nan, np.nan],
          'MetOxi':[K_O2, 5e-4, np.nan, np.nan, np.nan],
          'SulRed': [np.nan, np.nan, 4e-3, 1e-4, np.nan],
          'SulOxi': [K_O2, np.nan, np.nan, np.nan, 1e-3],
          'MetOxiSul':[np.nan, 5e-4, np.nan, 1e-4, np.nan]}   #HSC: half saturation constant

HSC_df = pd.DataFrame(HSC_df)
HSC_df.index = ['Monod_o2', 'Monod_ch4', 'Monod_dom', 'Monod_so4', 'Monod_h2s']
# There are six numbers for each term, the first number is the maximum reaction rate, the next five numbers are the Monod
# terms for O2, CH4, DOM, SO4, H2S, respectively. If a certain molecule is not in the Monod expression of that
# reaction, I set it to be 0
HSC = np.array(HSC_df)

# 3. data frame for monod inhibition constants
I_df = {'DOMAer': [np.nan, np.nan, np.nan, np.nan, np.nan],
          'Met': [2.5e-4, np.nan, np.nan, np.nan, np.nan],
          'MetOxi':[np.nan, np.nan, np.nan, np.nan, np.nan],
          'SulRed': [np.nan, np.nan, np.nan, np.nan, np.nan],
          'SulOxi': [np.nan, np.nan, np.nan, np.nan, np.nan],
          'MetOxiSul': [np.nan, np.nan, np.nan, np.nan, np.nan]}   # half saturation constants for monod inhibition, I = inhb/(inhb + conc)

I_df = pd.DataFrame(I_df)
I_df.index = ['Monod_o2', 'Monod_ch4', 'Monod_dom', 'Monod_so4', 'Monod_h2s']
I = np.array(I_df)




#%% This function calculates the actual reaction rates based on the maximum reaction rates, monod constants, and concentrations
nreac = 6
nspecies = 5

def rate_calc(K, HSC, I, Conc):
    Rates = np.zeros(shape = (ngrids, ntimepoint, nreac), dtype = np.float32)
    
    for n in range(0,nreac):                      #calculate the rates for each reaction one by one                
        Monod = np.ones(shape = (ngrids, ntimepoint))  #init a Monod matrix for accumulative multiplication of the monod terms 
        Inhb = np.ones(shape = (ngrids, ntimepoint))  #init a  matrix for accumulative multiplication of the inhibition terms 
        
        for j in range(0,nspecies):
            hsc = HSC[j,n]
            if not np.isnan(hsc):             #if the hsc is not NAN, it means that the reaction rate is dependent on this species
                monod_temp = Conc[:,:,j] / (Conc[:,:,j] + hsc)   #use the concentration matrix of this species times the half saturation constant
                Monod = Monod * monod_temp
        
        for j in range(0,nspecies):
            inhb = I[j,n]
            if not np.isnan(inhb):             #if the inhb is not NAN, it means that the reaction rate is dependent on this species
                inhb_temp = inhb / (Conc[:,:,j] + inhb)   #use the concentration matrix of this species times the half saturation constant
                Inhb = Inhb * inhb_temp
        
        Rates[:,:,n] = K[0,n] * Monod * Inhb            #save the rate for this timepoint and this reaction, then go into the next loop, i.e. next timepoint    
                                                        #if not considering Inhibition, the inhibition terms needs to be turned off
    return Rates
        
            
            
#%% calculate reaction rates for all reactions
Conc = Full_Data[:,:,1:6]  #concentration of the five species investigated
Rates = rate_calc(K, HSC, I, Conc)


#%% verify the rates calculation
reac = 1
grid = 10
t = 10

oxygen = Full_Data[grid,t,1]
dom = Full_Data[grid,t,3]
monod = HSC[2,1]
inhb = I[0,1]


rate = K[0,reac] *    dom / (monod + dom)    *    inhb/(inhb + oxygen)

print((rate - Rates[grid,t,reac])/rate)



#%% Plot Rates vs concentration to investigate the nonlinear response of rate to concentration changes
#%% create concentration series for plotting Monod curve
x_o2 = np.arange(0, 2.5e-4, 1e-6)
x_ch4 = np.arange(0, 3e-2, 1e-4)
x_doc = np.arange(0, 3e-2, 1e-4)
x_so4 = np.arange(0, 3e-2, 1e-4)
x_h2s = np.arange(0, 3e-2, 1e-4)


#%% Extract concentration and rate data for the O2 injection layer

# reaction ID: 0: DOM Aerobic decomposition; 1: Methanogenesis; 2: CH4 oxidation
# 3: Sulfate reduction; 4: H2S oxidation


#%% 1) DOM aerobic decomposition dependence on O2 conc
t = 10  #timepoint
var_id = 1
reac_id = 0

K_o2 = 8e-6
conv = 1/2.5e-4*100
#o2_injection = np.zeros(100)
#o2_injection[[13,17,26,27,28,46,52,57,58,61,65,72,75,81,82]] = 1.3 * 1e3 *(1e-8/15/3600)/(0.01*0.01*0.075*1e3) * 1800  #concentration change caused by O2 injection, unit: mol L-1 s-1


sub = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
sup = str.maketrans("0123456789+-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻")

plt.rcParams.update({'font.size': 15})
# the Monod curve
y = x_o2 / (x_o2 + K_o2)
plt.plot(x_o2 * conv, y, 'k--')
plt.xlabel('O2 (%Air Sat.)'.translate(sub))
plt.ylabel('Monod')


conc = Data_Het[300:400, t, var_id]
rates = conc / (conc + K_o2)
plt.plot(conc *conv, rates, 'ro', label = 'Rate of each cell')

# the mean of actual rate
plt.plot(np.mean(conc) * conv, np.mean(rates), 'r*', markersize = 10, label = 'Mean rate')

# the rate calculated by mean of concentration (not considering the spatial heterogeneity of O2)
plt.plot(np.mean(conc) * conv, np.mean(conc)/(np.mean(conc) + K_o2), 'k*', markersize = 10, label = 'Rate of mean O2')



# the homogeneity mode
o2_homo = np.mean(Data_Homo[300:400, t, var_id])
plt.plot(o2_homo * conv, o2_homo/(o2_homo + K_o2), 'b*', markersize = 10, label = 'Homo rate')

# Calculate the Pearson's correlation coefficient to assess the linearity/nonlinearity
Pcoef1 = pearsonr(conc, rates)
Pcoef2 = pearsonr(x_o2, y)


print(Pcoef1[0], Pcoef2[0])
#plt.text(60, 0.55, 'K_O2=' + str(K_o2) + ' M\n' + 'Pearsons coef\n' + str(round(Pcoef1[0],2)))
#plt.text(55, 0.55, 'O2 injection=\n60' +' mmol m-2 d-1'.translate(sup) + '\nPearsons coef ' + str(round(Pcoef1[0],2)))
plt.text(55, 0.55, 'μmax=\n1e-6' +' mol L-1 s-1'.translate(sup) + '\nLinearity ' + str(round(Pcoef1[0],2)))
plt.legend(loc = 0)
plt.xlim(-5,102)
plt.ylim(0,1)

#%% Plot the time series of mean reaction rates of Het vs Homo
Monod_t = np.zeros((ntimepoint,3), dtype = float)

K_O2 = 1e-4
for t in range(0,ntimepoint):
    conc_o2 = Data_NO[300:400, t, 1]
    Monod_t[t,0] = np.mean(conc_o2 / (conc_o2 + K_O2))
    
    conc_o2 = Data_Homo[300:400, t, 1]
    Monod_t[t,1] = np.mean(conc_o2 / (conc_o2 + K_O2))
    
    conc_o2 = Data_Het[300:400, t, 1]
    Monod_t[t,2] = np.mean(conc_o2 / (conc_o2 + K_O2))


plt.plot(Monod_t[:,0], '-', color ='#303030', label = 'no O2 release'.translate(subscript))
plt.plot(Monod_t[:,1], '-',  color = '#24AEDB', label = 'Homogeneity')
plt.plot(Monod_t[:,2], '-', color = '#D02F5E', label = 'Heterogeneity')

plt.legend(loc = 'upper left')
plt.xlabel('Time (day)')
plt.ylabel('Monod')
plt.ylim(0,0.1)


#%%
conc = Full_Data[300:400, 10, 5]
rates = conc / (conc + 1e-3)
plt.plot(conc, rates, 'bo')


#%% Plot the theoretical Monod curve with different K and umax
x = x_o2
conv = 1 / 2.5e-4 * 100
y1 = x_o2 / (x_o2 + 8e-6)
y2 = x_o2 / (x_o2 + 3.75e-5)
y3 = x_o2 / (x_o2 + 1e-4) 
plt.plot(x*conv, y1, 'b-', label = 'K(O2) = 8e-6 [M]')
plt.plot(x*conv, y2, 'k-', label = 'K(O2) = 3.75e-5 [M]')
plt.plot(x*conv, y3, 'r-', label = 'K(O2) = 1e-4 [M]')

plt.legend(loc = 0)
plt.xlabel('O2 %Air Sat.'.translate(sub))
plt.ylabel('Monod')

#%%
x = x_o2
conv = 1 / 2.5e-4 * 100
y1 = 5e-8 * x_o2 / (x_o2 + 8e-6)
y2 = 1e-7 * x_o2 / (x_o2 + 8e-6)
y3 = 5e-7 * x_o2 / (x_o2 + 8e-6) 
plt.plot(x*conv, y1, 'b-', label = 'μmax=10'+'-8 mol L-1 s-1'.translate(sup))
plt.plot(x*conv, y2, 'k-', label = 'μmax=10'+'-7 mol L-1 s-1'.translate(sup))
plt.plot(x*conv, y3, 'r-', label = 'μmax=10'+'-6 mol L-1 s-1'.translate(sup))

plt.legend(loc = 0)
plt.xlabel('O2 %Air Sat.'.translate(sub))
plt.ylabel('Rate (mol L-1 s-1)'.translate(sup))
#%% Calculate the O2 consumption rate

R_o2_Het = Rates_Het[300:400, t, 0] + Rates_Het[300:400, t, 2] + Rates_Het[300:400, t, 4]   # the sum of DOC aerobic decomposition rate, CH4 aerobic oxidation rate and H2S oxidation rate
conc = Data_Het[300:400, t, 1]
Pcoef = pearsonr(conc, R_o2_Het)
plt.plot(conc / 2.5e-4 * 100, R_o2_Het, 'bo', label = 'Het rate of each cell')

plt.plot(np.mean(conc) / 2.5e-4 * 100, np.mean(R_o2_Het), 'b*', label = 'Het mean rate')

R_o2_Homo = Rates_Homo[300:400, t, 0] + Rates_Homo[300:400, t, 2] + Rates_Homo[300:400, t, 4]
conc = Data_Homo[300:400, t, 1]
plt.plot(conc / 2.5e-4 * 100, R_o2_Homo, 'r*', label = 'Homo rate')

plt.xlim(-5,60)
plt.xlabel('% O2 sat')
plt.ylabel('O2 consumption rate (M s-1)')
plt.legend(loc = 0)


plt.text(30, 1.5e-8, 'K_O2=' + str(K_o2) + ' [M]\n' + 'Pearsons correlation\ncoefficient ' + str(round(Pcoef[0],2)))


#%%
y = x_o2 / (x_o2 + 1e-4)
plt.plot(x_o2, y, 'b-')

y = 5* x_o2 / (x_o2 + 1e-3)
plt.plot(x_o2, y, 'r-')



#%% 2) DOM aerobic decomposition dependece on DOM conc
var_id = 3
reac_id = 0

conc = Full_Data[300:400, timepoint, var_id]
rates = K[0, reac_id] * conc / (conc + 5e-3)
plt.plot(conc, rates, 'bo')

# the Monod curve
y = K[0, reac_id] * x_doc / (x_doc + 5e-3)
plt.plot(x_doc, y, 'b-')


#%% CH4 oxidation dependece on O2 conc
var_id = 1
reac_id = 2

conc = Full_Data[300:400, 15, var_id]
rates = conc / (conc + 8e-6)
plt.plot(conc, rates, 'bo')

# the Monod curve
y = x_o2 / (x_o2 + 8e-6)
plt.plot(x_o2, y, 'b-')

# Calculate the Pearson's correlation coefficient to assess the linearity/nonlinearity
Pcoef1 = pearsonr(conc, rates)
Pcoef2 = pearsonr(x_o2, y)

print(Pcoef1[0], Pcoef2[0])


#%% compare rates of all reactions
grid_id = 361
t = 30

for i in range(0,5):
    plt.bar(i, Rates[grid_id,t,i], color = 'lightblue', width = 0.3)

plt.xticks(np.arange(0,5), K_df.keys())
plt.ylabel('Rate (umol L-1 s-1)')
#plt.ylim(0,0.25e-8)


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


#%% compare the modeled reaction rates with the meaured rates from literatures

#measured reaction rates from literatures, units: mol L-1 porewater s-1
R_df = {'DOMAer': [1e-8], 'Met': [9e-11], 'MetOxi': [1.6e-10], 'SulRed': [2.26e-9],'SulOxi': [1.6e-10]}
R_df = pd.DataFrame(R_df)
R = np.array(R_df)



t = 30       #specify the point time
reac = 1     #specify the reaction 
layer = 6        #specify which layer, layer 1 is the top layer
i_start = (nz - layer) * nx * ny
i_end = i_start + nx * ny

plt.bar(np.arange(1,101,1),Rates[i_start:i_end,t,reac])
plt.bar(101, R[0, reac], width = 0.8, color = 'r')


#%%
plt.plot(np.arange(1,901,1),Rates[:,t,reac], 'ro')



#%%  reaction rate for the ebullition process

V = 4e-7
K = 1e-2
c = np.arange(0,1.2e-3,1e-4)
R = V*(c/(c+K))*(c/(c+K))*(c/(c+K))*(c/(c+K))

plt.plot(c, R)

#%% Monod and Inverse Monod inhibition
V = 4e-7
K1 = 0.1
K2 = 0.1
c = np.arange(0,0.1,1e-4)
R0 = V * (c/(c+K1))
R1 = V * (c/(c+K1)) * (c/(c+K2))
R2 = V * (c/(c+K1)) * (c/(c+K2)) * (c/(c+K2))

plt.plot(c, R0, 'g-')
plt.plot(c, R1, 'r-')
plt.plot(c, R2, 'b-')


#%%
cth = 1e-4
f = 1e4 / cth
x = np.arange(0,1e-3,1e-5)
y = np.arctan((x - cth)*f) / 3.14 + 0.5
  
plt.plot(x,y)
plt.xlabel('Concentration (M)')


#%%
time = [0,1,2,3,4,5]
ch4 = np.array([2.000000E-03, 1.136085E-03, 9.907725E-04, 9.876367E-04, 9.852182E-04, 9.831822E-04])

plt.plot(time, ch4, label = 'changes in CH4 over time')
plt.plot([0,5], [1e-3,1e-3], 'r-', label = 'threshold concentration')
plt.ylim([0.0005,0.0022])

plt.xlabel('Time (day)')
plt.ylabel('CH4 concentration (M)')
plt.legend(loc = 0)