# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 14:52:04 2022

@author: YZ60069
"""

# This file compiles model output at all different time points
# and plot the mean concentration profiles of different model set up

from contextlib import AsyncExitStack
import pandas as pd
#from socket import AF_X25
import numpy as np
import pickle
import statistics as st
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import os
os.chdir('C:/Users/yz60069/TAI/TAI_fresh')
#os.chdir('C:/MBL/Research/PFLOTRAN DATA/pflotran outputs/OxyHet/Marsh Interior/All inundated/ root')
import scipy.io
from scipy.stats import pearsonr
#mat = scipy.io.loadmat('WaterTableData.mat')
#%% specify the pflotran output structure
nx = 10
ny = 10
nz = 9
ngrids = nx * ny * nz
ntimepoint = 11


#%% import data
nvars = 9;  #number of variables that I want to investigate
Full_Data = np.empty( shape = (ngrids,ntimepoint,nvars), dtype = np.float32)
Var_str = {0: 'Liquid Saturation', 1: 'Total O2(aq) [M]', 2: 'Total CH4(aq) [M]', 3: 'Total DOM1 [M]', 4: 'Total SO4-- [M]',
           5: 'Total H2S(aq) [M]', 6: 'Total Tracer1 [M]', 7: 'Total Tracer2 [M]', 8: 'Total Tracer3 [M]',
           9: 'Total Tracer5 [M]', 10: 'Total Tracer6 [M]'}

#
# the dictionary in which the variable ID is coupled with the variable name

thefile = 'TAI_wetland2'
for var_id in range(0,nvars):
    
    var_str = Var_str[var_id]
    # Read in data from PFLOTRAN
    Data_Page = np.empty( shape = (ngrids,ntimepoint), dtype = np.float32)
    w = 0
    
    
    
    for i in range(0,ntimepoint):
       
        if i < 10:
           file_name = thefile + '-00' + str(i) + '.tec'
        elif (i == 10 or i > 10) and i < 100:
           file_name = thefile + '-0' + str(i) + '.tec'
        elif (i == 100 or i > 100) and i < 1000:
           file_name = thefile + '-' + str(i) + '.tec'
        
        with open(file_name,'r') as inputFile:
            read_lines = inputFile.readlines()
           
        parameter_str = read_lines[1] #read in the 2nd line, getting variable names as strings
        parameter_list = parameter_str.replace('"','').replace('\n','').split(',')
        data_list = [] #define a list 
        results = {} #define a dic called results where the numbers are paired with the variable names
        
        for k in range(3, len(read_lines)):
            data_str = read_lines[k].replace(' -','  -').replace('    ','  ').split('  ')
            data_list.append(np.array(data_str, dtype = np.float32))
            
          
        data_list = np.array(data_list)   #convert the list into numpy array
        
        for j in range(0, len(parameter_list)):
            results[parameter_list[j]] = data_list[:,j]
        
        Data_Page[:,w] = results[var_str]
        w = w + 1

    

    Full_Data[:,:,var_id] = Data_Page



# the coordinates
Coord = np.empty( shape = (ngrids,3) , dtype = np.float32 )
Coord[:,0] = results['VARIABLES=X [m]']
Coord[:,1] = results['Y [m]']
Coord[:,2] = results['Z [m]']



#%% Plot the depth profiles of the investigated variable for different timepoints
var_id = 1 #specify which variable to plot
var_str = Var_str[var_id]
interval = nx * ny 
depths = Coord[0: ngrids :interval,2] - 0.7  #minus the depth of the soil profile

if var_id == 1:
    conv = 1/2.5e-4*100
else:
    conv = 1e6

plt.rcParams.update({'font.size': 15})

for i in range(0,ntimepoint,1):
    conc = Full_Data[37 : ngrids : interval, i, var_id] * conv    # 37 is a root cell
    plt.plot(conc, depths)
    
plt.ylabel('Soil Depth (m)')
plt.xlabel(var_str[0:len(var_str)-4] + ' (uM)')
#plt.xlim(-1e-5,1e-3)
#plt.xlabel('%O2 sat')
#plt.xticks(np.arange(0, 2e-4, step = 5e-5))   
#plt.xticks(np.arange(5.8e-4, 6.2e-4, step = 1e-5)) 
# plt.ylim(-0.02,0)
#plt.xlim([1990,2010])

#%% plot the time series of the variable
var_id = 1
plt.plot(Full_Data[325, 0:30, var_id])   #bigger row number means closer to the soil surface
ax=plt.gca()
ax.set_xlabel('Time (day)')
ax.set_ylabel(var_str)
plt.rcParams.update({'font.size': 13})

#%% plot the depth profiles of all different columns
ncols = nx * ny
t = 30

for i in range(0, ncols):
    conc = Full_Data[i:ngrids:ncols, t, var_id] * 1e6
    plt.plot(conc, depths)
    
plt.xlabel(var_str[0:len(var_str) - 4] + ' uM')
#plt.xlabel('O2 sat(%)')
plt.ylabel('Soil Depth (m)')
plt.rcParams.update({'font.size': 12})
#plt.title('Conc. profiles of all columns')
# plt.ylim(-0.2, -0.03)
# plt.xlim(0,20)

Data_varin = Full_Data[:,:, var_id].reshape(nz, nx*ny, ntimepoint)
MeanProfs = Data_varin.mean(axis = 1)
depths = Coord[0:ngrids:interval, 2] - 0.7
plt.plot(MeanProfs[:,ntimepoint-1], depths)
conc = Full_Data[47:ngrids:interval, ntimepoint-1, var_id]
plt.plot(conc, depths, 'k*')

ax = plt.gca()
ax.set_xlabel(var_str)
ax.set_ylabel('Soil Depth (m)')
plt.rcParams.update({'font.size': 12})



#%% save data
filename = 'NO.pickle'
with open('' + filename, 'wb') as handle:
    pickle.dump([Full_Data, depths, Mass_df, Rates], handle)



#%% import the processed pflotran output
import pickle 
filename = 'growing_season.pickle'
with open('C:/MBL/Research/PFLOTRAN DATA/pflotran outputs/OxyHet/Creek Bank/' + filename, 'rb') as handle:
    Growing_Season_CB = pickle.load(handle)
 

#%%
Data_NO = Full_Data
MetF_NO = MetF


#%%      
Data_Homo = Full_Data
MetF_Homo = MetF



#%%
Data_Het = Full_Data
MetF_Het = MetF


#%%
Data_Het_30roots = Full_Data
MetF_Het_30roots = MetF
Rates_Het_30roots = Rates


#%%
Data_Het_50roots = Full_Data
MetF_Het_50roots = MetF
Rates_Het_50roots = Rates


#%%
Data_Het_70roots = Full_Data
MetF_Het_70roots = MetF
Rates_Het_70roots = Rates


#%% Calculate and Plot the mean profiles of different O2 injection modes
# Calculate the mean profiles
t = 10       #specify the time point, here extract the data on day 30
MP_NO = np.zeros((9,ntimepoint, 6), dtype = float)
MP_Homo = np.zeros((9, ntimepoint, 6), dtype = float)
MP_Het = np.zeros((9, ntimepoint, 6), dtype = float)
#MP_Het_30roots = np.zeros((9, ntimepoint, 6), dtype = float)
#MP_Het_50roots = np.zeros((9, ntimepoint, 6), dtype = float)
#MP_Het_70roots = np.zeros((9, ntimepoint, 6), dtype = float)


for t in range(0, ntimepoint):
    for var_id in range(0,6):
        #reshape the matrix of concentration to a 9*100 matrix, each row is the data of one soil layer containing 100 cells, and calculate the mean of all cells within the same layer
        MP_NO[:, t, var_id] = Data_NO[:, t, var_id].reshape(9,100).mean(axis = 1)      
        MP_Homo[:, t, var_id] = Data_Homo[:, t, var_id].reshape(9,100).mean(axis = 1)
        MP_Het[:,t, var_id] = Data_Het[:, t, var_id].reshape(9,100).mean(axis = 1)
        #MP_Het_30roots[:,t, var_id] = Data_Het_30roots[:, t, var_id].reshape(9,100).mean(axis = 1)
        #MP_Het_50roots[:,t, var_id] = Data_Het_50roots[:, t, var_id].reshape(9,100).mean(axis = 1)
        #MP_Het_70roots[:,t, var_id] = Data_Het_70roots[:, t, var_id].reshape(9,100).mean(axis = 1)
        
        
#%% Plot the profiles from different set up
plt.rcParams.update({'font.size': 15})

t = 10
var_id = 5
 
if var_id == 1:
    conv = 1/2.5e-4*100
else:
    conv = 1e3

y = depths[1:11]*100   #convert depth values to cm

sub = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
sup = str.maketrans("0123456789+-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻")

plt.plot(MP_NO[1:11, t, var_id] * conv, y, '-', color ='#303030', label = 'noROL')  #convert depth to cm
plt.plot(MP_Homo[1:11, t, var_id] * conv, y, '-', color = '#24AEDB', label = 'ROL_Homo')
plt.plot(MP_Het[1:11, t, var_id] * conv, y, '-', color = '#D02F5E', label = 'ROL_Het')
#plt.plot(MP_Het_30roots[1:11, t, var_id] * conv, y, '--', color = '#D02F5E', label = 'ROL_Het, 30roots')
#plt.plot(MP_Het_50roots[1:11, t, var_id] * conv, y, '-.', color = '#D02F5E', label = 'ROL_Het, 50roots')
#plt.plot(MP_Het_70roots[1:11, t, var_id] * conv, y, '-*', color = '#D02F5E', label = 'ROL_Het, 70roots')



if var_id == 1:
    xlab = 'O2'.translate(sub) +' %Air Sat.'.translate(sub)
    #plt.xlim(-2,52)
elif var_id == 2:
    xlab = 'CH4 (aq, mmol'.translate(sub) + ' L-1)'.translate(sup)
elif var_id == 3:
    xlab = 'DOC (mmol' + ' L-1)'.translate(sup)
    plt.xlim(0,8)
elif var_id == 4:
    xlab = 'SO4'.translate(sub) + '2- (mmol L-1)'.translate(sup)
    #plt.xlim(4.9, 9.2)
elif var_id == 5:    
    xlab = 'H2S(aq,'.translate(sub)  + ' mmol L-1)'.translate(sup)

plt.xlabel(xlab)
plt.ylabel('Depth(cm)')
if var_id == 1:
    plt.legend(loc = 'lower right')


#%% Plot the time series of concentration at the rooting zone
var_id = 1 
if var_id == 1:
    conv = 1/2.5e-4*100
else:
    conv = 1e3
    
plt.plot(MP_NO[3, :, var_id] * conv, '-', color ='#303030', label = 'no O2 release'.translate(subscript))  #convert depth to cm
plt.plot(MP_Homo[3, :, var_id] * conv, '-', color = '#24AEDB', label = 'Homogeneity')
plt.plot(MP_Het[3, :, var_id] * conv, '-', color = '#D02F5E', label = 'Heterogeneity')

if var_id == 1:
    ylab = 'O2 saturation (%)'
elif var_id == 2:
    ylab = 'CH4(aq, mmol'.translate(sub) + ' L-1)'.translate(sup)
elif var_id == 3:
    ylab = 'DOC (mmol L-1)'.translate(sup)
elif var_id == 4:
    ylab = 'SO4'.translate(sub) + '2-(μM)'.translate(sup)
elif var_id == 5:
    ylab = 'H2S(aq, mmol L-1)'.translate(sub)

plt.xlabel('Time (day)')
plt.ylabel(ylab)

if var_id == 1:
    plt.legend(loc = 0)

#%% compile the species concentration at the root layer
ConcCmpr = np.zeros((6,5), dtype = float)
ConcCmpr[0,0:5] = MP_NO[3,t,1:6]
ConcCmpr[1,0:5] = MP_Homo[3,t,1:6]
ConcCmpr[2,0:5] = MP_Het[3,t,1:6]

#%% Calculate the percentage difference between Het and Homo
ConcCmpr[3,] = (ConcCmpr[2,] - ConcCmpr[1,]) / ConcCmpr[1,] * 100  #Het vs Homo
ConcCmpr[4,] = (ConcCmpr[1,] - ConcCmpr[0,]) / ConcCmpr[0,] * 100  #Homo vs noROL
ConcCmpr[5,] = (ConcCmpr[2,] - ConcCmpr[0,]) / ConcCmpr[0,] * 100   #Het vs noROL



#%% Compile the CH4 fluxes
FluxCmpr = np.zeros((6,4), dtype = float)
FluxCmpr[0:3,:] = np.array([MetF_NO, MetF_Homo, MetF_Het]).reshape(3,4)

# calculate the percentage differences
FluxCmpr[3,:] = (FluxCmpr[2,:] - FluxCmpr[1,:]) / FluxCmpr[1,:] * 100
FluxCmpr[4,:] = (FluxCmpr[1,:] - FluxCmpr[0,:]) / FluxCmpr[0,:] * 100
FluxCmpr[5,:] = (FluxCmpr[2,:] - FluxCmpr[0,:]) / FluxCmpr[0,:] * 100


FluxCmpr_df = pd.DataFrame(FluxCmpr)
FluxCmpr_df.columns = ['Total Flux', 'Diffusion', 'Plant-mediated', 'Ebullition']
FluxCmpr_df.index = ['NO', 'Homo', 'Het', 'Diff_Het_Homo', 'Diff_Homo_NO', 'Diff_Het_NO']


#%% Plot the CH4 fluxes results
plt.rcParams.update({'font.size': 15})
group_names = ['Total', 'Surface D.', 'Plant-F.', 'Ebullition']
bar_labels = ['noROL','ROL_Homo', 'ROL_Het']

# Set the bar width and positions
bar_width = 0.2
x = np.arange(len(group_names))

colors = ['#585858','#58ACFA', '#F7819F']
# Plot the bars for each group
for i in range(0,3):
    plt.bar(x + i * bar_width, FluxCmpr[i], width=bar_width, label=bar_labels[i], color = colors[i])

# Set the x-axis tick labels
plt.xticks(x + (len(bar_labels) - 1) * bar_width / 2, group_names)

# Set the axis labels and title

#plt.xlim(0,15)
#plt.ylim(-0.5,3)
#plt.xticks([0, 2, 4, 6, 8, 10])

# Add a legend
plt.legend(loc=0)


sub = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
sup = str.maketrans("0123456789+-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻")
plt.ylabel('CH4'.translate(sub) + ' Flux\n(mmol m-2 d-1)'.translate(sup))




#%%  S cycling analysis
Data_Homo_S = Full_Data
MetF_Homo_S = MetF
Rates_Homo_S = Rates

#%%
Data_Homo_noS = Full_Data
MetF_Homo_noS = MetF
Rates_Homo_noS = Rates

#%%      
Data_S = Full_Data
MetF_S = MetF
Rates_S = Rates


#%%
Data_noS = Full_Data
MetF_noS = MetF
Rates_noS = Rates


#%% Calculate and Plot the mean profiles of different O2 injection modes
# Calculate the mean profiles
t = 10       #specify the time point, here extract the data on day 30
MP_Homo_S = np.zeros((9,ntimepoint, 6), dtype = float)
MP_Homo_noS = np.zeros((9,ntimepoint, 6), dtype = float)
MP_S = np.zeros((9, ntimepoint, 6), dtype = float)
MP_noS = np.zeros((9, ntimepoint, 6), dtype = float)

for t in range(0, ntimepoint):
    for var_id in range(0,6):
        #reshape the matrix of concentration to a 9*100 matrix, each row is the data of one soil layer containing 100 cells, and calculate the mean of all cells within the same layer
        MP_Homo_S[:, t, var_id] = Data_Homo_S[:, t, var_id].reshape(9,100).mean(axis = 1)      
        MP_Homo_noS[:, t, var_id] = Data_Homo_noS[:, t, var_id].reshape(9,100).mean(axis = 1)      
        MP_S[:, t, var_id] = Data_S[:, t, var_id].reshape(9,100).mean(axis = 1)
        MP_noS[:,t, var_id] = Data_noS[:, t, var_id].reshape(9,100).mean(axis = 1)

#%% Plot the profiles from different set up
plt.rcParams.update({'font.size': 15})

t = 10
var_id = 1
 
if var_id == 1:
    conv = 1/2.5e-4*100
else:
    conv = 1e3

y = depths[1:11]*100   #convert depth values to cm

#plt.plot(MP_Homo_S[1:11, t, var_id] * conv, y, '-', color ='#0174DF', label = 'with S cycling'.translate(subscript))  #convert depth to cm
#plt.plot(MP_Homo_noS[1:11, t, var_id] * conv, y, '--', color ='#0174DF', label = 'no S cycling'.translate(subscript))  #convert depth to cm

plt.plot(MP_S[1:11, t, var_id] * conv, y, '-', color = '#0174DF', label = 'with S cycling')
plt.plot(MP_noS[1:11, t, var_id] * conv, y, '--', color = '#0174DF', label = 'no S cycling')

sub = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
sup = str.maketrans("0123456789+-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻")

if var_id == 1:
    xlab = 'O2'.translate(sub) +' %Air Sat.'.translate(sub)
    #plt.xlim(-2,52)
elif var_id == 2:
    xlab = 'CH4 (aq, mmol'.translate(sub) + ' L-1)'.translate(sup)
    plt.xlim(0, 1.5)
elif var_id == 3:
    xlab = 'DOC (mmol' + ' L-1)'.translate(sup)
    plt.xlim(0,15)
elif var_id == 4:
    xlab = 'SO4'.translate(sub) + '2- (mmol L-1)'.translate(sup)
    #plt.xlim(4.9, 9.2)
elif var_id == 5:    
    xlab = 'H2S(aq,'.translate(sub)  + ' mmol L-1)'.translate(sup)

plt.xlabel(xlab)
plt.ylabel('Depth(cm)')
if var_id == 1:
    plt.legend(loc = 0)




#%% Analysis of S cycling data, Compile results
ConcCmpr = np.zeros((4,5), dtype = float)
ConcCmpr[0,:] = MP_Homo_S[3,10,1:6]        #first row, ROL_Homo with S
ConcCmpr[1,:] = MP_S[3,10,1:6]        #second row, ROL_Het with S
ConcCmpr[2,:] = MP_Homo_noS[3,10,1:6]        #thrid row, ROL_Homo no S
ConcCmpr[3,:] = MP_noS[3,10,1:6]        #forth row, ROL_Het no S



#%% Plot CH4 emissions data
MetF = np.zeros((2,4), dtype = float)
MetF[0,:] = np.array(MetF_S)
MetF[1,:] = np.array(MetF_noS)


#%%
import matplotlib.pyplot as plt
import numpy as np


# Generate some sample data
group_names = ['Total', 'Surface D.', 'Plant-F.', 'Ebullition']
bar_labels = ['with S cycling', 'no S cycling']

# Set the bar width and positions
bar_width = 0.2
x = np.arange(len(group_names))

#colors = ['#2E2E2E', '#BDBDBD']
colors = ['#2E9AFE', '#2E9AFE']
# Plot the bars for each group
for i in range(0,2):
    if i == 0:
        plt.bar(x + i * bar_width, MetF[i], width=bar_width, label=bar_labels[i], color = 'w', edgecolor = '#F5A9BC', hatch = '//', linewidth = 2)
    elif i == 1:
        plt.bar(x + i * bar_width, MetF[i], width=bar_width, label=bar_labels[i], color = 'w', edgecolor = '#F5A9BC', hatch = '', linewidth = 2)

# Set the x-axis tick labels
plt.xticks(x + (len(bar_labels) - 1) * bar_width / 2, group_names)

# Set the axis labels and title

#plt.xlim(0,15)
#plt.ylim(-0.5,3)
#plt.xticks([0, 2, 4, 6, 8, 10])

# Add a legend
plt.legend(loc=0)


sub = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
sup = str.maketrans("0123456789+-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻")
plt.ylabel('CH4'.translate(sub) + ' Flux\n(mmol m-2 d-1)'.translate(sup))
#plt.title('Homogeneity Mode')

#%%
# # # field data
# # Conc = np.array(FieldData['Sulfide'])
# # depths = -np.array(FieldData['Depth']) * 1e-2
# # plt.plot(Conc, depths, 'ko', label = 'Field')

# # Conc = Data_noroot[1][:,:,var_id].reshape(nz, nx*ny, ntimepoint)
# # MeanProfs = Conc.mean(axis = 1) 
# # depths = Data_noroot[2]
# # plt.plot(MeanProfs[:,timepoint], depths, 'k-', label = 'no heterogeneity')  #plot data of the last timepoint



# # Conc = Data_1root[1][:,:,var_id].reshape(nz, nx*ny, ntimepoint)
# # MeanProfs = Conc.mean(axis = 1) 
# # depths = Data_1root[2]
# # plt.plot(MeanProfs[:,timepoint], depths, 'r-', label = '1 root')

# t = 30

# Conc = Full_Data_roots[:,t,var_id].reshape(nz, nx*ny, ntimepoint)
# MeanProfs = Conc.mean(axis = 1) 
# depths = depths
# plt.plot(MeanProfs[:,t], depths, 'g-', label = '9 roots')



# Conc = Data_noO2inj[1][:,:,var_id].reshape(nz, nx*ny, ntimepoint)
# MeanProfs = Conc.mean(axis = 1) 
# depths = Data_noO2inj[2]
# plt.plot(MeanProfs[:,timepoint], depths, 'b-', label = 'no plant O2 injection')

# #saturation line for oxygen
# #plt.plot([8, 8], [0, -0.7] , 'b--')


# plt.xlabel(Var_str[var_id])
# plt.ylabel('Depth (m)')
# plt.legend(loc = 0)

# # os.chdir('C:\MBL\Conferences\AGU 2022\Poster')
# # fig.savefig('Sulfide.eps', dpi = 600, format='eps')

#%% Plot the time series of different set up
plt.rcParams.update({'font.size': 10})
layer_id = 4
var_id = 5

Conc = Data_noroot[1][:,:,var_id].reshape(nz, nx*ny, ntimepoint)
MeanProfs = Conc.mean(axis = 1)
plt.plot(MeanProfs[layer_id, :], 'k-', label = 'no heterogeneity')  #plot data of the last timepoint



Conc = Data_1root[1][:,:,var_id].reshape(nz, nx*ny, ntimepoint)
MeanProfs = Conc.mean(axis = 1)
plt.plot(MeanProfs[layer_id, :], 'r-', label = '1 root')



Conc = Data_9roots[1][:,:,var_id].reshape(nz, nx*ny, ntimepoint)
MeanProfs = Conc.mean(axis = 1)
plt.plot(MeanProfs[layer_id, :], 'b-', label = '9 roots')


Conc = Data_noO2inj[1][:,:,var_id].reshape(nz, nx*ny, ntimepoint)
MeanProfs = Conc.mean(axis = 1)
plt.plot(MeanProfs[layer_id, :], 'g-', label = 'no plant O2 injection')



plt.xlabel('Time (day)')
plt.ylabel('Concentration (M)')
plt.legend(loc = 0)


#%%
joey = Full_Data[:,168]
joey = joey.reshape(5,49)
#%%
plt.plot(conc_49cols, depths, 'b-')
plt.plot(Full_Data[:,168], depths, 'r-')

 #%% plot O2 and liquid saturation profile
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

ax1.plot(Full_Data_ls[:,4], 'b-')   #larger column number, closer to the soil surface
ax1.spines['left'].set_color('b')
ax1.set_ylabel('Liquid Saturation')

ax2.plot(Full_Data_o2[:,4], 'r-')
ax2.spines['right'].set_color('r')
ax2.set_ylabel('O2(aq) [M]')

#%% import the observation data of water table depth 
os.chdir('C:/MBL/Research/Typha data/water table')
WTMean = []


from csv import reader
file_name = "WTMeanCB.csv"

with open (file_name, 'r') as csv_file:
    csv_data = reader(csv_file)
    header = next(csv_file)
    
    for row in csv_data:
        WTMean.append(row)

WTMean = np.array(WTMean)

#%%  Plot the observed water table with the modeled liquid saturation
fig, ax1 = plt.subplots()

ax1.plot(Full_Data[4,:], 'r-', label = 'Liquid Saturation (PFLOTRAN)')  
ax1.set_ylim(0,1.1)
ax1.set_ylabel('Liquid Saturation', color = 'r')

ax2 = ax1.twinx()
ydata = np.array(WTMean[1515:1850, 1], dtype = float)
ax2.plot(range(0,335), ydata, 'b-', label = 'Water Table')
ax2.plot([0, 335], [0.95, 0.95], 'k-', label = 'Soil Surface')
ax2.spines['right'].set_color('b')
ax2.spines['left'].set_color('r')
ax2.set_ylabel('Water Table(m)', color = 'b')
#ax2.set_ylim(2.95, 3.5)

ax1.set_xlabel('Time (hr)')
plt.rcParams.update({'font.size': 13})
fig.legend(loc = 'lower left', bbox_to_anchor=(0,0.05), bbox_transform=ax.transAxes, fontsize = 10)

#%% Convert the liquid saturation to water table level
WT_mod = []
WaterHeight = []
D = [0.2]*5
Area = 1 * 1
soil_depth = 1
Vol = np.array(D) * Area


for i in range():
    WaterHeight.append(sum(arr * Vol) /Area)
    WT_mod.append(sum(arr * Vol) / Area + (0.95 - soil_depth) )





#%% compare the modeled water level with the observation
plt.plot(range(1,25), np.array(WTMean[200:248:2,1], dtype = np.float32))
plt.plot(WT_mod, 'r.')



#%% Compare modeled O2 and observed O2
#%% import O2 data
os.chdir('C:/MBL/Research/Typha data/Redox')
OxyConc = []


from csv import reader
file_name = "Oxygen_Conc_CB.csv"

with open (file_name, 'r') as csv_file:
    csv_data = reader(csv_file)
    header = next(csv_file)
    
    for row in csv_data:
        OxyConc.append(row)

OxyConc = np.array(OxyConc)

#%% Plot the modeled O2 against observed O2
plt.plot(Full_Data[1,:]*32*1000, 'b--', label = 'Modeled O2')

ydata = np.array(OxyConc[3773:7805:12,24], dtype = float)
plt.plot(ydata, 'b-', label = 'Estimated O2 (redox)')
plt.legend(loc = 0)

ax = plt.gca()
ax.set_xlabel('Time (hr)')
ax.set_ylabel('O2(aq) (mg/L)')
ax.set_ylim(-0.1, 4.2)


#%% Draw the discretization of the model along depths
plt.plot(np.ones(len(depths)), depths, 'ko')



#%% relationship between %O2 sat and other variables
layer = 6
i_start = (nz - layer) * nx * ny
i_end = i_start + nx * ny
t = 30

oxy_conc = Full_Data[i_start:i_end, t, 1] / 2.5e-4 * 100
ch4_conc = Full_Data[i_start:i_end, t, 3] * 1e6
plt.plot(oxy_conc, ch4_conc, 'ro')
plt.xlabel('%O2 sat')
plt.ylabel ('DOM umol L-1')


#%%
dom_conc = Full_Data[i_start:i_end, t, 3] * 1e6
ch4_conc = Full_Data[i_start:i_end, t, 2] * 1e6
plt.plot([1500, 3000],[500,1000],'k-')
plt.plot(dom_conc, ch4_conc, 'ro')

plt.xlabel('DOM umol L-1')
plt.ylabel ('CH4 (aq) umol L-1')

#%% plot CH4 vs DOM only for samples with low O2
# dig out the row numbers for grids with low O2, lower than 20% sat
LowOxyList = [];
for i in range(i_start, i_end):
    if Full_Data[i,t,1] < (20 / 100 * 2.5e-4):
        LowOxyList.append(i)

LowOxyList = np.array(LowOxyList)        
plt.plot(Full_Data[LowOxyList,t,1], 'ko')


#%% plot CH4 vs DOM
x = Full_Data[LowOxyList,t,1] / 2.5e-4 * 100
y = Full_Data[LowOxyList,t,3] * 1e6
plt.plot(x, y, 'ko')
plt.xlabel('DOM umol/L')
plt.ylabel('CH4 umol/L')

#%% CH4 concentration vs methane oxidation rate
x = Rates[i_start:i_end, t, 1]
y = ch4_conc
plt.plot(x, y, 'ro')




#%% export to csv
df = pd.DataFrame(ConcCmpr)
df.to_csv('concentration_compare', index = False)

