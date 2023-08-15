# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 14:52:04 2022

@author: YZ60069
"""

# This file reads in the PFLOTRAN model output files in the .tec format, compile
# data into a 3-dimensional array consisting of the concentration of different 
# porewater substances at each cell at each time point (the Full_Data variable)

#%% 1. Import packages

from contextlib import AsyncExitStack
import pandas as pd
import numpy as np
import pickle
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import os
os.chdir('C:/Users/yz60069/TAI/TAI_fresh')
import scipy.io


#%% 2. Specify the simulation domain structure and number of timepoints
nx = 10   # number of cells along the x axis
ny = 10   # number of cells along the y axis
nz = 9    # number of cells along the z axis, i.e., along depth
ngrids = nx * ny * nz   #total number of cells
ntimepoint = 11   #total number of timepoints, we ran simulations for 10 days, plus the initial timepoint


#%% 3. Read in the .tec files
nvars = 11;  #number of output variables
Full_Data = np.empty( shape = (ngrids,ntimepoint,nvars), dtype = np.float32)

# create a dictionary, establishing number handles for each variable
Var_str = {0: 'Liquid Saturation', 1: 'Total O2(aq) [M]', 2: 'Total CH4(aq) [M]', 3: 'Total DOM1 [M]', 4: 'Total SO4-- [M]',
           5: 'Total H2S(aq) [M]', 6: 'Total Tracer1 [M]', 7: 'Total Tracer2 [M]', 8: 'Total Tracer3 [M]',
           9: 'Total Tracer5 [M]', 10: 'Total Tracer6 [M]'}       


thefile = 'TAI_wetland2'   # the main part of the file name

# Read in the .tec files one by one and extract data and texts separately  
for i in range(0,ntimepoint):  # one timepoint is one tec file
   
    if i < 10:
       file_name = thefile + '-00' + str(i) + '.tec'    # create the full file name
    elif (i == 10 or i > 10) and i < 100:
       file_name = thefile + '-0' + str(i) + '.tec'
    elif (i == 100 or i > 100) and i < 1000:
       file_name = thefile + '-' + str(i) + '.tec'
    
    with open(file_name,'r') as inputFile:
        read_lines = inputFile.readlines()
       
    parameter_str = read_lines[1] #read in the 2nd line, getting variable names as strings
    parameter_list = parameter_str.replace('"','').replace('\n','').split(',')
    data_list = [] #define a list 
    
    for k in range(3, len(read_lines)):
        data_str = read_lines[k].replace(' -','  -').replace('    ','  ').split('  ')
        data_list.append(np.array(data_str, dtype = np.float32))
        
    #convert the list into numpy array and save it to the Full_Data
    data_list = np.array(data_list)   
    
    
    # add the data of each variable to Full_Data for the timepoint currently being processed
    
    for j in range(0, len(Var_str)):
        index = parameter_list.index(Var_str[j])   #find out the index of the variable in the parameter list
        Full_Data[:,i,j] = data_list[:, index]     #extract the data of this variable from the index list, and pair it with its number handle
        
               


# In the Full_Data variable, axis 0 is looking at different cells, axis 1 is looking at different timepoints
# axis 2 is looking at different variables

# compile the coordinates
Coord = np.empty( shape = (ngrids,3) , dtype = np.float32 )
Coord[:,0] = data_list[:, parameter_list.index('VARIABLES=X [m]')]
Coord[:,1] = data_list[:, parameter_list.index('Y [m]')]
Coord[:,2] = data_list[:, parameter_list.index('Z [m]')]



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



#%% 4. Compile data of different ROL modes
# calculate the average concentration profiles of different ROL modes and plot them together

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

t = 10       #specify the time point, here extract the data on day 30
MP_NO = np.zeros((9,ntimepoint, 6), dtype = float)
MP_Homo = np.zeros((9, ntimepoint, 6), dtype = float)
MP_Het = np.zeros((9, ntimepoint, 6), dtype = float)
#MP_Het_30roots = np.zeros((9, ntimepoint, 6), dtype = float)
#MP_Het_50roots = np.zeros((9, ntimepoint, 6), dtype = float)
#MP_Het_70roots = np.zeros((9, ntimepoint, 6), dtype = float)

# Calculate the mean profiles
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
var_id = 1
 
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
    
plt.plot(MP_NO[3, :, var_id] * conv, '-', color ='#303030', label = 'no O2 release'.translate(sub))  #convert depth to cm
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




#%%  5. S cycling simulation analysis      
Data_S = Full_Data
MetF_S = MetF



#%%
Data_noS = Full_Data
MetF_noS = MetF



#%% Calculate and Plot the mean profiles of different O2 injection modes
# Calculate the mean profiles
t = 10       #specify the time point, here extract the data on day 30
MP_S = np.zeros((9, ntimepoint, 6), dtype = float)
MP_noS = np.zeros((9, ntimepoint, 6), dtype = float)

for t in range(0, ntimepoint):
    for var_id in range(0,6):
        #reshape the matrix of concentration to a 9*100 matrix, each row is the data of one soil layer containing 100 cells, and calculate the mean of all cells within the same layer    
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

plt.plot(MP_S[1:11, t, var_id] * conv, y, '-', color ='#0174DF', label = 'with S cycling')  #convert depth to cm
plt.plot(MP_noS[1:11, t, var_id] * conv, y, '--', color ='#0174DF', label = 'no S cycling')  #convert depth to cm



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
ConcCmpr = np.zeros((3,5), dtype = float)
ConcCmpr[0,:] = MP_S[3,10,1:6]       
ConcCmpr[1,:] = MP_noS[3,10,1:6]        

ConcCmpr[2,:] = (ConcCmpr[1,:] - ConcCmpr[0,:]) / ConcCmpr[0,:] * 100

#%% Plot CH4 emissions data
MetF = np.zeros((3,4), dtype = float)
MetF[0,:] = np.array(MetF_S)
MetF[1,:] = np.array(MetF_noS)

MetF[2,:] = (MetF[1,:] - MetF[0,:]) / MetF[0,:] * 100



#%% Bar plot for the CH4 fluxes
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



