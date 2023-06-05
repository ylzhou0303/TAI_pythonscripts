# -*- coding: utf-8 -*-
"""
Created on Tue May 16 10:50:13 2023

@author: YZ60069
"""

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
import os
os.chdir('C:/Users/yz60069/TAI/TAI_fresh')
#os.chdir('C:/MBL/Research/PFLOTRAN DATA/pflotran outputs/OxyHet/Marsh Interior/All inundated/ root')
import scipy.io
#mat = scipy.io.loadmat('WaterTableData.mat')
#%% specify the pflotran output structure
nx = 1
ny = 1
nz = 7
ngrids = nx * ny * nz
ntimepoint = 6


#%% import data
nvars = 3;  #number of variables that I want to investigate
Full_Data = np.empty( shape = (ngrids,ntimepoint,nvars), dtype = np.float32)
Var_str = {0: 'Liquid Saturation', 1: 'Total O2(aq) [M]', 2: 'Total CH4(aq) [M]'}
# the dictionary in which the variable ID is coupled with the variable name


for var_id in range(0,nvars):
    
    var_str = Var_str[var_id]
    # Read in data from PFLOTRAN
    Data_Page = np.empty( shape = (ngrids,ntimepoint), dtype = np.float32)
    w = 0
    
    
    
    for i in range(0,ntimepoint):
       
        if i < 10:
           file_name = 'testwetlands-00' + str(i) + '.tec'
        elif (i == 10 or i > 10) and i < 100:
           file_name = 'testwetlands-0' + str(i) + '.tec'
        elif (i == 100 or i > 100) and i < 1000:
           file_name = 'testwetlands-' + str(i) + '.tec'
        
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
var_id = 0 #specify which variable to plot
var_str = Var_str[var_id]
interval = nx * ny 
depths = Coord[0: ngrids :interval,2] - 1  #minus the depth of the soil profile

if var_id == 1:
    conv = 1/2.5e-4*100
elif var_id == 0:
    conv = 1
else:
    conv = 1e6

plt.rcParams.update({'font.size': 15})

for i in range(0,6,1):
    conc = Full_Data[0 : ngrids : interval, i, var_id] * conv
    plt.plot(conc, depths, '-o')
    
plt.ylabel('Soil Depth (m)')
plt.xlabel(var_str[0:len(var_str)-4] + ' (uM)')
#plt.xlim(-1e-5,1e-3)
#plt.xlabel('water saturation')
#plt.xlabel('O2 sat%')
#plt.xticks(np.arange(0, 2e-4, step = 5e-5))   
#plt.xticks(np.arange(5.8e-4, 6.2e-4, step = 1e-5)) 
#plt.ylim(-1,-0.05)
#plt.xlim([1990,2010])

#%%
plt.plot(no_liquidloss[0:ngrids:interval, 5, 1], depths, 'r-')
plt.plot(with_liquidloss[0:ngrids:interval, 5, 1], depths, 'b-')

#%% plot the time series of the variable
var_id = 2
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
filename = 'OxyHet.pickle'
with open('C:/MBL/Research/PFLOTRAN DATA/pflotran outputs/OxyHet/Marsh Interior/' + filename, 'wb') as handle:
    pickle.dump([Full_Data, depths, variable_list_cor, mass_bal], handle)


#%% import field data
import pandas
FieldData = pandas.read_csv('C:\MBL\Research\PFLOTRAN DATA\pflotran outputs\OxyHet\FieldData_Jul_MI.csv')



#%% import the processed pflotran output
import pickle 
filename = 'growing_season.pickle'
with open('C:/MBL/Research/PFLOTRAN DATA/pflotran outputs/OxyHet/Creek Bank/' + filename, 'rb') as handle:
    Growing_Season_CB = pickle.load(handle)
       

#%% Calculate and Plot the mean profiles of different O2 injection modes
# Calculate the mean profiles
t = 30        #specify the time point, here extract the data on day 30
MP_NO = np.zeros((9,6), dtype = float)
MP_Homo = np.zeros((9,6), dtype = float)
MP_Het = np.zeros((9,6), dtype = float)

for var_id in range(0,6):
    #reshape the matrix of concentration to a 9*100 matrix, each row is the data of one soil layer containing 100 cells, and calculate the mean of all cells within the same layer
    MP_NO[:,var_id] = Data_NO[:, t, var_id].reshape(9,100).mean(axis = 1)      
    MP_Homo[:,var_id] = Data_Homo[:, t, var_id].reshape(9,100).mean(axis = 1)
    MP_Het[:,var_id] = Data_Het[:, t, var_id].reshape(9,100).mean(axis = 1)

#%% Plot the profiles from different set up
plt.rcParams.update({'font.size': 15})
fig, ax = plt.subplots()

var_id = 5   #choose which variable to plot
if var_id == 1:
    conv = 1/2.5e-4*100
else:
    conv = 1e6

y = depths[1:8]*100   #convert depth values to cm

plt.plot(MP_NO[1:8, var_id] * conv, y, '-', color ='#303030', label = 'no O2 release')  #convert depth to cm
plt.plot(MP_Homo[1:8, var_id] * conv, y, '-', color = '#24AEDB', label = 'Homogeneity')
plt.plot(MP_Het[1:8, var_id] * conv, y, '-', color = '#D02F5E', label = 'Heterogeneity')

subscript = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
superscript = str.maketrans("0123456789+-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻")

if var_id == 1:
    xlab = 'O2 saturation (%)'
elif var_id == 2:
    xlab = 'CH4(μM)'.translate(subscript)
elif var_id == 3:
    xlab = 'DOC (μM)'
elif var_id == 4:
    xlab = 'SO4'.translate(subscript) + '2-(μM)'.translate(superscript)
elif var_id == 5:
    xlab = 'H2S(aq) (μM)'.translate(subscript)

plt.xlabel(xlab)
plt.ylabel('Depth(cm)')
plt.legend(loc = 0)

#%% compile the species concentration at the root layer
ConcCmpr = np.zeros((3,6), dtype = float)
ConcCmpr[0,] = MP_NO[3,]
ConcCmpr[1,] = MP_Homo[3,]
ConcCmpr[2,] = MP_Het[3,]

#%% Compile the CH4 fluxes
FluxCmpr = np.array([MetF_NO, MetF_Homo, MetF_Het]).reshape(3,4)
FluxCmpr_df = pd.DataFrame(FluxCmpr)
FluxCmpr_df.columns = ['Total Flux', 'Diffusion', 'Plant-mediated', 'Ebullition']
FluxCmpr_df.index = ['NO', 'Homo', 'Het']



#%% S cycling
# Calculate and plot the mean profiles under different SO4-- concentrations

t = 30        #specify the time point, here extract the data on day 30
MP_noS = np.zeros((9,6), dtype = float)
MP_lowS = np.zeros((9,6), dtype = float)
MP_medS = np.zeros((9,6), dtype = float)
MP_highS = np.zeros((9,6), dtype = float)

for var_id in range(0,6):
    #reshape the matrix of concentration to a 9*100 matrix, each row is the data of one soil layer containing 100 cells, and calculate the mean of all cells within the same layer
    MP_noS[:,var_id] = noS[:, t, var_id].reshape(9,100).mean(axis = 1)      
    MP_lowS[:,var_id] = lowS[:, t, var_id].reshape(9,100).mean(axis = 1)
    MP_medS[:,var_id] = medS[:, t, var_id].reshape(9,100).mean(axis = 1)
    MP_highS[:,var_id] = highS[:, t, var_id].reshape(9,100).mean(axis = 1)


#%%

plt.rcParams.update({'font.size': 15})
fig, ax = plt.subplots()

var_id = 5   #choose which variable to plot
if var_id == 1:
    conv = 1/2.5e-4*100
else:
    conv = 1e6

y = depths[1:8]*100   #convert depth values to cm

plt.plot(MP_noS[1:8, var_id] * conv, y, '-', color ='#303030', label = 'no S')  #convert depth to cm
plt.plot(MP_lowS[1:8, var_id] * conv, y, '-', color = '#24AEDB', label = 'low S')
plt.plot(MP_medS[1:8, var_id] * conv, y, '-', color = 'y', label = 'medium S')
plt.plot(MP_highS[1:8, var_id] * conv, y, '-', color = '#D02F5E', label = 'high S')

subscript = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
superscript = str.maketrans("0123456789+-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻")

if var_id == 1:
    xlab = 'O2 saturation (%)'
elif var_id == 2:
    xlab = 'CH4(μM)'.translate(subscript)
elif var_id == 3:
    xlab = 'DOC (μM)'
elif var_id == 4:
    xlab = 'SO4'.translate(subscript) + '2-(μM)'.translate(superscript)
elif var_id == 5:
    xlab = 'H2S(aq) (μM)'.translate(subscript)

plt.xlabel(xlab)
plt.ylabel('Depth(cm)')
plt.legend(loc = 0)




#%% compile the species concentration at the root layer
ConcCmpr = np.zeros((4,6), dtype = float)
ConcCmpr[0,] = MP_noS[3,]
ConcCmpr[1,] = MP_lowS[3,]
ConcCmpr[2,] = MP_medS[3,]
ConcCmpr[3,] = MP_highS[3,]
ConcCmpr = pd.DataFrame(ConcCmpr)
ConcCmpr.index = ['noS', 'lowS', 'medS (standard run)', 'highS']



#%% Compile the CH4 fluxes
FluxCmpr = np.array([MetFlux_noS, MetFlux_lowS, MetFlux_medS, MetFlux_highS]).reshape(4,3)
FluxCmpr_df = pd.DataFrame(FluxCmpr)
FluxCmpr_df.columns = ['Total Flux', 'Diffusion', 'Ebullition']
FluxCmpr_df.index = ['noS', 'lowS', 'medS', 'highS']







#%% plot the S cycling vs no S cycling
plt.rcParams.update({'font.size': 15})
fig, ax = plt.subplots()

var_id = 2
var_str = Var_str[var_id]
t = 30

if var_id == 1:
    conv = 1/2.5e-4*100
else:
    conv = 1e6


# calculate and plot mean profiles
MeanProfs = []
for z in range(0,nz):
    i_start = nx * ny * z
    i_end = nx * ny * (z + 1)
    temp_mean = np.mean(Data_noS[i_start:i_end, :, var_id] , axis = 0)
    MeanProfs.append(temp_mean)

MeanProfs = np.array(MeanProfs) * conv
plt.plot(MeanProfs[1:8,t], depths[1:8]*100, '-', color ='#303030', label = 'no S cycling')  #convert depth to cm



MeanProfs = []
for z in range(0,nz):
    i_start = nx * ny * z
    i_end = nx * ny * (z + 1)
    temp_mean = np.mean(Data_S[i_start:i_end, :, var_id] , axis = 0)
    MeanProfs.append(temp_mean)

MeanProfs = np.array(MeanProfs) * conv
plt.plot(MeanProfs[1:8,t], depths[1:8]*100, '-', color = '#24AEDB', label = 'with S cycling')

subscript = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
superscript = str.maketrans("0123456789+-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻")
plt.xlabel('CH4(μM)'.translate(subscript))
plt.ylabel('Depth(cm)')
plt.legend(loc = 0)



#%% save plot
fn = r'C:\MBL\Research\Oxy Het MS\Figures\MeanProfs\oxygen.eps'
fig.savefig(fn, format='eps')

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
