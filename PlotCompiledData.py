# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 17:55:23 2023

@author: YZ60069
"""

#%% This file analyzes the difference in the effect of heterogeneity under different system nonlinearity

Conc_s = ConcCmpr
MetF_s = FluxCmpr

#%%
Conc_m = ConcCmpr
MetF_m = FluxCmpr

#%%
Conc_l = ConcCmpr
MetF_l = FluxCmpr



#%% Varying K


#%%
MetF = np.zeros((3,3), dtype = float)
MetF[:,0] = MetF_s[0:3,0]
MetF[:,1] = MetF_m[0:3,0]
MetF[:,2] = MetF_l[0:3,0]

Conc = np.zeros((3,6), dtype = float)
Conc[0, 0:5] = Conc_s[3,]
Conc[1, 0:5] = Conc_m[3,]
Conc[2, 0:5] = Conc_l[3,]

Conc[:,5] = ((MetF[2,:] - MetF[1,:]) / MetF[1,:]).transpose() *100      # difference in total CH4 flux


Conc_abs = abs(Conc)

fig, ax = plt.subplots()

x = np.arange(0,6)
y = np.arange(0,3)
X, Y = np.meshgrid(x,y)
plt.scatter(X, Y , s = Conc_abs, c = '#24AEDB', alpha = 0.7)

sub = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
sup = str.maketrans("0123456789+-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻")

xlabels = ['[O2]'.translate(sub), '[CH4]'.translate(sub), '[DOC]', '[SO4'.translate(sub)+'2-]'.translate(sup), '[H2S]'.translate(sub),'CH4 flux'.translate(sub)]
ylabels = ['3%', '15%', '40%']
plt.xticks(x, xlabels)
plt.yticks(y, ylabels)
plt.xlim(-0.5,5.5)
plt.ylim(-0.5,2.5)
plt.ylabel('K(O2) '.translate(sub) + '%Air saturation')
#plt.title('%Difference in [c], Homo vs Het')

#%%
MetF = np.zeros((3,3), dtype = float)
MetF[:,0] = MetF_s[:,0]
MetF[:,1] = MetF_m[:,0]
MetF[:,2] = MetF_l[:,0]

#%%
import matplotlib.pyplot as plt
import numpy as np


# Generate some sample data
group_names = ['noROL', 'ROL_Homo', 'ROL_Het']
bar_labels = ['3%', '15%', '40%']

# Set the bar width and positions
bar_height = 0.2
x = np.arange(len(bar_labels))

colors = ['#303030', '#24AEDB', '#D02F5E']
# Plot the bars for each group
for i in range(1,3):
    plt.barh(y + i * bar_height, MetF[i], height=bar_height, label=group_names[i], color = colors[i])

# Set the x-axis tick labels
plt.yticks(y + (len(group_names) - 1) * bar_height / 2, bar_labels)

# Set the axis labels and title
plt.ylabel('K(O2) '.translate(sub) + '%Air saturation')
plt.xlabel('CH4'.translate(sub)+' emissions (mmol m-2 d-1)'.translate(sup))
plt.xlim(0,15)
plt.ylim(-0.5,4)
plt.xticks([0, 2, 4, 6, 8, 10])

# Add a legend
plt.legend(loc='upper right')

# Show the plot
plt.show()



#%% Varying O2 injection rate
#%%
MetF = np.zeros((3,3), dtype = float)
MetF[:,0] = MetF_s[0:3,0]
MetF[:,1] = MetF_m[0:3,0]
MetF[:,2] = MetF_l[0:3,0]


Conc = np.zeros((3,6), dtype = float)
Conc[0, 0:5] = Conc_s[3,]
Conc[1, 0:5] = Conc_m[3,]
Conc[2, 0:5] = Conc_l[3,]
Conc[:,5] = ((MetF[2,:] - MetF[1,:]) / MetF[1,:]).transpose() *100

Conc_abs = abs(Conc)

fig, ax = plt.subplots()

x = np.arange(0,6)
y = np.arange(0,3)
X, Y = np.meshgrid(x,y)
plt.scatter(X, Y , s = Conc_abs, c = '#24AEDB', alpha = 0.7)


sub = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
sup = str.maketrans("0123456789+-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻")

xlabels = ['[O2]'.translate(sub), '[CH4]'.translate(sub), 'DOC', '[SO4'.translate(sub)+'2-]'.translate(sup), '[H2S]'.translate(sub), 'CH4 flux'.translate(sub)]
ylabels = ['60', '72', '100']
plt.xticks(x, xlabels)
plt.yticks(y, ylabels)
plt.xlim(-0.5,5.5)
plt.ylim(-0.5,2.5)
plt.ylabel('O2 '.translate(sub) + 'injection rate \n(mmol m-2 d-1)'.translate(sup))
#plt.title('%Difference in [c], Homo vs Het')



#%% To make legends for the circle sizes, I need to plot the legend as a figure
plt.scatter(1, 1 , s = 2, c = '#24AEDB', alpha = 0.7)

#%%
MetF = np.zeros((3,3), dtype = float)
MetF[:,0] = MetF_s[:,0]
MetF[:,1] = MetF_m[:,0]
MetF[:,2] = MetF_s[:,0]

#%%
import matplotlib.pyplot as plt
import numpy as np


# Generate some sample data
group_names = ['NO', 'Homo', 'Het']
bar_labels = ['60', '72', '100']

# Set the bar width and positions
bar_height = 0.2
x = np.arange(len(bar_labels))

colors = ['#303030', '#24AEDB', '#D02F5E']
# Plot the bars for each group
for i in range(1,len(group_names)):
    plt.barh(y + i * bar_height, MetF[i], height=bar_height, label=group_names[i], color = colors[i])

# Set the x-axis tick labels
plt.yticks(y + (len(group_names) - 1) * bar_height / 2, bar_labels)

# Set the axis labels and title
plt.ylabel('O2 '.translate(sub) + 'injection rate\n(mmol m-2 d-1)'.translate(sup))
plt.xlabel('CH4'.translate(sub)+' emissions (mmol m-2 d-1)'.translate(sup))
plt.xlim(0,15)
plt.ylim(-0.5,3)
plt.xticks([0, 2, 4, 6, 8, 10])

# Add a legend
plt.legend(loc='lower right')

# Show the plot
plt.show()






#%% Varying number of roots (varing the extent of heterogeneity)

# Import data
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



#%%
Data_Het_50roots = Full_Data
MetF_Het_50roots = MetF



#%%
Data_Het_70roots = Full_Data
MetF_Het_70roots = MetF



#%% Calculate and Plot the mean profiles of different O2 injection modes
# Calculate the mean profiles
t = 10       #specify the time point, here extract the data on day 30
MP_NO = np.zeros((9,ntimepoint, 6), dtype = float)
MP_Homo = np.zeros((9, ntimepoint, 6), dtype = float)
MP_Het = np.zeros((9, ntimepoint, 6), dtype = float)
MP_Het_30roots = np.zeros((9, ntimepoint, 6), dtype = float)
MP_Het_50roots = np.zeros((9, ntimepoint, 6), dtype = float)
MP_Het_70roots = np.zeros((9, ntimepoint, 6), dtype = float)


for t in range(0, ntimepoint):
    for var_id in range(0,6):
        #reshape the matrix of concentration to a 9*100 matrix, each row is the data of one soil layer containing 100 cells, and calculate the mean of all cells within the same layer
        MP_NO[:, t, var_id] = Data_NO[:, t, var_id].reshape(9,100).mean(axis = 1)      
        MP_Homo[:, t, var_id] = Data_Homo[:, t, var_id].reshape(9,100).mean(axis = 1)
        MP_Het[:,t, var_id] = Data_Het[:, t, var_id].reshape(9,100).mean(axis = 1)
        MP_Het_30roots[:,t, var_id] = Data_Het_30roots[:, t, var_id].reshape(9,100).mean(axis = 1)
        MP_Het_50roots[:,t, var_id] = Data_Het_50roots[:, t, var_id].reshape(9,100).mean(axis = 1)
        MP_Het_70roots[:,t, var_id] = Data_Het_70roots[:, t, var_id].reshape(9,100).mean(axis = 1)
        
#%% Compile the average concentration and fluxes        

ConcCmpr = np.zeros((10,5), dtype = float)
ConcCmpr[0,0:5] = MP_NO[3,t,1:6]
ConcCmpr[1,0:5] = MP_Homo[3,t,1:6]
ConcCmpr[2,0:5] = MP_Het[3,t,1:6]
ConcCmpr[3,0:5] = MP_Het_30roots[3,t,1:6]
ConcCmpr[4,0:5] = MP_Het_50roots[3,t,1:6]
ConcCmpr[5,0:5] = MP_Het_70roots[3,t,1:6]

ConcCmpr[6,0:5] = (MP_Het[3,t,1:6] - MP_Homo[3,t,1:6])/MP_Homo[3,t,1:6]*100
ConcCmpr[7,0:5] = (MP_Het_30roots[3,t,1:6] - MP_Homo[3,t,1:6])/MP_Homo[3,t,1:6]*100
ConcCmpr[8,0:5] = (MP_Het_50roots[3,t,1:6] - MP_Homo[3,t,1:6])/MP_Homo[3,t,1:6]*100
ConcCmpr[9,0:5] = (MP_Het_70roots[3,t,1:6] - MP_Homo[3,t,1:6])/MP_Homo[3,t,1:6]*100

#%%

Conc = np.zeros((4,6), dtype = float)
Conc[0:5, 0:5] = ConcCmpr[6:10,]

kk = np.float(MetF_Homo['Total Flux'])
Conc[0,5] = (np.float(MetF_Het['Total Flux']) - kk) / kk *100
Conc[1,5] = (np.float(MetF_Het_30roots['Total Flux']) - kk) / kk *100
Conc[2,5] = (np.float(MetF_Het_50roots['Total Flux']) - kk) / kk *100
Conc[3,5] = (np.float(MetF_Het_70roots['Total Flux']) - kk) / kk *100

Conc_abs = abs(Conc)

fig, ax = plt.subplots()

x = np.arange(0,6)
y = np.arange(0,4)
X, Y = np.meshgrid(x,y)
plt.scatter(X, Y , s = Conc_abs, c = '#24AEDB', alpha = 0.7)

sub = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
sup = str.maketrans("0123456789+-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻")

xlabels = ['[O2]'.translate(sub), '[CH4]'.translate(sub), '[DOC]', '[SO4'.translate(sub)+'2-]'.translate(sup), '[H2S]'.translate(sub),'CH4 flux'.translate(sub)]
ylabels = ['15', '30', '50','70']
plt.xticks(x, xlabels)
plt.yticks(y, ylabels)
plt.xlim(-0.5,5.5)
plt.ylim(-0.5,3.5)
plt.ylabel('Number of Roots')
#plt.title('%Difference in [c], Homo vs Het')