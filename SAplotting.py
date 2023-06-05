# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 14:59:27 2023

@author: YZ60069
"""

#%% this file plot the results of sensitivity analysis
#%% Calculate the mean profiles
t = 30
MP_lo = np.zeros((9,6), dtype = float)  
MP_med = np.zeros((9,6), dtype = float)
MP_hi = np.zeros((9,6), dtype = float)


for i in range(0,6):
    MP_lo[:,i] = Data_lo[:,t,i].reshape(9,100).mean(axis = 1)  #calculate the mean concentration profile
    MP_med[:,i] = Data_med[:,t,i].reshape(9,100).mean(axis = 1)
    MP_hi[:,i] = Data_hi[:,t,i].reshape(9,100).mean(axis = 1)
    
#%% Plot the results

var_id = 1
if var_id == 1:        # convert the O2 unit to %saturation
    conv = 1/2.5e-4*100
else:                 # convert
    conv = 1e6
    
    
y = depths[1:8]*100   #convert depth values to cm

plt.rcParams.update({'font.size': 15})
fig, ax = plt.subplots()

subscript = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
superscript = str.maketrans("0123456789+-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻")

plt.plot(MP_lo[1:8, var_id] * conv, y, '-', color = '#24AEDB', label = 'x0.1')  #convert depth to cm
plt.plot(MP_med[1:8, var_id] * conv, y, '-', color = '#303030', label = 'x1')
plt.plot(MP_hi[1:8, var_id] * conv, y, '-', color = '#D02F5E', label = 'x10')


plt.legend(loc = 0)

if var_id == 1:
    xlab = 'O2 saturation (%)'
elif var_id == 2:
    xlab = 'CH4'.translate(subscript) + ' (μmol L-1)'.translate(superscript)
elif var_id == 3:
    xlab = 'DOC' + ' (μmol L-1)'.translate(superscript)
elif var_id == 4:
    xlab = 'SO4'.translate(subscript) + '2-(μmol L-1)'.translate(superscript)
elif var_id == 5:
    xlab = 'H2S(aq)'.translate(subscript) + ' (μmol L-1)'.translate(superscript)

plt.xlabel(xlab)
plt.ylabel('Depth(cm)')


#%% Compile the concentration at root layer depth and the CH4 fluxes
ConcCompiled = np.vstack((MP_lo[3,], MP_med[3,], MP_hi[3,]))
ConcCompiled = pd.DataFrame(ConcCompiled, index = ['Low','Medium','High'])

#%%
MetFCompiled = np.vstack((MetF_lo, MetF_med, MetF_hi))
MetFCompiled = pd.DataFrame(MetFCompiled, index = ['Low', 'Medium', 'High'])


#%% plot results of number of roots
plt.rcParams.update({'font.size': 15})
fig, ax = plt.subplots()

var_id = 5
var_str = Var_str[var_id]
t = 30

if var_id == 1:
    conv = 1/2.5e-4*100
else:
    conv = 1e6


# calculate and plot mean profiles
MeanProfs_homo = []
for z in range(0,nz):
    i_start = nx * ny * z
    i_end = nx * ny * (z + 1)
    temp_mean = np.mean(Data_homo[i_start:i_end, :, var_id] , axis = 0)
    MeanProfs_homo.append(temp_mean)

MeanProfs_homo = np.array(MeanProfs_homo) * conv
plt.plot(MeanProfs_homo[1:8,t], depths[1:8]*100, '-', label = 'Homogeneity')  #convert depth to cm



MeanProfs_15roots = []
for z in range(0,nz):
    i_start = nx * ny * z
    i_end = nx * ny * (z + 1)
    temp_mean = np.mean(Data_15roots[i_start:i_end, :, var_id] , axis = 0)
    MeanProfs_15roots.append(temp_mean)

MeanProfs_15roots = np.array(MeanProfs_15roots) * conv
plt.plot(MeanProfs_15roots[1:8,t], depths[1:8]*100, '-', label = '15 roots')


MeanProfs_30roots = []
for z in range(0,nz):
    i_start = nx * ny * z
    i_end = nx * ny * (z + 1)
    temp_mean = np.mean(Data_30roots[i_start:i_end, :, var_id] , axis = 0)
    MeanProfs_30roots.append(temp_mean)

MeanProfs_30roots = np.array(MeanProfs_30roots) * conv
plt.plot(MeanProfs_30roots[1:8,t], depths[1:8]*100, '-', label = '30 roots')



MeanProfs_45roots = []
for z in range(0,nz):
    i_start = nx * ny * z
    i_end = nx * ny * (z + 1)
    temp_mean = np.mean(Data_45roots[i_start:i_end, :, var_id] , axis = 0)
    MeanProfs_45roots.append(temp_mean)

MeanProfs_45roots = np.array(MeanProfs_45roots) * conv
plt.plot(MeanProfs_45roots[1:8,t], depths[1:8]*100, '-', label = '45 roots')


MeanProfs_60roots = []
for z in range(0,nz):
    i_start = nx * ny * z
    i_end = nx * ny * (z + 1)
    temp_mean = np.mean(Data_60roots[i_start:i_end, :, var_id] , axis = 0)
    MeanProfs_60roots.append(temp_mean)

MeanProfs_60roots = np.array(MeanProfs_60roots) * conv
plt.plot(MeanProfs_60roots[1:8,t], depths[1:8]*100, '-', label = '60 roots')


MeanProfs_75roots = []
for z in range(0,nz):
    i_start = nx * ny * z
    i_end = nx * ny * (z + 1)
    temp_mean = np.mean(Data_75roots[i_start:i_end, :, var_id] , axis = 0)
    MeanProfs_75roots.append(temp_mean)

MeanProfs_75roots = np.array(MeanProfs_75roots) * conv
plt.plot(MeanProfs_75roots[1:8,t], depths[1:8]*100, '-', label = '75 roots')


MeanProfs_90roots = []
for z in range(0,nz):
    i_start = nx * ny * z
    i_end = nx * ny * (z + 1)
    temp_mean = np.mean(Data_90roots[i_start:i_end, :, var_id] , axis = 0)
    MeanProfs_90roots.append(temp_mean)

MeanProfs_90roots = np.array(MeanProfs_90roots) * conv
plt.plot(MeanProfs_90roots[1:8,t], depths[1:8]*100, '-', label = '90 roots')


subscript = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
superscript = str.maketrans("0123456789+-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻")

#plt.legend(loc = 0)
#plt.xlabel(var_str[6:len(var_str)-4] + ' (μM)')

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



#%% calculate the percentage difference between each rootnumber scenario and the homogeneity
t = 30
z = 3 #the depth layer where O2 was injected
AA = MeanProfs_homo[z, t]

Diff = []
Diff.append( (MeanProfs_15roots[z, t] - AA)/AA *100 )
Diff.append( (MeanProfs_30roots[z, t] - AA)/AA *100 )
Diff.append( (MeanProfs_45roots[z, t] - AA)/AA *100 )
Diff.append( (MeanProfs_60roots[z, t] - AA)/AA *100 )
Diff.append( (MeanProfs_75roots[z, t] - AA)/AA *100 )
Diff.append( (MeanProfs_90roots[z, t] - AA)/AA *100 )




plt.plot([15, 30, 45, 60, 75, 90], Diff, 'o-')


plt.xlabel('Number of Roots')
plt.ylabel('%Difference')
plt.title('%Difference in H2S concentration')

#%% Difference in CH4 emissions
CH4_emissions = np.array( [9.4279, 9.5094, 9.6557, 9.6859, 9.6834, 9.6968])
BB = 9.608
Diff = (CH4_emissions - BB)/BB*100

plt.plot([15, 30, 45, 60, 75, 90], Diff, 'o-')
plt.xlabel('Number of Roots')
plt.ylabel('%Difference')
plt.title('%Difference in CH4 emissions')