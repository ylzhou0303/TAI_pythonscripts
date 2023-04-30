# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 14:59:27 2023

@author: YZ60069
"""

#%% this file plot the results of sensitivity analysis
plt.rcParams.update({'font.size': 15})
fig, ax = plt.subplots()

var_id = 5
var_str = Var_str[var_id]
t = 30

if var_id == 1:
    conv = 1/2.5e-4*100
else:
    conv = 1e6
# for i in range(0, ncols):D
#     conc = Data_oxyhet[i:ngrids:ncols, t, var_id] * conv
#     plt.plot(conc, depths,color = 'skyblue', linestyle = '-')

# plt.plot(conc,depths, color = 'skyblue', linestyle = '-', label = 'O2 het, each column')

# calculate and plot mean profiles
MeanProfs = []
for z in range(0,nz):
    i_start = nx * ny * z
    i_end = nx * ny * (z + 1)
    temp_mean = np.mean(Data_lo[i_start:i_end, :, var_id] , axis = 0)
    MeanProfs.append(temp_mean)

MeanProfs = np.array(MeanProfs) * conv
plt.plot(MeanProfs[1:8,t], depths[1:8]*100, '-', color = '#24AEDB', label = '0.1x')



MeanProfs = []
for z in range(0,nz):
    i_start = nx * ny * z
    i_end = nx * ny * (z + 1)
    temp_mean = np.mean(Data_m[i_start:i_end, :, var_id] , axis = 0)
    MeanProfs.append(temp_mean)

MeanProfs = np.array(MeanProfs) * conv
plt.plot(MeanProfs[1:8,t], depths[1:8]*100, '-', color ='#303030', label = '1x')  #convert depth to cm




MeanProfs = []
for z in range(0,nz):
    i_start = nx * ny * z
    i_end = nx * ny * (z + 1)
    temp_mean = np.mean(Data_hi[i_start:i_end, :, var_id] , axis = 0)
    MeanProfs.append(temp_mean)

MeanProfs = np.array(MeanProfs) * conv
plt.plot(MeanProfs[1:8,t], depths[1:8]*100, '-', color = '#D02F5E', label = '10x')





subscript = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
superscript = str.maketrans("0123456789+-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻")

plt.legend(loc = 0)
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





#%% plot results of inhibition term
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
MeanProfs = []
for z in range(0,nz):
    i_start = nx * ny * z
    i_end = nx * ny * (z + 1)
    temp_mean = np.mean(Data_lo[i_start:i_end, :, var_id] , axis = 0)
    MeanProfs.append(temp_mean)

MeanProfs = np.array(MeanProfs) * conv
plt.plot(MeanProfs[1:8,t], depths[1:8]*100, '-', color ='#303030', label = 'K_I = 2.5e-5')  #convert depth to cm



MeanProfs = []
for z in range(0,nz):
    i_start = nx * ny * z
    i_end = nx * ny * (z + 1)
    temp_mean = np.mean(Data_m[i_start:i_end, :, var_id] , axis = 0)
    MeanProfs.append(temp_mean)

MeanProfs = np.array(MeanProfs) * conv
plt.plot(MeanProfs[1:8,t], depths[1:8]*100, '-', color = '#24AEDB', label = 'K_I = 2.5e-4')


MeanProfs = []
for z in range(0,nz):
    i_start = nx * ny * z
    i_end = nx * ny * (z + 1)
    temp_mean = np.mean(Data_hi[i_start:i_end, :, var_id] , axis = 0)
    MeanProfs.append(temp_mean)

MeanProfs = np.array(MeanProfs) * conv
plt.plot(MeanProfs[1:8,t], depths[1:8]*100, '-', color = '#D02F5E', label = 'K_I = 2.5e-3')



MeanProfs = []
for z in range(0,nz):
    i_start = nx * ny * z
    i_end = nx * ny * (z + 1)
    temp_mean = np.mean(Data_noinhibition[i_start:i_end, :, var_id] , axis = 0)
    MeanProfs.append(temp_mean)

MeanProfs = np.array(MeanProfs) * conv
plt.plot(MeanProfs[1:8,t], depths[1:8]*100, '-', color = 'g', label = 'No inhibition')

subscript = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
superscript = str.maketrans("0123456789+-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻")

plt.legend(loc = 0)
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